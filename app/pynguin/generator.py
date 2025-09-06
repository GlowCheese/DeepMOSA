from __future__ import annotations

import sys
import json
import enum
import inspect
import datetime
import threading
import importlib
import functools

import environ
from pathlib import Path
from grappa import should
from vendor.custom_logger import getLogger
from typing import TYPE_CHECKING, Callable, cast

from pynguin.config import (
    Algorithm,
    CoverageMetric,
    MutationStrategy,
    StatisticsBackend,
    AssertionGenerator
)
from pynguin.config.main import ExportStrategy
from pynguin.testcase import export
from pynguin.execution import (
    ExecutionTracer,
    TestCaseExecutor,
    AssertionExecutionObserver
)
from pynguin.constants import (
    EmptyConstantProvider,
    RestrictedConstantPool,
    DynamicConstantProvider,
    collect_static_constants,
    DelegatingConstantProvider
)

from pynguin.globl import Globl
from pynguin.utils import randomness
from pynguin.setup import ModuleTestCluster
from pynguin.runtimevar import RuntimeVariable
from pynguin.setup.testclustergenerator import generate_test_cluster
from pynguin.instrumentation import install_import_hook
from pynguin.instrumentation import InstrumentationFinder
from pynguin.utils.exceptions import ConfigurationException
from pynguin.ga.testcasechromosome import TestCaseChromosome
from pynguin.slicer.statementslicingobserver import StatementSlicingObserver

import pynguin.assertion.assertiongenerator as ag
import pynguin.assertion.mutation_analysis.mutators as mu
import pynguin.assertion.mutation_analysis.operators as mo
import pynguin.assertion.mutation_analysis.strategies as ms
from pynguin.assertion.mutation_analysis.transformer import ParentNodeTransformer

from pynguin.report import (
    CoverageReportJson,
    get_coverage_report,
    render_coverage_report,
    render_xml_coverage_report
)

import pynguin.ga.postprocess as pp
import pynguin.ga.computations as ff
import pynguin.ga.chromosome as chrom
import pynguin.ga.chromosomevisitor as cv
import pynguin.ga.testsuitechromosome as tsc
import pynguin.ga.generationalgorithmfactory as gaf
from pynguin.statistics.stats import StatisticsTracker
from pynguin.deepmosa import DeepMOSALanguageModel
from pynguin.codamosa.model import CodaMOSALanguageModel
from pynguin.algo.generationalgorithm import GenerationAlgorithm
from pynguin.deepmosa import DeepMOSASeeding
from pynguin.codamosa.llmseeding import CodaMOSASeeding


if TYPE_CHECKING:
    from pynguin.testcase import testcase as tc
    from pynguin.seeding import InitialPopulationProvider
    from pynguin.assertion.mutation_analysis.operators.base import MutationOperator


_strategies: dict[MutationStrategy, Callable[[int], ms.HOMStrategy]] = {
    MutationStrategy.FIRST_TO_LAST: ms.FirstToLastHOMStrategy,
    MutationStrategy.BETWEEN_OPERATORS: ms.BetweenOperatorsHOMStrategy,
    MutationStrategy.RANDOM: ms.RandomHOMStrategy,
    MutationStrategy.EACH_CHOICE: ms.EachChoiceHOMStrategy,
}

logger = getLogger(__name__)


@enum.unique
class ReturnCode(enum.IntEnum):
    """Return codes for Pynguin to signal result."""

    OK = 0
    """Symbolises that the execution ended as expected."""

    SETUP_FAILED = 1
    """Symbolises that the execution failed in the setup phase."""

    NO_TESTS_GENERATED = 2
    """Symbolises that no test could be generated."""

    NO_PREDICATE_FOUND = 3
    """Symbolises that there's no predicate under test to cover."""


def _prepare_run_pynguin(fun):
    """
    Prepare everything before main function and do cleanup at the end.
    """
    @functools.wraps(fun)
    async def f(*args, **kwargs):
        """ SETUP SEED """
        logger.info("Using seed %d", Globl.seed)
        randomness.set_seed(Globl.seed)


        """ SETUP PATH
        Add project_path to sys.path, which allows
        Python to load (import) modules in this project. """
        logger.info('Setting up path for %s', Globl.project_path)
        try:
            sys.path.insert(0, Globl.project_path)

            """ SETUP CONSTANT SEEDING
            Collect constants from SUT, if enabled. """
            Globl.constant_provider = EmptyConstantProvider()
            if Globl.seeding_conf.constant_seeding:
                logger.info("Collecting static constants from module under test")
                constant_pool = collect_static_constants(Globl.project_path)
                if len(constant_pool) == 0:
                    logger.info("No constants found")
                else:
                    logger.info("Constants found: %s", len(constant_pool))
                    # Probability of 1.0 -> if a value is requested and available -> return it.
                    Globl.constant_provider = DelegatingConstantProvider(
                        constant_pool, Globl.constant_provider, 1.0
                    )

                if Globl.seeding_conf.dynamic_constant_seeding:
                    logger.info("Setting up runtime collection of constants")
                    Globl.dynamic_constant_provider = DynamicConstantProvider(
                        RestrictedConstantPool(
                            max_size=Globl.seeding_conf.max_dynamic_pool_size
                        ),
                        Globl.constant_provider,
                        Globl.seeding_conf.seeded_dynamic_values_reuse_probability,
                        Globl.seeding_conf.max_dynamic_length
                    )
                    Globl.constant_provider = Globl.dynamic_constant_provider


            """ SETUP REPORT DIR
            Report dir only needs to be created
            when statistics or coverage report is enabled.
            """
            if (
                Globl.statistics_conf.create_coverage_report
                or Globl.statistics_conf.statistics_backend != StatisticsBackend.NONE
            ):
                report_dir = Path(Globl.report_dir).absolute()
                try:
                    report_dir.mkdir(parents=True, exist_ok=True)
                except (OSError, FileNotFoundError):
                    logger.exception(f"Cannot create report dir {Globl.report_dir}")
                    return ReturnCode.SETUP_FAILED
                

            """ SETUP IMPORT HOOK """
            logger.debug('Setting up instrument for %s', Globl.module_name)

            tracer = ExecutionTracer()
            
            coverage_metrics = set(Globl.coverage_metrics)
            install_import_hook(
                Globl.module_name, tracer,
                coverage_metrics,
                Globl.dynamic_constant_provider
            )


            """ LOAD SUT
            Use importlib.import_module to load it (for later test_cluster). """
            try:
                # We need to set the current thread ident so the import trace is recorded.
                tracer.current_thread_identifier = threading.current_thread().ident
                if Globl.module_name in sys.modules:
                    importlib.reload(sys.modules[Globl.module_name])
                else:
                    importlib.import_module(Globl.module_name)
                
            except ImportError:
                # A module could not be imported because some dependencies
                # are missing or it is malformed
                logger.exception("Failed to load SUT")
                return ReturnCode.SETUP_FAILED
            

            """ SETUP TEST CLUSTER """
            # Analyzing the SUT should not cause any coverage.
            tracer.disable()
            test_cluster = Globl.test_cluster = generate_test_cluster()

            if test_cluster.num_accessible_objects_under_test() == 0:
                logger.error("SUT contains nothing we can test.")
                return ReturnCode.SETUP_FAILED

            tracer.enable()


            """ SETUP TEST CASE EXECUTOR """
            stop = Globl.stopping_conf
            executor = kwargs['executor'] = TestCaseExecutor(
                tracer,
                maximum_test_execution_timeout=stop.maximum_test_execution_timeout,
                test_execution_time_per_statement=stop.test_execution_time_per_statement
            )


            """ TRACK SUT STATISTICS """
            stat = Globl.statistics_tracker = StatisticsTracker()
            if len(tracer.get_subject_properties().existing_code_objects) == 0:
                return ReturnCode.NO_PREDICATE_FOUND

            stat.track_output_variable(
                RuntimeVariable.CodeObjects,
                len(tracer.get_subject_properties().existing_code_objects),
            )
            stat.track_output_variable(
                RuntimeVariable.Predicates,
                len(tracer.get_subject_properties().existing_predicates),
            )
            stat.track_output_variable(
                RuntimeVariable.Lines,
                len(tracer.get_subject_properties().existing_lines),
            )

            cyclomatic_complexities: list[int] = [
                code.original_cfg.cyclomatic_complexity
                for code in tracer.get_subject_properties().existing_code_objects.values()
            ]
            stat.track_output_variable(
                RuntimeVariable.McCabeCodeObject, json.dumps(cyclomatic_complexities)
            )

            test_cluster.track_statistics_values(stat.track_output_variable)
            if CoverageMetric.BRANCH in Globl.coverage_metrics:
                stat.track_output_variable(
                    RuntimeVariable.ImportBranchCoverage,
                    ff.compute_branch_coverage(tracer.import_trace, tracer.get_subject_properties()),
                )
            if CoverageMetric.LINE in Globl.coverage_metrics:
                stat.track_output_variable(
                    RuntimeVariable.ImportLineCoverage,
                    ff.compute_line_coverage(tracer.import_trace, tracer.get_subject_properties()),
                )
            

            """ SETUP LANGUAGE MODEL SEEDING """
            if Globl.algorithm in (Algorithm.CODAMOSA, Algorithm.DEEPMOSA):
                assert environ.OPENAI_API_KEY is not None, (
                    "Environment variable DEEPSEEK_API_KEY should be "
                    "set in order to generate test cases using "
                    f"{Globl.algorithm.value} strategy!"
                )
            
                if Globl.seeding_conf.large_language_model_mutation:
                    logger.error(
                        "Mutation currently unsupported --- the OpenAI edit models throttle."
                    )

                logger.info("Trying to set up the large language model.")

                if Globl.algorithm == Algorithm.CODAMOSA:
                    languagemodel = CodaMOSALanguageModel()
                    Globl.llmseeding = CodaMOSASeeding(test_cluster)
                else:
                    languagemodel = DeepMOSALanguageModel()
                    Globl.llmseeding = DeepMOSASeeding(test_cluster, Globl.conf.deepmosa.tau)

                with open(Globl.module_path, encoding="UTF-8") as module_file:
                    languagemodel.test_src = module_file.read()

                Globl.llmseeding.model = languagemodel
                Globl.llmseeding.executor = executor
                Globl.llmseeding.sample_with_replacement = (
                    Globl.seeding_conf.sample_with_replacement
                )


            """ MAIN RUN_PYNGUIN CALL """
            return await fun(*args, **kwargs)
        finally:
            sys.path.remove(Globl.project_path)

    return f


def _get_report_for_tcc(
    coverage_metrics: list[CoverageMetric],
    algorithm: Algorithm,
    executor: ExecutionTracer,
    test_case_chromosomes: list[TestCaseChromosome],
    original_test_suite: tsc.TestSuiteChromosome = None
):
    test_suite = original_test_suite \
        or algorithm.create_test_suite(test_case_chromosomes)

    chromosome_export_results = _export_chromosome(
        test_suite, Globl.test_cluster,
        Globl.output_conf.export_strategies
    )
    original_report = get_coverage_report(
        executor.tracer, test_suite, coverage_metrics
    )
    test_suite_coverage_report = CoverageReportJson.make_from(original_report)

    test_case_coverage_report = [
        CoverageReportJson.make_from(get_coverage_report(
            executor.tracer,
            algorithm.create_test_suite([individual]),
            coverage_metrics
        ))
        for individual in test_case_chromosomes
    ]

    return original_report, {
        "testcases": chromosome_export_results,
        "test_case_coverage_report": test_case_coverage_report,
        "test_suite_coverage_report": test_suite_coverage_report
    }


@_prepare_run_pynguin
async def run_pynguin(*, executor: TestCaseExecutor=None):
    logger.info('Start Pynguin Testing for %s...', Globl.module_name)

    if CoverageMetric.CHECKED in Globl.coverage_metrics:
        executor.add_observer(StatementSlicingObserver(executor.tracer))

    algorithm: GenerationAlgorithm = _instantiate_test_generation_strategy(executor)

    generation_result = algorithm.generate_tests()
    if inspect.isawaitable(generation_result):
        generation_result: tsc.TestSuiteChromosome = await generation_result

    if algorithm.resources_left():
        logger.info("Algorithm stopped before using all resources.")
    else:
        logger.info("Stopping condition reached")
        for stop in algorithm.stopping_conditions:
            logger.info("%s", stop)
    logger.info("Stop generating test cases")

    # Executions that happen after this point should not influence the
    # search statistics
    executor.clear_observers()

    stat = Globl.statistics_tracker
    _track_search_metrics(algorithm, generation_result)
    _remove_statements_after_exceptions(generation_result)
    _generate_assertions(executor, generation_result)
    tracked_metrics = _track_final_metrics(
        algorithm, executor, generation_result
    )

    original_report, report = _get_report_for_tcc(
        tracked_metrics, algorithm, executor,
        generation_result.test_case_chromosomes,
        generation_result
    )

    if Globl.statistics_conf.create_coverage_report:
        render_coverage_report(
            original_report,
            Path(Globl.report_dir) / "cov_report.html",
            datetime.datetime.now(),  # noqa: DTZ005
        )
        render_xml_coverage_report(
            original_report,
            Path(Globl.report_dir) / "cov_report.xml",
            datetime.datetime.now(),  # noqa: DTZ005
        )

    _collect_miscellaneous_statistics()
    try:
        assert stat.write_statistics()
        logger.info("Statistics were written successfully")
    except Exception as e:
        logger.exception("Failed to write statistics")

    return report


def _instantiate_test_generation_strategy(
    executor: TestCaseExecutor
) -> GenerationAlgorithm:
    factory = gaf.TestSuiteGenerationAlgorithmFactory(executor)
    return factory.get_search_algorithm()


def _track_search_metrics(
    algorithm: GenerationAlgorithm,
    generation_result: tsc.TestSuiteChromosome
) -> None:
    """Track multiple set coverage metrics of the generated test suites.

    This possibly re-executes the test suites.

    Args:
        algorithm: The test generation strategy
        generation_result:  The resulting chromosome of the generation strategy
        coverage_metrics: The selected coverage metrics to guide the search
    """
    stat = Globl.statistics_tracker

    for metric, runtime, fitness_type in [
        (
            CoverageMetric.LINE,
            RuntimeVariable.LineCoverage,
            ff.TestSuiteLineCoverageFunction,
        ),
        (
            CoverageMetric.BRANCH,
            RuntimeVariable.BranchCoverage,
            ff.TestSuiteBranchCoverageFunction,
        ),
        (
            CoverageMetric.CHECKED,
            RuntimeVariable.StatementCheckedCoverage,
            ff.TestSuiteStatementCheckedCoverageFunction,
        ),
    ]:
        if metric in Globl.coverage_metrics:
            coverage_function: ff.TestSuiteCoverageFunction = _get_coverage_ff_from_algorithm(
                algorithm, cast(type[ff.TestSuiteCoverageFunction], fitness_type)
            )
            stat.track_output_variable(
                runtime, generation_result.get_coverage_for(coverage_function)
            )
    # Write overall coverage data of result
    stat.current_individual(generation_result)


def _remove_statements_after_exceptions(generation_result):
    truncation = pp.ExceptionTruncation()
    generation_result.accept(truncation)
    if Globl.output_conf.post_process:
        unused_primitives_removal = pp.TestCasePostProcessor([pp.UnusedStatementsTestCaseVisitor()])
        generation_result.accept(unused_primitives_removal)
        # TODO(fk) add more postprocessing stuff.


def _generate_assertions(executor: TestCaseExecutor, generation_result):
    ass_gen = Globl.output_conf.assertion_generation
    if ass_gen != AssertionGenerator.NONE:
        logger.info("Start generating assertions")
        generator: cv.ChromosomeVisitor
        if ass_gen == AssertionGenerator.MUTATION_ANALYSIS:
            generator = _setup_mutation_analysis_assertion_generator(executor)
        else:
            generator = ag.AssertionGenerator(executor)
        generation_result.accept(generator)


def _track_final_metrics(
    algorithm,
    executor: TestCaseExecutor,
    generation_result: tsc.TestSuiteChromosome
) -> set[CoverageMetric]:
    """Track the final coverage metrics.

    Re-loads all required instrumentations for metrics that were not already
    calculated and tracked during the result generation.
    These metrics are then also calculated on the result, which is executed
    once again with the new instrumentation.

    Args:
        algorithm: the used test-generation algorithm
        executor: the testcase executor of the run
        generation_result: the generated testsuite containing assertions

    Returns:
        The set of tracked coverage metrics, including the ones that we optimised for.
    """
    # Alias for shorter lines
    stat = Globl.statistics_tracker
    cov_metrics = Globl.coverage_metrics
    output_variables = Globl.output_variables
    metrics_for_reinstrumenation: set[CoverageMetric] = set(cov_metrics)

    to_calculate: list[tuple[RuntimeVariable, ff.TestSuiteCoverageFunction]] = []

    add_additional_metrics(
        algorithm=algorithm,
        cov_metrics=cov_metrics,
        executor=executor,
        metrics_for_reinstrumentation=metrics_for_reinstrumenation,
        output_variables=output_variables,
        to_calculate=to_calculate,
    )

    # Assertion Checked Coverage is special...
    if RuntimeVariable.AssertionCheckedCoverage in output_variables:
        metrics_for_reinstrumenation.add(CoverageMetric.CHECKED)
        executor.set_instrument(True)
        executor.add_observer(AssertionExecutionObserver(executor.tracer))
        assertion_checked_coverage_ff = ff.TestSuiteAssertionCheckedCoverageFunction(executor)
        to_calculate.append((
            RuntimeVariable.AssertionCheckedCoverage,
            assertion_checked_coverage_ff,
        ))

    # re-instrument the files
    _reload_instrumentation_loader(metrics_for_reinstrumenation, executor.tracer)

    # force new execution of the test cases after new instrumentation
    _reset_cache_for_result(generation_result)

    # set value for each newly calculated variable
    for runtime_variable, coverage_ff in to_calculate:
        generation_result.add_coverage_function(coverage_ff)
        logger.info(f"Calculating resulting {runtime_variable.value}")  # noqa: G004
        stat.track_output_variable(
            runtime_variable, generation_result.get_coverage_for(coverage_ff)
        )

    ass_gen = Globl.output_conf.assertion_generation
    if (
        ass_gen == AssertionGenerator.CHECKED_MINIMIZING
        and RuntimeVariable.AssertionCheckedCoverage in output_variables
    ):
        _minimize_assertions(stat, generation_result)

    # Collect other final stats on result
    stat.track_output_variable(RuntimeVariable.FinalLength, generation_result.length())
    stat.track_output_variable(RuntimeVariable.FinalSize, generation_result.size())

    # reset whether to instrument tests and assertions as well as the SUT
    instrument_test = CoverageMetric.CHECKED in cov_metrics
    executor.set_instrument(instrument_test)
    return metrics_for_reinstrumenation


def _export_chromosome(
    chromosome: chrom.Chromosome,
    test_cluster: ModuleTestCluster,
    export_strategies: list[ExportStrategy] = [],
    file_name_suffix: str = "",
):
    """Export the given chromosome.

    Args:
        chromosome: the chromosome to export.
        coverage_report: use to export testcase-level coverage report in
            arguexporter.
        file_name_suffix: Suffix that can be added to the file name to distinguish
            between different results e.g., failing and succeeding test cases.

    Returns:
        the generated test cases in string.
    """
    export_visitor = export.PyTestChromosomeToAstVisitor()
    chromosome.accept(export_visitor)
    module = export_visitor.to_module()

    result = None

    for export_strategy in export_strategies:
        if export_strategy == ExportStrategy.NONE:
            output = export.module_to_output_str(
                module,
                format_with_black=Globl.output_conf.format_with_black
            )
            logger.info("Generated testcases module has been converted to `str`")
            result = output

        elif export_strategy == ExportStrategy.PY_TEST:   
            module_name = Globl.module_name.replace('.', '_')
            target_file = (
                Path(Globl.output_conf.output_path).resolve()
                / f'test_{module_name}{file_name_suffix}.py'
            )
            output = export.save_module_to_file(
                export_visitor.to_module(),
                target_file,
                format_with_black=Globl.output_conf.format_with_black,
            )
            logger.info("Written %i test cases to %s", chromosome.size(), target_file)
            result = output
        
        else:
            logger.error("Error: unexpected export_strategy: %s", export_strategy)
            raise Exception
        
    return result


def _get_coverage_ff_from_algorithm(
    algorithm: GenerationAlgorithm, function_type: type[ff.TestSuiteCoverageFunction]
) -> ff.TestSuiteCoverageFunction:
    """Retrieve the coverage function for a test suite of a given coverage type.

    Args:
        algorithm: The test generation strategy
        function_type: the type of coverage function to receive

    Returns:
        The coverage function for a test suite for this run of the given type
    """
    test_suite_coverage_func = None
    for coverage_func in algorithm.test_suite_coverage_functions:
        if isinstance(coverage_func, function_type):
            test_suite_coverage_func = coverage_func
    assert test_suite_coverage_func, "The required coverage function was not initialised"
    return test_suite_coverage_func


def _setup_mutation_analysis_assertion_generator(
    executor: TestCaseExecutor,
) -> ag.MutationAnalysisAssertionGenerator:
    logger.info("Setup mutation generator")
    mutant_generator = _setup_mutant_generator()

    logger.info("Import module %s", Globl.module_name)
    module = importlib.import_module(Globl.module_name)

    logger.info("Build AST for %s", module.__name__)
    executor.tracer.current_thread_identifier = threading.current_thread().ident
    module_source_code = inspect.getsource(module)
    module_ast = ParentNodeTransformer.create_ast(module_source_code)

    logger.info("Mutate module %s", module.__name__)
    mutation_tracer = ExecutionTracer()
    mutation_controller = ag.InstrumentedMutationController(
        mutant_generator, module_ast, module, mutation_tracer
    )
    assertion_generator = ag.MutationAnalysisAssertionGenerator(executor, mutation_controller)

    logger.info("Generated %d mutants", mutation_controller.mutant_count())
    return assertion_generator


def _setup_mutant_generator() -> mu.Mutator:
    operators: list[type[MutationOperator]] = [
        *mo.standard_operators,
        *mo.experimental_operators,
    ]

    mutation_strategy = Globl.output_conf.mutation_strategy

    if mutation_strategy == MutationStrategy.FIRST_ORDER_MUTANTS:
        return mu.FirstOrderMutator(operators)

    order = Globl.output_conf.mutation_order

    if order <= 0:
        raise ConfigurationException("Mutation order should be > 0.")

    if mutation_strategy in _strategies:
        hom_strategy = _strategies[mutation_strategy](order)
        return mu.HighOrderMutator(operators, hom_strategy=hom_strategy)

    raise ConfigurationException("No suitable mutation strategy found.")


def add_additional_metrics(  # noqa: D103
    *,
    algorithm,
    cov_metrics,
    executor,
    metrics_for_reinstrumentation,
    output_variables,
    to_calculate,
):
    if (
        RuntimeVariable.FinalLineCoverage in output_variables
        and CoverageMetric.LINE not in cov_metrics
    ):
        metrics_for_reinstrumentation.add(CoverageMetric.LINE)
        line_cov_ff = ff.TestSuiteLineCoverageFunction(executor)
        to_calculate.append((RuntimeVariable.FinalLineCoverage, line_cov_ff))
    elif CoverageMetric.LINE in cov_metrics:
        # If we optimised for lines, we still want to get the final line coverage.
        to_calculate.append((
            RuntimeVariable.FinalLineCoverage,
            _get_coverage_ff_from_algorithm(algorithm, ff.TestSuiteLineCoverageFunction),
        ))
    if (
        RuntimeVariable.FinalBranchCoverage in output_variables
        and CoverageMetric.BRANCH not in cov_metrics
    ):
        metrics_for_reinstrumentation.add(CoverageMetric.BRANCH)
        branch_cov_ff = ff.TestSuiteBranchCoverageFunction(executor)
        to_calculate.append((RuntimeVariable.FinalBranchCoverage, branch_cov_ff))
    elif CoverageMetric.BRANCH in cov_metrics:
        # If we optimised for branches, we still want to get the final branch coverage.
        to_calculate.append((
            RuntimeVariable.FinalBranchCoverage,
            _get_coverage_ff_from_algorithm(algorithm, ff.TestSuiteBranchCoverageFunction),
        ))


def _reload_instrumentation_loader(
    coverage_metrics: set[CoverageMetric],
    tracer: ExecutionTracer,
):
    module = importlib.import_module(Globl.module_name)
    tracer.current_thread_identifier = threading.current_thread().ident
    first_finder: InstrumentationFinder | None = None
    for finder in sys.meta_path:
        if isinstance(finder, InstrumentationFinder):
            first_finder = finder
            break
    assert first_finder is not None
    first_finder.update_instrumentation_metrics(
        tracer=tracer,
        coverage_metrics=coverage_metrics,
        dynamic_constant_provider=Globl.dynamic_constant_provider
    )
    importlib.reload(module)


def _reset_cache_for_result(generation_result):
    generation_result.invalidate_cache()
    for test_case in generation_result.test_case_chromosomes:
        test_case.invalidate_cache()
        test_case.remove_last_execution_result()


def _minimize_assertions(
    stat: StatisticsTracker,
    generation_result: tsc.TestSuiteChromosome
):
    logger.info("Minimizing assertions based on checked coverage")
    assertion_minimizer = pp.AssertionMinimization()
    generation_result.accept(assertion_minimizer)
    stat.track_output_variable(
        RuntimeVariable.Assertions, len(assertion_minimizer.remaining_assertions)
    )
    stat.track_output_variable(
        RuntimeVariable.DeletedAssertions,
        len(assertion_minimizer.deleted_assertions),
    )


def _collect_miscellaneous_statistics() -> None:
    Globl.test_cluster.log_cluster_statistics()
    stat = Globl.statistics_tracker
    stat.track_output_variable(RuntimeVariable.TargetModule, Globl.module_name)
    stat.track_output_variable(RuntimeVariable.RandomSeed, Globl.seed)
    stat.track_output_variable(
        RuntimeVariable.ConfigurationId,
        Globl.statistics_conf.configuration_id,
    )
    stat.track_output_variable(RuntimeVariable.RunId, Globl.statistics_conf.run_id)
    stat.track_output_variable(
        RuntimeVariable.ProjectName, Globl.project_name
    )
    for runtime_variable, value in stat.variables_generator:
        stat.set_output_variable_for_runtime_variable(runtime_variable, value)
