#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Pynguin is an automated unit test generation framework for Python.

The framework generates unit tests for a given Python module.  For this it
supports various approaches, such as a random approach, similar to Randoop or a
whole-suite approach, based on a genetic algorithm, as implemented in EvoSuite.  The
framework allows to export test suites in various styles, i.e., using the `unittest`
library from the Python standard library or tests in the style used by the PyTest
framework.

Pynguin is supposed to be used as a standalone command-line application but it
can also be used as a library by instantiating this class directly.
"""

from __future__ import annotations

import datetime
import importlib
import inspect
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import pynguin.assertion.assertiongenerator as ag
import pynguin.assertion.mutation_analysis.mutators as mu
import pynguin.assertion.mutation_analysis.operators as mo
import pynguin.assertion.mutation_analysis.strategies as ms
import pynguin.ga.chromosome as chrom
import pynguin.ga.chromosomevisitor as cv
import pynguin.ga.computations as ff
import pynguin.ga.generationalgorithmfactory as gaf
import pynguin.ga.postprocess as pp
import pynguin.ga.testsuitechromosome as tsc
import pynguin.utils.statistics.stats as stat
from pynguin.analyses.constants import (
    ConstantProvider,
    DelegatingConstantProvider,
    DynamicConstantProvider,
    EmptyConstantProvider,
    RestrictedConstantPool,
    collect_static_constants,
    set_constant_provider,
)
from pynguin.assertion.mutation_analysis.transformer import ParentNodeTransformer
from pynguin.configuration import (
    Algorithm,
    AssertionGenerator,
    CoverageMetric,
    ExportStrategy,
    MutationStrategy,
    StatisticsBackend,
    config,
)
from pynguin.ga.algorithms.generationalgorithm import GenerationAlgorithm
from pynguin.instrumentation.machinery import InstrumentationFinder, install_import_hook
from pynguin.setup.testcluster import ModuleTestCluster
from pynguin.setup.testclustergenerator import generate_test_cluster
from pynguin.slicer.statementslicingobserver import StatementSlicingObserver
from pynguin.testcase import export
from pynguin.testcase.execution import (
    AssertionExecutionObserver,
    ExecutionTracer,
    TestCaseExecutor,
)
from pynguin.utils import randomness
from libs.custom_logger import getLogger
from pynguin.utils.exceptions import ConfigurationException
from pynguin.utils.report import (
    get_coverage_report,
    render_coverage_report,
    render_xml_coverage_report,
)
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

from . import environ

if TYPE_CHECKING:
    from pynguin.assertion.mutation_analysis.operators.base import MutationOperator


_strategies: dict[MutationStrategy, Callable[[int], ms.HOMStrategy]] = {
    MutationStrategy.FIRST_TO_LAST: ms.FirstToLastHOMStrategy,
    MutationStrategy.BETWEEN_OPERATORS: ms.BetweenOperatorsHOMStrategy,
    MutationStrategy.RANDOM: ms.RandomHOMStrategy,
    MutationStrategy.EACH_CHOICE: ms.EachChoiceHOMStrategy,
}

logger = getLogger(__name__)


def prepare_everything():
    """Prepare everything before main function."""

    """ SET RANDOM SEED """
    logger.info("Using seed %d", config.seeding.seed)
    randomness.set_seed(config.seeding.seed)

    """ SETUP PATH
    Add project_path to sys.path, which allows
    Python to import modules in this project."""

    logger.info("Setting up path for %s", config.project_path)
    sys.path.insert(0, config.project_path)

    """ SETUP CONSTANT SEEDING
    Collect constants from SUT, if enabled."""

    constant_provider = EmptyConstantProvider()
    dynamic_constant_provider: DynamicConstantProvider | None = None

    if config.seeding.constant_seeding:
        logger.info("Collecting static constants from module under test")
        constant_pool = collect_static_constants(config.project_path)
        if len(constant_pool) == 0:
            logger.info("No constants found")
        else:
            logger.info("Constants found: %s", len(constant_pool))
            # Probability of 1.0 -> if a value is requested and available -> return it.
            constant_provider = DelegatingConstantProvider(constant_pool, constant_provider, 1.0)

        if config.seeding.dynamic_constant_seeding:
            logger.info("Setting up runtime collection of constants")
            dynamic_constant_provider = DynamicConstantProvider(
                RestrictedConstantPool(max_size=config.seeding.max_dynamic_pool_size),
                constant_provider,
                config.seeding.seeded_dynamic_values_reuse_probability,
                config.seeding.max_dynamic_length,
            )
            constant_provider = dynamic_constant_provider

    set_constant_provider(constant_provider)

    """ SETUP REPORT DIR
    Report dir only needs to be created
    when statistics or coverage report is enabled.
    """
    if (
        config.statistics_output.create_coverage_report
        or config.statistics_output.statistics_backend != StatisticsBackend.NONE
    ):
        report_dir = Path(config.statistics_output.report_dir).absolute()
        report_dir.mkdir(parents=True, exist_ok=True)

    """ SETUP IMPORT HOOK """
    logger.debug("Setting up instrument for %s", config.module_name)

    tracer = ExecutionTracer()

    coverage_metrics = set(config.statistics_output.coverage_metrics)
    install_import_hook(
        config.module_name,
        tracer,
        coverage_metrics,
        dynamic_constant_provider,
    )

    """ LOAD SUT
    Use importlib.import_module to load it (for later test_cluster). """
    try:
        # We need to set the current thread ident so the import trace is recorded.
        tracer.current_thread_identifier = threading.current_thread().ident
        if config.module_name in sys.modules:
            importlib.reload(sys.modules[config.module_name])
        else:
            importlib.import_module(config.module_name)

    except ImportError:
        logger.error(
            "Failed to load SUT: A module could not be imported"
            "because some dependencies are missing or it is malformed"
        )
        raise

    """ SETUP TEST CLUSTER """
    # Analyzing the SUT should not cause any coverage.
    tracer.disable()
    test_cluster = generate_test_cluster()

    if test_cluster.num_accessible_objects_under_test() == 0:
        raise Exception("SUT contains nothing we can test.")

    tracer.enable()

    """ SETUP TEST CASE EXECUTOR """
    executor = TestCaseExecutor(
        tracer,
        maximum_test_execution_timeout=config.stopping.maximum_test_execution_timeout,
        test_execution_time_per_statement=config.stopping.test_execution_time_per_statement,
    )

    """ TRACK SUT STATISTICS """

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
    if CoverageMetric.BRANCH in config.statistics_output.coverage_metrics:
        stat.track_output_variable(
            RuntimeVariable.ImportBranchCoverage,
            ff.compute_branch_coverage(tracer.import_trace, tracer.get_subject_properties()),
        )
    if CoverageMetric.LINE in config.statistics_output.coverage_metrics:
        stat.track_output_variable(
            RuntimeVariable.ImportLineCoverage,
            ff.compute_line_coverage(tracer.import_trace, tracer.get_subject_properties()),
        )

    """ SETUP LANGUAGE MODEL SEEDING """
    if config.algorithm in (Algorithm.CODAMOSA, Algorithm.DEEPMOSA):
        assert environ.OPENAI_API_KEY is not None, (
            "Environment variable DEEPSEEK_API_KEY should be "
            "set in order to generate test cases using "
            f"{config.algorithm.value} strategy!"
        )

        if config.seeding.large_language_model_mutation:
            logger.error("Mutation currently unsupported --- the OpenAI edit models throttle.")

        logger.info("Setting up large language model.")

        module_src = config.module_path.open(encoding="UTF-8").read()

        if config.algorithm == Algorithm.CODAMOSA:
            from pynguin.llm.codamosa.llmseeding import codamosaseeding
            from pynguin.llm.codamosa.model import codamosalanguagemodel

            codamosaseeding.executor = executor
            codamosaseeding.test_cluster = test_cluster
            codamosalanguagemodel.test_src = module_src

        elif config.algorithm == Algorithm.DEEPMOSA:
            from pynguin.llm.deepmosa.llmseeding import deepmosaseeding
            from pynguin.llm.deepmosa.model import deepmosalanguagemodel

            deepmosaseeding.executor = executor
            deepmosaseeding.test_cluster = test_cluster
            deepmosalanguagemodel.test_src = module_src

    return test_cluster, executor, constant_provider


async def run_pynguin():
    logger.info("Start Pynguin Testing for %s...", config.module_name)

    test_cluster, executor, constant_provider = prepare_everything()

    if CoverageMetric.CHECKED in config.statistics_output.coverage_metrics:
        executor.add_observer(StatementSlicingObserver(executor.tracer))

    algorithm: GenerationAlgorithm = _instantiate_test_generation_strategy(executor, test_cluster)

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

    _track_search_metrics(algorithm, generation_result)
    _remove_statements_after_exceptions(generation_result)
    _generate_assertions(executor, generation_result)
    tracked_metrics = _track_final_metrics(
        algorithm, executor, generation_result, constant_provider
    )

    # Export the generated test suites
    if config.test_case_output.export_strategy == ExportStrategy.PY_TEST:
        _export_chromosome(generation_result)

    if config.statistics_output.create_coverage_report:
        coverage_report = get_coverage_report(
            executor.tracer,
            generation_result,
            tracked_metrics,
        )
        render_coverage_report(
            coverage_report,
            Path(config.statistics_output.report_dir) / "cov_report.html",
            datetime.datetime.now(),
        )
        render_xml_coverage_report(
            coverage_report,
            Path(config.statistics_output.report_dir) / "cov_report.xml",
            datetime.datetime.now(),
        )

    _collect_miscellaneous_statistics(test_cluster)
    try:
        assert stat.write_statistics()
        logger.info("Statistics were written successfully")
    except Exception:
        logger.exception("Failed to write statistics")


def _instantiate_test_generation_strategy(
    executor: TestCaseExecutor, test_cluster: ModuleTestCluster
) -> GenerationAlgorithm:
    factory = gaf.TestSuiteGenerationAlgorithmFactory(executor, test_cluster)
    return factory.get_search_algorithm()


def _track_search_metrics(
    algorithm: GenerationAlgorithm, generation_result: tsc.TestSuiteChromosome
) -> None:
    """Track multiple set coverage metrics of the generated test suites.

    This possibly re-executes the test suites.

    Args:
        algorithm: The test generation strategy
        generation_result:  The resulting chromosome of the generation strategy
        coverage_metrics: The selected coverage metrics to guide the search
    """
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
        if metric in config.statistics_output.coverage_metrics:
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
    if config.test_case_output.post_process:
        unused_primitives_removal = pp.TestCasePostProcessor([pp.UnusedStatementsTestCaseVisitor()])
        generation_result.accept(unused_primitives_removal)
        # TODO(fk) add more postprocessing stuff.


def _generate_assertions(executor: TestCaseExecutor, generation_result):
    ass_gen = config.test_case_output.assertion_generation
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
    generation_result: tsc.TestSuiteChromosome,
    constant_provider: ConstantProvider,
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
    cov_metrics = config.statistics_output.coverage_metrics
    output_variables = config.statistics_output.output_variables
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
        to_calculate.append(
            (
                RuntimeVariable.AssertionCheckedCoverage,
                assertion_checked_coverage_ff,
            )
        )

    # re-instrument the files
    dynamic_constant_provider = None
    if isinstance(constant_provider, DynamicConstantProvider):
        dynamic_constant_provider = constant_provider
    _reload_instrumentation_loader(
        metrics_for_reinstrumenation, executor.tracer, dynamic_constant_provider
    )

    # force new execution of the test cases after new instrumentation
    _reset_cache_for_result(generation_result)

    # set value for each newly calculated variable
    for runtime_variable, coverage_ff in to_calculate:
        generation_result.add_coverage_function(coverage_ff)
        logger.info(f"Calculating resulting {runtime_variable.value}")
        stat.track_output_variable(
            runtime_variable, generation_result.get_coverage_for(coverage_ff)
        )

    ass_gen = config.test_case_output.assertion_generation
    if (
        ass_gen == AssertionGenerator.CHECKED_MINIMIZING
        and RuntimeVariable.AssertionCheckedCoverage in output_variables
    ):
        _minimize_assertions(generation_result)

    # Collect other final stats on result
    stat.track_output_variable(RuntimeVariable.FinalLength, generation_result.length())
    stat.track_output_variable(RuntimeVariable.FinalSize, generation_result.size())

    # reset whether to instrument tests and assertions as well as the SUT
    instrument_test = CoverageMetric.CHECKED in cov_metrics
    executor.set_instrument(instrument_test)
    return metrics_for_reinstrumenation


def _export_chromosome(chromosome: chrom.Chromosome):
    """Export the given chromosome.

    Args:
        chromosome: the chromosome to export.

    Returns:
        The string representation of generated test cases.
    """
    export_visitor = export.PyTestChromosomeToAstVisitor()
    chromosome.accept(export_visitor)

    module_name = config.module_name.replace(".", "_")
    target_file = Path(config.test_case_output.output_path).resolve() / f"test_{module_name}.py"

    result = export.save_module_to_file(
        export_visitor.to_module(),
        target_file,
        format_with_black=config.test_case_output.format_with_black,
    )
    logger.info("Written %i test cases to %s", chromosome.size(), target_file)

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

    logger.info("Import module %s", config.module_name)
    module = importlib.import_module(config.module_name)

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

    mutation_strategy = config.test_case_output.mutation_strategy

    if mutation_strategy == MutationStrategy.FIRST_ORDER_MUTANTS:
        return mu.FirstOrderMutator(operators)

    order = config.test_case_output.mutation_order

    if order <= 0:
        raise ConfigurationException("Mutation order should be > 0.")

    if mutation_strategy in _strategies:
        hom_strategy = _strategies[mutation_strategy](order)
        return mu.HighOrderMutator(operators, hom_strategy=hom_strategy)

    raise ConfigurationException("No suitable mutation strategy found.")


def add_additional_metrics(
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
        to_calculate.append(
            (
                RuntimeVariable.FinalLineCoverage,
                _get_coverage_ff_from_algorithm(algorithm, ff.TestSuiteLineCoverageFunction),
            )
        )
    if (
        RuntimeVariable.FinalBranchCoverage in output_variables
        and CoverageMetric.BRANCH not in cov_metrics
    ):
        metrics_for_reinstrumentation.add(CoverageMetric.BRANCH)
        branch_cov_ff = ff.TestSuiteBranchCoverageFunction(executor)
        to_calculate.append((RuntimeVariable.FinalBranchCoverage, branch_cov_ff))
    elif CoverageMetric.BRANCH in cov_metrics:
        # If we optimised for branches, we still want to get the final branch coverage.
        to_calculate.append(
            (
                RuntimeVariable.FinalBranchCoverage,
                _get_coverage_ff_from_algorithm(algorithm, ff.TestSuiteBranchCoverageFunction),
            )
        )


def _reload_instrumentation_loader(
    coverage_metrics: set[CoverageMetric],
    tracer: ExecutionTracer,
    dynamic_constant_provider: DynamicConstantProvider | None,
):
    module = importlib.import_module(config.module_name)
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
        dynamic_constant_provider=dynamic_constant_provider,
    )
    importlib.reload(module)


def _reset_cache_for_result(generation_result):
    generation_result.invalidate_cache()
    for test_case in generation_result.test_case_chromosomes:
        test_case.invalidate_cache()
        test_case.remove_last_execution_result()


def _minimize_assertions(generation_result: tsc.TestSuiteChromosome):
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


def _collect_miscellaneous_statistics(test_cluster: ModuleTestCluster) -> None:
    test_cluster.log_cluster_statistics()
    stat.track_output_variable(RuntimeVariable.TargetModule, config.module_name)
    stat.track_output_variable(RuntimeVariable.RandomSeed, config.seeding.seed)
    stat.track_output_variable(
        RuntimeVariable.ConfigurationId,
        config.statistics_output.configuration_id,
    )
    stat.track_output_variable(RuntimeVariable.RunId, config.statistics_output.run_id)
    stat.track_output_variable(RuntimeVariable.ProjectName, config.statistics_output.project_name)
    for runtime_variable, value in stat.variables_generator:
        stat.set_output_variable_for_runtime_variable(runtime_variable, value)
