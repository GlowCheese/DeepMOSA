from __future__ import annotations

import os
import sys
import json
import enum
import threading
import importlib
import functools

from pathlib import Path
from grappa import should
from typing import TYPE_CHECKING
from vendor.custom_logger import getLogger

from pynguin.codamosa.outputfixers import rewrite_tests as rewrite_coda
from pynguin.deepmosa.outputfixers import rewrite_tests as rewrite_deep
from pynguin.config import Algorithm, CoverageMetric, StatisticsBackend
from pynguin.deepmosa.llmseeding import deserialize_code_to_testcases as deserialize_deep
from pynguin.codamosa.llmseeding import deserialize_code_to_testcases as deserialize_coda
from pynguin.export.pytestexporter import PyTestExporter
import pynguin.ga.testcasechromosome as tcc
from pynguin.testcase import export
from pynguin.execution import ExecutionTracer, TestCaseExecutor
from pynguin.constants import (
    EmptyConstantProvider,
    RestrictedConstantPool,
    DynamicConstantProvider,
    collect_static_constants,
    DelegatingConstantProvider
)

from pynguin.globl import Globl
from pynguin.utils import randomness
from pynguin.runtimevar import RuntimeVariable
from pynguin.setup.testclustergenerator import generate_test_cluster
from pynguin.instrumentation import install_import_hook
from pynguin.slicer.statementslicingobserver import StatementSlicingObserver

import pynguin.assertion.mutation_analysis.strategies as ms

import pynguin.stmt as stmt
import pynguin.ga.computations as ff
from pynguin.deepmosa import DeepMOSASeeding
from pynguin.deepmosa import DeepMOSALanguageModel
import pynguin.ga.generationalgorithmfactory as gaf
from pynguin.statistics.stats import StatisticsTracker
from pynguin.codamosa.llmseeding import CodaMOSASeeding
from pynguin.codamosa.model import CodaMOSALanguageModel
from pynguin.algo.generationalgorithm import GenerationAlgorithm

if TYPE_CHECKING:
    import pynguin.testcase.testfactory as tf
    import pynguin.testcase.defaulttestcase as dtc

_logger = getLogger(__name__)


@enum.unique
class ReturnCode(enum.IntEnum):
    """Return codes for Pynguin to signal result."""

    OK = 0
    """Symbolises that the execution ended as expected."""

    SETUP_FAILED = 1
    """Symbolises that the execution failed in the setup phase."""

    NO_TESTS_GENERATED = 2
    """Symbolises that no test could be generated."""


def _prepare_run_pynguin(fun):
    """
    Prepare everything before main function and do cleanup at the end.
    """
    @functools.wraps(fun)
    async def f(*args, **kwargs):
        """ SETUP SEED """
        _logger.info("Using seed %d", Globl.seed)
        randomness.set_seed(Globl.seed)


        """ SETUP PATH
        Add project_path to sys.path, which allows
        Python to load (import) modules in this project. """
        _logger.debug('Setting up path for %s', Globl.project_path)
        sys.path.insert(0, Globl.project_path)


        """ SETUP CONSTANT SEEDING
        Collect constants from SUT, if enabled. """
        Globl.constant_provider = EmptyConstantProvider()
        if Globl.seeding_conf.constant_seeding:
            _logger.info("Collecting static constants from module under test")
            constant_pool = collect_static_constants(Globl.project_path)
            if len(constant_pool) == 0:
                _logger.info("No constants found")
            else:
                _logger.info("Constants found: %s", len(constant_pool))
                # Probability of 1.0 -> if a value is requested and available -> return it.
                Globl.constant_provider = DelegatingConstantProvider(
                    constant_pool, Globl.constant_provider, 1.0
                )

            if Globl.seeding_conf.dynamic_constant_seeding:
                _logger.info("Setting up runtime collection of constants")
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
                _logger.exception(f"Cannot create report dir {Globl.report_dir}")
                return ReturnCode.SETUP_FAILED
            

        """ SETUP IMPORT HOOK """
        _logger.debug('Setting up instrument for %s', Globl.module_name)

        tracer =  ExecutionTracer()
        
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
            
        except ImportError as e:
            # A module could not be imported because some dependencies
            # are missing or it is malformed
            _logger.info("Failed to load SUT: %s", e)
            return ReturnCode.SETUP_FAILED
        

        """ SETUP TEST CLUSTER """
        # Analyzing the SUT should not cause any coverage.
        tracer.disable()
        test_cluster = Globl.test_cluster = generate_test_cluster()

        if test_cluster.num_accessible_objects_under_test() == 0:
            _logger.error("SUT contains nothing we can test.")
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
            assert os.getenv('OPENAI_API_KEY') is not None, (
                "Environment variable DEEPSEEK_API_KEY should be "
                "set in order to generate test cases using "
                f"{Globl.algorithm.value} strategy!"
            )
        
            if Globl.seeding_conf.large_language_model_mutation:
                _logger.error(
                    "Mutation currently unsupported --- the OpenAI edit models throttle."
                )

            _logger.info("Trying to set up the large language model.")

            if Globl.algorithm == Algorithm.CODAMOSA:
                languagemodel = CodaMOSALanguageModel()
                Globl.llmseeding = CodaMOSASeeding(test_cluster)
            else:
                languagemodel = DeepMOSALanguageModel()
                Globl.llmseeding = DeepMOSASeeding(test_cluster)

            with open(Globl.module_path, encoding="UTF-8") as module_file:
                languagemodel.test_src = module_file.read()

            Globl.llmseeding.model = languagemodel
            Globl.llmseeding.executor = executor
            Globl.llmseeding.sample_with_replacement = (
                Globl.seeding_conf.sample_with_replacement
            )


        """ MAIN RUN_PYNGUIN CALL """
        try:
            return await fun(*args, **kwargs)
        finally:
            sys.path.remove(Globl.project_path)

    return f


def _instantiate_test_generation_strategy(
    executor: TestCaseExecutor
) -> GenerationAlgorithm:
    factory = gaf.TestSuiteGenerationAlgorithmFactory(executor)
    return factory.get_search_algorithm()


@_prepare_run_pynguin
async def run_pynguin(*, executor: TestCaseExecutor=None):
    _logger.info('Start Pynguin Testing for %s...', Globl.module_name)

    if CoverageMetric.CHECKED in Globl.coverage_metrics:
        executor.add_observer(StatementSlicingObserver(executor.tracer))

    algorithm: GenerationAlgorithm = _instantiate_test_generation_strategy(executor)

    src = ""
    with open("app/temp/src.py", "r") as file:
        src = file.read()

    _logger.info("--------------------------------------------------------------------------")
    _logger.info(f"> TESTING {Globl.algorithm.name} deserializer:")
    _logger.info("")

    previewer = TestCasePreviewer(executor, algorithm.test_factory)

    if Globl.algorithm == Algorithm.DEEPMOSA:
        src = '\n'.join(rewrite_deep(src).values())
        testcases = deserialize_deep(src, Globl.test_cluster, True)[0]

    elif Globl.algorithm == Algorithm.CODAMOSA:
        src = '\n'.join(rewrite_coda(src).values())
        testcases = deserialize_coda(src, Globl.test_cluster, True)[0]

    previewer.preview_pre_testcase(src)

    for i, testcase in enumerate(testcases, 1):
        previewer.preview_testcase(testcase, i)


class TestCasePreviewer:
    def __init__(self, executor: TestCaseExecutor, test_factory: tf.TestFactory):
        self.executor = executor
        self.test_factory = test_factory
        self.exporter = PyTestExporter(wrap_code=False)

    def _display_code(self, source: str):
        for i, line in enumerate(source.splitlines(), 1):
            _logger.info(f"{i:>2} |    {line}")

    def _display_section_title(self, title: str):
        _logger.info(f"{title}:")
        _logger.info("----------------------")

    def _display_section_block(self, title: str):
        _logger.info("")
        _logger.info("-" * (16 + len(title)))
        _logger.info(f"||      {title}      ||")
        _logger.info("-" * (16 + len(title)))
        _logger.info("")

    def preview_pre_testcase(self, pre_testcase: str):
        self._display_section_title("Pre-Deserialization")
        self._display_code(pre_testcase)
        _logger.info("")

    def preview_testcase(self, testcase: dtc.DefaultTestCase, idx: int):
        testcase_str = self.exporter.export_sequences_to_str([testcase])
        chromosome = tcc.TestCaseChromosome(testcase, self.test_factory)

        self._display_section_block(f"Test case {idx:>2}")

        self._display_raw_testcase(testcase_str)

        self._display_execution_report(testcase)

        self._display_ast_assign_stmt_infomations(testcase)

        self._display_mutated_testcase(testcase)

    def _display_raw_testcase(self, testcase_str: str):
        self._display_section_title("## RAW TEST CASE")
        self._display_code(testcase_str)
        _logger.info("")

    def _display_execution_report(self, testcase: dtc.DefaultTestCase):
        self._display_section_title("## Executing Test Case")
        result = self.executor.execute(testcase)
        covered = len(result.execution_trace.covered_line_ids)
        _logger.info(covered)
        _logger.info("")

    def _display_ast_assign_stmt_infomations(self, testcase: dtc.DefaultTestCase):
        self._display_section_title("## ASTAssignStatements")
        for statement in testcase.statements:
            if isinstance(statement, stmt.ASTAssignStatement):
                _logger.info(statement.get_variable_references())
        _logger.info("")

    def _display_mutated_testcase(self, testcase: dtc.DefaultTestCase):
        chromosome = tcc.TestCaseChromosome(testcase, self.test_factory)
        chromosome.changed = False
        
        while not chromosome.changed:
            chromosome.mutate()
        mutated_testcase_str = self.exporter.export_sequences_to_str(
            [chromosome.test_case]
        )

        self._display_section_title("## Mutated Test Case")
        self._display_code(mutated_testcase_str)
        _logger.info("")

        self._display_section_title("## Executing Mutated Test Case")
        result = self.executor.execute(chromosome.test_case)
        covered = len(result.execution_trace.covered_line_ids)
        _logger.info(covered)
        _logger.info("")