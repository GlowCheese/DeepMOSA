#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides factories for the generation algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import pynguin.ga.algorithms.archive as arch
import pynguin.utils.statistics.statisticsobserver as sso
from pynguin.analyses.seeding import InitialPopulationProvider
from pynguin.configuration import (
    Algorithm,
    CoverageMetric,
    Selection,
    config,
)
from pynguin.ga.algorithms.codamosaalgorithm import CodaMOSAAlgorithm
from pynguin.ga.algorithms.deepmosaalgorithm import DeepMOSAAlgorithm
from pynguin.ga.algorithms.dynamosaalgorithm import DynaMOSAAlgorithm
from pynguin.ga.algorithms.mioalgorithm import MIOAlgorithm
from pynguin.ga.algorithms.mosaalgorithm import MOSAAlgorithm
from pynguin.ga.algorithms.randomalgorithm import RandomAlgorithm
from pynguin.ga.algorithms.randomsearchalgorithm import (
    RandomTestCaseSearchAlgorithm,
    RandomTestSuiteSearchAlgorithm,
)
from pynguin.ga.algorithms.wholesuitealgorithm import WholeSuiteAlgorithm
from pynguin.ga.operators.crossover import SinglePointRelativeCrossOver
from pynguin.ga.operators.ranking import RankBasedPreferenceSorting
from pynguin.ga.operators.selection import RankSelection, TournamentSelection
from pynguin.ga.stoppingcondition import (
    CoveragePlateauStoppingCondition,
    MaxCoverageStoppingCondition,
    MaxIterationsStoppingCondition,
    MaxSearchTimeStoppingCondition,
    MaxStatementExecutionsStoppingCondition,
    MaxTestExecutionsStoppingCondition,
    MinimumCoveragePlateauStoppingCondition,
)
from pynguin.setup.testcluster import FilteredModuleTestCluster
from pynguin.testcase.execution import AbstractTestCaseExecutor, TypeTracingTestCaseExecutor
from pynguin.testcase.testfactory import TestFactory
from pynguin.utils.custom_logger import getLogger
from pynguin.utils.exceptions import ConfigurationException
from pynguin.utils.orderedset import OrderedSet

from . import computations as ff
from . import coveragegoals as bg
from . import testcasechromosomefactory as tccf
from . import testcasefactory as tcf
from .chromosome import Chromosome
from .searchobserver import LogSearchObserver
from .testsuitechromosome import TestSuiteChromosome
from .testsuitechromosomefactory import TestSuiteChromosomeFactory

if TYPE_CHECKING:
    import pynguin.ga.chromosomefactory as cf
    from pynguin.ga.algorithms.generationalgorithm import GenerationAlgorithm
    from pynguin.ga.operators.crossover import CrossOverFunction
    from pynguin.ga.operators.ranking import RankingFunction
    from pynguin.ga.operators.selection import SelectionFunction
    from pynguin.ga.stoppingcondition import StoppingCondition
    from pynguin.setup.testcluster import ModuleTestCluster


C = TypeVar("C", bound=Chromosome)


class GenerationAlgorithmFactory(ABC, Generic[C]):
    """A generic generation algorithm factory."""

    _logger = getLogger(__name__)

    _DEFAULT_MAX_SEARCH_TIME = 600

    def get_stopping_conditions(self) -> list[StoppingCondition]:
        """Instantiates the stopping conditions depending on the configuration settings.

        Returns:
            A list of stopping conditions
        """
        stopping = config.stopping
        conditions: list[StoppingCondition] = []
        if (max_iter := stopping.maximum_iterations) >= 0:
            conditions.append(MaxIterationsStoppingCondition(max_iter))
        if (max_stmt := stopping.maximum_statement_executions) >= 0:
            conditions.append(MaxStatementExecutionsStoppingCondition(max_stmt))
        if (max_test_exec := stopping.maximum_test_executions) >= 0:
            conditions.append(MaxTestExecutionsStoppingCondition(max_test_exec))
        if (max_search_time := stopping.maximum_search_time) >= 0:
            conditions.append(MaxSearchTimeStoppingCondition(max_search_time))
        if (max_coverage := stopping.maximum_coverage) < 100:
            conditions.append(MaxCoverageStoppingCondition(max_coverage))
        if (iterations := stopping.maximum_coverage_plateau) >= 0:
            conditions.append(CoveragePlateauStoppingCondition(iterations))
        if (min_coverage := stopping.minimum_coverage) < 100:
            if min_coverage <= 0:
                raise AssertionError("Coverage has to be larger 0 but less than 100%")
            plateau_iterations = stopping.minimum_plateau_iterations
            if plateau_iterations <= 0:
                raise AssertionError("Minimum Plateau Iterations has to be larger 0")
            conditions.append(
                MinimumCoveragePlateauStoppingCondition(min_coverage, plateau_iterations)
            )
        if len(conditions) == 0:
            self._logger.info("No stopping condition configured!")
            self._logger.info(
                "Using fallback timeout of %i seconds",
                GenerationAlgorithmFactory._DEFAULT_MAX_SEARCH_TIME,
            )
            conditions.append(
                MaxSearchTimeStoppingCondition(GenerationAlgorithmFactory._DEFAULT_MAX_SEARCH_TIME)
            )
        return conditions

    @abstractmethod
    def get_search_algorithm(self) -> GenerationAlgorithm:
        """Initialises and sets up the test-generation strategy to use.

        Returns:
            A fully configured test-generation strategy
        """


class TestSuiteGenerationAlgorithmFactory(GenerationAlgorithmFactory[TestSuiteChromosome]):
    """A factory for a search algorithm generating test-suites."""

    _strategies: ClassVar[dict[Algorithm, Callable[[], GenerationAlgorithm]]] = {
        Algorithm.CODAMOSA: CodaMOSAAlgorithm,
        Algorithm.DYNAMOSA: DynaMOSAAlgorithm,
        Algorithm.DEEPMOSA: DeepMOSAAlgorithm,
        Algorithm.MIO: MIOAlgorithm,
        Algorithm.MOSA: MOSAAlgorithm,
        Algorithm.RANDOM: RandomAlgorithm,
        Algorithm.RANDOM_TEST_SUITE_SEARCH: RandomTestSuiteSearchAlgorithm,
        Algorithm.RANDOM_TEST_CASE_SEARCH: RandomTestCaseSearchAlgorithm,
        Algorithm.WHOLE_SUITE: WholeSuiteAlgorithm,
    }

    _selections: ClassVar[dict[Selection, Callable[[], SelectionFunction]]] = {
        Selection.TOURNAMENT_SELECTION: TournamentSelection,
        Selection.RANK_SELECTION: RankSelection,
    }

    def __init__(self, executor: AbstractTestCaseExecutor, test_cluster: ModuleTestCluster):
        """Initializes the factory."""
        if config.type_inference.type_tracing:
            executor = TypeTracingTestCaseExecutor(executor, test_cluster)
        self._executor = executor
        self._test_cluster = test_cluster

    def _get_chromosome_factory(self, strategy: GenerationAlgorithm) -> cf.ChromosomeFactory:
        """Provides a chromosome factory.

        Args:
            strategy: The strategy that is currently configured.

        Returns:
            A chromosome factory
        """
        # TODO add conditional returns/other factories here
        test_case_factory: tcf.TestCaseFactory = tcf.RandomLengthTestCaseFactory(
            strategy.test_factory, strategy.test_cluster
        )
        if config.seeding.initial_population_seeding:
            self._logger.info("Using population seeding")
            population_provider = InitialPopulationProvider(
                test_cluster=strategy.test_cluster,
                test_factory=strategy.test_factory,
            )
            self._logger.info("Collecting and parsing provided testcases.")
            population_provider.collect_testcases(config.seeding.initial_population_data)
            if len(population_provider) == 0:
                self._logger.info("Could not parse any test case")
            else:
                self._logger.info("Parsed testcases: %s", len(population_provider))
                test_case_factory = tcf.SeededTestCaseFactory(
                    test_case_factory, population_provider
                )

        test_case_chromosome_factory: cf.ChromosomeFactory = tccf.TestCaseChromosomeFactory(
            strategy.test_factory,
            test_case_factory,
            strategy.test_case_fitness_functions,
        )
        if config.seeding.seed_from_archive:
            self._logger.info("Using archive seeding")
            test_case_chromosome_factory = tccf.ArchiveReuseTestCaseChromosomeFactory(
                test_case_chromosome_factory, strategy.archive
            )
        if config.algorithm in {
            Algorithm.DYNAMOSA,
            Algorithm.CODAMOSA,
            Algorithm.DEEPMOSA,
            Algorithm.MIO,
            Algorithm.MOSA,
            Algorithm.RANDOM_TEST_CASE_SEARCH,
        }:
            return test_case_chromosome_factory
        return TestSuiteChromosomeFactory(
            test_case_chromosome_factory,
            strategy.test_suite_fitness_functions,
            strategy.test_suite_coverage_functions,
        )

    def get_search_algorithm(self) -> GenerationAlgorithm:
        """Initialises and sets up the test-generation strategy to use.

        Returns:
            A fully configured test-generation strategy
        """
        strategy = self._get_generation_strategy()
        strategy.branch_goal_pool = bg.BranchGoalPool(
            self._executor.tracer.get_subject_properties()
        )
        strategy.test_case_fitness_functions = self._get_test_case_fitness_functions(strategy)
        strategy.test_suite_fitness_functions = self._get_test_suite_fitness_functions()
        strategy.test_suite_coverage_functions = self._get_test_suite_coverage_functions()
        strategy.archive = self._get_archive(strategy)

        strategy.executor = self._executor
        strategy.test_cluster = self._get_test_cluster(strategy)
        strategy.test_factory = self._get_test_factory(strategy)
        strategy.chromosome_factory = self._get_chromosome_factory(strategy)

        selection_function = self._get_selection_function()
        selection_function.maximize = False
        strategy.selection_function = selection_function

        stopping_conditions = self.get_stopping_conditions()
        strategy.stopping_conditions = stopping_conditions
        for stop in stopping_conditions:
            strategy.add_search_observer(stop)
            if stop.observes_execution:
                self._executor.add_observer(stop)
        strategy.add_search_observer(LogSearchObserver())
        strategy.add_search_observer(sso.SequenceStartTimeObserver())
        strategy.add_search_observer(sso.IterationObserver())
        strategy.add_search_observer(sso.BestIndividualObserver())

        crossover_function = self._get_crossover_function()
        strategy.crossover_function = crossover_function

        ranking_function = self._get_ranking_function()
        strategy.ranking_function = ranking_function

        return strategy

    @classmethod
    def _get_generation_strategy(cls) -> GenerationAlgorithm:
        """Provides a generation strategy.

        Returns:
            A generation strategy

        Raises:
            ConfigurationException: if an unknown algorithm was requested
        """
        if config.algorithm in cls._strategies:
            strategy = cls._strategies.get(config.algorithm)
            assert strategy, "Strategy cannot be defined as None"
            cls._logger.info("Using strategy: %s", config.algorithm)
            return strategy()
        raise ConfigurationException("No suitable generation strategy found.")

    @classmethod
    def _get_selection_function(cls) -> SelectionFunction[TestSuiteChromosome]:
        """Provides a selection function for the selected algorithm.

        Returns:
            A selection function

        Raises:
            ConfigurationException: if an unknown function was requested
        """
        if config.search_algorithm.selection in cls._selections:
            strategy = cls._selections.get(config.search_algorithm.selection)
            assert strategy, "Selection function cannot be defined as None"
            cls._logger.info(
                "Using selection function: %s",
                config.search_algorithm.selection,
            )
            return strategy()
        raise ConfigurationException("No suitable selection function found.")

    def _get_crossover_function(self) -> CrossOverFunction[TestSuiteChromosome]:
        """Provides a crossover function for the selected algorithm.

        Returns:
            A crossover function
        """
        self._logger.info("Using crossover function: SinglePointRelativeCrossOver")
        return SinglePointRelativeCrossOver()

    def _get_archive(self, strategy: GenerationAlgorithm) -> arch.Archive:
        if config.algorithm == Algorithm.MIO:
            self._logger.info("Using MIOArchive")
            size = config.mio.initial_config.number_of_tests_per_target
            return arch.MIOArchive(
                strategy.test_case_fitness_functions,
                initial_size=size,
            )
        # Use CoverageArchive as default, even if the algorithm does not use it.
        self._logger.info("Using CoverageArchive")
        if config.algorithm in (Algorithm.DYNAMOSA, Algorithm.DEEPMOSA):
            # DynaMOSA gradually adds its fitness functions, so we initialize
            # with an empty set.
            return arch.CoverageArchive(OrderedSet())
        return arch.CoverageArchive(OrderedSet(strategy.test_case_fitness_functions))

    def _get_ranking_function(self) -> RankingFunction:
        self._logger.info("Using ranking function: RankBasedPreferenceSorting")
        return RankBasedPreferenceSorting()

    def _get_test_case_fitness_functions(
        self, strategy: GenerationAlgorithm
    ) -> OrderedSet[ff.TestCaseFitnessFunction]:
        """Creates the fitness functions for test cases.

        Args:
            strategy: The currently configured strategy

        Returns:
            A list of fitness functions
        """

        if config.algorithm in {
            Algorithm.DYNAMOSA,
            Algorithm.CODAMOSA,
            Algorithm.DEEPMOSA,
            Algorithm.MIO,
            Algorithm.MOSA,
            Algorithm.RANDOM_TEST_CASE_SEARCH,
            Algorithm.WHOLE_SUITE,
        }:
            fitness_functions: OrderedSet[ff.TestCaseFitnessFunction] = OrderedSet()
            coverage_metrics = config.statistics_output.coverage_metrics

            if CoverageMetric.LINE in coverage_metrics:
                fitness_functions.update(bg.create_line_coverage_fitness_functions(self._executor))

            if CoverageMetric.BRANCH in coverage_metrics:
                fitness_functions.update(
                    bg.create_branch_coverage_fitness_functions(
                        self._executor, strategy.branch_goal_pool
                    )
                )

            if CoverageMetric.CHECKED in coverage_metrics:
                fitness_functions.update(
                    bg.create_checked_coverage_fitness_functions(self._executor)
                )
            self._logger.info("Instantiated %d fitness functions", len(fitness_functions))

            return fitness_functions
        return OrderedSet()

    def _get_test_suite_fitness_functions(
        self,
    ) -> OrderedSet[ff.TestSuiteFitnessFunction]:
        test_suite_ffs: OrderedSet[ff.TestSuiteFitnessFunction] = OrderedSet()
        coverage_metrics = config.statistics_output.coverage_metrics
        if CoverageMetric.LINE in coverage_metrics:
            test_suite_ffs.update([ff.LineTestSuiteFitnessFunction(self._executor)])
        if CoverageMetric.BRANCH in coverage_metrics:
            test_suite_ffs.update([ff.BranchDistanceTestSuiteFitnessFunction(self._executor)])
        if CoverageMetric.CHECKED in coverage_metrics:
            test_suite_ffs.update([ff.StatementCheckedTestSuiteFitnessFunction(self._executor)])
        return test_suite_ffs

    def _get_test_suite_coverage_functions(
        self,
    ) -> OrderedSet[ff.TestSuiteCoverageFunction]:
        test_suite_ffs: OrderedSet[ff.TestSuiteCoverageFunction] = OrderedSet()
        coverage_metrics = config.statistics_output.coverage_metrics
        if CoverageMetric.LINE in coverage_metrics:
            test_suite_ffs.update([ff.TestSuiteLineCoverageFunction(self._executor)])
        if CoverageMetric.BRANCH in coverage_metrics:
            test_suite_ffs.update([ff.TestSuiteBranchCoverageFunction(self._executor)])
        if CoverageMetric.CHECKED in coverage_metrics:
            test_suite_ffs.update([ff.TestSuiteStatementCheckedCoverageFunction(self._executor)])
        # do not add TestSuiteAssertionCheckedCoverageFunction here, since it must
        # be added and calculated after the assertion generation
        return test_suite_ffs

    def _get_test_cluster(self, strategy: GenerationAlgorithm):
        if config.search_algorithm.filter_covered_targets_from_test_cluster:
            # Wrap test cluster in filter.
            return FilteredModuleTestCluster(
                self._test_cluster,
                strategy.archive,
                self._executor.tracer.get_subject_properties(),
                strategy.test_case_fitness_functions,
            )
        return self._test_cluster

    @staticmethod
    def _get_test_factory(strategy: GenerationAlgorithm):
        return TestFactory(strategy.test_cluster)
