from __future__ import annotations

import os
import asyncio
import networkx as nx
from copy import copy
from grappa import should
from datetime import datetime
from vendor.custom_logger import getLogger
from typing import TYPE_CHECKING, Set, List, cast

import pynguin.ga.coveragegoals as bg
import pynguin.ga.testcasechromosome as tcc

from pynguin.globl.main import Globl
from pynguin.utils import randomness
from vendor.orderedset import OrderedSet
from networkx.drawing.nx_pydot import to_pydot
from pynguin.runtimevar import RuntimeVariable
from pynguin.export.pytestexporter import PyTestExporter
from pynguin.utils.exceptions import ConstructionFailedException
from pynguin.operators.ranking import fast_epsilon_dominance_assignment

from pynguin.deepmosa import EarlyStopTargetting
from .abstractmosaalgorithm import AbstractMOSAAlgorithm

if TYPE_CHECKING:
    from .archive import CoverageArchive
    from pynguin.deepmosa import DeepMOSASeeding
    from pynguin.execution import SubjectProperties

    import pynguin.ga.computations as ff
    import pynguin.testcase.testcase as tc
    import pynguin.ga.testsuitechromosome as tsc

_logger = getLogger(__name__)


class DeepMOSAAlgorithm(AbstractMOSAAlgorithm):
    """No description."""

    def __init__(self) -> None:
        super().__init__()
        self._goals_manager: _GoalsManager
        self._deepmosa = Globl.conf.deepmosa
        self._plateau_len = self._deepmosa.max_plateau_len

        self._total_tests_added: int = 0
        self._llms_testcases: List[tcc.TestCaseChromosome] = []

    async def generate_tests(self) -> tsc.TestSuiteChromosome:
        Globl.llmseeding.analyse_project()

        self.before_search_start()
        self._goals_manager = _GoalsManager(
            self._test_case_fitness_functions,  # type: ignore[arg-type]
            self._archive,
            self.executor.tracer.get_subject_properties(),
        )
        self._number_of_goals = len(self._test_case_fitness_functions)

        Globl.statistics_tracker.set_output_variable_for_runtime_variable(
            RuntimeVariable.Goals, self._number_of_goals
        )

        self._population = self._get_random_population()
        self._goals_manager.update(self._population)

        # Calculate dominance ranks and crowding distance
        fronts = self._ranking_function.compute_ranking_assignment(
            self._population, self._goals_manager.current_goals
        )
        for i in range(fronts.get_number_of_sub_fronts()):
            fast_epsilon_dominance_assignment(
                fronts.get_sub_front(i), self._goals_manager.current_goals
            )

        self.before_first_search_iteration(
            self.create_test_suite(self._archive.solutions)
        )

        last_num_covered_goals = len(self._archive.covered_goals)
        its_without_update = 0
        while self.resources_left() and len(self._archive.uncovered_goals) > 0:
            num_covered_goals = len(self._archive.covered_goals)
            if num_covered_goals == last_num_covered_goals:
                its_without_update += 1
            else:
                its_without_update = 0
            last_num_covered_goals = num_covered_goals

            if its_without_update > self._plateau_len:
                its_without_update = 0
                await self.evolve_targeted()
            else:
                self.evolve()

        self.after_search_finish()
        return self.create_test_suite(
            self._archive.solutions
            if len(self._archive.solutions) > 0
            else self._get_best_individuals()
        )

    async def evolve_targeted(self):
        """Runs an evolution step that targets uncovered functions.

        Args:
            test_suite: the test suite to base coverage off of.
        """

        original_population: Set[tc.TestCase] = {
            chrom.test_case for chrom in self._population
        }

        llmseeding: DeepMOSASeeding = Globl.llmseeding
        if not self._deepmosa.target_low_coverage_functions:
            # only focus on target uncovered functions
            raise Exception()

        testcases_count = 0
        for _ in range(self._deepmosa.num_seeds_to_inject):
            if not self.resources_left() or len(self._archive.uncovered_goals) == 0:
                break

            try:
                coro = llmseeding.target_uncovered_functions(
                    list(self._archive.solutions),
                    list(self._goals_manager.current_goals)
                )
                if not self._deepmosa.async_enabled:
                    # wait until we receive the test cases
                    test_cases = await coro
                else:
                    # create a task and run in background
                    task = asyncio.create_task(coro)

                    while not task.done():
                        if not self.resources_left() or len(self._archive.uncovered_goals) == 0:
                            raise EarlyStopTargetting()
                        self.evolve()
                        await asyncio.sleep(0.1)  # avoid blocking loop

                    (test_cases := task.result()) | should.be.a(list)

                if len(test_cases) == 0:
                    _logger.warning("len(test_cases) is zero")
                    continue

                testcases_count += len(test_cases)
                test_case_chromosomes = [
                    tcc.TestCaseChromosome(test_case, self.test_factory)
                    for test_case in test_cases
                ]

                self._llms_testcases.extend(test_case_chromosomes)

                new_offspring: List[tcc.TestCaseChromosome] = []
                while len(new_offspring) < len(test_cases) * Globl.ga_conf.population / self._deepmosa.num_seeds_to_inject:
                    offspring_1 = randomness.choice(test_case_chromosomes).clone()
                    offspring_2 = randomness.choice(self._llms_testcases).clone()

                    if randomness.chance(Globl.ga_conf.crossover_rate):
                        try:
                            self._crossover_function.cross_over(offspring_1, offspring_2)
                        except ConstructionFailedException:
                            _logger.debug("CrossOver failed.")
                            continue

                    # Apply mutation on offspring_1
                    for _ in range(Globl.ga_conf.number_of_mutations):
                        self._mutate(offspring_1)
                    if offspring_1.changed and offspring_1.size() > 0:
                        new_offspring.append(offspring_1)

                    # Apply mutation on offspring_2
                    for _ in range(Globl.ga_conf.number_of_mutations):
                        self._mutate(offspring_2)
                    if offspring_2.changed and offspring_2.size() > 0:
                        new_offspring.append(offspring_2)

                self.evolve_common(test_case_chromosomes + new_offspring)

            except EarlyStopTargetting:
                break

        _logger.info(f"evolve_targeted has contributed {testcases_count} testcases")

        self._llms_testcases = self._llms_testcases[-Globl.ga_conf.population:]

        added_tests = sum(
            chrom.test_case not in original_population
            for chrom in self._population
        )

        if not added_tests:
            self._plateau_len *= 2
        else:
            self._total_tests_added += added_tests
            Globl.statistics_tracker.track_output_variable(
                RuntimeVariable.LLMStageSavedTests, self._total_tests_added
            )


    def evolve(self):
        """Runs one evolution step."""
        offspring_population = self._breed_next_generation()
        self.evolve_common(offspring_population)

    
    def evolve_common(self, offspring_population: list[tcc.TestCaseChromosome]):
        """Runs one evolution step."""

        # Create union of parents and offspring
        union: list[tcc.TestCaseChromosome] = []
        union.extend(self._population)
        union.extend(offspring_population)

        # Ranking the union
        _logger.debug("Union Size = %d", len(union))
        # Ranking the union using the best rank algorithm
        fronts = self._ranking_function.compute_ranking_assignment(
            union, self._goals_manager.current_goals
        )

        # Form the next population using “preference sorting and non-dominated
        # sorting” on the updated set of goals
        remain = max(
            Globl.ga_conf.population,
            len(fronts.get_sub_front(0)),
        )
        index = 0
        self._population.clear()

        # Obtain the first front
        front = fronts.get_sub_front(index)

        while remain >= len(front):
            # Assign crowding distance to individuals
            fast_epsilon_dominance_assignment(front, self._goals_manager.current_goals)
            # Add the individuals of this front
            self._population.extend(front)
            # Decrement remain
            remain -= len(front)
            # Obtain the next front
            index += 1
            if remain > 0:
                front = fronts.get_sub_front(index)

        # Remain is less than len(front[index]), insert only the best one
        if 0 < remain < len(front):
            fast_epsilon_dominance_assignment(front, self._goals_manager.current_goals)
            front.sort(key=lambda t: t.distance, reverse=True)
            self._population.extend(front[k] for k in range(remain))

        try:
            self._population | should.have.length(Globl.ga_conf.population)
        except AssertionError:
            _logger.warning("Population size: %s", len(self._population))
            _logger.warning("No sub fronts: %s", fronts.get_number_of_sub_fronts())
            _logger.warning("fronts: %s", fronts.fronts)
            raise

        self._goals_manager.update(self._population)
        self.after_search_iteration(self.create_test_suite(self._archive.solutions))


class _GoalsManager:
    """Manages goals and provides dynamically selected ones for the generation."""

    def __init__(
        self,
        fitness_functions: OrderedSet[ff.FitnessFunction],
        archive: CoverageArchive,
        subject_properties: SubjectProperties,
    ) -> None:
        self._archive = archive
        branch_fitness_functions: OrderedSet[bg.BranchCoverageTestFitness] = OrderedSet()
        for fit in fitness_functions:
            # NOTE: DynaMOSA does not evaluate Line Coverage
            if isinstance(fit, bg.BranchCoverageTestFitness):
                branch_fitness_functions.add(fit)
        self._graph = _BranchFitnessGraph(branch_fitness_functions, subject_properties)
        self._current_goals: OrderedSet[bg.BranchCoverageTestFitness] = self._graph.root_branches
        self._archive.add_goals(self._current_goals)  # type: ignore[arg-type]

    @property
    def current_goals(self) -> OrderedSet[ff.FitnessFunction]:
        """Provides the set of current goals.

        Returns:
            The set of current goals
        """
        return self._current_goals  # type: ignore[return-value]

    def update(self, solutions: list[tcc.TestCaseChromosome]) -> None:
        """Updates the information on the current goals from the found solutions.

        Args:
            solutions: The previously found solutions
        """
        # We must keep iterating, as long as new goals are added.

        new_goals_added = True
        while new_goals_added:
            self._archive.update(solutions)

            covered = self._archive.covered_goals
            new_goals: OrderedSet[bg.BranchCoverageTestFitness] = OrderedSet()
            new_goals_added = False
            for old_goal in self._current_goals:
                if old_goal in covered:
                    children = self._graph.get_structural_children(old_goal)
                    for child in children:
                        if child not in self._current_goals and child not in covered:
                            new_goals.add(child)
                            new_goals_added = True
                else:
                    new_goals.add(old_goal)
            self._current_goals = new_goals
            self._archive.add_goals(self._current_goals)  # type: ignore[arg-type]

        _logger.debug("current goals after update: %s", self._current_goals)


class _BranchFitnessGraph:
    """Best effort re-implementation of EvoSuite's BranchFitnessGraph.

    Arranges the fitness functions for all branches according to their control
    dependencies in the CDG. Each node represents a fitness function. A directed edge
    (u -> v) states that fitness function v should be added for consideration
    only when fitness function u has been covered.
    """

    def __init__(
        self,
        fitness_functions: OrderedSet[bg.BranchCoverageTestFitness],
        subject_properties: SubjectProperties,
    ):
        self._graph = nx.DiGraph()
        # Branch less code objects and branches that are not control dependent on other
        # branches.
        self._root_branches: OrderedSet[bg.BranchCoverageTestFitness] = OrderedSet()
        self._build_graph(fitness_functions, subject_properties)

    def _build_graph(
        self,
        fitness_functions: OrderedSet[bg.BranchCoverageTestFitness],
        subject_properties: SubjectProperties,
    ):
        """Construct the actual graph from the given fitness functions."""
        for fitness in fitness_functions:
            self._graph.add_node(fitness)

        for fitness in fitness_functions:
            if fitness.goal.is_branchless_code_object:
                self._root_branches.add(fitness)
                continue
            assert fitness.goal.is_branch
            branch_goal = cast(bg.BranchGoal, fitness.goal)
            predicate_meta_data = subject_properties.existing_predicates[branch_goal.predicate_id]
            code_object_meta_data = subject_properties.existing_code_objects[
                predicate_meta_data.code_object_id
            ]
            if code_object_meta_data.cdg.is_control_dependent_on_root(predicate_meta_data.node):
                self._root_branches.add(fitness)

            dependencies = code_object_meta_data.cdg.get_control_dependencies(
                predicate_meta_data.node
            )
            for dependency in dependencies:
                goal = bg.BranchGoal(
                    predicate_meta_data.code_object_id,
                    dependency.predicate_id,
                    value=dependency.branch_value,
                )
                dependent_ff = self._goal_to_fitness_function(fitness_functions, goal)
                self._graph.add_edge(dependent_ff, fitness)

        # Sanity check
        assert {n for n in self._graph.nodes if self._graph.in_degree(n) == 0}.issubset(
            self._root_branches
        ), "Root branches cannot depend on other branches."

    @property
    def dot(self):
        """Return DOT representation of this graph."""
        dot = to_pydot(self._graph)
        return dot.to_string()

    @property
    def root_branches(self) -> OrderedSet[bg.BranchCoverageTestFitness]:
        """Return the root branches, i.e., the fitness functions without conditions."""
        return OrderedSet(self._root_branches)

    @staticmethod
    def _goal_to_fitness_function(
        search_in: OrderedSet[bg.BranchCoverageTestFitness], goal: bg.BranchGoal
    ) -> bg.BranchCoverageTestFitness:
        """Little helper to find the fitness function associated with a certain goal.

        Args:
            search_in: The list to search in
            goal: The goal to search for

        Returns:
            The found fitness function.
        """
        for fitness in search_in:
            if fitness.goal == goal:
                return fitness
        raise RuntimeError(f"Could not find fitness function for goal: {goal}")

    def get_structural_children(
        self, fitness_function: bg.BranchCoverageTestFitness
    ) -> OrderedSet[bg.BranchCoverageTestFitness]:
        """Get the fitness functions that are structural children of the given one.

        Args:
            fitness_function: The fitness function whose structural children should be
            returned.

        Returns:
            The structural children fitness functions of the given fitness function.
        """
        return OrderedSet(self._graph.successors(fitness_function))
