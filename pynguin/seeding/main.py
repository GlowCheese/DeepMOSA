#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Implements simple constant seeding strategies."""

from __future__ import annotations

import ast
import os

from pathlib import Path
from custom_logger import getLogger
from typing import TYPE_CHECKING, AnyStr

from pynguin.globl import Globl
from pynguin.utils import randomness
from pynguin.runtimevar import RuntimeVariable
from pynguin.seeding import StatementDeserializer
from pynguin.testcase.defaulttestcase import DefaultTestCase
from pynguin.ga.testcasechromosome import TestCaseChromosome
from pynguin.seeding.ast_to_testcase import AstToTestCaseVisitor

if TYPE_CHECKING:
    import pynguin.testcase.testcase as tc
    import pynguin.testcase.testfactory as tf
    from pynguin.setup import ModuleTestCluster
    from pynguin.constants import ConstantProvider

_logger = getLogger(__name__)


class InitialPopulationProvider:
    """Class for seeding the initial population with previously existing testcases."""

    def __init__(
        self,
        test_cluster: ModuleTestCluster,
        test_factory: tf.TestFactory,
        constant_provider: ConstantProvider,
    ):
        """Create new population provider.

        Args:
            test_cluster: Test cluster used to construct test cases
            test_factory: Test factory used to construct test cases
            constant_provider: Constant provider for primitives
        """
        self._testcases: list[DefaultTestCase] = []
        self._test_cluster: ModuleTestCluster = test_cluster
        self._test_factory: tf.TestFactory = test_factory
        self._constant_provider: ConstantProvider = constant_provider

    @staticmethod
    def _get_ast_tree(module_path: AnyStr | os.PathLike[AnyStr]) -> ast.Module | None:
        """Returns the ast tree from a module.

        Args:
            module_path: The path to the project's root

        Returns:
            The ast tree of the given module.
        """
        stat = Globl.statistics_tracker

        module_name = Globl.module_name.rsplit(".", maxsplit=1)[-1]
        _logger.debug("Module name: %s", module_name)
        result: list[Path] = []
        for root, _, files in os.walk(module_path):
            root_path = Path(root).resolve()  # type: ignore[arg-type]
            for name in files:
                assert isinstance(name, str)
                if module_name in name and "test_" in name:
                    result.append(root_path / name)
                    break
        try:
            if len(result) > 0:
                _logger.debug("Module name found: %s", result[0])
                stat.track_output_variable(RuntimeVariable.SuitableTestModule, value=True)
                with result[0].open(mode="r", encoding="utf-8") as module_file:
                    return ast.parse(module_file.read())
            else:
                _logger.debug("No suitable test module found.")
                stat.track_output_variable(RuntimeVariable.SuitableTestModule, value=False)
                return None
        except BaseException as exception:
            _logger.exception("Cannot read module: %s", exception)
            stat.track_output_variable(RuntimeVariable.SuitableTestModule, value=False)
            return None

    def collect_testcases(self, module_path: AnyStr | os.PathLike[AnyStr]) -> None:
        """Collect all test cases from a module.

        Args:
            module_path: Path to the module to collect the test cases from
        """
        stat = Globl.statistics_tracker
        
        tree = self._get_ast_tree(module_path)
        if tree is None:
            _logger.info("Provided testcases are not used.")
            return
        visitor = AstToTestCaseVisitor(
            include_nontest_functions=True,  # TODO: make this configurable
            statement_deserializer=StatementDeserializer(
                self._test_cluster,
                True in Globl.seeding_conf.uninterpreted_statements.value
            )
        )
        visitor.visit(tree)
        self._testcases = visitor.testcases
        stat.track_output_variable(RuntimeVariable.FoundTestCases, len(self._testcases))
        stat.track_output_variable(RuntimeVariable.CollectedTestCases, len(self._testcases))
        self._mutate_testcases_initially()

    def _mutate_testcases_initially(self):
        """Mutates the initial population."""
        for _ in range(Globl.conf.seeding.initial_population_mutations):
            for testcase in self._testcases:
                testcase_wrapper = TestCaseChromosome(testcase, self._test_factory)
                testcase_wrapper.mutate()
                if not testcase_wrapper.test_case.statements:
                    self._testcases.remove(testcase)  # noqa: B909

    def random_testcase(self) -> tc.TestCase:
        """Provides a random seeded test case.

        Returns:
            A random test case
        """
        return randomness.choice(self._testcases)

    def __len__(self) -> int:
        """Number of parsed test cases.

        Returns:
            Number of parsed test cases
        """
        return len(self._testcases)
