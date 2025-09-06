"""A class to transform an AST module into a TestCase"""

from __future__ import annotations

import ast
from vendor.custom_logger import getLogger
from typing import Any, TYPE_CHECKING

from pynguin.config.main import AssertionGenerator
from pynguin.globl import Globl

from .stmtdeserializer import StatementDeserializer

if TYPE_CHECKING:
    from pynguin.setup import ModuleTestCluster
    import pynguin.testcase.defaulttestcase as dtt

_logger = getLogger(__name__)


def _count_all_statements(node) -> int:
    """Counts the number of statements in node and all blocks, not including `node`

    Args:
        node: node to count statements for

    Returns:
        the number of child statements to node

    """
    num_non_assert_statements = 0
    for nm, value in ast.iter_fields(node):
        # For all blocks
        if isinstance(value, list) and all(
            isinstance(elem, ast.stmt) for elem in value
        ):
            for elem in value:
                if isinstance(elem, ast.Assert):
                    continue
                num_non_assert_statements += 1
                num_non_assert_statements += _count_all_statements(elem)
    return num_non_assert_statements


class AstToTestCaseVisitor(ast.NodeVisitor):
    """Transforms a Python AST into our internal test-case representation."""

    def __init__(  # noqa: D107
        self,
        *,
        include_nontest_functions: bool,
        statement_deserializer: StatementDeserializer
    ):
        self._current_parsable: bool = True
        self._testcases: list[dtt.DefaultTestCase] = []
        self._number_found_testcases: int = 0
        self._create_assertions = \
            Globl.output_conf.assertion_generation != \
            AssertionGenerator.NONE
        self._deserializer = statement_deserializer
        self._include_nontest_functions = include_nontest_functions

        # Used for statistics only
        self.total_statements = 0
        self.total_parsed_statements = 0
        self._current_parsed_statements = 0
        self._current_max_num_statements = 0


    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: D102, N802
        if (
            not self._include_nontest_functions
            and not node.name.startswith("test_")
            and not node.name.startswith("seed_test_")
        ):
            return

        self._number_found_testcases += 1
        self._deserializer.reset()
        self._current_parsable = True

        self._current_parsed_statements = 0
        self._current_max_num_statements = _count_all_statements(node)

        self.generic_visit(node)
        current_testcase = self._deserializer.test_case

        self.total_statements += self._current_max_num_statements
        self.total_parsed_statements += self._current_parsed_statements

        if self._current_parsable:
            self._testcases.append(current_testcase)
            _logger.info("Successfully imported %s.", node.name)
        else:
            if (
                self._current_parsed_statements > 0
                and Globl.seeding_conf.include_partially_parsable
            ):
                _logger.info(
                    "Partially parsed %s. Retrieved %s/%s statements.",
                    node.name,
                    self._current_parsed_statements,
                    self._current_max_num_statements,
                )
                self._testcases.append(current_testcase)
            else:
                _logger.info("Failed to parse %s.", node.name)


    def visit_Assign(self, node: ast.Assign) -> Any:
        if (
            self._current_parsable
            or Globl.seeding_conf.include_partially_parsable
        ):
            if self._deserializer.add_assign_stmt(node):
                self._current_parsed_statements += 1
            else:
                self._current_parsable = False


    def visit_Assert(self, node: ast.Assert) -> Any:
        if self._create_assertions and (
            self._current_parsable
            or Globl.seeding_conf.include_partially_parsable
        ):
            self._deserializer.add_assert_stmt(node)


    @property
    def testcases(self) -> list[dtt.DefaultTestCase]:
        """Provides the testcases that could be generated from the given AST.

        It is possible that not every aspect of the AST could be transformed
        to our internal representation.

        Returns:
            The generated testcases.
        """
        return self._testcases
