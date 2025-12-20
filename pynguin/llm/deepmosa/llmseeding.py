from __future__ import annotations

import ast
import datetime
import inspect
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import numpy as np

import pynguin.utils.statistics.stats as stat
from pynguin.configuration import config
from pynguin.export.pytestexporter import PyTestExporter
from pynguin.ga.coveragegoals import BranchGoal
from pynguin.llm.ast_to_testcase import AstToTestCaseVisitor
from pynguin.llm.deepmosa.stmtdeserializer_v2 import StatementDeserializerV2
from pynguin.testcase.statement import ASTAssignStatement
from pynguin.utils.custom_logger import getLogger
from pynguin.utils.generic import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)
from pynguin.utils.orderedset import OrderedSet
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

from .model import deepmosalanguagemodel
from .outputfixers import fixup_imports

if TYPE_CHECKING:
    import pynguin.ga.testcasechromosome as tcc
    import pynguin.testcase.defaulttestcase as dtc
    import pynguin.testcase.testcase as tc
    from pynguin.ga.coveragegoals import BranchCoverageTestFitness
    from pynguin.setup.testcluster import TestCluster
    from pynguin.testcase.execution import TestCaseExecutor


logger = getLogger(__name__)


class EarlyStopTargetting(Exception):
    pass


def deserialize_code_to_testcases(
    test_file_contents: str,
    test_cluster: TestCluster,
    use_uninterpreted_statements: bool = False,
) -> Tuple[List[dtc.DefaultTestCase], int, int]:
    """Extracts as many TestCase objects as possible from the given code.

    Args:
        test_file_contents: code containing tests
        test_cluster: the TestCluster to deserialize with
        use_uninterpreted_statements: whether or not to allow ASTAssignStatements

    Returns:
        A tuple consisting of (1) a list of TestCase extracted from the given code
        (2) the number of parsable statements in the given code (3) the number
        of successfully parsed statements from that code
    """
    visitor = AstToTestCaseVisitor(
        include_nontest_functions=False,
        statement_deserializer=StatementDeserializerV2(test_cluster, use_uninterpreted_statements),
    )
    visitor.visit(ast.parse(test_file_contents))
    return (
        visitor.testcases,
        visitor.total_parsed_statements,
        visitor.total_statements,
    )


class GenCallablesFinder(ast.NodeVisitor):
    def __init__(
        self,
        imported_modules: Dict[str, str],
        imported_callables: Dict[str, Tuple[str, str]],
        all_accessible_objects: List[GenericCallableAccessibleObject],
    ):
        self._imported_modules = imported_modules
        self._imported_callables = imported_callables
        self._all_accessible_objects = all_accessible_objects

    def _find_gen_callables(self, call_id: str, call_name: str):
        if call_id in self._imported_modules:
            expected = self._imported_modules[call_id]
        elif call_name in self._imported_callables:
            expected, call_name = self._imported_callables[call_name]
        else:
            return []

        for obj in self._all_accessible_objects:
            if isinstance(obj, (GenericConstructor, GenericMethod)):
                assert obj.owner is not None
                if call_name != obj.owner.name:
                    continue
                if expected in obj.owner.module:
                    return [obj]
            elif isinstance(obj, GenericFunction):
                if call_name != obj.function_name:
                    continue
                if expected in obj.callable.__module__:
                    return [obj]
        return []

    def generic_visit(self, node):
        return []

    def visit_Attribute(self, node):
        names = []
        curr = node.value
        while isinstance(curr, ast.Attribute):
            names.append(curr.attr)
            curr = curr.value
        if not isinstance(curr, ast.Name):
            return []
        names.append(curr.id)
        call_id = ".".join(reversed(names))
        return self._find_gen_callables(call_id, node.attr)

    def visit_Name(self, node):
        return self._find_gen_callables("", node.id)

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return []
        return self._find_gen_callables("", node.value)

    def visit_Tuple(self, node):
        result = []
        for e in node.elts:
            result.extend(self.visit(e))
        return result

    def visit_BinOp(self, node):
        return self.visit(node.left) + self.visit(node.right)

    def visit_Subscript(self, node):
        return self.visit(node.value) + self.visit(node.slice)

    def visit_Call(self, node):
        return self.visit(node.func)

    def visit_arg(self, node):
        if node.annotation is None:
            return []
        return self.visit(node.annotation)


class _DeepMOSASeeding:
    """Class for seeding the initial population with test cases generated by a large
    language model."""

    def __init__(self):
        self.test_cluster: TestCluster
        self.executor: TestCaseExecutor

        self._parsed_statements = 0
        self._parsable_statements = 0
        self._uninterp_statements = 0

        self._code_id_to_gao: Dict[int, GenericCallableAccessibleObject | None] = {}
        self._discarded_modules: OrderedSet[str] = OrderedSet()
        self._imported_modules: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._imported_callables_of: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
        self._gao_owner_str: Dict[GenericCallableAccessibleObject, str] = {}
        self._dependers: dict[
            GenericCallableAccessibleObject, OrderedSet[GenericCallableAccessibleObject]
        ] = defaultdict(OrderedSet)
        self._dependees: dict[
            GenericCallableAccessibleObject, OrderedSet[GenericCallableAccessibleObject]
        ] = defaultdict(OrderedSet)
        self._query_count_of: Dict[BranchCoverageTestFitness, int] = defaultdict(int)
        self._query_threshold: int = 4
        self._query_lim: int = 8

        # cache of target_uncovered_functions
        self._goals_fitness: List[float]
        self._goals_probability: List[float]
        self._uncovered_goals: List[BranchCoverageTestFitness]

    async def _get_targeted_testcase(
        self,
        gao: GenericCallableAccessibleObject,
        pred_lineno: int | None,
        pred_value: bool | None,
    ):
        """
        Generate a new test case targeting a GenericCallableAccessibleObject

        Args:
            gao: the GenericCallableAccessibleObject to target
            pred_lineno: line of the predicate that we want to cover
            pred_value: value of the predicate that we want to satisfy

        Returns:
            A sequence of generated test cases
        """
        str_test_case = await deepmosalanguagemodel.target_test_case(
            gao, self._gao_owner_str, self._dependers, pred_lineno, pred_value
        )
        # with open(".trash/hint.py", "r") as file:
        #     str_test_case = file.read()

        # Deserialize code to test cases
        ret_testcases: Set[tc.TestCase] = set()
        use_uninterp_tuple = config.seeding.uninterpreted_statements.value
        for use_uninterp in use_uninterp_tuple:
            logger.debug("Codex-generated testcase:\n%s", str_test_case)
            (
                testcases,
                parsed_statements,
                parsable_statements,
            ) = deserialize_code_to_testcases(str_test_case, self.test_cluster, use_uninterp)
            for testcase in testcases:
                exporter = PyTestExporter(wrap_code=False)
                testcase_str = exporter.export_sequences_to_str([testcase])
                logger.debug(
                    "Imported test case (%i/%i statements parsed):\n%s",
                    parsed_statements,
                    parsable_statements,
                    testcase_str,
                )

                report_dir = config.statistics_output.report_dir
                with open(
                    os.path.join(report_dir, "gen_after_parse.py"),
                    "a+",
                    encoding="UTF-8",
                ) as log_file:
                    log_file.write(
                        f"\n\n# ({config.module_name}) Generated at {datetime.datetime.now()}\n"
                    )
                    log_file.write(testcase_str)

                self._parsable_statements += parsable_statements
                self._parsed_statements += parsed_statements
                self._uninterp_statements += len(
                    [stmt for stmt in testcase.statements if isinstance(stmt, ASTAssignStatement)]
                )

                stat.track_output_variable(
                    RuntimeVariable.ParsableStatements, self._parsable_statements
                )
                stat.track_output_variable(
                    RuntimeVariable.ParsedStatements, self._parsed_statements
                )
                stat.track_output_variable(
                    RuntimeVariable.UninterpStatements, self._uninterp_statements
                )
            ret_testcases.update(testcases)
        return list(ret_testcases)

    def _process_import(self, module_path: str):
        # TODO: this feature can be enhance to be more precise!
        assert os.path.splitext(module_path)[1] == ".py"
        if module_path in self._imported_modules:
            return

        imported_modules = self._imported_modules[module_path] = {}
        imported_callables = self._imported_callables_of[module_path]

        dont_update_me = set(imported_callables.keys())

        # Callables and module outside SUT
        with open(module_path, "r", encoding="utf-8") as file:
            module_tree = ast.parse(file.read())

        for node in ast.walk(module_tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    alias = name.asname or name.name
                    imported_modules[alias] = name.name
            if not isinstance(node, ast.ImportFrom):
                continue

            # process ast.ImportFrom
            if node.level == 0:
                current_path = config.project_path
            else:
                current_path = module_path
            if node.level == 0:
                current_path = config.project_path
            for _ in range(node.level):
                current_path = os.path.dirname(current_path)

            # if node.module is None, it'll become f'{current_path}/'
            current_path = os.path.join(current_path, (node.module or "").replace(".", "/"))

            if (Path(current_path) / "__init__.py").exists():
                current_path = str(Path(current_path) / "__init__.py")
                self._process_import(current_path)
                for name in node.names:
                    if name.name == "*":
                        imported_modules.update(
                            {
                                k: v
                                for k, v in self._imported_modules[current_path].items()
                                if k not in dont_update_me
                            }
                        )
                        imported_callables.update(
                            {
                                k: v
                                for k, v in self._imported_callables_of[current_path].items()
                                if k not in dont_update_me
                            }
                        )
                    elif name.name not in dont_update_me:
                        alias = name.asname or name.name
                        if name.name in self._imported_modules[current_path]:
                            imported_modules[alias] = self._imported_modules[current_path][
                                name.name
                            ]
                            imported_callables.pop(alias, None)
                        elif name.name in self._imported_callables_of[current_path]:
                            imported_callables[alias] = self._imported_callables_of[current_path][
                                name.name
                            ]
                            imported_modules.pop(alias, None)
                        else:
                            imported_modules.pop(alias, None)
                            imported_callables.pop(alias, None)

            elif Path(current_path).with_suffix(".py").exists():
                for name in node.names:
                    if name.name == "*":
                        current_path = str(Path(current_path).with_suffix(".py"))
                        self._process_import(current_path)
                        imported_modules.update(
                            {
                                k: v
                                for k, v in self._imported_modules[current_path].items()
                                if k not in dont_update_me
                            }
                        )
                        imported_callables.update(
                            {
                                k: v
                                for k, v in self._imported_callables_of[current_path].items()
                                if k not in dont_update_me
                            }
                        )

                    elif name.name not in dont_update_me:
                        alias = name.asname or name.name
                        mod = os.path.relpath(current_path, config.project_path)
                        if mod.startswith("."):
                            continue  # module outside project
                        mod = mod.replace("/", ".")
                        imported_callables[alias] = (mod, name.name)

    def analyse_project(self):
        # Map code object ids to gaos
        _subject_properties = self.executor.tracer.get_subject_properties()

        for id, code_object in _subject_properties.existing_code_objects.items():
            code1 = code_object.code_object
            module1 = str(Path(config.project_path).resolve(True))
            module1 = os.path.relpath(code1.co_filename, module1)
            module1 = module1.partition(".")[0].replace("/", ".")
            for gao in self.test_cluster.all_accessible_objects:
                try:
                    if isinstance(gao, GenericConstructor):
                        assert gao.owner.module == module1  # type: ignore
                        assert gao.owner.name == code1.co_name  # type: ignore
                    elif isinstance(gao, (GenericFunction, GenericMethod)):
                        code2 = gao._callable.__code__  # type: ignore
                        assert code1.co_name == code2.co_name
                        assert code1.co_filename == code2.co_filename
                        assert code1.co_firstlineno == code2.co_firstlineno
                    else:
                        continue
                except AssertionError:
                    continue

                assert id not in self._code_id_to_gao
                self._code_id_to_gao[id] = gao

        all_accessible_objects = [
            o
            for o in self.test_cluster.all_accessible_objects
            if isinstance(o, GenericCallableAccessibleObject)
        ]

        for gao in all_accessible_objects:
            # Prepare gao owner str
            if isinstance(gao, (GenericConstructor, GenericMethod)):
                gao_owner = gao.owner.raw_type  # type: ignore
                try:
                    self._gao_owner_str[gao] = inspect.getsource(gao_owner)
                except (TypeError, OSError):
                    logger.debug("Cannot get source code for %s", gao_owner)

            # Find callables within module
            if isinstance(gao, GenericMethod):
                alias = gao.method_name
            elif isinstance(gao, GenericFunction):
                alias = gao.function_name
            else:
                assert isinstance(gao, GenericConstructor)
                alias = gao.owner.name  # type: ignore

            assert alias is not None
            self._imported_callables_of[gao.file_path][alias] = (gao.module_name, alias)

        # Find dependers and dependees
        for gao in all_accessible_objects:
            source = deepmosalanguagemodel._get_gao_str(gao)
            if source is None:
                continue
            self._process_import(gao.file_path)

            callables_finder = GenCallablesFinder(
                self._imported_modules[gao.file_path],
                self._imported_callables_of[gao.file_path],
                all_accessible_objects,
            )

            for node in ast.walk(deepmosalanguagemodel._safe_parse(source)):
                if not isinstance(node, (ast.arg, ast.Call)):
                    continue
                for callable in callables_finder.visit(node):
                    callable: GenericCallableAccessibleObject = callable
                    if (
                        callable.owner is None
                        or not issubclass(callable.owner.raw_type, BaseException)
                        or (gao.owner is not None and issubclass(gao.owner.raw_type, BaseException))
                    ):
                        self._dependers[gao].add(callable)
                        self._dependees[callable].add(gao)

            if isinstance(gao, GenericConstructor):
                for base in gao.owner.raw_type.__bases__:  # type: ignore
                    for callable in callables_finder._find_gen_callables(
                        base.__module__, base.__name__
                    ):
                        self._dependers[gao].add(callable)
                        self._dependees[callable].add(gao)

        # gao should neither be its depender or dependee!
        for gao, dep in self._dependers.items():
            dep.discard(gao)
        for gao, dep in self._dependees.items():
            dep.discard(gao)

    def get_gao_from_id(self, id: int):
        if id in self._code_id_to_gao:
            return self._code_id_to_gao[id]

        res = None

        id_trace = [id]
        _subject_properties = self.executor.tracer.get_subject_properties()
        code_object = _subject_properties.existing_code_objects[id]
        code = code_object.code_object

        while code.co_name != "<module>":
            # getting parent's info
            parent_id = code_object.parent_code_object_id
            assert parent_id is not None
            parent_code_object = _subject_properties.existing_code_objects[parent_id]
            parent_code = parent_code_object.code_object

            # if code object for parent is cached
            id_trace.append(parent_id)
            res = self._code_id_to_gao.get(parent_id)
            if res is not None:
                break

            # move up if fails
            id = parent_id
            code = parent_code
            code_object = parent_code_object

        if res is None:
            logger.warning("GenericAccessibleObject for code object id %s not found", id_trace[0])
        elif deepmosalanguagemodel._get_gao_str(res) is None:
            logger.warning("Not targetting %s as we can't find its source code", repr(res))

        self._code_id_to_gao.update({id: res for id in id_trace})
        return res

    def _prepare_example_test_str(self, sol: tc.TestCase):
        exporter = PyTestExporter(wrap_code=False)
        return fixup_imports(exporter.export_sequences_to_str([sol]))

    def _convert_fitness_to_probabilty(self):
        def transform_probs(probs: np.ndarray, temp: float) -> np.ndarray:
            scaled = probs ** (1 / temp)
            return scaled / scaled.sum()

        fixed = list(
            (self._query_count_of[goal] + 1) * self._goals_fitness[i]
            for i, goal in enumerate(self._uncovered_goals)
        )
        probs = transform_probs(1 / np.array(fixed), config.deepmosa.tau)

        self._goals_probability = probs.tolist()

    async def target_uncovered_functions(
        self,
        solutions: List[tcc.TestCaseChromosome],
        uncovered_goals: List[BranchCoverageTestFitness],
    ):
        """Generate test cases for functions that are less covered in `archive`.

        Returns:
            a list of LLM-generated test cases.
        """
        assert self.executor is not None

        if uncovered_goals == []:
            raise EarlyStopTargetting()

        self._uncovered_goals = []
        for _ in range(2):
            for goal in uncovered_goals:
                # Only targetting goals with containing gaos
                if self.get_gao_from_id(goal.code_object_id) is None:
                    continue
                # We spent too many query on this goal -> skip
                if self._query_count_of[goal] >= self._query_threshold:
                    continue
                self._uncovered_goals.append(goal)

            if len(self._uncovered_goals) != 0:
                break
            if self._query_threshold < self._query_lim:
                self._query_threshold += 1

        # Early exit if there's no goal to cover
        if len(self._uncovered_goals) == 0:
            raise EarlyStopTargetting()

        # Calculating best fitness for each goal
        self._goals_fitness = []
        for goal in self._uncovered_goals:
            self._goals_fitness.append(min(sol.get_fitness_for(goal) for sol in solutions))

        self._convert_fitness_to_probabilty()
        return await self._target_closest_goal()

    async def _target_closest_goal(self):
        """Target a goal with probability correspond to its fitness"""
        assert len(self._uncovered_goals) == len(self._goals_probability)

        goal = random.choices(self._uncovered_goals, weights=self._goals_probability)[0]

        subject_properties = self.executor.tracer.get_subject_properties()
        code_object = subject_properties.existing_code_objects[goal.code_object_id]
        gao = self.get_gao_from_id(goal.code_object_id)
        assert gao is not None

        pred_lineno = pred_value = None
        if isinstance(goal.goal, BranchGoal):
            pred = subject_properties.existing_predicates[goal.goal.predicate_id]
            pred_lineno = pred.line_no - code_object.code_object.co_firstlineno + 1
            pred_value = goal.goal.value

        test_cases = await self._get_targeted_testcase(gao, pred_lineno, pred_value)

        self._query_count_of[goal] += 1
        return test_cases


deepmosaseeding = _DeepMOSASeeding()

__all__ = ["deepmosaseeding"]
