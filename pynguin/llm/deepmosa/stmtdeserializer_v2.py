"""A class to deserialize AST nodes into Statements"""

from __future__ import annotations

import ast
from collections.abc import Collection
from copy import deepcopy
from inspect import Parameter
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, cast

import pynguin.testcase.statement as stmt
import pynguin.testcase.variablereference as vr
from libs.custom_logger import getLogger
from pynguin.llm.stmtdeserializer import StatementDeserializer
from pynguin.utils.generic import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)

if TYPE_CHECKING:
    import pynguin.assertion.assertion as ass

_logger = getLogger(__name__)


class _VarrefLinkerVisitor(ast.NodeVisitor):
    """Traverses AST nodes and recursively links them to corresponding
    VariableReferences. Useful for parsing ast.Assign statements.
    """

    def __init__(self, value: Union[vr.VariableReference, List[vr.VariableReference]]):
        self.value = value
        self.result: List[Tuple[str, vr.VariableReference]] = []

    def generic_visit(self, node):
        # Only whitelisted node types are supported. All
        # others return False until an extension is available
        return False

    def visit_Name(self, node):
        assert isinstance(self.value, vr.VariableReference)
        self.result.append((node.id, self.value))
        return True

    def _visit_sequence(self, nodes: list[ast.AST]):
        if isinstance(self.value, vr.VariableReference):
            statement = self.value.test_case.get_statement(self.value.get_statement_position())
            if isinstance(statement, stmt.DictStatement):
                self.value = [e[0] for e in statement.elements]
            elif isinstance(statement, stmt.NonDictCollection):
                self.value = statement.elements
            else:
                return False

        if not isinstance(self.value, Collection) or len(self.value) != len(nodes):
            return False

        original_value = self.value
        for node, value in zip(nodes, original_value):
            self.value = value
            if not self.visit(node):
                self.value = original_value
                return False

        self.value = original_value
        return True

    def visit_List(self, node):
        return self._visit_sequence(node.elts)  # type: ignore

    def visit_Tuple(self, node):
        return self._visit_sequence(node.elts)  # type: ignore


class StatementDeserializerV2(StatementDeserializer):
    """Extended from StatementDeserializer aiming to
    support more flexible syntax in LLM-generated test cases.
    """

    def __init__(self, test_cluster, use_uninterpreted_statements=False):
        super().__init__(test_cluster, use_uninterpreted_statements)

        # Override some default behaviors in parent class
        self._experimental_flag = True

    def add_assert_stmt(self, assert_: ast.Assert) -> bool:
        """Tries to add the assert in `assert_` to the current test case

        Args:
            assert_: The ast.Assert node

        Returns:
            True if the assert was parsed successfully, False otherwise
        """
        result = self.create_assert_stmts(assert_)

        for var_ref, assertion in result:
            self._testcase.get_statement(var_ref.get_statement_position()).add_assertion(assertion)

        return True

    def create_assert_stmts(self, assert_node: ast.Assert, layer: int = 0):
        """Creates many assert statements.
        Number of assert statements = len(assert_node.ops)

        Args:
            assert_node: the ast assert node.

        Returns:
            The corresponding assert statement.
        """

        if not isinstance(assert_node.test, ast.Compare) or layer == 1:
            assert_node = ast.Assert(
                test=ast.Compare(
                    left=ast.Call(
                        func=ast.Name("bool", ctx=ast.Load()),
                        args=[assert_node.test],
                        keywords=[],
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=True)],
                )
            )
            return self.create_assert_stmts(assert_node, 2)

        # assert_node.test | should.be.a(ast.Compare)

        result: List[Tuple[vr.VariableReference, ass.Assertion]] = []

        ops = assert_node.test.ops
        comparators = [assert_node.test.left] + assert_node.test.comparators
        sources: List[vr.VariableReference | None] = [None] * len(comparators)

        for i, op in enumerate(ops):
            lhs, rhs = sources[i], comparators[i + 1]
            if lhs is None:
                lhs = self.create_elements([comparators[i]])
                if lhs is not None:
                    sources[i] = lhs = lhs[0]

            if (
                not isinstance(op, (ast.Is, ast.Eq))
                or lhs is None
                or (assertion := self.create_assertion(lhs, rhs)) is None  # type: ignore
            ):
                if layer != 2:
                    result.extend(
                        self.create_assert_stmts(
                            ast.Assert(
                                test=ast.Compare(
                                    left=comparators[i],
                                    ops=[op],
                                    comparators=[comparators[i + 1]],
                                )
                            ),
                            1,
                        )
                    )
                continue

            result.append((lhs, assertion))

        return result

    def add_assign_stmt(self, assign: ast.Assign) -> bool:
        """Tries to add the assignment in `assign` to the current test case

        Args:
            assign: The ast.Assign node

        Returns:
            True if the assign was parsed successfully, False otherwise
        """
        result = self.create_assign_stmts(assign)
        if result is None:
            return False

        for ref_id, var_ref in result:
            self._ref_dict[ref_id] = var_ref

        return True

    def create_assign_stmts(
        self, assign: ast.Assign
    ) -> list[tuple[str, vr.VariableReference]] | None:
        """Creates the assign statements from an ast.Assign node.

        Args:
            assign: The ast.Assign node

        Returns:
            The statements or None if no statement type matches.
        """
        new_stmt: stmt.Statement | None
        assert len(assign.targets) == 1

        target = assign.targets[0]
        value = assign.value
        var_ref: vr.VariableReference | None = None

        if isinstance(value, ast.Name):
            var_ref = self._ref_dict.get(value.id)
            new_stmt = (
                var_ref.test_case.get_statement(var_ref.get_statement_position())
                if var_ref is not None
                else None
            )
        elif isinstance(value, ast.Constant):
            new_stmt = self.create_stmt_from_constant(value)
        elif isinstance(value, ast.UnaryOp):
            new_stmt = self.create_stmt_from_unaryop(value)
        elif isinstance(value, ast.Call):
            new_stmt = self.create_stmt_from_call(value)
        elif isinstance(value, (ast.List, ast.Set, ast.Dict, ast.Tuple)):
            new_stmt = self.create_stmt_from_collection(value)
        elif self._use_uninterpreted_statements:
            new_stmt = self.create_ast_assign_stmt(value)
        else:
            _logger.debug(f"Assign statement could not be parsed: {ast.unparse(assign)}")
            new_stmt = None
        if new_stmt is None:
            return None

        if isinstance(target, ast.Name):
            assert isinstance(new_stmt, stmt.VariableCreatingStatement)
            var_ref = var_ref or self._testcase.add_variable_creating_statement(new_stmt)
            return [(target.id, var_ref)]
        elif isinstance(new_stmt, stmt.CollectionStatement):
            if isinstance(new_stmt, stmt.NonDictCollection):
                colls = new_stmt.elements
            else:
                colls = [e[0] for e in new_stmt.elements]

            visitor = _VarrefLinkerVisitor(colls)
            if not visitor.visit(target):
                _logger.warning(f"VarrefLinkerVisitor failure: {ast.unparse(assign)}")
                return None

            return visitor.result
        else:
            return None

    def create_variable_references_from_call_args(
        self,
        call_args: list[vr.VariableReference | ast.Starred],
        call_keywords: list[ast.keyword],
        gen_callable: GenericCallableAccessibleObject,
    ):
        """Takes the arguments of an ast.Call node and returns the variable
        references of the corresponding statements.

        Args:
            call_args: the positional arguments
            call_keywords: the keyword arguments
            gen_callable: the callable that is called

        Returns:
            The dict with the variable references of the call_args.

        """
        var_refs: Dict[
            str,
            vr.VariableReference  # normal parameters
            | List[vr.VariableReference]  # *args
            | Dict[str, vr.VariableReference],  # **kwargs
        ] = {}

        # We have to ignore the first parameter (usually 'self') for regular methods and
        # constructors because it is filled by the runtime.
        # TODO(fk) also consider @classmethod, because their first argument is the
        # class, which is also filled by the runtime.
        parameters: List[Tuple[str, Parameter]] = list(
            reversed(list(gen_callable.inferred_signature.signature.parameters.items()))
        )
        if gen_callable.is_method() or gen_callable.is_constructor():
            parameters.pop()
        assert len(set(param for param, _ in parameters)) == len(parameters)

        args_name = kwargs_name = None

        def __add_positional_arg(arg: vr.VariableReference):
            nonlocal args_name

            name, param = parameters[-1]
            if param.kind in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            ):
                var_refs[name] = arg
                parameters.pop()
            else:
                assert param.kind == Parameter.VAR_POSITIONAL
                if name not in var_refs:
                    args_name = name
                    var_refs[name] = []

                assert isinstance(var_refs[name], list)
                var_refs[name].append(arg)  # type: ignore

        for call_arg in call_args:
            if isinstance(call_arg, vr.VariableReference):
                __add_positional_arg(call_arg)
            else:
                if isinstance(call_arg.value, list):
                    for element in call_arg.value:
                        __add_positional_arg(element)
                else:
                    assert isinstance(call_arg.value, vr.VariableReference)
                    name, param = parameters[-1]
                    assert param.kind == Parameter.VAR_POSITIONAL
                    assert name not in var_refs
                    var_refs[name] = call_arg.value

        if len(parameters) > 0:
            name, param = parameters[-1]
            if param.kind == Parameter.VAR_POSITIONAL:
                if name in var_refs:
                    assert name == args_name
                else:
                    args_name = name
                    var_refs[name] = []
                parameters.pop()

        if args_name is not None:
            tmp = var_refs[args_name]
            if isinstance(tmp, list):
                coll_type = self.get_collection_type(tmp, list)
                statement = stmt.ListStatement(self._testcase, coll_type, tmp)
                var_refs[args_name] = self._testcase.add_variable_creating_statement(statement)

        flag = True
        for name, param in parameters:
            if not flag:
                assert param.kind != Parameter.VAR_KEYWORD
            else:
                if param.kind == Parameter.VAR_KEYWORD:
                    kwargs_name = name
                    var_refs[name] = {}
                flag = False
            assert param.kind != Parameter.POSITIONAL_ONLY
            if param.kind == Parameter.VAR_POSITIONAL:
                args_name = name

        remainings: Dict[str, Parameter] = dict(parameters)
        if args_name is not None:
            remainings.pop(args_name)

        for call_keyword in call_keywords:
            kw_arg, kw_value = call_keyword.arg, call_keyword.value
            if kw_arg is None:
                assert kwargs_name is not None
                assert isinstance(var_refs[kwargs_name], dict)
                var_refs[kwargs_name] = call_keyword.value  # type: ignore
            elif kw_arg not in remainings:
                assert kwargs_name is not None
                assert kw_arg not in var_refs
                assert isinstance(var_refs[kwargs_name], dict)
                var_refs[kwargs_name][kw_arg] = kw_value  # type: ignore
            else:
                var_refs[kw_arg] = kw_value  # type: ignore
                remainings.pop(kw_arg)

        if kwargs_name is not None:
            tmp = var_refs[kwargs_name]
            if isinstance(tmp, dict):
                keys, values = [], list(tmp.values())
                for key in tmp.keys():
                    statement = stmt.StringPrimitiveStatement(self._testcase, key)
                    keys.append(self._testcase.add_variable_creating_statement(statement))
                coll_elems = list(zip(keys, values))
                coll_type = self.get_collection_type(coll_elems, dict)
                statement = stmt.DictStatement(self._testcase, coll_type, coll_elems)
                var_refs[kwargs_name] = self._testcase.add_variable_creating_statement(statement)

            remainings.pop(kwargs_name)

        for name, param in remainings.items():
            # default arguments should not appear in var_refs
            assert name not in var_refs
            assert param.default != Parameter.empty

        result: dict[str, vr.VariableReference] = {}
        for k, v in var_refs.items():
            assert isinstance(v, vr.VariableReference)
            result[k] = v

        return result

    def assemble_stmt_from_gen_callable(
        self, gen_callable: GenericCallableAccessibleObject, call: ast.Call
    ) -> stmt.ParametrizedStatement | None:
        """Takes a generic callable and assembles the corresponding
        parametrized statement from it.

        Args:
            gen_callable: the corresponding callable of the cluster
            call: the ast.Call statement

        Returns:
            The corresponding statement.
        """

        original_call = deepcopy(call)

        args = []
        for arg in call.args:
            if isinstance(arg, ast.Starred):
                # Trying to convert arg.value to list
                # if not possible, VariableReference instead

                if isinstance(arg.value, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
                    if isinstance(arg.value, ast.Dict):
                        coll = arg.value.keys
                    else:
                        coll = arg.value.elts
                    arg.value = self.create_elements(coll)  # type: ignore

                elif isinstance(arg.value, ast.Name):
                    var_ref = self._ref_dict.get(arg.value.id)
                    if var_ref is None:
                        return None

                    try:
                        statement = var_ref.test_case.get_statement(
                            var_ref.get_statement_position()
                        )
                        if isinstance(statement, stmt.DictStatement):
                            arg.value = [e[0] for e in statement.elements]  # type: ignore
                        elif isinstance(statement, stmt.NonDictCollection):
                            arg.value = statement.elements  # type: ignore
                        else:
                            arg.value = var_ref  # type: ignore
                    except Exception:
                        _logger.exception("Unable to assemble ast.Name")
                        arg.value = var_ref  # type: ignore
                else:
                    arg.value = self.create_elements([arg.value])  # type: ignore
                    if arg.value is None:
                        return None
                    arg.value = arg.value[0]  # type: ignore

                if arg.value is None:
                    return None

            elif isinstance(arg, ast.Name):
                arg = self._ref_dict.get(arg.id)
            else:
                arg = self.create_elements([arg])
                if arg is None:
                    return None
                arg = arg[0]

            if arg is None:
                return None
            args.append(arg)

        keywords: list[ast.keyword] = []
        for keyword in call.keywords:
            if not isinstance(keyword, ast.keyword):
                return None

            if keyword.arg is None:
                # Minimize the existence of **kwargs as much as possible
                # if no, keyword.value = VariableReference

                if isinstance(keyword.value, ast.Name):
                    var_ref = self._ref_dict.get(keyword.value.id)
                    if var_ref is None:
                        return None

                    try:
                        statement = var_ref.test_case.get_statement(
                            var_ref.get_statement_position()
                        )
                        if isinstance(statement, stmt.DictStatement):
                            keys = []
                            for e in statement.elements:
                                tmp = e[0].test_case.get_statement(e[0].get_statement_position())
                                assert isinstance(tmp, stmt.StringPrimitiveStatement)
                                keys.append(tmp._value)

                            keyword.value = ast.Dict(
                                keys=keys,
                                values=[e[1] for e in statement.elements],  # type: ignore
                            )
                            assert len(keyword.value.keys) == len(keyword.value.values)
                        else:
                            assert False

                    except Exception:
                        _logger.exception("Unable to assemble ast.Name")
                        keyword.value = var_ref  # type: ignore

                if isinstance(keyword.value, ast.Dict):
                    append_later: List[ast.keyword] = []
                    for key, value in zip(keyword.value.keys, keyword.value.values):
                        try:
                            assert key is not None
                            if isinstance(key, str):
                                pass
                            elif isinstance(key, ast.Constant):
                                if not isinstance(key.value, str):
                                    return None
                                key = key.value
                            elif isinstance(key, ast.Name):
                                var_ref = self._ref_dict.get(key.id)
                                if var_ref is None:
                                    return None
                                statement = var_ref.test_case.get_statement(
                                    var_ref.get_statement_position()
                                )
                                if isinstance(statement, stmt.StringPrimitiveStatement):
                                    key = statement._value
                                else:
                                    assert False
                            else:
                                _logger.warning("Key is %s", key)
                                assert False
                        except Exception:
                            _logger.exception("Failed to assemble ast.Dict")
                            break
                        append_later.append(ast.keyword(key, value))
                    else:
                        call.keywords.extend(append_later)
                        continue

                if not isinstance(keyword.value, vr.VariableReference):
                    keyword.value = self.create_elements([keyword.value])  # type: ignore
                    if keyword.value is None:
                        return None
                    keyword.value = keyword.value[0]  # type: ignore

            elif isinstance(keyword.value, ast.Name):
                keyword.value = self._ref_dict.get(keyword.value.id)  # type: ignore
            elif not isinstance(keyword.value, vr.VariableReference):
                keyword.value = self.create_elements([keyword.value])  # type: ignore
                if keyword.value is None:
                    return None
                keyword.value = keyword.value[0]  # type: ignore

            if keyword.value is None:
                return None
            keywords.append(keyword)

        try:
            var_refs = self.create_variable_references_from_call_args(args, keywords, gen_callable)
        except Exception as e:
            # args = ', '.join(args)
            # keywords = ', '.join(f'{kw.arg}: {kw.value}' for kw in keywords)
            _logger.exception(
                "Unable to create variable references for gen_callable!\n"
                f"{ast.unparse(original_call)}"
            )
            return None

        if isinstance(gen_callable, GenericFunction):
            return stmt.FunctionStatement(
                self._testcase,
                cast(GenericCallableAccessibleObject, gen_callable),
                var_refs,
            )
        if isinstance(gen_callable, GenericMethod):
            try:
                self._ref_dict[call.func.value.id]  # type: ignore
            except (KeyError, AttributeError):
                return None
            return stmt.MethodStatement(
                self._testcase,
                gen_callable,
                self._ref_dict[call.func.value.id],  # type: ignore
                var_refs,
            )
        if isinstance(gen_callable, GenericConstructor):
            return stmt.ConstructorStatement(
                self._testcase,
                cast(GenericCallableAccessibleObject, gen_callable),
                var_refs,
            )
        return None
