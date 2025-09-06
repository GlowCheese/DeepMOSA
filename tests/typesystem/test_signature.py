import pytest
import inspect

from grappa import should
from inspect import Parameter

from pynguin.typesystem import TypeSystem
from pynguin.typesystem import Instance, TypeInfo
from pynguin.config import TypeInferenceStrategy


# shorthand alias
mt = Parameter.empty
pos_only = Parameter.POSITIONAL_ONLY
pos_or_kw = Parameter.POSITIONAL_OR_KEYWORD
kw_only = Parameter.KEYWORD_ONLY
var_pos = Parameter.VAR_POSITIONAL  # *args
var_kw = Parameter.VAR_KEYWORD  # **kwargs


def __f0(x: int) -> int: return x
def __f1(x: int): pass
def __f2(*, y): pass
def __f3(x, *, y): pass
def __f4(x: str='z'): pass
def __f5(*, y='z'): pass
def __f6(*args): pass
def __f7(*args, y: Parameter): pass
def __f8(x, **kwargs): pass
def __f9(x, y='z'): pass
def __f10(a, b, /, c, *args, d, e=5, **kwargs): pass


@pytest.mark.parametrize(
    "func, expected",
    [
        (__f1, {'x': (pos_or_kw, mt, int)}),
        (__f2, {'y': kw_only}),
        (__f3, {'x': pos_or_kw, 'y': kw_only}),
        (__f4, {'x': (pos_or_kw, 'z', str)}),
        (__f5, {'y': (kw_only, 'z', mt)}),
        (__f6, {'args': var_pos}),
        (__f7, {'args': var_pos, 'y': (kw_only, mt, Parameter)}),
        (__f8, {'x': pos_or_kw, 'kwargs': var_kw}),
        (__f9, {'x': pos_or_kw, 'y': (pos_or_kw, 'z', mt)}),
        (__f10, {'a': pos_only, 'b': pos_only, 'c': pos_or_kw,
                 'args': var_pos, 'd': kw_only, 'e': (kw_only, 5, mt),
                 'kwargs': var_kw})
    ]
)
def test_inspect_signature(func, expected: dict):
    params = inspect.signature(func).parameters
    has_var_kw = False
    for name, prm in expected.items():
        has_var_kw | should.be.false
        if isinstance(prm, tuple):
            kind, default, annotation = prm
        else:
            kind, default, annotation = prm, mt, mt
        params[name].kind | should.equal(kind)
        params[name].default | should.equal(default)
        params[name].annotation | should.equal(annotation)
        if kind == var_kw:
            has_var_kw = True


@pytest.mark.parametrize(
    "func, infer_types, expected_parameters, expected_return",
    [
        pytest.param(
            __f0,
            TypeInferenceStrategy.TYPE_HINTS,
            {"x": Instance(TypeInfo(int))},
            Instance(TypeInfo(int))
        )
    ]
)
def test_infer_type_info(func, infer_types, expected_parameters, expected_return):
    type_system = TypeSystem()
    result = type_system.infer_type_info(func, type_inference_strategy=infer_types)
    result.original_parameters | should.equal(expected_parameters)
    result.return_type | should.equal(expected_return)
