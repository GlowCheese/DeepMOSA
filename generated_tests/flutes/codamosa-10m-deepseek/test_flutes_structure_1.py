# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.structure as module_0
import builtins as module_1
import typing as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_0.reverse_map(var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.map_structure(var_0, var_0)

def test_case_1():
    var_0 = {}
    var_1 = module_0.reverse_map(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.no_map_instance(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.map_structure(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.reverse_map(var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.register_no_map_class(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_1.bytes
    var_1 = module_0.register_no_map_class(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.no_map_instance(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = (var_0,)
    module_0.map_structure_zip(var_0, var_1)

def test_case_8():
    var_0 = 3276
    var_1 = module_0.no_map_instance(var_0)
    module_0.map_structure(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = 195
    var_2 = {var_1: var_1, var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = module_2._TypeVarLike
    var_4 = module_0.register_no_map_class(var_3)
    assert module_2.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_2.T).__module__}.{type(module_2.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.KT).__module__}.{type(module_2.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT).__module__}.{type(module_2.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.T_co).__module__}.{type(module_2.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.V_co).__module__}.{type(module_2.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.VT_co).__module__}.{type(module_2.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.T_contra).__module__}.{type(module_2.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.CT_co).__module__}.{type(module_2.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_2.AnyStr).__module__}.{type(module_2.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_2.TYPE_CHECKING is False
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.reverse_map(var_2)
    var_6 = module_0.reverse_map(var_2)
    module_0.map_structure_zip(var_2, var_6)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 3276
    module_0.map_structure(var_0, var_0)

def test_case_11():
    var_0 = 3276
    var_1 = module_0.no_map_instance(var_0)
    var_2 = -1511
    var_3 = (var_1, var_1)
    module_0.map_structure_zip(var_2, var_3)