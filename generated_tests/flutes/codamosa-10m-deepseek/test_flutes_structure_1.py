# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_2
import typing as module_0

import flutes.structure as module_1
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0._Final
    var_2 = 2447
    var_3 = {var_0: var_2}
    var_4 = module_1.reverse_map(var_3)
    assert module_0.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT).__module__}.{type(module_0.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.V_co).__module__}.{type(module_0.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.T_contra).__module__}.{type(module_0.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.CT_co).__module__}.{type(module_0.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.R).__module__}.{type(module_1.R).__qualname__}' == 'typing.TypeVar'
    var_5 = module_1.register_no_map_class(var_1)
    module_1.map_structure(var_0, var_0)

def test_case_1():
    var_0 = {}
    var_1 = module_1.reverse_map(var_0)
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.R).__module__}.{type(module_1.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.no_map_instance(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_1.map_structure(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_1.reverse_map(var_0)

def test_case_5():
    var_0 = None
    var_1 = module_1.register_no_map_class(var_0)
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.R).__module__}.{type(module_1.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_2.bytes
    var_1 = module_1.register_no_map_class(var_0)
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.R).__module__}.{type(module_1.R).__qualname__}' == 'typing.TypeVar'
    module_1.no_map_instance(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = [var_0, var_0]
    module_1.map_structure_zip(var_0, var_1)

def test_case_8():
    var_0 = 5
    var_1 = module_1.no_map_instance(var_0)
    var_2 = [var_1, var_1]
    var_3 = lambda x, y: x + y
    var_4 = module_1.map_structure_zip(var_3, var_2)
    module_1.map_structure(var_4, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 1
    var_1 = 3
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_2, var_4]
    var_6 = lambda x, y: x + y
    var_7 = module_1.map_structure_zip(var_6, var_5)
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.R).__module__}.{type(module_1.R).__qualname__}' == 'typing.TypeVar'
    var_8 = lambda x, y: x + y
    module_1.map_structure(var_2, var_8)

def test_case_10():
    var_0 = 12
    var_1 = -4483
    var_2 = 3
    var_3 = None
    var_4 = module_1.register_no_map_class(var_3)
    var_5 = 41
    var_6 = module_1.no_map_instance(var_5)
    var_7 = [var_6, var_6]
    var_8 = lambda x, y: x + y
    var_9 = module_1.map_structure_zip(var_8, var_7)
    var_10 = [var_0, var_1]
    var_11 = [var_2, var_5]
    var_12 = [var_10, var_11]
    module_1.map_structure(var_8, var_12)