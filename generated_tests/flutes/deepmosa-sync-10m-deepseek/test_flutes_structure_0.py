# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.structure as module_0
import typing as module_1
import builtins as module_2

def test_case_0():
    var_0 = b'\xb1 4$\xa3E\x85si=\xf0,\x9c\xb1\t'
    var_1 = -4001
    var_2 = False
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.reverse_map(var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.no_map_instance(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.map_structure(var_0, var_0)

def test_case_3():
    pass

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.reverse_map(var_0)

def test_case_5():
    var_0 = module_1._TypeVarLike
    var_1 = module_0.register_no_map_class(var_0)
    assert module_1.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_1.T).__module__}.{type(module_1.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT).__module__}.{type(module_1.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.V_co).__module__}.{type(module_1.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.T_contra).__module__}.{type(module_1.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CT_co).__module__}.{type(module_1.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.AnyStr).__module__}.{type(module_1.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_1.TYPE_CHECKING is False
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 10
    module_0.map_structure(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_2.str
    module_0.no_map_instance(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_2.str
    var_1 = {var_0, var_0}
    module_0.map_structure(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_2.str
    var_1 = None
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 'None'
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.map_structure_zip(var_2, var_2)

def test_case_10():
    var_0 = module_2.str
    var_1 = module_0.map_structure(var_0, var_0)
    var_2 = module_0.no_map_instance(var_1)
    module_0.map_structure(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = lambda x, y: x - y
    var_1 = [var_0, var_0, var_0, var_0]
    module_0.map_structure_zip(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = lambda x, y: x - y
    var_1 = module_0.no_map_instance(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = [var_0, var_0]
    module_0.map_structure_zip(var_0, var_2)