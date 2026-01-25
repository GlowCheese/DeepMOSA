# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.structure as module_0
import builtins as module_1

def test_case_0():
    var_0 = b'\xb1 4$\xa3E\x85si=\xf0,\x9c\xb1\t'
    var_1 = -4001
    var_2 = False
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.reverse_map(var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

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
    var_0 = lambda x, y: x + y
    module_0.map_structure(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = b'\r\xc8\x08\\\xdf '
    module_0.reverse_map(var_0)

def test_case_5():
    var_0 = module_1.bytes
    var_1 = module_0.register_no_map_class(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_1.type
    var_1 = None
    var_2 = module_0.register_no_map_class(var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.map_structure(var_0, var_1)
    module_0.no_map_instance(var_3)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = b'\r\xc8\x08\\\xdf '
    var_1 = None
    module_0.map_structure_zip(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.map_structure(var_0, var_0)

def test_case_9():
    var_0 = 1
    var_1 = module_0.no_map_instance(var_0)
    var_2 = lambda x, y: x + y
    var_3 = [var_2, var_1]
    module_0.map_structure_zip(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 1
    var_1 = 2
    var_2 = lambda x, y: x + y
    var_3 = [var_0, var_1]
    var_4 = module_0.no_map_instance(var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.map_structure(var_3, var_2)

def test_case_11():
    var_0 = 4
    var_1 = module_0.no_map_instance(var_0)
    var_2 = [var_1, var_1]
    var_3 = lambda x, y: x + y
    var_4 = module_0.map_structure_zip(var_3, var_2)
    var_5 = [var_0, var_0]
    module_0.map_structure(var_5, var_3)