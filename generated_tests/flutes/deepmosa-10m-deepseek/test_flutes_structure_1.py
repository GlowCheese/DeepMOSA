# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_1

import flutes.structure as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = {var_0: var_0}
    var_2 = module_0.reverse_map(var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.map_structure_zip(var_1, var_2)

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

def test_case_4():
    pass

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.reverse_map(var_0)

def test_case_6():
    var_0 = None
    var_1 = module_0.register_no_map_class(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'n-'
    module_0.map_structure(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = lambda x, y: x + y
    var_1 = [var_0, var_0]
    module_0.map_structure_zip(var_0, var_1)
    assert var_2 == 'abcd'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_1.dict
    module_0.no_map_instance(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_1.set
    module_0.map_structure_zip(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = lambda x, y: x + y
    var_1 = module_0.no_map_instance(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.map_structure(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = lambda x, y: x + y
    var_1 = [var_0, var_0]
    var_2 = module_0.no_map_instance(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.map_structure_zip(var_0, var_1)
    assert var_3 == 'abcd'