# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.structure as module_0
import builtins as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = module_0.reverse_map(var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.map_structure_zip(var_2, var_3)

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
    var_0 = set()

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
    var_0 = 'wranfrmed'
    module_0.map_structure(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = lambda x, y: x + y
    var_1 = (var_0, var_0)
    module_0.map_structure_zip(var_0, var_1)
    assert var_2 == 'helloworld'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = {}
    var_1 = module_0.reverse_map(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_1.set
    var_3 = {}
    var_4 = module_0.reverse_map(var_3)
    var_5 = module_0.register_no_map_class(var_2)
    module_0.no_map_instance(var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'wranfrmed'
    module_0.map_structure_zip(var_0, var_0)

def test_case_11():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = module_0.no_map_instance(var_1)
    var_3 = 10
    var_4 = [var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 15
    module_0.map_structure(var_2, var_2)

def test_case_12():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = module_0.no_map_instance(var_1)
    var_3 = 10
    var_4 = [var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    assert var_5 == 15
    module_0.map_structure_zip(var_0, var_0)