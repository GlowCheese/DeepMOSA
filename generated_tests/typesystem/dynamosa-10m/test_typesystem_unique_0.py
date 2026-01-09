# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.unique as module_0


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.make_hashable(var_0)
    var_3 = (var_2,)
    var_4 = [var_3]
    module_0.Uniqueness(var_4)

def test_case_1():
    var_0 = module_0.Uniqueness()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'

def test_case_2():
    var_0 = None
    var_1 = []
    var_2 = module_0.Uniqueness(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_3 = var_2.add(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.Uniqueness()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_0.add(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = False
    var_2 = [var_1, var_1]
    var_3 = module_0.Uniqueness()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_4 = var_3.make_hashable(var_2)
    var_4.make_hashable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.Uniqueness()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_1 = None
    var_2 = var_0.make_hashable(var_1)
    var_3 = module_0.Uniqueness()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_4 = 467.3086
    var_5 = False
    var_6 = {var_4: var_5, var_5: var_5, var_4: var_5, var_5: var_5}
    var_7 = var_3.make_hashable(var_6)
    var_3.__contains__(var_7)

def test_case_6():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.add(var_0)
    var_4 = None
    var_5 = module_0.Uniqueness(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_6 = var_5.__contains__(var_3)
    assert var_6 is False

def test_case_7():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = True
    var_4 = 159.22
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Uniqueness(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.unique.Uniqueness'