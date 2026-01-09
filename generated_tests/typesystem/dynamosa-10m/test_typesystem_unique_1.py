# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.unique as module_0


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = []
    var_1 = b'\x12\xd8'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    module_0.Uniqueness(var_3)

def test_case_1():
    var_0 = module_0.Uniqueness()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'

def test_case_2():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.add(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = True
    var_2 = module_0.Uniqueness()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_3 = var_2.make_hashable(var_1)
    var_3.make_hashable(var_0)

def test_case_4():
    var_0 = None
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.Uniqueness()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_3 = var_2.add(var_0)
    var_4 = module_0.Uniqueness()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_5 = module_0.Uniqueness()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_6 = module_0.Uniqueness()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_7 = var_5.make_hashable(var_0)
    var_8 = var_4.add(var_1)
    var_9 = None
    var_10 = module_0.Uniqueness()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_11 = var_10.add(var_8)
    var_12 = var_2.add(var_9)
    var_13 = var_10.make_hashable(var_9)
    var_14 = var_2.make_hashable(var_8)
    var_15 = module_0.Uniqueness()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_16 = var_5.add(var_8)
    var_17 = var_2.make_hashable(var_11)

def test_case_5():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = module_0.Uniqueness(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_3 = var_2.make_hashable(var_0)
    var_4 = var_1.__contains__(var_0)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = False
    var_1 = None
    var_2 = [var_0, var_1]
    var_3 = module_0.Uniqueness(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_4 = None
    var_5 = module_0.Uniqueness()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_6 = var_3.make_hashable(var_4)
    var_6.add(var_4)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = module_0.Uniqueness(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    var_5 = [var_0, var_0, var_0]
    var_6 = var_3.make_hashable(var_5)
    var_7 = None
    var_8 = module_0.Uniqueness()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_9 = var_8.make_hashable(var_7)
    var_10 = var_3.make_hashable(var_1)
    var_10.make_hashable(var_7)