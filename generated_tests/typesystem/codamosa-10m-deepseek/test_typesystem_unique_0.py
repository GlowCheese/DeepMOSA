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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.add(var_0)
    var_4 = var_1.__contains__(var_0)
    assert var_4 is True
    var_5 = var_1.add(var_4)
    var_6 = module_0.Uniqueness()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_7 = var_6.make_hashable(var_0)
    var_8 = var_6.add(var_7)
    var_9 = var_1.make_hashable(var_3)
    var_7.__contains__(var_7)

def test_case_3():
    var_0 = b'\x01\xf8\xbd\x8b'
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    with pytest.raises(AssertionError):
        var_1.make_hashable(var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.make_hashable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
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
def test_case_6():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = module_0.Uniqueness()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    var_5 = var_1.__contains__(var_0)
    assert var_5 is False
    var_6 = 467.3086
    var_7 = False
    var_8 = var_3.add(var_0)
    var_9 = {var_6: var_7, var_7: var_7, var_6: var_7, var_7: var_7}
    var_10 = var_3.make_hashable(var_9)
    var_11 = var_3.__contains__(var_0)
    assert var_11 is True
    var_1.__contains__(var_10)

def test_case_7():
    var_0 = None
    var_1 = module_0.Uniqueness()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.add(var_0)
    var_4 = None
    var_5 = None
    var_6 = module_0.Uniqueness(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.unique.Uniqueness'
    var_7 = var_6.__contains__(var_4)
    assert var_7 is False

def test_case_8():
    var_0 = None
    var_1 = []
    var_2 = module_0.Uniqueness(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.unique.Uniqueness'
    assert f'{type(module_0.Uniqueness.TRUE).__module__}.{type(module_0.Uniqueness.TRUE).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.Uniqueness.FALSE).__module__}.{type(module_0.Uniqueness.FALSE).__qualname__}' == 'builtins.object'
    var_3 = var_2.add(var_0)