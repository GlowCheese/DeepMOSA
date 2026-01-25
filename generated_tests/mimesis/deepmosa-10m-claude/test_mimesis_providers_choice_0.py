# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.choice as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = b'Y\x06b\x01\xb9y\xf5\xd7\x83\xf1\n\xec\xbf'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.__call__(var_0)
    module_0.Choice(random=var_2)

def test_case_1():
    var_0 = None
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(TypeError):
        var_1.__call__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1.choice(**var_0)

def test_case_3():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2, var_2, var_1, var_2]
    var_4 = 338
    var_5 = var_0(items=var_3, length=var_4)
    with pytest.raises(TypeError):
        var_6 = bool(var_5 in var_4)
    assert var_6 is True

def test_case_4():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = 2
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 2

def test_case_5():
    var_0 = None
    var_1 = True
    var_2 = '61'
    var_3 = module_0.Choice()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_4 = var_3.__call__(var_2, var_1)
    with pytest.raises(TypeError):
        var_3.__call__(var_0, var_0)

def test_case_6():
    var_0 = True
    var_1 = ''
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(ValueError):
        var_2.__call__(var_1, var_0)

def test_case_7():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 5
    var_5 = True
    with pytest.raises(ValueError):
        var_6 = var_0(items=var_3, length=var_4, unique=var_5)

def test_case_8():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 3
    var_6 = True
    var_7 = var_0(items=var_4, length=var_5, unique=var_6)
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = set(var_7)
    var_10 = set(var_4)
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

def test_case_9():
    var_0 = '`'
    var_1 = -2265
    var_2 = True
    var_3 = module_0.Choice()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(ValueError):
        var_3.__call__(var_0, var_1, var_2)