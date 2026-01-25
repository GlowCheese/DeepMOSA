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
    var_3 = (var_1, var_2, var_1)
    var_4 = 5
    var_5 = var_0(items=var_3, length=var_4)
    var_6 = len(var_5)
    assert var_6 == 5

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = True
    var_1 = '61'
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.__call__(var_1, var_0)
    var_3.choice(**var_3)

def test_case_5():
    var_0 = True
    var_1 = ''
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(ValueError):
        var_2.__call__(var_1, var_0)

def test_case_6():
    var_0 = True
    var_1 = '@671K'
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.__call__(var_1, var_0)
    var_4 = module_0.Choice(seed=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    var_5 = var_4.__call__(var_3, var_0, var_4)
    var_6 = module_0.Choice()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_6.seed).__module__}.{type(var_6.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_7 = False
    with pytest.raises(TypeError):
        var_2.__call__(var_4, unique=var_7)

def test_case_7():
    var_0 = '`'
    var_1 = -2265
    var_2 = True
    var_3 = module_0.Choice()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(ValueError):
        var_3.__call__(var_0, var_1, var_2)

def test_case_8():
    var_0 = True
    var_1 = '671'
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.__call__(var_1, var_0)
    var_4 = var_2.__call__(var_3)
    var_5 = module_0.Choice(seed=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    var_6 = 889
    with pytest.raises(ValueError):
        var_2.__call__(var_3, var_6, var_4)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_1, var_2]
    var_4 = 5
    var_5 = False
    var_6 = var_0(items=var_3, length=var_4, unique=var_5)
    var_6.__call__(var_6)