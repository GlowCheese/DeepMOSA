# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.providers.choice as module_0
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.Choice(random=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = True
    with pytest.raises(TypeError):
        var_1.__call__(var_0, unique=var_2)

def test_case_1():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = True
    var_3 = var_0.__call__(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_0.choice(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = var_0.__call__(var_1)
    var_2.choice(*var_2)

def test_case_4():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = var_0.__call__(var_1)
    var_3 = var_0.__call__(var_1)
    assert var_3 == 'i'
    var_4 = -3907
    with pytest.raises(ValueError):
        var_0.__call__(var_3, var_4)

def test_case_5():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = None
    var_3 = module_0.Choice(seed=var_2, random=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert var_3.seed is None
    var_4 = var_0.__call__(var_1)
    assert var_4 == 'i'
    var_5 = True
    var_6 = var_3.__call__(var_4, var_5, var_4)
    assert var_6 == 'i'
    var_7 = var_3.__call__(var_6)
    assert var_7 == 'i'

def test_case_6():
    var_0 = []
    var_1 = 2287
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(ValueError):
        var_2.__call__(var_0, var_1, var_1)

def test_case_7():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = var_0.__str__()
    assert var_2 == 'Choice'
    var_3 = None
    var_4 = module_0.Choice(seed=var_3, random=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert var_4.seed is None
    var_5 = var_0.__call__(var_2)
    var_6 = var_4.__str__()
    assert var_6 == 'Choice'
    var_7 = module_0.Choice(seed=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_7.random).__module__}.{type(var_7.random).__qualname__}' == 'mimesis.random.Random'
    assert var_7.seed is None
    var_8 = 3733
    with pytest.raises(ValueError):
        var_0.__call__(var_5, var_8, var_5)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = None
    var_3 = module_0.Choice(seed=var_2, random=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert var_3.seed is None
    var_4 = var_0.__call__(var_1)
    var_5 = var_0.__call__(var_1)
    var_6 = module_0.Choice()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_6.seed).__module__}.{type(var_6.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_7 = False
    var_8 = var_3.__call__(var_4, unique=var_7)
    var_9 = True
    var_10 = (var_2, var_7, var_5, var_5)
    var_11 = module_0.Choice(seed=var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_11.random).__module__}.{type(var_11.random).__qualname__}' == 'mimesis.random.Random'
    assert var_11.seed is None
    var_12 = var_11.__call__(var_10, var_9)
    var_8.__call__(var_8, unique=var_8)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = None
    var_3 = module_0.Choice(seed=var_2, random=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert var_3.seed is None
    var_4 = var_3.__str__()
    assert var_4 == 'Choice'
    var_5 = var_0.__str__()
    assert var_5 == 'Choice'
    var_6 = None
    var_7 = module_0.Choice(random=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_7.random).__module__}.{type(var_7.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_7.seed).__module__}.{type(var_7.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_8 = True
    var_9 = [var_1]
    var_10 = var_0.__call__(var_9, var_8)
    var_3.choice()