# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.__call__(var_0)

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
    var_0 = ''
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(ValueError):
        var_1.__call__(var_0)

def test_case_4():
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.reseed()
    var_3 = var_1.__call__(var_0)
    assert var_3 == '|'
    var_4 = var_1.choice(*var_3)
    assert var_4 == '|'
    var_5 = module_0.Choice()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_6 = module_0.Choice()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_6.seed).__module__}.{type(var_6.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_7 = -1358
    with pytest.raises(ValueError):
        var_6.__call__(var_4, var_7)

def test_case_5():
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = 611
    var_3 = var_1.__call__(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = 'I^_\r_@Nt,'
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.reseed()
    var_4 = var_2.__call__(var_1)
    var_5 = var_2.choice(*var_4)
    var_6 = None
    var_7 = module_0.Choice(seed=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_7.random).__module__}.{type(var_7.random).__qualname__}' == 'mimesis.random.Random'
    assert var_7.seed is None
    var_8 = var_4.__str__()
    var_9 = True
    var_10 = var_0.__str__()
    assert var_10 == 'None'
    var_11 = var_2.__call__(var_4, var_9, var_9)
    var_12 = var_2.__str__()
    assert var_12 == 'Choice'
    var_12.__call__(var_3, var_9)

def test_case_7():
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.reseed()
    var_3 = var_1.__call__(var_0)
    var_4 = None
    var_5 = module_0.Choice(seed=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert var_5.seed is None
    var_6 = 599
    var_7 = var_1.__call__(var_3, var_6)
    var_8 = module_0.Choice()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_8.random).__module__}.{type(var_8.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_8.seed).__module__}.{type(var_8.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_9 = var_8.__str__()
    assert var_9 == 'Choice'
    var_10 = var_7.__str__()
    var_11 = module_0.Choice()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_11.random).__module__}.{type(var_11.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_11.seed).__module__}.{type(var_11.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_12 = True
    var_13 = 2439
    var_14 = var_8.__str__()
    assert var_14 == 'Choice'
    with pytest.raises(ValueError):
        var_1.__call__(var_7, var_13, var_12)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '|\t }M*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.reseed()
    var_3 = var_1.__call__(var_0)
    assert var_3 == '\t'
    var_4 = var_1.choice(*var_3)
    assert var_4 == '\t'
    var_5 = None
    var_6 = module_0.Choice(seed=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert var_6.seed is None
    var_7 = var_6.__call__(var_3)
    assert var_7 == '\t'
    var_8 = var_4.__str__()
    assert var_8 == '\t'
    var_9 = (var_6,)
    var_10 = 678
    var_11 = var_1.__call__(var_9, var_10)
    var_11.__call__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = var_1.__call__(var_2, var_3)
    var_1.choice()