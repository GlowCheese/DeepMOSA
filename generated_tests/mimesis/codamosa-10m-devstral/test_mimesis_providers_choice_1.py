# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Choice(random=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = True
    with pytest.raises(TypeError):
        var_1.__call__(var_0, unique=var_2)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = var_0.__call__(var_1)
    assert var_2 == 'c'
    var_2.choice(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_0.choice(*var_2)

def test_case_3():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = True
    var_3 = var_0.__call__(var_1, var_2, var_2)

def test_case_4():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = True
    var_3 = False
    var_4 = var_0.__call__(var_1, var_2, var_3)

def test_case_5():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = var_0.__str__()
    assert var_2 == 'Choice'
    var_3 = True
    var_4 = var_0.__call__(var_2, var_3, var_3)
    var_5 = -4349
    with pytest.raises(ValueError):
        var_0.__call__(var_4, var_5)

def test_case_6():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = True
    var_3 = True
    var_4 = var_0.__call__(var_1, var_2, var_3)
    var_5 = module_0.Choice()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_6 = var_0.choice(*var_4)
    var_7 = 272
    with pytest.raises(ValueError):
        var_5.__call__(var_4, var_7, var_4)

def test_case_7():
    var_0 = ''
    var_1 = None
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(ValueError):
        var_2.__call__(var_0, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = None
    var_3 = var_0.reseed()
    var_4 = True
    var_5 = var_0.__str__()
    assert var_5 == 'Choice'
    var_6 = module_0.Choice()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_6.seed).__module__}.{type(var_6.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_7 = [var_1]
    var_8 = var_6.__call__(var_7, var_4)
    var_8.__call__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = True
    var_3 = (var_2,)
    var_4 = var_0.__call__(var_3, var_2)
    var_4.choice()