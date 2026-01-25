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
    var_4 = var_1.choice(*var_3)
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
    var_0 = True
    var_1 = (var_0,)
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.__call__(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = '|\t }M@?*Ttsx'
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.reseed()
    var_4 = var_2.__call__(var_1)
    var_5 = var_2.choice(*var_4)
    var_6 = module_0.Choice(seed=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert var_6.seed is None
    var_7 = 599
    var_8 = var_2.__call__(var_5, var_7)
    var_5.validate_enum(var_2, var_3)

def test_case_7():
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.reseed()
    var_3 = var_1.__call__(var_0)
    var_4 = var_1.choice(*var_3)
    var_5 = module_0.Choice(seed=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    var_6 = 583
    var_7 = var_1.__call__(var_4, var_6)
    var_8 = module_0.Choice()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_8.random).__module__}.{type(var_8.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_8.seed).__module__}.{type(var_8.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_9 = 130
    with pytest.raises(ValueError):
        var_5.__call__(var_3, var_9, var_7)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.__call__(var_0)
    var_3 = module_0.Choice(seed=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    var_4 = True
    var_5 = 583
    var_6 = var_1.__call__(var_0, var_5)
    var_7 = var_3.__call__(var_2, var_4, var_6)
    var_0.__call__(var_2)