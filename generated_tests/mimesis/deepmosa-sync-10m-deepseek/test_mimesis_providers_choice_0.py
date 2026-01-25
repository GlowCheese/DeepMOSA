# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.choice as module_0

def test_case_0():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'aa'
    var_2 = 3
    var_3 = True
    with pytest.raises(ValueError):
        var_4 = var_0(items=var_1, length=var_2, unique=var_3)

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
    var_0 = '|\t }M@?*Ttsx'
    var_1 = module_0.Choice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.__call__(var_0)
    var_3 = None
    var_4 = None
    var_5 = module_0.Choice(random=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_5.seed).__module__}.{type(var_5.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    with pytest.raises(TypeError):
        var_1.__call__(var_3)

def test_case_4():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = (var_1, var_2, var_3)
    var_5 = 5
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 5

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

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = True
    var_1 = '671K'
    var_2 = module_0.Choice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.__call__(var_1, var_0)
    var_4 = var_2.__call__(var_3)
    var_5 = module_0.Choice(seed=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    var_6 = var_5.__call__(var_3, var_0, var_5)
    module_0.Choice(random=var_3)

def test_case_8():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = -1
    with pytest.raises(ValueError):
        var_6 = var_0(items=var_4, length=var_5)

def test_case_9():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = var_0(items=var_4, length=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0]