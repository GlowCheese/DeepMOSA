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
    assert var_3 == 'o'

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
    assert var_4 == 'c'
    var_5 = -4349
    with pytest.raises(ValueError):
        var_0.__call__(var_4, var_5)

def test_case_6():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = var_0.__call__(var_4, var_5)

def test_case_7():
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
    var_7 = True
    var_8 = var_0.__call__(var_4, var_7, var_4)
    var_9 = 192
    with pytest.raises(ValueError):
        var_0.__call__(var_4, var_9, var_6)

def test_case_8():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = []
    var_2 = None
    with pytest.raises(ValueError):
        var_0.__call__(var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Choice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.choice.Choice'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.__str__()
    assert var_1 == 'Choice'
    var_2 = set()
    var_3 = True
    var_4 = {}
    var_5 = (var_0, var_2, var_3, var_4)
    var_6 = True
    var_7 = var_0.__call__(var_5, var_6)
    var_8 = True
    var_9 = var_0.__call__(var_1, var_8, var_8)
    var_10 = var_0.choice(*var_9)
    var_11 = var_0.__call__(var_9, var_8, var_9)
    var_12 = None
    var_12.validate_enum(var_11, var_11)