# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.food as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.spices()

def test_case_1():
    var_0 = module_0.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.vegetable()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.fruit()
    var_2 = var_0.vegetable()
    var_3 = var_0.drink()
    var_4 = None
    var_0.validate_enum(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.dish()
    var_2 = module_0.Food()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    var_3 = var_2.drink()
    var_4 = None
    var_5 = var_2.dish()
    module_1.BaseDataProvider(*var_4)