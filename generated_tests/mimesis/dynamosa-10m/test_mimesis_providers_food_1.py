# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.providers.food as module_0
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.Food(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed is None
    assert var_1.locale == 'en'
    var_2 = var_1.vegetable()

@pytest.mark.xfail(strict=True)
def test_case_1():
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
def test_case_2():
    var_0 = module_0.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.vegetable()
    var_2 = var_0.spices()
    var_3 = var_0.dish()
    var_4 = var_0.drink()
    var_5 = var_0.vegetable()
    var_6 = None
    module_0.Food(**var_6)