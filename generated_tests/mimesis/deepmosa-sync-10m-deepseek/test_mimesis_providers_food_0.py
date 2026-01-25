# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.food as module_0

def test_case_0():
    var_0 = 127
    var_1 = module_0.Food(seed=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert var_1.seed == 127
    assert var_1.locale == 'en'
    var_2 = var_1.drink()
    assert var_2 == 'Bay Breeze'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.vegetable()
    var_2 = {var_0, var_0, var_0, var_0}
    var_3 = var_0.override_locale(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_3.gen).__module__}.{type(var_3.gen).__qualname__}' == 'builtins.generator'
    assert f'{type(var_3.args).__module__}.{type(var_3.args).__qualname__}' == 'builtins.tuple'
    assert len(var_3.args) == 2
    assert var_3.kwds == {}
    var_4 = {}
    var_5 = var_0.vegetable()
    var_6 = var_0.vegetable()
    var_7 = var_0.spices()
    var_8 = module_0.Food(**var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_8.random).__module__}.{type(var_8.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_8.seed).__module__}.{type(var_8.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_8.locale == 'en'
    var_9 = var_0.dish()
    var_10 = var_0.dish()
    var_11 = var_0.get_current_locale()
    assert var_11 == 'en'
    var_12 = None
    module_0.Food(var_12, var_12, *var_12)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.Food()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    var_2 = var_1.fruit()
    var_3 = [var_0, var_0, var_0, var_0]
    var_4 = 'eX9i\ns]}]O'
    var_5 = "1x'fWL]\t]x-M"
    var_6 = {var_4: var_0, var_5: var_0}
    module_0.Food(*var_3, **var_6)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Food(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_2.locale == 'en'
    var_3 = var_2.dish()
    module_0.Food(**var_0)