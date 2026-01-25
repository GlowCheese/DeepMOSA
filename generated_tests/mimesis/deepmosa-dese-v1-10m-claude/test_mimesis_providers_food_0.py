# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.base as module_0
import mimesis.providers.food as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = (-3998.7247+232.465j)
    var_1 = module_0.BaseProvider()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.DATADIR).__module__}.{type(module_0.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_0.LOCALE_SEP == '-'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1.validate_enum(var_0, var_0)

def test_case_1():
    var_0 = module_1.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.vegetable()
    var_2 = var_0.drink()
    var_3 = var_0.vegetable()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = var_0.fruit()
    var_2 = None
    module_1.Food(*var_2, **var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'H<_O@aG]'
    var_1 = module_1.Food()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.locale == 'en'
    var_2 = var_1.dish()
    var_3 = var_1.spices()
    var_4 = var_1.vegetable()
    var_5 = module_1.Food(seed=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_5.random).__module__}.{type(var_5.random).__qualname__}' == 'mimesis.random.Random'
    assert var_5.seed == 'H<_O@aG]'
    assert var_5.locale == 'en'
    var_6 = var_5.vegetable()
    assert var_6 == 'Avocado'
    var_7 = var_1.fruit()
    var_8 = var_1.vegetable()
    var_9 = [var_0]
    module_1.Food(*var_9, seed=var_9)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_1.Food()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.food.Food'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.locale == 'en'
    var_1 = 'baOmA9dlze`i"j'
    var_2 = 'U ?2Bo82t%?GQ'
    var_3 = None
    var_4 = '!.\\`&)UWC+'
    var_5 = 'i'
    var_6 = {var_2: var_3, var_4: var_3, var_2: var_3, var_5: var_3}
    var_7 = var_0.spices()
    module_0.BaseDataProvider(var_1, var_1, **var_6)