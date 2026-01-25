# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = 'FIXED-STRING'
    var_2 = '&'
    var_3 = var_0.generate_string_by_mask(char=var_2)
    var_4 = var_0.generate_string_by_mask(var_1)
    assert var_4 == 'FIXED-STRING'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.Random()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2 = None
    var_3 = None
    var_4 = var_1.generate_string_by_mask()
    var_5 = module_0.Random()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.random.Random'
    assert var_5.gauss_next is None
    var_6 = 2700
    var_7 = {var_0: var_6}
    var_8 = var_5.weighted_choice(var_7)
    var_5.randints(var_3, var_2, var_8)

def test_case_2():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = None
    var_2 = module_0.Random()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.random.Random'
    assert var_2.gauss_next is None
    with pytest.raises(ValueError):
        var_2.weighted_choice(var_1)

def test_case_3():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = None
    var_2 = module_0.Random()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.random.Random'
    assert var_2.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2.uniform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.randbytes()
    var_2 = None
    var_3 = var_0.generate_string_by_mask()
    var_4 = var_0.generate_string_by_mask()
    var_5 = module_0.Random()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.random.Random'
    assert var_5.gauss_next is None
    var_0.paretovariate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_0.choice_enum_item(var_0)

def test_case_7():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.randints()
    var_2 = None
    with pytest.raises(ValueError):
        var_0.weighted_choice(var_2)

def test_case_8():
    var_0 = None
    var_1 = module_0.Random()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2 = module_0.Random()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.random.Random'
    assert var_2.gauss_next is None
    var_3 = False
    with pytest.raises(ValueError):
        var_2.randints(var_3, var_0)

def test_case_9():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = '@@##'
    var_2 = '@'
    with pytest.raises(ValueError):
        var_0.generate_string_by_mask(var_1, var_2, var_2)