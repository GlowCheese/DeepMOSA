# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.random as module_0
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2 = -640
    with pytest.raises(ValueError):
        var_1.randints(var_2)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = -1769
    var_1 = module_0.Random()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2 = var_1.uniform(var_0, var_0)
    assert var_2 == pytest.approx(-1769.0, abs=0.01, rel=0.01)
    var_3 = None
    var_1.randints(b=var_3)

def test_case_2():
    var_0 = None
    var_1 = module_0.Random()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2 = var_1.generate_string_by_mask()
    var_3 = '9'
    var_4 = var_1.generate_string_by_mask(digit=var_3)
    with pytest.raises(ValueError):
        var_1.weighted_choice(var_0)

def test_case_3():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None

def test_case_4():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.generate_string_by_mask()
    var_2 = '9'
    var_3 = var_0.generate_string_by_mask(digit=var_2)

def test_case_5():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = -547
    var_2 = var_0.randints(a=var_1)
    var_3 = var_0.generate_string_by_mask()
    var_4 = '<'
    var_5 = var_0.generate_string_by_mask(digit=var_4)

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
    var_1 = var_0.generate_string_by_mask()
    var_2 = '<'
    var_3 = var_0.generate_string_by_mask(digit=var_2)
    var_0.weighted_choice(var_3)

def test_case_7():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.generate_string_by_mask()
    var_2 = var_0.randbytes()
    var_3 = '9'
    var_4 = var_0.generate_string_by_mask(digit=var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = -1769
    var_1 = module_0.Random()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2 = var_1.uniform(var_0, var_0)
    assert var_2 == pytest.approx(-1769.0, abs=0.01, rel=0.01)
    var_3 = None
    var_1.choice_enum_item(var_3)

def test_case_9():
    var_0 = None
    var_1 = '4l\nY$D~'
    var_2 = module_0.Random()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.random.Random'
    assert var_2.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_3 = var_2.randints()
    var_4 = 676
    var_5 = 598.253
    var_6 = {var_2: var_4, var_1: var_4, var_0: var_5, var_4: var_5}
    var_7 = var_2.weighted_choice(var_6)
    assert var_7 == 676
    var_8 = -547
    var_9 = var_2.randints(a=var_8)
    var_10 = var_2.generate_string_by_mask()
    var_11 = '<'
    with pytest.raises(ValueError):
        var_2.generate_string_by_mask(char=var_11, digit=var_11)