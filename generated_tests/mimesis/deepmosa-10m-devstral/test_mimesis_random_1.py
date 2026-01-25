# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.random as module_0
import random as module_1

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
    var_1 = None
    var_2 = module_0.Random(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.random.Random'
    assert var_2.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    with pytest.raises(ValueError):
        var_2.weighted_choice(var_0)

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
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.generate_string_by_mask()
    var_2 = var_0.triangular()
    var_2.randints()

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
    var_1 = None
    var_2 = module_0.Random()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.random.Random'
    assert var_2.gauss_next is None
    var_3 = var_2.randints()
    var_2.randbytes(var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.Random(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.random.Random'
    assert var_3.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_4 = ()
    var_5 = True
    var_6 = {var_4: var_5, var_1: var_5}
    var_7 = var_3.weighted_choice(var_6)
    var_3.uniform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
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

def test_case_8():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.generate_string_by_mask()
    var_2 = 'B'
    var_3 = var_0.generate_string_by_mask(digit=var_2)
    var_4 = module_1.Random()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'random.Random'
    assert var_4.gauss_next is None
    assert module_1.TWOPI == pytest.approx(6.283185307179586, abs=0.01, rel=0.01)
    assert module_1.NV_MAGICCONST == pytest.approx(1.7155277699214135, abs=0.01, rel=0.01)
    assert module_1.LOG4 == pytest.approx(1.3862943611198906, abs=0.01, rel=0.01)
    assert module_1.SG_MAGICCONST == pytest.approx(2.504077396776274, abs=0.01, rel=0.01)
    assert module_1.BPF == 53
    assert module_1.RECIP_BPF == pytest.approx(1.1102230246251565e-16, abs=0.01, rel=0.01)
    assert module_1.Random.VERSION == 3

def test_case_9():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = '#'
    with pytest.raises(ValueError):
        var_0.generate_string_by_mask(char=var_1, digit=var_1)