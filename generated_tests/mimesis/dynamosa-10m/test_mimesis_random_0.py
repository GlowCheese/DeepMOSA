# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.random as module_0
import pytest


def test_case_0():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.generate_string_by_mask()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_0.weighted_choice(var_0)

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
    var_0.uniform(var_0, var_0, var_0)

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
    var_0 = '\x0cZXT~.yFvbzVv\nJ0X+;'
    var_1 = module_0.Random()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_2 = var_1.generate_string_by_mask(var_0)
    assert var_2 == '\x0cZXT~.yFvbzVv\nJ0X+;'

def test_case_8():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = var_0.randints()

def test_case_9():
    var_0 = module_0.Random()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.random.Random'
    assert var_0.gauss_next is None
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.global_seed).__module__}.{type(module_0.global_seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.random).__module__}.{type(module_0.random).__qualname__}' == 'mimesis.random.Random'
    assert module_0.random.gauss_next is None
    var_1 = -176
    with pytest.raises(ValueError):
        var_0.randints(var_1)

def test_case_10():
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
        var_0.generate_string_by_mask(char=var_1)