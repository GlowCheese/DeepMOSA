# Check out: https://github.com/GlowCheese/deepmosa
import decimal as module_1

import mimesis.providers.base as module_2
import mimesis.providers.numeric as module_0
import pytest


def test_case_0():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.increment()
    assert var_1 == 1
    var_2 = var_0.increment()
    assert var_2 == 2

def test_case_1():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.floats()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = -411
    var_2 = module_0.Numeric()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = module_0.Numeric()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_4 = var_2.complexes(var_1)
    module_0.Numeric(**var_0)

def test_case_3():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.decimals()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 52
    var_1 = None
    var_2 = module_0.Numeric()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.decimals()
    var_4 = var_2.matrix()
    var_5 = var_2.float_number()
    var_6 = -2210
    var_7 = var_2.float_number(precision=var_6)
    assert var_7 == pytest.approx(-0.0, abs=0.01, rel=0.01)
    var_8 = var_2.floats(n=var_0, precision=var_1)
    var_5.integer_number(end=var_1)

def test_case_5():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'

def test_case_6():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.decimal_number()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.integer_number()
    var_2 = None
    var_3 = module_2.BaseProvider(seed=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'
    var_3.validate_enum(var_2, var_0)

def test_case_8():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.integers()
    var_2 = var_0.float_number()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.decimal_number()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_2 = None
    var_0.complex_number(var_1, start_imag=var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = "2+4hjG+=::,OMqO8'N"
    var_2 = True
    var_3 = var_0.decimals()
    var_4 = var_0.floats(end=var_2)
    var_5 = var_0.increment(var_1)
    assert var_5 == 1
    var_6 = var_0.integer_number()
    var_7 = var_0.integers()
    var_8 = None
    var_9 = var_0.float_number()
    var_10 = var_0.reseed()
    var_11 = var_0.increment(var_8)
    assert var_11 == 1
    var_12 = None
    var_0.decimal_number(var_12)