# Check out: https://github.com/GlowCheese/deepmosa
import decimal as module_1

import mimesis.providers.numeric as module_0
import pytest


def test_case_0():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.increment()
    assert var_1 == 1

def test_case_1():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.complexes()

def test_case_2():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = {}
    var_2 = module_0.Numeric(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = var_2.complexes()
    var_4 = -625
    var_5 = None
    var_6 = var_0.complex_number(precision_real=var_5, precision_imag=var_4)
    var_7 = var_0.decimals()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Numeric()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1.matrix(n=var_0)

def test_case_4():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.matrix()

def test_case_5():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'

def test_case_6():
    var_0 = -28.965285
    var_1 = module_0.Numeric()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.decimal_number(end=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_0.integer_number(var_0)

def test_case_8():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = {}
    var_2 = module_0.Numeric(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_3 = 2468
    var_4 = var_2.integers(var_3)
    var_5 = var_2.complexes()
    var_6 = var_2.increment()
    assert var_6 == 1

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
    var_3 = var_0.increment()
    assert var_3 == 1
    assert f'{type(module_1.DefaultContext).__module__}.{type(module_1.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_1.HAVE_CONTEXTVAR is True
    assert module_1.HAVE_THREADS is True
    assert f'{type(module_1.BasicContext).__module__}.{type(module_1.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_1.ExtendedContext).__module__}.{type(module_1.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_1.MAX_PREC == 999999999999999999
    assert module_1.MAX_EMAX == 999999999999999999
    assert module_1.MIN_EMIN == -999999999999999999
    assert module_1.MIN_ETINY == -1999999999999999997
    assert module_1.ROUND_UP == 'ROUND_UP'
    assert module_1.ROUND_DOWN == 'ROUND_DOWN'
    assert module_1.ROUND_CEILING == 'ROUND_CEILING'
    assert module_1.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_1.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_1.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_1.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_1.ROUND_05UP == 'ROUND_05UP'
    var_4 = var_0.complexes(var_3)
    var_5 = var_0.increment(var_3)
    assert var_5 == 1
    var_0.complex_number(end_real=var_2, precision_real=var_2)