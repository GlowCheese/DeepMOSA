# Check out: https://github.com/GlowCheese/deepmosa
import decimal as module_2

import mimesis.providers.base as module_1
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

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.floats()
    var_2 = None
    var_3 = None
    var_4 = module_1.BaseProvider(seed=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert var_4.seed is None
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_4.validate_enum(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.floats()
    var_2 = var_0.increment()
    assert var_2 == 1
    var_3 = var_0.float_number()
    var_4 = None
    var_5 = var_0.floats(end=var_2)
    var_6 = module_0.Numeric()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_6.random).__module__}.{type(var_6.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_6.seed).__module__}.{type(var_6.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_7 = var_6.complexes()
    var_8 = var_6.decimal_number(end=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    module_0.Numeric(*var_4)

@pytest.mark.xfail(strict=True)
def test_case_3():
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

def test_case_4():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.decimal_number()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_5():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.increment()
    assert var_1 == 1
    var_2 = var_0.increment()
    assert var_2 == 2
    var_3 = var_0.integer_number()

def test_case_6():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.integers()
    var_2 = 1085
    var_3 = var_0.float_number(var_2, var_2)
    assert var_3 == pytest.approx(1085.0, abs=0.01, rel=0.01)
    var_4 = var_0.float_number()

def test_case_7():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.floats()
    var_2 = var_0.increment()
    assert var_2 == 1
    var_3 = None
    var_4 = var_0.decimal_number()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = var_0.float_number(precision=var_2)
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_6 = var_0.decimal_number()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = var_0.complex_number(precision_real=var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = "2+4hjG+=::,OMqO8'N"
    var_2 = False
    var_3 = var_0.floats(end=var_2)
    var_4 = var_0.increment(var_1)
    assert var_4 == 1
    var_5 = var_0.integers()
    var_6 = None
    var_7 = var_0.float_number()
    var_8 = var_0.reseed()
    var_9 = var_0.increment(var_6)
    assert var_9 == 1
    var_10 = None
    var_0.decimal_number(var_10)