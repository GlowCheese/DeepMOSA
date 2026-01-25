# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.numeric as module_0
import decimal as module_1
import mimesis.providers.base as module_2

def test_case_0():
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
    var_4 = var_2.increment()
    assert var_4 == 1

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = None
    var_2 = var_0.decimals()
    var_0.validate_enum(var_1, var_1)

def test_case_2():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.matrix()
    var_2 = None
    var_3 = var_0.increment(var_2)
    assert var_3 == 1

def test_case_3():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.decimal_number()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_2 = var_0.increment()
    assert var_2 == 1
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
    var_3 = module_2.BaseProvider()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.Numeric()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.integer_number()
    var_1.validate_enum(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.increment()
    assert var_1 == 1
    var_2 = None
    var_0.integers(var_1, n=var_2)

def test_case_6():
    var_0 = module_0.Numeric()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.numeric.Numeric'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_1 = var_0.decimal_number()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_2 = var_0.increment(var_0)
    assert var_2 == 1
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
    var_3 = module_2.BaseProvider()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_3.random).__module__}.{type(var_3.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_3.seed).__module__}.{type(var_3.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.DATADIR).__module__}.{type(module_2.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.LOCALE_SEP == '-'
    assert f'{type(module_2.MissingSeed).__module__}.{type(module_2.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_2.Seed).__module__}.{type(module_2.Seed).__qualname__}' == 'types.UnionType'