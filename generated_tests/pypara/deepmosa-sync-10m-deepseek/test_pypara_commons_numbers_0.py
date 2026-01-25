# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.commons.numbers as module_0
import decimal as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = -2612
    var_1 = {var_0: var_0}
    var_2 = module_0.isum(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.weirdiv(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
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
    var_4 = module_0.normalize(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    module_0.isum(var_4, var_2)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.isum(var_0)

def test_case_2():
    var_0 = module_1.Decimal()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'decimal.Decimal'
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
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_1 = module_0.normalize(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = False
    var_1 = module_0.sign(var_0)
    assert var_1 == 0
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    var_2 = module_1.Decimal()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
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
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = None
    var_4 = module_0.weirdiv(var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    module_0.make_quantizer(var_3)

def test_case_4():
    var_0 = module_1.Decimal()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'decimal.Decimal'
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
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_1 = module_0.weirdiv(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_5():
    module_0.NaturalNumber()

def test_case_6():
    var_0 = False
    var_1 = module_0.sign(var_0)
    assert var_1 == 0
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 1855
    var_1 = {var_0: var_0}
    var_2 = module_0.isum(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.weirdiv(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
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
    var_4 = module_0.sign(var_2)
    assert var_4 == 1
    var_5 = module_0.normalize(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    var_6 = None
    var_7 = module_0.weirdiv(var_6, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    module_0.normalize(var_6)

def test_case_8():
    var_0 = -1
    var_1 = module_0.sign(var_0)
    assert var_1 == -1
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 0
    var_1 = [var_0]
    module_0.PositiveInteger(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 11
    var_1 = [var_0]
    var_2 = None
    var_3 = 1168
    var_4 = module_0.PositiveInteger(*var_1)
    assert var_4 == 11
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    var_4.__new__(var_2, var_3)

def test_case_11():
    var_0 = 0
    var_1 = [var_0]
    var_2 = module_0.NaturalNumber(*var_1)
    assert var_2 == 0
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.weirdiv(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_2 = module_0.normalize(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
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
    var_3 = -40
    var_4 = [var_3]
    module_0.NaturalNumber(*var_4)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = -2572
    var_1 = {var_0: var_0}
    var_2 = module_0.isum(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.weirdiv(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
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
    module_0.NaturalNumber(**var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 1855
    var_1 = {var_0: var_0}
    var_2 = module_0.isum(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.weirdiv(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
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
    var_4 = module_0.sign(var_2)
    assert var_4 == 1
    var_5 = module_0.normalize(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    var_6 = None
    var_7 = module_0.weirdiv(var_6, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.weirdiv(var_2, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = 'n8>bm\\?tG?/7\rS\tf'
    var_10 = module_0.make_quantize_func(var_6)
    module_0.PositiveInteger(*var_9)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 1864
    var_1 = {var_0: var_0}
    var_2 = module_0.isum(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.make_quantizer(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
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
    var_4 = module_0.sign(var_2)
    assert var_4 == 1
    var_5 = 348
    var_6 = module_0.sign(var_5)
    assert var_6 == 1
    var_7 = module_0.normalize(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.weirdiv(var_2, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.make_quantize_func(var_8)
    var_10 = module_0.normalize(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    var_11 = 'n8>bm\\?tG?/7\rS\tf'
    var_12 = module_0.make_quantize_func(var_2)
    module_0.PositiveInteger(*var_11)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 1855
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.isum(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.DecimalLike).__module__}.{type(module_0.DecimalLike).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.CENT).__module__}.{type(module_0.CENT).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.HUNDRED).__module__}.{type(module_0.HUNDRED).__qualname__}' == 'decimal.Decimal'
    assert module_0.MaxPrecision == 12
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer2).__module__}.{type(module_0.Quantizer2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer4).__module__}.{type(module_0.Quantizer4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer8).__module__}.{type(module_0.Quantizer8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Quantizer12).__module__}.{type(module_0.Quantizer12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_1.Decimal.real).__module__}.{type(module_1.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.Decimal.imag).__module__}.{type(module_1.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.weirdiv(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
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
    var_4 = module_0.weirdiv(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    var_5 = 348
    var_6 = module_0.sign(var_5)
    assert var_6 == 1
    var_7 = module_0.normalize(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.weirdiv(var_2, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.make_quantize_func(var_8)
    var_10 = module_0.weirdiv(var_7, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    var_11 = module_0.normalize(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    var_12 = 'n8>bm\\?tG?/7\rS\tf'
    module_0.PositiveInteger(*var_12)