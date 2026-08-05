# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.commons.numbers as module_0
import decimal as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.isum(var_0, var_0)

def test_case_1():
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

def test_case_2():
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

def test_case_3():
    pass

def test_case_4():
    var_0 = 3126
    var_1 = module_0.make_quantizer(var_0)
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
    var_2 = module_0.weirdiv(var_1, var_1)
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

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '/\t\x0bl8v"Oomm|-'
    var_1 = False
    module_0.isum(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 3126
    var_1 = module_0.make_quantizer(var_0)
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
    var_3 = [var_2]
    var_4 = module_0.NaturalNumber(*var_3)
    assert var_4 == 0
    module_0.PositiveInteger(*var_3)

def test_case_7():
    var_0 = 3126
    var_1 = module_0.make_quantizer(var_0)
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
    var_2 = module_0.weirdiv(var_1, var_1)
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
    var_3 = module_0.normalize(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    var_4 = None
    var_5 = module_0.weirdiv(var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    var_6 = b'Q&\xb2wvF\x93\x81P\x83\x17\xe1\xdc\xfa{'
    var_7 = module_0.isum(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.weirdiv(var_7, var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.normalize(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    var_10 = module_0.sign(var_0)
    assert var_10 == 1

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = -247
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
    var_2 = None
    var_3 = [var_2, var_2]
    module_0.PositiveInteger(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = -1066
    var_1 = module_0.make_quantizer(var_0)
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
    var_3 = module_0.weirdiv(var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    var_4 = None
    var_5 = module_0.weirdiv(var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    var_6 = module_0.sign(var_5)
    assert var_6 == 0
    var_6.__new__(var_4, var_4)

def test_case_10():
    var_0 = 3105
    var_1 = module_0.make_quantizer(var_0)
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
    var_2 = [var_1]
    var_3 = module_0.NaturalNumber(*var_2)
    assert var_3 == 0
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

def test_case_11():
    var_0 = 3126
    var_1 = module_0.make_quantizer(var_0)
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
    var_3 = [var_0]
    var_4 = module_0.NaturalNumber(*var_3)
    assert var_4 == 3126
    var_5 = -2357
    var_6 = module_0.make_quantizer(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.PositiveInteger(*var_3)
    assert var_7 == 3126

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = -1426
    var_1 = [var_0]
    module_0.NaturalNumber(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 3126
    var_1 = module_0.make_quantizer(var_0)
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
    var_2 = None
    var_3 = None
    var_4 = module_0.weirdiv(var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
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
    var_5 = b'Q&\xb2wvF\x93\x81P\x83\x17\xe1\xdc\xfa{'
    var_6 = module_0.isum(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.weirdiv(var_6, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_6.__new__(var_2, var_6)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 3114
    var_1 = module_0.make_quantizer(var_0)
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
    var_3 = -2357
    var_4 = None
    var_5 = module_0.make_quantize_func(var_4)
    var_6 = module_0.make_quantizer(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.weirdiv(var_1, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = b'Q&\xb2wvF\x93\x81P\x83\x17\xe1\xdc\xfa{'
    var_9 = module_0.isum(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    var_10 = module_0.weirdiv(var_9, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    module_0.normalize(var_4)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 3126
    var_1 = module_0.make_quantizer(var_0)
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
    var_3 = -2357
    var_4 = None
    var_5 = module_0.make_quantize_func(var_4)
    var_6 = module_0.make_quantizer(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = None
    var_8 = module_0.make_quantizer(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = b'Q&\xb2wvF\x93\x81P\x83\x17\xe1\xdc\xfa{'
    var_10 = module_0.isum(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    var_11 = module_0.weirdiv(var_10, var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    var_12 = module_0.weirdiv(var_10, var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'decimal.Decimal'
    module_0.normalize(var_7)

def test_case_16():
    var_0 = 3126
    var_1 = module_0.make_quantizer(var_0)
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
    var_3 = None
    var_4 = module_0.make_quantize_func(var_3)
    var_5 = None
    var_6 = module_0.weirdiv(var_1, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = b'Q&\xb2wvF\x93\x81P\x83\x17\xe1\xdc\xfa{'
    var_8 = module_0.isum(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.weirdiv(var_8, var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    var_10 = module_0.weirdiv(var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    var_11 = module_0.normalize(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    var_12 = module_0.sign(var_8)
    assert var_12 == 1