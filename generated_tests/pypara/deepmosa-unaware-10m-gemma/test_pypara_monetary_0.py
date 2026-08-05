# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.monetary as module_0
import pypara.commons.errors as module_1
import decimal as module_2

def test_case_0():
    var_0 = module_0.NoneMoney()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'

def test_case_1():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.NoneMoney()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_2 = var_1.scalar_add(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_3 = None
    var_4 = var_2.round()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_5 = var_4.is_some(var_1)
    assert var_5 is False
    var_6 = var_2.dov_or_none()
    var_7 = var_1.scalar_add(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = module_0.NoneMoney()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7.qty_or_else(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.ccy_or(var_0)
    var_4 = var_3.__ge__(var_0)
    var_2.divide(var_0)

def test_case_4():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.ccy_or_none()
    assert var_4 is True
    var_5 = var_2.qty_or_else(var_3)
    assert var_5 is True
    var_6 = var_2.gte(var_2)
    assert var_6 is True
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_4, strict=var_3)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.NoneMoney()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_2 = var_1.__round__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_1.qty_or_else(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.NoneMoney()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_2 = var_1.ccy_or_none()
    var_3 = var_1.subtract(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_2.__setattr__(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.NoneMoney()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_1.dimap(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = {}
    var_2 = module_0.NoneMoney(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.ccy_or_none()
    var_3.__setattr__(var_0, var_0)

def test_case_9():
    var_0 = []
    var_1 = module_0.NoneMoney(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_2 = var_1.qty_or_zero()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = var_1.abs()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
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
    var_4 = var_3.dov_or_none()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.lt(var_0)
    assert var_1 is False
    var_2 = var_0.__abs__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.__floordiv__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.times(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.ccy_or_none()
    var_6 = var_0.ccy_or(var_5)
    var_7 = var_0.scalar_subtract(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8 = var_7.add(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.NonePrice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_2 = var_1.convert(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_1.__pos__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_1.round()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4.__lt__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.NonePrice()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_2 = var_1.__floordiv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.__mul__(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_1.ccy_or_none()
    var_5 = var_4.__ge__(var_0)
    var_5.ccy_or(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.subtract(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = var_0.times(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.ccy_or_none()
    var_7 = None
    var_8 = var_5.qty_or(var_6)
    var_9 = var_1.ccy_or(var_7)
    var_10 = module_0.NoneMoney()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_11 = var_10.__round__()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_10.qty_or_else(var_4)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = var_0.with_ccy(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.__floordiv__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.__round__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_0.__sub__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = None
    var_0.dimap(var_1, var_1)

def test_case_17():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = var_0.negative()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = None
    var_5 = var_0.ccy_or(var_3)
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.__round__(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_5.__repr__()
    assert var_8 == 'None'

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.qty_or_none()
    var_2 = var_0.ccy_or_none()
    var_3 = var_2.__repr__()
    assert var_3 == 'None'
    var_0.qty_or_else(var_2)

def test_case_19():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.qty_or_none()
    var_2 = var_0.dov_or(var_1)
    var_3 = var_0.convert(var_1, strict=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.positive()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_3.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = None
    var_6 = var_3.lt(var_3)
    assert var_6 is False
    var_7 = var_2.ccy_or_none()
    assert var_7 is False
    var_8 = var_2.is_some(var_4)
    assert var_8 is True
    var_8.dov_or(var_5)

def test_case_21():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.qty_or_none()
    var_2 = var_0.dov_or(var_1)
    var_3 = var_0.with_qty(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_3.with_dov(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.dov_or_none()
    var_3 = var_0.__le__(var_1)
    assert var_3 is True
    var_2.floor_divide(var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.positive()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.lte(var_3)
    assert var_5 is True
    var_6 = var_2.subtract(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_6.lt(var_4)
    assert var_7 is False
    var_8 = module_0.NoneMoney()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_9 = var_6.__add__(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    var_10 = var_9.gt(var_4)
    assert var_10 is True
    var_11 = module_0.NoneMoney()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_12 = var_4.lte(var_2)
    assert var_12 is True
    var_2.subtract(var_12)

def test_case_24():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.as_boolean()
    assert var_5 is False
    var_6 = var_4.__floordiv__(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.gte(var_2)
    assert var_7 is False
    var_8 = var_6.lte(var_6)
    assert var_8 is True
    var_9 = var_2.gt(var_6)
    assert var_9 is True

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.divide(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.__floordiv__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = None
    var_6 = None
    var_7 = var_0.times(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_8 = var_7.ccy_or_none()
    var_9 = var_0.ccy_or(var_8)
    var_10 = var_9.__lt__(var_5)
    var_10.qty_map(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.qty_or_else(var_0)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_0.qty_map(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.__floordiv__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = None
    var_6 = None
    var_7 = var_0.times(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_8 = var_7.ccy_or_none()
    var_9 = var_0.ccy_or(var_8)
    var_10 = var_7.is_equal(var_5)
    assert var_10 is False
    var_10.qty_map(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.NoneMoney()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_0.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = None
    var_2 = module_0.NoneMoney()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.positive()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4 = var_2.multiply(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_1.round(var_1)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.NoneMoney()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_2 = var_1.__floordiv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_3 = var_1.multiply(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4 = var_2.round(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_1.or_else(var_2)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.lt(var_2)
    assert var_3 is False
    var_4 = var_2.with_ccy(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.gt(var_5)
    assert var_6 is False
    var_7 = var_2.gte(var_3)
    assert var_7 is True
    module_2.Decimal(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.NoneMoney()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_2 = var_1.with_qty(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_0.__setattr__(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = None
    var_2 = module_0.NoneMoney()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.divide(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4 = var_3.dov_or(var_0)
    var_5 = var_2.ccy_or(var_0)
    var_6 = var_5.__repr__()
    assert var_6 == 'None'
    var_6.qty_or_zero()

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.lte(var_0)
    assert var_1 is True
    var_2 = var_0.with_ccy(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.__floordiv__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_0.__gt__(var_3)
    assert var_8 is False
    var_9 = var_6.with_ccy(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_10 = var_0.convert(var_7, var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NonePrice'
    var_10.as_float()

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = None
    var_4 = var_0.is_equal(var_2)
    assert var_4 is False
    var_5 = var_1.add(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_6 = var_3.__repr__()
    assert var_6 == 'None'
    var_6.qty_or_else(var_0)

def test_case_37():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = None
    var_7 = module_0.NonePrice()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_8 = var_7.round()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NonePrice'
    var_9 = var_8.__gt__(var_6)
    assert var_9 is False
    var_10 = var_2.add(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_10) == 3
    var_11 = var_10.as_boolean()
    assert var_11 is False
    var_12 = var_2.gte(var_2)
    assert var_12 is True
    var_13 = var_2.lte(var_4)
    assert var_13 is True
    var_14 = module_0.SomePrice(*var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_15 = var_14.__add__(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = var_15.subtract(var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_14.lte(var_15)
    assert var_17 is True
    var_18 = var_14.lt(var_15)
    assert var_18 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_18)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.gte(var_0)
    assert var_1 is True
    var_2 = var_0.__round__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.__floordiv__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.times(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.ccy_or_none()
    var_6 = var_0.ccy_or(var_5)
    var_7 = var_0.scalar_subtract(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8 = var_7.qty_or(var_5)
    var_9 = var_7.add(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NonePrice'
    var_9.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.lt(var_0)
    assert var_1 is False
    var_2 = var_0.qty_or_zero()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = var_0.__round__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
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
    var_4 = None
    var_5 = var_3.ccy_or_none()
    var_6 = var_5.__repr__()
    assert var_6 == 'None'
    var_6.subtract(var_4)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = var_0.abs()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.__floordiv__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_0.fmap(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NonePrice'
    var_9 = var_0.ccy_or(var_7)
    var_10 = var_0.__gt__(var_3)
    assert var_10 is False
    var_2.as_integer()

def test_case_42():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.lt(var_2)
    assert var_3 is False
    var_4 = var_2.qty_or_none()
    assert var_4 is False
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_2.__eq__(var_4)
    assert var_6 is False
    var_7 = module_0.NoneMoney()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_2.gte(var_7)
    assert var_8 is True
    var_9 = var_2.subtract(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    var_10 = var_2.qty_or_else(var_4)
    assert var_10 is False
    var_11 = var_5.__add__(var_10)
    assert var_11 is False

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.lt(var_0)
    assert var_1 is False
    var_2 = var_0.scalar_add(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.__floordiv__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_6.fmap(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_9 = var_8.ccy_or_none()
    var_4.qty_map(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.__floordiv__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_0.ccy_or(var_4)
    var_9 = module_0.NoneMoney()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_9.qty_map(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = var_0.with_ccy(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.__floordiv__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_0.ccy_or(var_7)
    var_9 = var_6.__lt__(var_6)
    assert var_9 is False
    var_2.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_5.lte(var_5)
    assert var_6 is True
    var_7 = var_2.qty_or_else(var_3)
    assert var_7 is False
    var_8 = var_2.gt(var_2)
    assert var_8 is False
    var_5.qty_map(var_7, var_7)

def test_case_48():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.ccy_or_none()
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_2.lt(var_6)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is False
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.is_none(var_11)
    assert var_12 is False
    var_13 = var_11.__eq__(var_5)
    assert var_13 is False
    var_14 = var_11.__add__(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_4.with_dov(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_16) == 3
    var_17 = var_14.subtract(var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_18 = var_14.negative()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_17.__gt__(var_14)
    assert var_19 is False
    var_20 = var_11.lte(var_11)
    assert var_20 is True
    var_21 = var_14.__eq__(var_18)
    assert var_21 is True
    var_22 = var_11.lt(var_17)
    assert var_22 is False
    var_23 = var_5.__repr__()
    assert var_23 == 'False'
    with pytest.raises(module_1.ProgrammingError):
        var_11.convert(var_23, var_23, var_13)

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.__floordiv__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = var_0.times(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.qty_or_none()
    var_0.qty_or_else(var_2)

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = var_0.with_ccy(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.__floordiv__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_0.ccy_or(var_7)
    var_9 = var_0.__gt__(var_3)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is False
    var_11 = var_1.add(var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5.qty_or_else(var_5)

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_2.scalar_subtract(var_0)

@pytest.mark.xfail(strict=True)
def test_case_52():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = var_1.abs()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.or_else(var_2)

@pytest.mark.xfail(strict=True)
def test_case_53():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_2.abs()

@pytest.mark.xfail(strict=True)
def test_case_54():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.with_ccy(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_2.positive()

@pytest.mark.xfail(strict=True)
def test_case_55():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.lt(var_2)
    assert var_3 is False
    var_4 = var_2.add(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.lte(var_4)
    assert var_6 is True
    module_0.SomePrice()

@pytest.mark.xfail(strict=True)
def test_case_56():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = var_1.abs()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.__floordiv__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = None
    var_7 = None
    var_8 = var_0.times(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_9 = var_8.ccy_or_none()
    var_10 = var_0.ccy_or(var_9)
    var_11 = var_8.is_equal(var_6)
    assert var_11 is False
    var_12 = var_8.convert(var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_11.qty_map(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_57():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.as_boolean()
    assert var_4 is True
    var_5 = var_2.with_dov(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.subtract(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gt(var_6)
    assert var_7 is True
    var_8 = var_2.add(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    var_9 = var_2.gte(var_2)
    assert var_9 is True
    var_10 = var_2.is_equal(var_6)
    assert var_10 is False
    var_11 = var_8.lte(var_8)
    assert var_11 is True
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = var_12.with_ccy(var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_12.__add__(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_12.__add__(var_13)

@pytest.mark.xfail(strict=True)
def test_case_58():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_0.ccy_or(var_4)
    var_9 = module_0.NoneMoney()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_10 = var_6.__round__(var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_11 = var_10.scalar_subtract(var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_59():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.__floordiv__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_0.ccy_or(var_7)
    var_9 = var_6.with_dov(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_60():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = -1940
    var_2 = var_0.__round__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = None
    var_5 = None
    var_6 = var_0.times(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.ccy_or_none()
    var_8 = var_2.ccy_or(var_3)
    var_9 = var_7.__lt__(var_4)
    var_9.qty_map(var_9, var_9)

@pytest.mark.xfail(strict=True)
def test_case_61():
    var_0 = module_0.NonePrice()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.__floordiv__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = None
    var_5 = var_0.times(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.ccy_or_none()
    var_5.__float__()

def test_case_62():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.qty_or_none()
    assert var_3 is True
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.gt(var_0)
    assert var_5 is True
    var_6 = var_2.lte(var_2)
    assert var_6 is True

def test_case_63():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.__bool__()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_6.is_none(var_2)
    assert var_7 is False
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_2.lte(var_4)
    assert var_9 is False
    var_10 = var_6.lt(var_2)
    assert var_10 is False
    var_11 = var_6.as_boolean()
    assert var_11 is True
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = var_12.__add__(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = None
    var_15 = var_4.with_dov(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_15) == 3
    var_16 = var_13.subtract(var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_16.__gt__(var_13)
    assert var_17 is False
    var_18 = var_12.lte(var_12)
    assert var_18 is True
    var_19 = var_16.__int__()
    assert var_19 == 1
    var_20 = var_16.abs()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_20) == 3
    var_21 = var_12.lte(var_20)
    assert var_21 is True
    var_22 = var_12.lt(var_20)
    assert var_22 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_14)

def test_case_64():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.add(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gte(var_2)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_65():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.lt(var_2)
    assert var_3 is False
    var_2.floor_divide(var_1)

@pytest.mark.xfail(strict=True)
def test_case_66():
    var_0 = True
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.SomeMoney(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_4 = None
    var_5 = var_3.gt(var_4)
    assert var_5 is True
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_3.subtract(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = 'WO\x0b]*k2NM\t.gade2'
    module_0.IncompatibleCurrencyError(var_0, var_4, var_8)

@pytest.mark.xfail(strict=True)
def test_case_67():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    var_5 = None
    var_6 = var_2.gt(var_2)
    assert var_6 is False
    var_7 = var_2.gt(var_2)
    assert var_7 is False
    var_8 = var_2.add(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    var_9 = module_0.SomePrice(*var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_9) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_9.divide(var_5)

@pytest.mark.xfail(strict=True)
def test_case_68():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = module_0.NoneMoney()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_2.gt(var_2)
    assert var_4 is False
    var_5 = var_2.subtract(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    module_0.SomePrice()

def test_case_69():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.add(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3

@pytest.mark.xfail(strict=True)
def test_case_70():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_2.with_qty(var_0)

@pytest.mark.xfail(strict=True)
def test_case_71():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.gte(var_3)
    assert var_4 is True
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_3.gt(var_5)
    assert var_6 is True
    var_7 = module_0.NoneMoney()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_2.gt(var_2)
    assert var_8 is False
    var_9 = var_2.subtract(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    module_0.SomePrice()

def test_case_72():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_5.lte(var_3)
    assert var_6 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_3, strict=var_3)

@pytest.mark.xfail(strict=True)
def test_case_73():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_2.scalar_add(var_0)

def test_case_74():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.ccy_or_none()
    assert var_4 is True
    var_5 = var_2.lt(var_3)
    assert var_5 is False
    var_6 = var_2.gte(var_2)
    assert var_6 is True
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_7.lte(var_3)
    assert var_8 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_4, strict=var_3)

def test_case_75():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_float()
    assert var_3 == pytest.approx(1.0, abs=0.01, rel=0.01)
    var_4 = var_2.lt(var_2)
    assert var_4 is False
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.gt(var_5)
    assert var_6 is False
    var_7 = module_0.NoneMoney()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_5.lte(var_7)
    assert var_8 is True
    var_9 = var_2.subtract(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3

@pytest.mark.xfail(strict=True)
def test_case_76():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.qty_or_none()
    assert var_3 is True
    var_4 = var_2.qty_or(var_3)
    assert var_4 is True
    var_5 = var_2.qty_or(var_4)
    assert var_5 is True
    var_6 = var_2.lt(var_2)
    assert var_6 is False
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = module_0.NoneMoney()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_9 = var_7.gt(var_8)
    assert var_9 is True
    var_10 = module_0.NoneMoney()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_11 = var_8.lte(var_7)
    assert var_11 is True
    var_12 = var_2.subtract(var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_12) == 3
    module_0.SomePrice()

def test_case_77():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.gt(var_2)
    assert var_4 is False
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.gte(var_2)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_78():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.as_boolean()
    assert var_4 is False
    var_5 = var_2.subtract(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.gt(var_5)
    assert var_6 is False
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_2.lte(var_5)
    assert var_9 is True
    var_10 = var_7.lt(var_2)
    assert var_10 is False
    var_11 = var_7.as_boolean()
    assert var_11 is False
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = var_12.__add__(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_13.subtract(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = var_13.__le__(var_3)
    assert var_15 is False
    var_2.dimap(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_79():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.lt(var_2)
    assert var_3 is False
    var_4 = var_2.qty_or_none()
    assert var_4 is False
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_2.__eq__(var_4)
    assert var_6 is False
    var_7 = module_0.NoneMoney()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_5.lte(var_5)
    assert var_8 is True
    var_9 = var_2.with_dov(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    var_10 = var_2.subtract(var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_10) == 3
    module_0.SomePrice(**var_4)

def test_case_80():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_11.__eq__(var_12)
    assert var_13 is False
    var_14 = var_11.__add__(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_4.with_dov(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_16) == 3
    var_17 = var_14.subtract(var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_18 = var_11.abs()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_17.__gt__(var_14)
    assert var_19 is False
    var_20 = var_11.lte(var_11)
    assert var_20 is True
    var_21 = var_17.__int__()
    assert var_21 == 1
    var_22 = var_11.lt(var_17)
    assert var_22 is False
    var_23 = var_16.dov_or(var_9)
    var_24 = var_2.is_equal(var_15)
    assert var_24 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_24)

@pytest.mark.xfail(strict=True)
def test_case_81():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_5.lte(var_5)
    assert var_6 is True
    var_7 = var_2.gt(var_2)
    assert var_7 is False
    var_8 = var_2.subtract(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    var_9 = var_2.ccy_or_none()
    assert var_9 is False
    var_10 = var_2.lt(var_8)
    assert var_10 is False
    var_11 = var_3.is_equal(var_1)
    assert var_11 is False
    var_2.multiply(var_9)

def test_case_82():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.ccy_or_none()
    assert var_4 is True
    var_5 = var_4.__gt__(var_4)
    assert var_5 is False
    var_6 = var_2.gte(var_2)
    assert var_6 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_4, strict=var_3)

@pytest.mark.xfail(strict=True)
def test_case_83():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.add(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_4.gt(var_5)
    assert var_6 is True
    var_7 = var_5.lt(var_3)
    assert var_7 is True
    var_8 = var_2.subtract(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    module_0.SomePrice()

def test_case_84():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.as_boolean()
    assert var_4 is True
    var_5 = var_2.subtract(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.gt(var_5)
    assert var_6 is True
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_7.gte(var_2)
    assert var_9 is True
    var_10 = var_7.lte(var_7)
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_2.or_else(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_12) == 3
    var_13 = var_11.__add__(var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_13.__pos__()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = var_11.gt(var_3)
    assert var_15 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_15)

@pytest.mark.xfail(strict=True)
def test_case_85():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.gt(var_2)
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is False
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_12.subtract(var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_11.lte(var_12)
    assert var_14 is True
    var_15 = None
    var_16 = var_11.lt(var_15)
    assert var_16 is False
    var_17 = var_12.with_qty(var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_17.qty_map(var_15, var_15)

def test_case_86():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.ccy_or_none()
    assert var_3 is True
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.as_integer()
    assert var_5 == 1
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gt(var_2)
    assert var_7 is False
    var_8 = var_6.lte(var_6)
    assert var_8 is True
    var_9 = var_2.qty_or_none()
    assert var_9 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_3)

@pytest.mark.xfail(strict=True)
def test_case_87():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_3.is_equal(var_3)
    assert var_6 is True
    var_7 = var_2.qty_or_else(var_3)
    assert var_7 is False
    var_8 = var_2.gt(var_2)
    assert var_8 is False
    var_9 = var_2.subtract(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    var_2.fmap(var_7)

@pytest.mark.xfail(strict=True)
def test_case_88():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_5.lte(var_5)
    assert var_7 is True
    var_5.__round__(var_4)

def test_case_89():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.is_equal(var_2)
    assert var_6 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_3, strict=var_3)

def test_case_90():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = None
    var_14 = var_12.subtract(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = var_11.abs()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = var_14.__gt__(var_12)
    assert var_16 is False
    var_17 = var_11.lte(var_11)
    assert var_17 is True
    var_18 = var_14.__int__()
    assert var_18 == 1
    var_19 = var_11.lt(var_14)
    assert var_19 is False
    var_20 = var_2.dov_or_none()
    assert var_20 is True
    var_21 = var_20.__le__(var_13)
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_21)

@pytest.mark.xfail(strict=True)
def test_case_91():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_3.is_equal(var_3)
    assert var_6 is True
    var_7 = var_2.qty_or_else(var_3)
    assert var_7 is False
    var_8 = var_2.gt(var_2)
    assert var_8 is False
    var_9 = var_2.subtract(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    var_10 = var_9.divide(var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    module_0.SomePrice()

def test_case_92():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_11.__eq__(var_12)
    assert var_13 is False
    var_14 = var_11.__add__(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_14.subtract(var_11)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_6.qty_or_zero()
    assert var_17 == 1
    var_18 = var_11.abs()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_16.__gt__(var_14)
    assert var_19 is False
    var_20 = var_11.lte(var_11)
    assert var_20 is True
    var_21 = var_16.__int__()
    assert var_21 == 1
    var_22 = var_11.lt(var_16)
    assert var_22 is False
    var_23 = var_15.__le__(var_15)
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_23)

def test_case_93():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.gte(var_2)
    assert var_3 is True
    var_4 = var_2.add(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0, strict=var_0)

@pytest.mark.xfail(strict=True)
def test_case_94():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.negative()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_3) == 3
    var_4 = var_2.gte(var_2)
    assert var_4 is True
    var_5 = var_2.negative()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_5.lte(var_5)
    assert var_6 is True
    var_7 = var_3.or_else(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_7.qty_or_else(var_4)
    assert var_8 == -1
    var_9 = var_2.with_ccy(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    var_10 = None
    var_11 = var_2.dov_or(var_10)
    assert var_11 is True
    var_12 = var_5.ccy_or_none()
    assert var_12 is True
    var_2.lt(var_9)

def test_case_95():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.SomePrice(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_4 = var_3.with_ccy(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = var_3.subtract(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3
    var_6 = var_5.__pos__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_6) == 3
    var_7 = var_3.gte(var_5)
    assert var_7 is True

@pytest.mark.xfail(strict=True)
def test_case_96():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.lt(var_3)
    assert var_4 is False
    var_5 = var_2.with_ccy(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_2.gt(var_5)

def test_case_97():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.ccy_or_none()
    assert var_4 is True
    var_5 = var_2.lt(var_3)
    assert var_5 is False
    var_6 = var_2.negative()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_6.as_integer()
    assert var_8 == -1
    var_9 = var_2.add(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_9) == 3
    var_10 = var_2.gt(var_2)
    assert var_10 is False
    var_11 = var_6.lte(var_9)
    assert var_11 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_4, var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_98():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.gt(var_2)
    assert var_4 is False
    var_5 = var_2.gt(var_2)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = module_0.SomePrice(*var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_7) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_7.divide(var_3)

def test_case_99():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.ccy_or_none()
    assert var_5 is True
    var_6 = var_2.gt(var_4)
    assert var_6 is True
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_2.lte(var_4)
    assert var_9 is False
    var_10 = var_7.lt(var_2)
    assert var_10 is False
    var_11 = var_7.as_boolean()
    assert var_11 is True
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = var_12.__eq__(var_5)
    assert var_13 is False
    var_14 = var_12.__add__(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_4.with_dov(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_16) == 3
    var_17 = var_14.subtract(var_12)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_18 = var_12.multiply(var_5)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_14.negative()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_19) == 3
    var_20 = var_17.__gt__(var_14)
    assert var_20 is False
    var_21 = var_12.lte(var_12)
    assert var_21 is True
    var_22 = var_14.__eq__(var_19)
    assert var_22 is False
    var_23 = var_12.lt(var_17)
    assert var_23 is False
    var_24 = var_5.__repr__()
    assert var_24 == 'True'
    with pytest.raises(module_1.ProgrammingError):
        var_12.convert(var_24, var_24, var_13)

def test_case_100():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_11.__gt__(var_11)
    assert var_13 is False
    var_14 = var_11.__add__(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_4.with_dov(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_16) == 3
    var_17 = var_14.subtract(var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_18 = var_11.or_else(var_4)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_17.__gt__(var_8)
    assert var_19 is True
    var_20 = var_17.__int__()
    assert var_20 == 1
    var_21 = var_11.lt(var_11)
    assert var_21 is False
    var_22 = var_11.dov_or_none()
    assert var_22 is True
    var_23 = var_18.gt(var_15)
    assert var_23 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_12, strict=var_15)

@pytest.mark.xfail(strict=True)
def test_case_101():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.with_ccy(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_6.gte(var_2)

def test_case_102():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.as_boolean()
    assert var_9 is True
    var_10 = module_0.SomePrice(*var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11 = var_10.__add__(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    var_12 = None
    var_13 = var_11.subtract(var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_10.abs()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = var_13.__gt__(var_11)
    assert var_15 is False
    var_16 = var_10.lte(var_10)
    assert var_16 is True
    var_17 = var_13.__int__()
    assert var_17 == 1
    var_18 = var_10.lt(var_13)
    assert var_18 is False
    var_19 = var_6.ccy_or_none()
    assert var_19 is True
    var_20 = var_19.__le__(var_12)
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_20, var_20)

def test_case_103():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_3.__pos__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3

def test_case_104():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3

@pytest.mark.xfail(strict=True)
def test_case_105():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.SomePrice(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_4 = var_3.__add__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = var_4.__pos__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3
    var_6 = var_4.dov_or_none()
    assert var_6 is False
    var_3.dimap(var_2, var_2)

def test_case_106():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__pos__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_2.gt(var_3)
    assert var_4 is False
    with pytest.raises(module_1.ProgrammingError):
        var_3.convert(var_4)

def test_case_107():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.ccy_or_none()
    assert var_5 is False
    var_6 = var_2.gt(var_4)
    assert var_6 is False
    var_7 = var_2.add(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_2.lte(var_4)
    assert var_9 is True
    var_10 = var_2.lt(var_7)
    assert var_10 is False
    var_11 = var_7.as_boolean()
    assert var_11 is False
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = None
    var_14 = var_12.__eq__(var_13)
    assert var_14 is False
    var_15 = var_12.__add__(var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = var_15.subtract(var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_12.lte(var_12)
    assert var_17 is True
    var_18 = var_15.__eq__(var_12)
    assert var_18 is True
    var_19 = var_12.lt(var_16)
    assert var_19 is False
    var_20 = var_5.__repr__()
    assert var_20 == 'False'
    with pytest.raises(module_1.ProgrammingError):
        var_12.convert(var_20, var_20, var_14)

@pytest.mark.xfail(strict=True)
def test_case_108():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_6.gte(var_2)
    assert var_8 is True
    var_9 = var_6.lte(var_6)
    assert var_9 is True
    var_10 = module_0.SomePrice(*var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_10.times(var_2)

@pytest.mark.xfail(strict=True)
def test_case_109():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_3.positive()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = None
    var_6 = var_3.subtract(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_6) == 3
    var_7 = var_6.__gt__(var_3)
    assert var_7 is False
    var_8 = var_2.is_equal(var_5)
    assert var_8 is False
    var_9 = var_2.lte(var_5)
    assert var_9 is False
    var_10 = var_3.qty_or_else(var_5)
    assert var_10 == 2
    var_10.ccy_or(var_5)

def test_case_110():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_2.gt(var_2)
    assert var_4 is False

def test_case_111():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.gte(var_2)
    assert var_6 is True
    var_7 = var_2.lte(var_4)
    assert var_7 is False
    var_8 = var_5.lt(var_2)
    assert var_8 is False
    var_9 = var_5.as_boolean()
    assert var_9 is True
    var_10 = module_0.SomePrice(*var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11 = None
    var_12 = var_10.__eq__(var_11)
    assert var_12 is False
    var_13 = var_10.__add__(var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = None
    var_15 = var_4.with_dov(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_15) == 3
    var_16 = var_10.abs()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_13.__gt__(var_13)
    assert var_17 is False
    var_18 = var_10.lte(var_10)
    assert var_18 is True
    var_19 = var_13.__int__()
    assert var_19 == 2
    var_20 = var_10.lt(var_13)
    assert var_20 is True
    var_21 = var_10.qty_or_none()
    assert var_21 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_18)

def test_case_112():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.__lt__(var_2)
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_11.__eq__(var_12)
    assert var_13 is False
    var_14 = var_11.__add__(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_4.with_dov(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_16) == 3
    var_17 = var_14.subtract(var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_18 = var_17.with_dov(var_13)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_17.__gt__(var_14)
    assert var_19 is False
    var_20 = var_11.lte(var_11)
    assert var_20 is True
    var_21 = var_17.__int__()
    assert var_21 == 1
    var_22 = var_11.lt(var_8)
    assert var_22 is False
    var_23 = var_6.ccy_or_none()
    assert var_23 is True
    var_24 = var_23.__le__(var_10)
    assert var_24 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_24, var_15)

def test_case_113():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    var_5 = var_2.__add__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3
    var_6 = var_2.negative()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_6) == 3
    var_7 = var_2.lte(var_2)
    assert var_7 is True
    var_8 = var_6.__int__()
    assert var_8 == 0

def test_case_114():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.__le__(var_4)
    assert var_8 is False
    var_9 = var_2.lte(var_4)
    assert var_9 is False
    var_10 = var_6.lt(var_2)
    assert var_10 is False
    var_11 = var_6.as_boolean()
    assert var_11 is True
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = None
    var_14 = var_12.__eq__(var_13)
    assert var_14 is False
    var_15 = var_12.__add__(var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = None
    var_17 = var_15.subtract(var_12)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_18 = var_15.round()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_17.__gt__(var_15)
    assert var_19 is False
    var_20 = var_12.as_boolean()
    assert var_20 is True
    var_21 = var_17.__int__()
    assert var_21 == 1
    var_22 = var_12.lt(var_17)
    assert var_22 is False
    var_23 = var_12.qty_or_none()
    assert var_23 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_16, var_16)

def test_case_115():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.gt(var_2)
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.lte(var_4)
    assert var_7 is False
    var_8 = var_6.gte(var_2)
    assert var_8 is True
    var_9 = var_6.lte(var_6)
    assert var_9 is True
    var_10 = module_0.SomePrice(*var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11 = var_10.__add__(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    var_12 = var_11.scalar_add(var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_10.gt(var_12)
    assert var_13 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_13)

def test_case_116():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.subtract(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_2.lte(var_2)
    assert var_4 is True
    var_5 = var_2.lt(var_3)
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_117():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.as_boolean()
    assert var_4 is True
    var_5 = var_2.subtract(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.gt(var_5)
    assert var_6 is True
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_2.as_boolean()
    assert var_9 is True
    var_10 = var_7.lte(var_7)
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11.fmap(var_3)

def test_case_118():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_3.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3

def test_case_119():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is False
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_12.subtract(var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_11.lte(var_12)
    assert var_14 is True
    var_15 = var_11.lt(var_12)
    assert var_15 is False
    var_16 = None
    var_17 = var_11.lt(var_16)
    assert var_17 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_16, var_13)

def test_case_120():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.SomePrice(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_4 = var_3.__add__(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = var_4.subtract(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3
    var_6 = var_3.ccy_or(var_2)
    assert var_6 is False
    var_7 = var_4.__le__(var_5)
    assert var_7 is True

@pytest.mark.xfail(strict=True)
def test_case_121():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.as_boolean()
    assert var_4 is True
    var_5 = var_2.add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.gt(var_5)
    assert var_6 is False
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.as_boolean()
    assert var_8 is True
    var_9 = var_2.is_equal(var_5)
    assert var_9 is False
    var_10 = var_7.lte(var_7)
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.with_ccy(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_11.__pos__()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_11.gte(var_12)

@pytest.mark.xfail(strict=True)
def test_case_122():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = None
    var_4 = var_2.as_boolean()
    assert var_4 is True
    var_5 = var_2.with_dov(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_6 = var_2.subtract(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gt(var_6)
    assert var_7 is True
    var_8 = var_2.add(var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    var_9 = var_2.gte(var_2)
    assert var_9 is True
    var_10 = var_2.is_equal(var_6)
    assert var_10 is False
    var_11 = var_8.lte(var_8)
    assert var_11 is True
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = var_12.with_ccy(var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_12.__add__(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = var_13.__pos__()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = var_14.gte(var_3)
    assert var_16 is True
    var_2.floor_divide(var_15)

def test_case_123():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_3.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = var_2.lte(var_3)
    assert var_5 is True

def test_case_124():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_2)
    assert var_8 is True
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_11.dov_or(var_12)
    assert var_13 is True
    var_14 = None
    var_15 = var_11.__eq__(var_14)
    assert var_15 is False
    var_16 = var_11.__add__(var_11)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = None
    var_18 = var_4.with_dov(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_18) == 3
    var_19 = var_16.subtract(var_11)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_19) == 3
    var_20 = var_11.abs()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_20) == 3
    var_21 = var_19.__gt__(var_16)
    assert var_21 is False
    var_22 = var_11.lte(var_11)
    assert var_22 is True
    var_23 = var_19.__int__()
    assert var_23 == 1
    var_24 = var_11.lt(var_19)
    assert var_24 is False
    var_25 = var_11.qty_or_none()
    assert var_25 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0)

@pytest.mark.xfail(strict=True)
def test_case_125():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomePrice(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__add__(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    var_4 = var_3.positive()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = None
    var_6 = var_3.subtract(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_6) == 3
    var_7 = var_6.__gt__(var_3)
    assert var_7 is False
    var_8 = var_2.is_equal(var_5)
    assert var_8 is False
    var_9 = var_2.lte(var_5)
    assert var_9 is False
    var_10 = var_3.ccy_or(var_5)
    assert var_10 is True
    var_4.convert(var_5, strict=var_9)

@pytest.mark.xfail(strict=True)
def test_case_126():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.as_boolean()
    assert var_7 is False
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is False
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11.floor_divide(var_11)

def test_case_127():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is False
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__truediv__(var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_13 = var_12.negative()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.NonePrice'
    var_14 = var_12.subtract(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = var_13.__le__(var_13)
    assert var_15 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_15)

@pytest.mark.xfail(strict=True)
def test_case_128():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.__bool__()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = module_0.SomePrice(*var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11 = var_10.__add__(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    var_12 = var_11.negative()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_10.scalar_subtract(var_10)

@pytest.mark.xfail(strict=True)
def test_case_129():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.lte(var_4)
    assert var_5 is False
    var_6 = var_2.gt(var_4)
    assert var_6 is True
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_2.lte(var_4)
    assert var_9 is False
    var_10 = var_7.lt(var_2)
    assert var_10 is False
    var_11 = var_7.lte(var_4)
    assert var_11 is False
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = var_12.__add__(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_13.subtract(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_12.as_float()
    assert var_16 == pytest.approx(1.0, abs=0.01, rel=0.01)
    var_16.with_dov(var_15)

def test_case_130():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_6.as_boolean()
    assert var_9 is False
    var_10 = module_0.SomePrice(*var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11 = var_10.__add__(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    var_12 = var_10.with_qty(var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_10.lte(var_11)
    assert var_13 is True
    var_14 = var_12.__gt__(var_10)
    assert var_14 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_14)

@pytest.mark.xfail(strict=True)
def test_case_131():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.__ge__(var_2)
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_11.qty_or_zero()
    assert var_13 is True
    var_14 = var_12.positive()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    module_0.SomePrice()

def test_case_132():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = None
    var_14 = var_12.qty_or(var_13)
    assert var_14 == 2
    var_15 = var_12.subtract(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = var_15.__gt__(var_12)
    assert var_16 is False
    var_17 = var_11.lt(var_12)
    assert var_17 is True
    var_18 = var_11.lte(var_13)
    assert var_18 is False
    var_19 = var_11.lt(var_15)
    assert var_19 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_13)

@pytest.mark.xfail(strict=True)
def test_case_133():
    var_0 = False
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is False
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.lte(var_6)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is True
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = module_0.SomePrice(*var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_11 = var_10.__add__(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    var_12 = var_11.subtract(var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_10.negative()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_13) == 3
    var_14 = var_10.lte(var_11)
    assert var_14 is True
    var_15 = None
    var_16 = var_12.__gt__(var_15)
    assert var_16 is True
    var_17 = var_10.lt(var_8)
    assert var_17 is False
    var_18 = var_10.lte(var_15)
    assert var_18 is False
    var_19 = var_10.floor_divide(var_5)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_20 = var_10.lt(var_15)
    assert var_20 is False
    var_21 = var_10.qty_or_else(var_17)
    assert var_21 is False
    var_21.__setattr__(var_15, var_9)

@pytest.mark.xfail(strict=True)
def test_case_134():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = None
    var_14 = var_4.with_dov(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_14) == 3
    var_15 = var_12.subtract(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = var_15.__gt__(var_12)
    assert var_16 is False
    var_17 = var_11.lt(var_12)
    assert var_17 is True
    var_18 = var_15.with_ccy(var_13)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_11.lt(var_18)

def test_case_135():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = None
    var_14 = var_4.with_dov(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_14) == 3
    var_15 = var_12.subtract(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = var_11.abs()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_15.__gt__(var_12)
    assert var_17 is False
    var_18 = var_11.lte(var_11)
    assert var_18 is True
    var_19 = var_12.__eq__(var_16)
    assert var_19 is False
    var_20 = var_6.ccy_or_none()
    assert var_20 is True
    var_21 = var_20.__le__(var_13)
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_21)

@pytest.mark.xfail(strict=True)
def test_case_136():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_11.__eq__(var_12)
    assert var_13 is False
    var_14 = var_11.with_ccy(var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_4.with_dov(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_16) == 3
    var_14.subtract(var_11)

def test_case_137():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.ccy_or(var_2)
    assert var_5 is True
    var_6 = var_2.gt(var_4)
    assert var_6 is True
    var_7 = var_2.add(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_2.gte(var_2)
    assert var_8 is True
    var_9 = var_2.lte(var_4)
    assert var_9 is False
    var_10 = var_7.lt(var_2)
    assert var_10 is False
    var_11 = var_7.as_boolean()
    assert var_11 is True
    var_12 = module_0.SomePrice(*var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_13 = None
    var_14 = var_12.__eq__(var_13)
    assert var_14 is False
    var_15 = var_12.with_dov(var_4)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_16 = None
    var_17 = var_4.with_dov(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_17) == 3
    var_18 = var_15.subtract(var_12)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_18) == 3
    var_19 = var_12.abs()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_19) == 3
    var_20 = var_2.gte(var_4)
    assert var_20 is True
    var_21 = var_12.lte(var_12)
    assert var_21 is True
    var_22 = var_12.__eq__(var_3)
    assert var_22 is False
    var_23 = var_12.lt(var_16)
    assert var_23 is False
    var_24 = var_5.__repr__()
    assert var_24 == 'True'
    with pytest.raises(AttributeError):
        var_24.convert(var_16)

@pytest.mark.xfail(strict=True)
def test_case_138():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = None
    var_13 = var_11.__eq__(var_12)
    assert var_13 is False
    var_14 = var_11.__add__(var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_14) == 3
    var_15 = None
    var_16 = var_4.with_dov(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_16) == 3
    var_17 = var_11.with_ccy(var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_17) == 3
    var_18 = var_11.__gt__(var_12)
    assert var_18 is True
    var_11.lte(var_17)

@pytest.mark.xfail(strict=True)
def test_case_139():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.gt(var_4)
    assert var_5 is True
    var_6 = var_2.add(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.gte(var_2)
    assert var_7 is True
    var_8 = var_2.lte(var_4)
    assert var_8 is False
    var_9 = var_6.lt(var_2)
    assert var_9 is False
    var_10 = var_6.as_boolean()
    assert var_10 is True
    var_11 = module_0.SomePrice(*var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_12 = var_11.__add__(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = None
    var_14 = var_4.with_dov(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_14) == 3
    var_15 = var_11.with_ccy(var_2)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_15) == 3
    var_11.__gt__(var_15)

@pytest.mark.xfail(strict=True)
def test_case_140():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.SomeMoney(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_2) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.as_boolean()
    assert var_3 is True
    var_4 = var_2.subtract(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = None
    var_6 = var_2.with_ccy(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_2.lte(var_6)