# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.monetary as module_0
import pypara.commons.errors as module_1
import decimal as module_2
import pypara.currencies as module_3

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.__add__(var_3)
    assert var_5 is True
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_2.qty_or_none()
    var_8 = var_2.lt(var_0)
    assert var_8 is False
    var_9 = var_4.is_equal(var_0)
    assert var_9 is False
    var_10 = var_2.gte(var_0)
    assert var_10 is True
    var_2.__gt__(var_2)

def test_case_2():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.positive()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_2.lt(var_0)
    assert var_6 is False
    var_7 = var_2.__le__(var_0)
    assert var_7 is False
    var_8 = var_4.round()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_8)

def test_case_3():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_2.lt(var_0)
    assert var_5 is False
    var_6 = var_2.__le__(var_0)
    assert var_6 is False
    var_7 = var_4.__add__(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_7)

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_2 = var_1.__bool__()
    assert var_2 is False
    var_1.dimap(var_0, var_0)

def test_case_5():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_2.lt(var_0)
    assert var_5 is False
    var_6 = var_2.as_boolean()
    assert var_6 is False
    var_7 = var_2.__ge__(var_0)
    assert var_7 is True
    var_8 = var_4.ccy_or_none()
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_8)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = {}
    var_1 = module_0.NoneMoney(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_2 = None
    var_1.qty_or_else(var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
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
    var_2 = var_0.scalar_subtract(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.times(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = None
    var_6 = var_0.add(var_5)
    var_7 = var_0.qty_or(var_5)
    var_8 = var_0.abs()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NonePrice'
    var_9 = var_8.__gt__(var_8)
    assert var_9 is False
    var_8.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_0.as_float()

def test_case_9():
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
    var_2 = None
    var_3 = var_0.positive()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_5 = var_0.__eq__(var_2)
    assert var_5 is False
    var_6 = var_3.__add__(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = var_6.negative()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'

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
    var_2 = None
    var_3 = var_0.positive()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.or_else(var_2)

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.scalar_subtract(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.times(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_0.with_qty(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_4.__round__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.qty_or_else(var_1)

def test_case_12():
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
    var_1 = var_0.ccy_or_none()
    var_2 = var_0.__truediv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.__round__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_0.times(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'

def test_case_13():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.__truediv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.times(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_2.with_qty(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = module_0.NonePrice()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = var_3.__abs__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_7.qty_or(var_1)
    var_9 = var_0.gt(var_5)
    assert var_9 is False

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.times(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_0.add(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4 = var_0.__ge__(var_3)
    assert var_4 is True
    var_5 = var_0.__round__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_6 = var_2.lte(var_1)
    assert var_6 is True
    var_2.__int__()

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.with_dov(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_0.is_none(var_2)
    assert var_3 is True
    var_4 = var_0.__truediv__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_4.times(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_4.with_qty(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = var_6.__round__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8 = var_0.as_boolean()
    assert var_8 is False
    var_9 = var_0.ccy_or(var_1)

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
    var_1 = var_0.qty_or_none()
    var_2 = var_0.__round__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.__round__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = None
    var_2 = module_0.NonePrice()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_3 = var_2.__neg__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2.qty_map(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
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
    var_3 = module_0.NoneMoney()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_2.qty_or_else(var_0)
    var_5 = var_4.__le__(var_0)
    var_6 = var_2.with_ccy(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_6) == 3
    var_2.__le__(var_6)

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.is_none(var_0)
    assert var_2 is True
    var_3 = var_0.times(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_0.with_qty(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = var_4.__round__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = var_3.lte(var_1)
    assert var_7 is True
    var_8 = var_4.qty_or_zero()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_9 = var_3.floor_divide(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NoneMoney'
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
    var_10 = var_3.__mul__(var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = var_2.ccy_or_none()
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.__add__(var_3)
    assert var_6 is True
    var_7 = var_5.subtract(var_6)
    assert var_7 == -1
    var_8 = var_2.lt(var_0)
    assert var_8 is False
    var_9 = var_2.gt(var_8)
    assert var_9 is True
    var_10 = var_2.__gt__(var_0)
    assert var_10 is True
    var_11 = var_5.ccy_or(var_4)
    var_12 = var_2.subtract(var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_12) == 3
    var_13 = var_2.lte(var_0)
    assert var_13 is False
    var_2.add(var_12)

def test_case_21():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.subtract(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_2.lt(var_0)
    assert var_6 is False
    var_7 = var_2.__le__(var_0)
    assert var_7 is False
    var_8 = var_2.__gt__(var_0)
    assert var_8 is True
    var_9 = var_4.round()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_10 = var_2.subtract(var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_10) == 3
    var_11 = var_9.dov_or_none()
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0, var_11)

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
    var_2 = var_0.scalar_subtract(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_0.times(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = None
    var_5 = var_0.add(var_4)
    var_6 = var_0.qty_or(var_4)
    var_7 = var_0.abs()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8 = var_7.__gt__(var_7)
    assert var_8 is False
    var_9 = var_0.subtract(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NonePrice'
    var_10 = var_2.add(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NonePrice'
    var_11 = var_3.__neg__()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.NoneMoney'

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_1 = var_0.ccy_or_none()
    var_2 = var_0.__truediv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.__round__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.dimap(var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.__truediv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.__round__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.qty_or_else(var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.__truediv__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_5.__truediv__(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7 = var_2.lt(var_0)
    assert var_7 is False
    var_8 = var_6.as_boolean()
    assert var_8 is False
    var_9 = var_2.__ge__(var_0)
    assert var_9 is True
    var_10 = var_4.add(var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_2.subtract(var_2)

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
    var_1 = var_0.lt(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = var_0.times(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3.as_float()

@pytest.mark.xfail(strict=True)
def test_case_27():
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
    var_1 = None
    var_2 = var_0.gt(var_1)
    assert var_2 is False
    var_3 = var_0.qty_or_none()
    var_0.lt(var_3)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
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
    var_3 = var_2.fmap(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4 = {}
    var_5 = module_0.SomePrice(*var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_5.add(var_5)

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.times(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_0.__neg__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_0.__ge__(var_3)
    assert var_4 is True
    var_5 = var_0.__round__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_6 = var_2.lte(var_1)
    assert var_6 is True
    var_2.qty_map(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.is_some(var_3)
    assert var_4 is True
    var_3.__add__(var_3)

def test_case_31():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.__round__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'

@pytest.mark.xfail(strict=True)
def test_case_32():
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
    var_2 = var_1.positive()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_3 = var_1.positive()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_2.is_some(var_0)

def test_case_33():
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
    var_1 = var_0.__truediv__(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_2.lt(var_0)
    assert var_5 is False
    var_6 = var_4.abs()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7 = var_4.lt(var_6)
    assert var_7 is False
    var_8 = var_2.__le__(var_0)
    assert var_8 is False
    var_9 = var_2.__gt__(var_0)
    assert var_9 is True
    var_2.__ge__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.__abs__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_4.subtract(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_2.__gt__(var_2)

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
    var_1 = var_0.qty_or_none()
    var_2 = var_0.__round__(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.__round__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_0.lte(var_3)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_37():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.with_dov(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_0.is_none(var_2)
    assert var_3 is True
    var_4 = var_2.times(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.with_dov(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_5.with_qty(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7 = None
    var_8 = var_6.__round__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_9 = var_4.lte(var_7)
    assert var_9 is True
    var_10 = var_4.lte(var_4)
    assert var_10 is True
    var_8.__ge__(var_7)

def test_case_38():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = module_0.NonePrice()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.NonePrice.defined).__module__}.{type(module_0.NonePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.undefined).__module__}.{type(module_0.NonePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NonePrice.money).__module__}.{type(module_0.NonePrice.money).__qualname__}' == 'builtins.property'
    var_5 = var_3.__sub__(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3

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
    var_1 = var_0.__round__()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'

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
    var_1 = var_0.dov_or_none()
    var_2 = None
    var_3 = var_0.with_dov(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_0.__truediv__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_4.times(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_3.__neg__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = var_4.dov_or(var_1)
    var_5.is_none(var_1)

def test_case_41():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = var_2.ccy_or_none()
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_5.with_ccy(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7 = var_5.subtract(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_2.lt(var_0)
    assert var_8 is False
    var_9 = var_2.gt(var_8)
    assert var_9 is True
    var_10 = var_2.__gt__(var_4)
    assert var_10 is True
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0, strict=var_0)

@pytest.mark.xfail(strict=True)
def test_case_42():
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
    var_2 = var_1.convert(var_0, strict=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_3 = module_0.MonetaryOperationException()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.MonetaryOperationException'
    var_4 = var_1.convert(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_3.multiply(var_0)

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
    var_1 = None
    var_2 = var_0.floor_divide(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = None
    var_4 = var_0.times(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4.as_float()

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.__truediv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.times(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_2.with_qty(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_4.__round__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.qty_or_else(var_1)

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
    var_1 = var_0.lt(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = var_0.times(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_0.ccy_or_none()
    var_5 = var_0.with_ccy(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4.__delattr__(var_4)

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.times(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.with_qty(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4 = var_2.dov_or(var_1)
    var_5 = var_0.__neg__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_6 = var_0.__ge__(var_5)
    assert var_6 is True
    var_7 = var_3.__round__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_0.qty_or_else(var_1)

@pytest.mark.xfail(strict=True)
def test_case_47():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.with_dov(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_0.__truediv__(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_3.times(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_3.with_qty(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'
    var_6 = var_5.__round__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_0.qty_or_else(var_1)

def test_case_48():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.times(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_0.__neg__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_3.__round__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_2.lte(var_1)
    assert var_5 is True

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
    var_1 = var_0.ccy_or_none()
    var_2 = var_0.fmap(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.__round__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_0.times(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'

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
    var_1 = var_0.positive()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2 = None
    var_3 = var_0.scalar_add(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_0.dov_or_none()
    var_5 = var_3.__round__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NonePrice'

@pytest.mark.xfail(strict=True)
def test_case_51():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.__truediv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.times(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_0.convert(var_1, strict=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = None
    var_6 = var_4.fmap(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = module_0.NonePrice()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2.dimap(var_7, var_5)

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.__truediv__(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_2.times(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_2.with_qty(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_3.qty_or_none()
    var_6 = None
    var_7 = var_4.__round__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8 = module_0.NonePrice()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NonePrice'
    var_9 = var_7.__mul__(var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.NonePrice'
    var_2.dimap(var_8, var_6)

@pytest.mark.xfail(strict=True)
def test_case_53():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.times(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_3 = var_2.with_qty(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_4 = var_0.__neg__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_0.__ge__(var_4)
    assert var_5 is True
    var_6 = var_3.__round__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7 = var_2.lte(var_1)
    assert var_7 is True
    var_2.qty_map(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_54():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.with_dov(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_0.is_none(var_2)
    assert var_3 is True
    var_4 = var_2.times(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.with_dov(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_5.with_qty(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7 = None
    var_8 = var_6.__round__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_9 = var_4.lte(var_4)
    assert var_9 is True
    var_10 = var_4.multiply(var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_11 = var_6.qty_or_zero()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_12 = var_4.with_qty(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.NoneMoney'
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
    var_10.__int__()

def test_case_55():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.__truediv__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_2.lt(var_0)
    assert var_6 is False
    var_7 = var_2.__le__(var_0)
    assert var_7 is False
    var_8 = var_2.__gt__(var_0)
    assert var_8 is True
    var_9 = var_2.subtract(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_9) == 3
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0)

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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.dov_or(var_1)
    var_3 = var_0.with_dov(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NonePrice'
    var_4 = var_0.__truediv__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_4.times(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_4.with_qty(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = var_6.__round__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8 = var_5.lte(var_1)
    assert var_8 is True
    var_0.qty_or_else(var_1)

@pytest.mark.xfail(strict=True)
def test_case_57():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.is_none(var_3)
    assert var_4 is False
    var_5 = var_3.__gt__(var_0)
    assert var_5 is True
    var_3.subtract(var_3)

@pytest.mark.xfail(strict=True)
def test_case_58():
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_2.lt(var_4)
    assert var_5 is False
    var_6 = var_1.__gt__(var_0)
    var_7 = var_4.round()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_4.scalar_subtract(var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_2.convert(var_5)

@pytest.mark.xfail(strict=True)
def test_case_59():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.__mul__(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_2.lt(var_0)
    assert var_6 is False
    var_2.gte(var_2)

def test_case_60():
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
    var_2 = var_1.with_dov(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_3 = [var_0, var_0, var_0]
    var_4 = module_0.SomePrice(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_5 = var_4.gt(var_0)
    assert var_5 is True
    with pytest.raises(module_1.ProgrammingError):
        var_4.convert(var_3)

@pytest.mark.xfail(strict=True)
def test_case_61():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.scalar_add(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_2.gte(var_0)
    assert var_6 is True
    var_7 = var_2.__eq__(var_0)
    assert var_7 is False
    var_2.lt(var_2)

@pytest.mark.xfail(strict=True)
def test_case_62():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.with_dov(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_0.is_none(var_2)
    assert var_3 is True
    var_4 = var_2.times(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.with_dov(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_6 = var_5.with_qty(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_7 = None
    var_8 = var_6.__round__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_9 = var_4.lte(var_7)
    assert var_9 is True
    var_10 = var_4.lte(var_4)
    assert var_10 is True
    var_11 = var_5.__round__(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_12 = var_4.lte(var_4)
    assert var_12 is True
    var_4.qty_map(var_7, var_7)

@pytest.mark.xfail(strict=True)
def test_case_63():
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
    var_1 = var_0.dov_or_none()
    var_2 = var_0.with_dov(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.monetary.NonePrice'
    var_3 = var_0.is_none(var_2)
    assert var_3 is True
    var_4 = var_0.__truediv__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5 = var_4.times(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_4.with_qty(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NonePrice'
    var_7 = var_4.with_dov(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NonePrice'
    var_8 = var_7.__round__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.NonePrice'
    var_5.or_else(var_1)

@pytest.mark.xfail(strict=True)
def test_case_64():
    var_0 = None
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
    var_2.__gt__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_65():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.ccy_or_none()
    var_5 = var_3.__le__(var_2)
    assert var_5 is False
    var_6 = var_3.lt(var_4)
    assert var_6 is False
    var_3.abs()

@pytest.mark.xfail(strict=True)
def test_case_66():
    var_0 = None
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
    var_2.floor_divide(var_0)

@pytest.mark.xfail(strict=True)
def test_case_67():
    var_0 = None
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
    var_3 = var_2.lt(var_0)
    assert var_3 is False
    var_2.subtract(var_2)

@pytest.mark.xfail(strict=True)
def test_case_68():
    var_0 = None
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
    var_2.__le__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_69():
    var_0 = None
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
    var_2.__ge__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_70():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = {}
    var_4 = module_0.SomePrice(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_4.scalar_subtract(var_0)

@pytest.mark.xfail(strict=True)
def test_case_71():
    var_0 = None
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
    var_3 = var_2.as_boolean()
    assert var_3 is False
    var_2.lt(var_2)

@pytest.mark.xfail(strict=True)
def test_case_72():
    var_0 = None
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
    var_2.__round__()

@pytest.mark.xfail(strict=True)
def test_case_73():
    var_0 = None
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
    var_3 = var_2.__bool__()
    assert var_3 is False
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_2.__gt__(var_0)
    assert var_5 is True
    var_6 = var_2.__add__(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_6) == 3
    var_7 = var_4.fmap(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_6.ccy_or(var_0)
    var_6.dimap(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_74():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.ccy_or(var_0)
    var_5 = var_3.is_equal(var_2)
    assert var_5 is False
    var_3.__ge__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_75():
    var_0 = None
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
    var_2.lt(var_2)

@pytest.mark.xfail(strict=True)
def test_case_76():
    var_0 = None
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
    var_2.subtract(var_2)

@pytest.mark.xfail(strict=True)
def test_case_77():
    var_0 = None
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
    var_2.add(var_2)

@pytest.mark.xfail(strict=True)
def test_case_78():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_3.__pos__()

@pytest.mark.xfail(strict=True)
def test_case_79():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.with_dov(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_3.__ge__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_80():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = {}
    var_4 = module_0.SomePrice(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_4.qty_map(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_81():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_3.as_float()

@pytest.mark.xfail(strict=True)
def test_case_82():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.ccy_or(var_1)
    var_3.__ge__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_83():
    var_0 = None
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
    var_3 = var_2.gt(var_0)
    assert var_3 is True
    var_4 = var_2.qty_or_else(var_0)
    var_2.lt(var_2)

@pytest.mark.xfail(strict=True)
def test_case_84():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_3.negative()

@pytest.mark.xfail(strict=True)
def test_case_85():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = {}
    var_4 = module_0.SomePrice(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_5 = var_4.with_dov(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3
    var_6 = var_5.ccy_or(var_0)
    var_7 = var_5.dov_or(var_1)
    var_4.__ge__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_86():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.ccy_or_none()
    var_3.__ge__(var_3)

def test_case_87():
    var_0 = None
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
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0)

@pytest.mark.xfail(strict=True)
def test_case_88():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_3.scalar_add(var_0)

@pytest.mark.xfail(strict=True)
def test_case_89():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.qty_or_none()
    var_5 = var_3.qty_or_else(var_0)
    var_5.qty_map(var_5, var_0)

def test_case_90():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.qty_or_zero()
    with pytest.raises(module_1.ProgrammingError):
        var_3.convert(var_0)

def test_case_91():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.with_dov(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = var_4.__ge__(var_0)
    assert var_5 is True

def test_case_92():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.with_ccy(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = var_3.with_dov(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_5) == 3
    var_6 = var_5.__gt__(var_0)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_93():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = {}
    var_4 = module_0.SomePrice(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_4.times(var_0)

@pytest.mark.xfail(strict=True)
def test_case_94():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.or_else(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    var_5 = var_3.ccy_or_none()
    var_4.__ge__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_95():
    var_0 = None
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
    var_2.__truediv__(var_0)

def test_case_96():
    var_0 = None
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
    var_3 = var_2.with_qty(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_3) == 3
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0)

def test_case_97():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.lte(var_0)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_98():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_3.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_99():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = {}
    var_4 = module_0.SomePrice(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_4) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomePrice.defined).__module__}.{type(module_0.SomePrice.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.undefined).__module__}.{type(module_0.SomePrice.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomePrice.money).__module__}.{type(module_0.SomePrice.money).__qualname__}' == 'builtins.property'
    var_5 = var_4.dov_or_none()
    var_6 = var_4.__ge__(var_0)
    assert var_6 is True
    var_4.__gt__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_100():
    var_0 = None
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
    var_3.multiply(var_0)

@pytest.mark.xfail(strict=True)
def test_case_101():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_3.multiply(var_0)

def test_case_102():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.qty_or(var_0)
    with pytest.raises(module_1.ProgrammingError):
        var_3.convert(var_0)

@pytest.mark.xfail(strict=True)
def test_case_103():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_3.subtract(var_0)

@pytest.mark.xfail(strict=True)
def test_case_104():
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
    var_3 = var_2.qty_or_else(var_0)
    var_4 = var_2.or_else(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_2.__sub__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_105():
    var_0 = None
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
    var_3 = var_2.gt(var_0)
    assert var_3 is True
    var_2.__le__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_106():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_3.fmap(var_0)

@pytest.mark.xfail(strict=True)
def test_case_107():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomePrice(*var_1, **var_2)
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
    var_4 = var_3.__ge__(var_0)
    assert var_4 is True
    var_5 = var_3.__ge__(var_0)
    assert var_5 is True
    var_6 = module_0.SomeMoney(*var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_6.scalar_add(var_0)

@pytest.mark.xfail(strict=True)
def test_case_108():
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_2.abs()

def test_case_109():
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_110():
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_2.dimap(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_111():
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
    var_2.__floordiv__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_112():
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
    var_2.__lt__(var_2)

def test_case_113():
    var_0 = None
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
    var_3 = var_2.gt(var_0)
    assert var_3 is True
    var_4 = None
    var_5 = var_2.gte(var_0)
    assert var_5 is True
    var_6 = var_2.lt(var_4)
    assert var_6 is False
    with pytest.raises(module_1.ProgrammingError):
        var_2.convert(var_0, var_5)

def test_case_114():
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
    var_3 = var_2.as_boolean()
    assert var_3 is False

@pytest.mark.xfail(strict=True)
def test_case_115():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = None
    var_5 = var_2.__eq__(var_4)
    assert var_5 is False
    var_6 = var_2.lt(var_0)
    assert var_6 is False
    var_7 = var_2.__le__(var_0)
    assert var_7 is False
    var_8 = var_2.__gt__(var_0)
    assert var_8 is True
    var_2.__round__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_116():
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
    var_2.lte(var_2)

@pytest.mark.xfail(strict=True)
def test_case_117():
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
    var_2.gte(var_2)

@pytest.mark.xfail(strict=True)
def test_case_118():
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
    var_2.__gt__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_119():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.SomeMoney(*var_1, **var_2)
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
    var_3.scalar_subtract(var_0)

@pytest.mark.xfail(strict=True)
def test_case_120():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = module_0.SomeMoney(*var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_2.lt(var_0)
    assert var_6 is False
    var_7 = var_5.ccy_or_none()
    var_8 = var_2.__le__(var_0)
    assert var_8 is False
    var_2.__gt__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_121():
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_2.positive()

@pytest.mark.xfail(strict=True)
def test_case_122():
    var_0 = None
    var_1 = None
    var_2 = [var_0, var_0, var_0]
    var_3 = {}
    var_4 = module_0.SomeMoney(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Numeric).__module__}.{type(module_0.Numeric).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.NoMoney).__module__}.{type(module_0.NoMoney).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoPrice).__module__}.{type(module_0.NoPrice).__qualname__}' == 'pypara.monetary.NonePrice'
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.qty_or_else(var_1)
    var_4.qty_map(var_1, var_5)

@pytest.mark.xfail(strict=True)
def test_case_123():
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
    var_3 = var_2.lt(var_0)
    assert var_3 is False
    var_2.convert(var_0, var_0)

def test_case_124():
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
    var_3 = var_2.gt(var_0)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_125():
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
    var_3 = var_2.dov_or_none()
    var_2.convert(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_126():
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
    var_3 = var_2.qty_or_zero()
    var_2.convert(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_127():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.dov_or_none()
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.__add__(var_3)
    assert var_7 is True
    var_4.fmap(var_0)

@pytest.mark.xfail(strict=True)
def test_case_128():
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
    var_3 = var_2.dov_or_none()
    var_4 = module_0.NoneMoney()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_2.subtract(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_2.lt(var_5)

@pytest.mark.xfail(strict=True)
def test_case_129():
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
    var_2.__sub__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_130():
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
    var_2.divide(var_1)

@pytest.mark.xfail(strict=True)
def test_case_131():
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
    var_3 = var_2.qty_or(var_0)
    var_4 = var_2.lt(var_0)
    assert var_4 is False
    var_5 = var_2.or_else(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    var_2.convert(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_132():
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = var_2.dov_or(var_0)
    var_2.gt(var_2)

@pytest.mark.xfail(strict=True)
def test_case_133():
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
    var_2.add(var_2)

@pytest.mark.xfail(strict=True)
def test_case_134():
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
    var_2.convert(var_0, var_0)

def test_case_135():
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = var_2.is_equal(var_1)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_136():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.dov_or_none()
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4.with_qty(var_5)

@pytest.mark.xfail(strict=True)
def test_case_137():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.gte(var_0)
    assert var_5 is True
    var_6 = var_2.ccy_or(var_0)
    var_7 = var_4.ccy_or(var_6)
    var_8 = var_2.lt(var_0)
    assert var_8 is False
    var_9 = var_4.lt(var_5)
    assert var_9 is False
    var_10 = var_2.__gt__(var_0)
    assert var_10 is True
    var_2.subtract(var_2)

@pytest.mark.xfail(strict=True)
def test_case_138():
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
    var_3 = var_2.qty_or_none()
    var_2.convert(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_139():
    var_0 = None
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
    var_4 = var_2.gte(var_3)
    assert var_4 is True
    var_5 = module_0.SomeMoney(*var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_5) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5.negative()

@pytest.mark.xfail(strict=True)
def test_case_140():
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
    var_3 = module_0.NoneMoney()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_4 = var_2.subtract(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    var_5 = var_2.lt(var_3)
    assert var_5 is False
    var_6 = var_2.__le__(var_0)
    assert var_6 is False
    var_4.__floordiv__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_141():
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
    var_2.__float__()

def test_case_142():
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
    var_3 = var_2.qty_or_none()
    var_4 = var_2.with_ccy(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3

@pytest.mark.xfail(strict=True)
def test_case_143():
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
    var_2.as_integer()

@pytest.mark.xfail(strict=True)
def test_case_144():
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
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.lt(var_0)
    assert var_4 is False
    var_2.convert(var_0, strict=var_0)

@pytest.mark.xfail(strict=True)
def test_case_145():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.gte(var_0)
    assert var_5 is True
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_4.subtract(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_7) == 3
    var_8 = var_4.lt(var_6)
    assert var_8 is False
    var_9 = var_7.dov_or_none()
    var_10 = var_4.with_dov(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_10) == 3
    var_11 = var_6.round(var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_12 = var_4.lt(var_0)
    assert var_12 is False
    var_2.subtract(var_2)

@pytest.mark.xfail(strict=True)
def test_case_146():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.dov_or_none()
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.abs()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_4.subtract(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    var_9 = var_2.lt(var_0)
    assert var_9 is False
    var_10 = var_2.__gt__(var_5)
    assert var_10 is True
    var_11 = var_8.__add__(var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_11) == 3
    var_12 = var_7.multiply(var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_13 = var_4.subtract(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_13) == 3
    var_14 = var_4.gte(var_0)
    assert var_14 is True
    var_15 = var_2.__gt__(var_0)
    assert var_15 is True
    var_2.subtract(var_2)

@pytest.mark.xfail(strict=True)
def test_case_147():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.gte(var_0)
    assert var_5 is True
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_2.lt(var_0)
    assert var_7 is False
    var_8 = var_4.lt(var_6)
    assert var_8 is False
    var_9 = var_2.__gt__(var_0)
    assert var_9 is True
    var_10 = var_2.is_some(var_2)
    assert var_10 is True
    var_11 = var_2.subtract(var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_11) == 3
    var_12 = var_2.is_some(var_4)
    assert var_12 is True
    var_11.subtract(var_2)

@pytest.mark.xfail(strict=True)
def test_case_148():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = var_4.dov_or_none()
    var_6 = module_0.NoneMoney()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_7 = var_6.abs()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_8 = var_4.subtract(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    var_9 = 'p jn7vDz6:yOh'
    var_10 = 'It'
    var_11 = module_3.Currency(var_9, var_10, var_3, var_0, var_0, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.currencies.Currency'
    assert var_11.code == 'p jn7vDz6:yOh'
    assert var_11.name == 'It'
    assert var_11.decimals is True
    assert var_11.type is None
    assert var_11.quantizer is None
    assert var_11.hashcache is None
    assert f'{type(module_3.ZERO).__module__}.{type(module_3.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.MaxPrecisionQuantizer).__module__}.{type(module_3.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.Currencies).__module__}.{type(module_3.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_3.Currencies) == 188
    assert f'{type(module_3.Currency.of).__module__}.{type(module_3.Currency.of).__qualname__}' == 'builtins.method'
    var_12 = var_8.with_ccy(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_12) == 3
    var_4.lt(var_12)

def test_case_149():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_4.subtract(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_5.dov_or_none()
    var_8 = var_2.lt(var_0)
    assert var_8 is False
    var_9 = var_5.ccy_or_none()
    var_10 = var_5.with_ccy(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_11 = var_4.lt(var_5)
    assert var_11 is False
    var_12 = var_2.__le__(var_0)
    assert var_12 is False
    var_13 = var_2.__gt__(var_0)
    assert var_13 is True
    var_14 = var_4.with_dov(var_7)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_14) == 3
    var_15 = var_2.__eq__(var_9)
    assert var_15 is False
    var_16 = var_2.subtract(var_5)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_9.__repr__()
    assert var_17 == 'None'
    var_18 = var_6.__truediv__(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pypara.monetary.NoneMoney'

def test_case_150():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_4.subtract(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_5.dov_or_none()
    var_8 = var_2.lt(var_0)
    assert var_8 is False
    var_9 = var_5.ccy_or_none()
    var_10 = var_5.with_ccy(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_11 = var_4.lt(var_5)
    assert var_11 is False
    var_12 = var_2.__le__(var_0)
    assert var_12 is False
    var_13 = var_2.__gt__(var_0)
    assert var_13 is True
    var_14 = var_4.with_ccy(var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_14) == 3
    var_15 = var_2.__eq__(var_9)
    assert var_15 is False
    var_16 = var_2.subtract(var_5)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.monetary.SomePrice'
    assert len(var_16) == 3
    var_17 = var_9.__repr__()
    assert var_17 == 'None'
    var_18 = var_6.__gt__(var_7)
    assert var_18 is True
    var_19 = var_6.__floordiv__(var_17)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pypara.monetary.NoneMoney'
    var_20 = var_6.dov_or_none()

@pytest.mark.xfail(strict=True)
def test_case_151():
    var_0 = None
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
    var_3 = var_2.gte(var_0)
    assert var_3 is True
    var_4 = module_0.SomeMoney(*var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_4) == 3
    assert f'{type(module_0.SomeMoney.defined).__module__}.{type(module_0.SomeMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.undefined).__module__}.{type(module_0.SomeMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.SomeMoney.price).__module__}.{type(module_0.SomeMoney.price).__qualname__}' == 'builtins.property'
    var_5 = module_0.NoneMoney()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.monetary.NoneMoney'
    assert f'{type(module_0.NoneMoney.defined).__module__}.{type(module_0.NoneMoney.defined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.undefined).__module__}.{type(module_0.NoneMoney.undefined).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.NoneMoney.price).__module__}.{type(module_0.NoneMoney.price).__qualname__}' == 'builtins.property'
    var_6 = var_4.subtract(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_6) == 3
    var_7 = var_2.lt(var_0)
    assert var_7 is False
    var_8 = var_4.with_ccy(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.monetary.SomeMoney'
    assert len(var_8) == 3
    var_9 = var_2.__gt__(var_0)
    assert var_9 is True
    var_6.__add__(var_8)