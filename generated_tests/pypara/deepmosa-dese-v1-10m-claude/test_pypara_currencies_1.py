# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.commons.numbers as module_0
import decimal as module_1
import pypara.currencies as module_2
import dataclasses as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = "LpGwEl('6/SV"
    var_2 = '&H\x0bCFM6'
    var_3 = None
    var_4 = -1573
    var_5 = module_0.make_quantizer(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
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
    var_6 = module_2.Currency(var_1, var_2, var_3, var_3, var_5, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.currencies.Currency'
    assert var_6.code == "LpGwEl('6/SV"
    assert var_6.name == '&H\x0bCFM6'
    assert var_6.decimals is None
    assert var_6.type is None
    assert f'{type(var_6.quantizer).__module__}.{type(var_6.quantizer).__qualname__}' == 'decimal.Decimal'
    assert var_6.hashcache is None
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
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.Currency.of).__module__}.{type(module_2.Currency.of).__qualname__}' == 'builtins.method'
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = None
    module_0.make_quantizer(var_8)

def test_case_1():
    var_0 = module_2.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'

def test_case_2():
    var_0 = module_2.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    with pytest.raises(module_2.CurrencyLookupError):
        var_1 = var_0[var_0]

def test_case_3():
    var_0 = None
    var_1 = module_2.CurrencyLookupError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_1.code is None
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = False
    var_2 = module_2.Currency(var_0, var_0, var_1, var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.currencies.Currency'
    assert var_2.code is None
    assert var_2.name is None
    assert var_2.decimals is False
    assert var_2.type is None
    assert var_2.quantizer is None
    assert var_2.hashcache is None
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.Currency.of).__module__}.{type(module_2.Currency.of).__qualname__}' == 'builtins.method'
    var_3 = var_2.__lt__(var_0)
    var_2.quantize(var_3)

def test_case_5():
    var_0 = module_2.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = 'NON-EXISTING'
    var_2 = var_0.__len__()
    assert var_2 == 0
    with pytest.raises(module_2.CurrencyLookupError):
        var_3 = var_0[var_1]

def test_case_6():
    var_0 = module_2.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.__contains__(var_1)
    assert var_2 is False
    var_3 = None
    with pytest.raises(module_2.CurrencyLookupError):
        var_0.__getitem__(var_3)

def test_case_7():
    var_0 = module_2.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = 'UYD'
    var_2 = var_0.has(var_1)
    assert var_2 is False
    var_3 = 'EUR'
    with pytest.raises(module_2.CurrencyLookupError):
        var_4 = var_0[var_3]

def test_case_8():
    var_0 = None
    var_1 = None
    var_2 = module_2.CurrencyRegistry()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_2) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_3 = var_2.get(var_0)
    with pytest.raises(module_2.CurrencyLookupError):
        var_2.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = '<2KOn#y[@D_='
    var_2 = False
    var_3 = module_2.CurrencyType.ALTERNATIVE
    var_4 = module_2.Currency(var_0, var_1, var_2, var_3, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.Currency'
    assert var_4.code is None
    assert var_4.name == '<2KOn#y[@D_='
    assert var_4.decimals is False
    assert var_4.type == module_2.CurrencyType.ALTERNATIVE
    assert var_4.quantizer is None
    assert var_4.hashcache is None
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.Currency.of).__module__}.{type(module_2.Currency.of).__qualname__}' == 'builtins.method'
    var_5 = var_4.__ge__(var_0)
    var_6 = var_5.__ge__(var_0)
    var_7 = module_2.CurrencyLookupError(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_7.code is None
    var_8 = None
    var_9 = var_4.__ge__(var_0)
    var_10 = var_5.__gt__(var_6)
    var_11 = var_4.__lt__(var_8)
    var_12 = var_4.__hash__()
    var_13 = module_3.dataclass(order=var_0, slots=var_3)
    assert f'{type(module_3.MISSING).__module__}.{type(module_3.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_3.KW_ONLY).__module__}.{type(module_3.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_9.__exit__(var_10, var_0, var_8)

def test_case_10():
    var_0 = module_2.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = var_0.codes
    var_4 = var_0.codenames

def test_case_11():
    var_0 = module_2.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = var_0.codes
    var_2 = var_0.codenames

def test_case_12():
    var_0 = "LpGwEl('6/SV"
    var_1 = '&H\x0bCFM6'
    var_2 = None
    var_3 = -1573
    var_4 = module_0.make_quantizer(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
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
    var_5 = module_2.Currency(var_0, var_1, var_2, var_2, var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.Currency'
    assert var_5.code == "LpGwEl('6/SV"
    assert var_5.name == '&H\x0bCFM6'
    assert var_5.decimals is None
    assert var_5.type is None
    assert f'{type(var_5.quantizer).__module__}.{type(var_5.quantizer).__qualname__}' == 'decimal.Decimal'
    assert var_5.hashcache is None
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
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.Currency.of).__module__}.{type(module_2.Currency.of).__qualname__}' == 'builtins.method'
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = module_2.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_8 = var_7.__len__()
    assert var_8 == 0
    var_9 = None
    var_10 = var_5.__eq__(var_9)
    assert var_10 is False
    var_11 = '.~v\\@+#y7Gd'
    with pytest.raises(module_2.CurrencyLookupError):
        var_7.__getitem__(var_11)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = '&H\x0bCFM6'
    var_2 = None
    var_3 = -1573
    var_4 = module_0.make_quantizer(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
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
    var_5 = module_2.Currency(var_1, var_1, var_2, var_2, var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.Currency'
    assert var_5.code == '&H\x0bCFM6'
    assert var_5.name == '&H\x0bCFM6'
    assert var_5.decimals is None
    assert var_5.type is None
    assert f'{type(var_5.quantizer).__module__}.{type(var_5.quantizer).__qualname__}' == 'decimal.Decimal'
    assert var_5.hashcache is None
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
    assert f'{type(module_2.ZERO).__module__}.{type(module_2.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.MaxPrecisionQuantizer).__module__}.{type(module_2.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.Currencies).__module__}.{type(module_2.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_2.Currencies) == 0
    assert f'{type(module_2.Currency.of).__module__}.{type(module_2.Currency.of).__qualname__}' == 'builtins.method'
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = module_2.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    assert f'{type(module_2.CurrencyRegistry.all).__module__}.{type(module_2.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codes).__module__}.{type(module_2.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.CurrencyRegistry.codenames).__module__}.{type(module_2.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_8 = var_7.__len__()
    assert var_8 == 0
    var_7.__getitem__(var_5)