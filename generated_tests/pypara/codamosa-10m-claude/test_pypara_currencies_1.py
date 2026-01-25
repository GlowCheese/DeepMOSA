# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.currencies as module_0
import dataclasses as module_1

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = module_0.Currency(var_0, var_0, var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.Currency'
    assert f'{type(var_1.code).__module__}.{type(var_1.code).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1.code) == 0
    assert f'{type(var_1.name).__module__}.{type(var_1.name).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1.name) == 0
    assert f'{type(var_1.decimals).__module__}.{type(var_1.decimals).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1.decimals) == 0
    assert f'{type(var_1.type).__module__}.{type(var_1.type).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1.type) == 0
    assert f'{type(var_1.quantizer).__module__}.{type(var_1.quantizer).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1.quantizer) == 0
    assert f'{type(var_1.hashcache).__module__}.{type(var_1.hashcache).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1.hashcache) == 0
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.CurrencyRegistry()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_3) == 0
    var_4 = module_0.CurrencyRegistry()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_4) == 0
    var_5 = module_0.CurrencyRegistry()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_5) == 0
    var_6 = 'qGh:bO7Hq'
    with pytest.raises(module_0.CurrencyLookupError):
        var_4.__getitem__(var_6)

def test_case_1():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'

def test_case_2():
    var_0 = None
    var_1 = module_0.CurrencyRegistry()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    with pytest.raises(module_0.CurrencyLookupError):
        var_1.__getitem__(var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.CurrencyLookupError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_1.code is None
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = False
    var_2 = module_0.Currency(var_0, var_0, var_1, var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.currencies.Currency'
    assert var_2.code is None
    assert var_2.name is None
    assert var_2.decimals is False
    assert var_2.type is None
    assert var_2.quantizer is None
    assert var_2.hashcache is None
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_3 = var_2.__lt__(var_0)
    var_2.quantize(var_3)

def test_case_5():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = var_0.__len__()
    assert var_1 == 0
    var_2 = None
    var_3 = module_0.CurrencyRegistry()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_3) == 0
    with pytest.raises(module_0.CurrencyLookupError):
        var_3.__getitem__(var_2)

def test_case_6():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = None
    var_2 = var_0.__contains__(var_1)
    assert var_2 is False
    var_3 = None
    with pytest.raises(module_0.CurrencyLookupError):
        var_0.__getitem__(var_3)

def test_case_7():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = var_0.has(var_0)
    assert var_1 is False
    var_2 = module_0.Currency(var_1, var_1, var_1, var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.currencies.Currency'
    assert var_2.code is False
    assert var_2.name is False
    assert var_2.decimals is False
    assert var_2.type is False
    assert var_2.quantizer is False
    assert var_2.hashcache is False
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.CurrencyRegistry()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_4) == 0
    var_5 = module_0.CurrencyRegistry()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_5) == 0
    var_6 = module_0.CurrencyRegistry()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_6) == 0
    var_7 = 'qGh:bO7Hq'
    with pytest.raises(module_0.CurrencyLookupError):
        var_5.__getitem__(var_7)

def test_case_8():
    var_0 = None
    var_1 = None
    var_2 = module_0.CurrencyRegistry()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_2) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_3 = var_2.get(var_0)
    with pytest.raises(module_0.CurrencyLookupError):
        var_2.__getitem__(var_1)

def test_case_9():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = var_0.has(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = module_0.Currency(var_2, var_2, var_1, var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.Currency'
    assert var_3.code is None
    assert var_3.name is None
    assert var_3.decimals is False
    assert var_3.type is False
    assert var_3.quantizer is None
    assert var_3.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.CurrencyRegistry()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_5) == 0
    var_6 = var_3.__le__(var_2)
    var_7 = var_3.__hash__()
    var_8 = 'CO2YJp/UHDA'
    var_9 = var_0.get(var_8)
    var_10 = var_5.get(var_6, var_6)
    var_11 = var_9.__eq__(var_9)
    assert var_11 is True
    with pytest.raises(module_0.CurrencyLookupError):
        var_5.__getitem__(var_6)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = '..+qvi]!n<.ZES~ y\x0cx'
    var_2 = var_0.has(var_1)
    assert var_2 is False
    var_3 = None
    var_4 = 3062
    var_5 = module_0.Currency(var_3, var_3, var_4, var_3, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.Currency'
    assert var_5.code is None
    assert var_5.name is None
    assert var_5.decimals == 3062
    assert var_5.type is None
    assert var_5.quantizer is None
    assert var_5.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_6 = var_5.__gt__(var_3)
    var_7 = module_0.Currency(var_1, var_3, var_3, var_6, var_6, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.Currency'
    assert var_7.code == '..+qvi]!n<.ZES~ y\x0cx'
    assert var_7.name is None
    assert var_7.decimals is None
    assert f'{type(var_7.type).__module__}.{type(var_7.type).__qualname__}' == 'builtins.NotImplementedType'
    assert f'{type(var_7.quantizer).__module__}.{type(var_7.quantizer).__qualname__}' == 'builtins.NotImplementedType'
    assert var_7.hashcache is None
    var_8 = var_0.has(var_3)
    assert var_8 is False
    var_9 = var_5.__eq__(var_7)
    assert var_9 is True
    var_10 = module_0.CurrencyRegistry()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_10) == 0
    var_11 = var_7.__le__(var_6)
    var_12 = var_5.__hash__()
    var_10.get(var_7)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = var_0.has(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = 3319
    var_4 = module_0.Currency(var_2, var_2, var_3, var_3, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.Currency'
    assert var_4.code is None
    assert var_4.name is None
    assert var_4.decimals == 3319
    assert var_4.type == 3319
    assert var_4.quantizer is None
    assert var_4.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_5 = var_0.has(var_2)
    assert var_5 is False
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    var_8 = var_4.__le__(var_2)
    var_9 = var_4.__hash__()
    var_10 = var_0.get(var_8)
    var_11 = var_0.__exit__(var_2, var_2, var_10)
    var_12 = module_1.dataclass()
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_13 = module_0.CurrencyRegistry()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_13) == 0
    var_14 = var_7.__enter__()
    var_15 = module_0.CurrencyRegistry()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_15) == 0
    var_16 = var_4.__ge__(var_12)
    var_13.__getitem__(var_4)

def test_case_12():
    var_0 = module_0.CurrencyRegistry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_0) == 0
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_1 = len(var_0)
    var_2 = len(var_0)
    var_3 = var_0.all
    var_4 = len(var_3)
    var_5 = len(var_0)
    var_6 = var_0.codes
    var_7 = len(var_6)
    var_8 = len(var_0)
    var_9 = var_0.codenames
    var_10 = len(var_9)