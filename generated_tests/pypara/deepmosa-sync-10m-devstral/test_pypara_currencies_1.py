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
    var_1 = False
    var_2 = None
    var_3 = module_0.Currency(var_1, var_2, var_1, var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.Currency'
    assert var_3.code is False
    assert var_3.name is None
    assert var_3.decimals is False
    assert var_3.type is None
    assert var_3.quantizer is None
    assert var_3.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False

def test_case_1():
    pass

def test_case_2():
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

def test_case_3():
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
    with pytest.raises(module_0.CurrencyLookupError):
        var_0.__getitem__(var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.CurrencyLookupError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_1.code is None
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0

@pytest.mark.xfail(strict=True)
def test_case_5():
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
    var_1 = var_0.__len__()
    assert var_1 == 0
    var_2 = None
    var_3 = module_0.CurrencyRegistry()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_3) == 0
    with pytest.raises(module_0.CurrencyLookupError):
        var_3.__getitem__(var_2)

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
    var_1 = None
    var_2 = 3319
    var_3 = -1867
    var_4 = module_0.Currency(var_1, var_1, var_3, var_1, var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.Currency'
    assert var_4.code is None
    assert var_4.name is None
    assert var_4.decimals == -1867
    assert var_4.type is None
    assert var_4.quantizer is None
    assert var_4.hashcache == 3319
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_5 = var_4.__eq__(var_1)
    assert var_5 is False
    var_6 = module_0.CurrencyRegistry()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_6) == 0
    var_7 = var_6.__contains__(var_1)
    assert var_7 is False
    var_8 = var_4.__gt__(var_1)
    with pytest.raises(module_0.CurrencyLookupError):
        var_0.__getitem__(var_8)

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_3 = None
    var_4 = 3319
    var_5 = module_0.Currency(var_3, var_3, var_4, var_4, var_2, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.Currency'
    assert var_5.code is None
    assert var_5.name is None
    assert var_5.decimals == 3319
    assert var_5.type == 3319
    assert var_5.quantizer is None
    assert var_5.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = None
    var_8 = ''
    var_9 = module_1.dataclass(frozen=var_3, kw_only=var_7)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_9.__getitem__(var_8)

def test_case_9():
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
    var_1 = 3319
    var_2 = -1867
    var_3 = module_0.Currency(var_1, var_1, var_2, var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.Currency'
    assert var_3.code == 3319
    assert var_3.name == 3319
    assert var_3.decimals == -1867
    assert var_3.type == 3319
    assert var_3.quantizer == 3319
    assert var_3.hashcache == 3319
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_4 = module_0.CurrencyRegistry()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_4) == 0
    var_5 = var_4.__contains__(var_3)
    assert var_5 is False
    var_6 = None
    var_7 = module_0.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    with pytest.raises(module_0.CurrencyLookupError):
        var_7.__getitem__(var_6)

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
    var_1 = module_0.CurrencyRegistry()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1) == 0
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = var_0.all
    var_5 = bool(var_0.all == [])
    assert var_5 is True
    var_6 = var_0.codes
    var_7 = bool(var_0.codes == [])
    assert var_7 is True
    var_8 = var_0.codenames
    var_9 = bool(var_0.codenames == [])
    assert var_9 is True

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
    var_1 = False
    var_2 = None
    var_3 = var_0.__ge__(var_2)
    var_4 = module_0.Currency(var_1, var_0, var_1, var_0, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.Currency'
    assert var_4.code is False
    assert f'{type(var_4.name).__module__}.{type(var_4.name).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_4.name) == 0
    assert var_4.decimals is False
    assert f'{type(var_4.type).__module__}.{type(var_4.type).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_4.type) == 0
    assert f'{type(var_4.quantizer).__module__}.{type(var_4.quantizer).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_4.quantizer) == 0
    assert f'{type(var_4.hashcache).__module__}.{type(var_4.hashcache).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_4.hashcache) == 0
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_13():
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
    var_2 = [var_1, var_0, var_0, var_0]
    var_0.__getitem__(var_2)