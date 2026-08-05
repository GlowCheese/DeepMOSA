# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.currencies as module_0

@pytest.mark.xfail(strict=True)
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
    var_6 = var_0.__len__()
    assert var_6 == 0
    var_7 = var_5.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.CurrencyRegistry()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_8) == 0
    var_5.__delattr__(var_2)

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
        var_1 = var_0['USD']

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
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True

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
    var_3 = 'RE{9tl'
    with pytest.raises(module_0.CurrencyLookupError):
        var_0.__getitem__(var_3)

def test_case_7():
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
    var_3 = module_0.Currency(var_2, var_2, var_1, var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.Currency'
    assert var_3.code is None
    assert var_3.name is None
    assert var_3.decimals is False
    assert var_3.type is False
    assert var_3.quantizer is None
    assert var_3.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_4 = var_0.__len__()
    assert var_4 == 0
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_6 = module_0.CurrencyRegistry()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_6) == 0
    var_0.__getitem__(var_3)

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
    var_1 = var_0.all
    var_2 = bool(var_0.all == [])
    assert var_2 is True
    with pytest.raises(AttributeError):
        var_3 = var_2.codes

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
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = var_3.__eq__(var_1)
    assert var_6 is False
    var_7 = module_0.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    var_8 = '\x0cW`Ym4^'
    var_9 = var_7.get(var_8)
    with pytest.raises(module_0.CurrencyLookupError):
        var_7.__getitem__(var_2)