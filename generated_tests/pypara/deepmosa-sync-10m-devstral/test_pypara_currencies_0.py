# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.currencies as module_0

def test_case_0():
    pass

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

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = {}
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
    var_1.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_2 = var_0.get(var_1)
    var_3 = var_2.__eq__(var_0)
    var_4 = var_0.get(var_0)
    var_4.__delattr__(var_4)

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
    var_2 = var_0.codes
    var_3 = bool(var_0.codes == [])
    assert var_3 is True
    var_4 = var_0.codenames
    var_5 = bool(var_0.codenames == [])
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'EP\x0blT7O\r4['
    var_1 = module_0.CurrencyLookupError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_1.code == 'EP\x0blT7O\r4['
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    var_2 = None
    var_3 = module_0.Currency(var_2, var_2, var_2, var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.Currency'
    assert var_3.code is None
    assert var_3.name is None
    assert var_3.decimals is None
    assert var_3.type is None
    assert var_3.quantizer is None
    assert var_3.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_4 = var_3.__hash__()
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_1.__contains__(var_2)

@pytest.mark.xfail(strict=True)
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
    var_2 = var_0.__contains__(var_1)
    assert var_2 is False
    var_3 = None
    var_4 = None
    var_5 = None
    var_6 = var_0.get(var_4, var_3)
    var_7 = var_6.__gt__(var_4)
    var_6.__setattr__(var_7, var_5)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.CurrencyLookupError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_1.code is None
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    var_2 = ']Rt/:w;H/A0p*+IF\tA'
    var_3 = module_0.Currency(var_2, var_0, var_0, var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.Currency'
    assert var_3.code == ']Rt/:w;H/A0p*+IF\tA'
    assert var_3.name is None
    assert var_3.decimals is None
    assert var_3.type is None
    assert var_3.quantizer is None
    assert var_3.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_4 = var_3.__repr__()
    assert var_4 == "Currency(code=']Rt/:w;H/A0p*+IF\\tA', name=None, decimals=None, type=None, quantizer=None, hashcache=None)"
    var_5 = var_4.__lt__(var_4)
    assert var_5 is False
    var_6 = var_3.__lt__(var_4)
    var_7 = var_3.__eq__(var_0)
    assert var_7 is False
    var_8 = var_6.__repr__()
    assert var_8 == 'NotImplemented'
    var_3.quantize(var_4)

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
    var_1 = module_0.CurrencyRegistry()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_1) == 0
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_0.has(var_4)
    assert var_5 is False
    var_6 = bool(not var_5)
    assert var_6 is True
    var_7 = var_0.get(var_4)
    assert var_7 is None
    var_8 = var_0.all
    var_9 = bool(var_0.all == [])
    assert var_9 is True
    var_10 = var_0.codes
    var_11 = bool(var_0.codes == [])
    assert var_11 is True
    var_12 = var_0.codenames
    var_13 = bool(var_0.codenames == [])
    assert var_13 is True

def test_case_10():
    var_0 = 'EP\x0blT7O\r4['
    var_1 = module_0.CurrencyLookupError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_1.code == 'EP\x0blT7O\r4['
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    var_2 = None
    var_3 = module_0.Currency(var_2, var_2, var_2, var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.Currency'
    assert var_3.code is None
    assert var_3.name is None
    assert var_3.decimals is None
    assert var_3.type is None
    assert var_3.quantizer is None
    assert var_3.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_4 = var_3.__hash__()
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True
    var_6 = var_3.__hash__()
    var_7 = module_0.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    assert f'{type(module_0.CurrencyRegistry.all).__module__}.{type(module_0.CurrencyRegistry.all).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codes).__module__}.{type(module_0.CurrencyRegistry.codes).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.CurrencyRegistry.codenames).__module__}.{type(module_0.CurrencyRegistry.codenames).__qualname__}' == 'builtins.property'
    var_8 = "<d[Ukax%P*'l>u"
    with pytest.raises(module_0.CurrencyLookupError):
        var_7.__getitem__(var_8)