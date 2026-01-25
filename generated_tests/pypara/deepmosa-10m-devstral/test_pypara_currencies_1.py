# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.currencies as module_0
import dataclasses as module_1

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
    var_1 = var_0.__contains__(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = None
    var_4 = '2=CJrub'
    var_5 = module_0.CurrencyType.MONEY
    var_6 = module_0.Currency(var_2, var_4, var_2, var_5, var_2, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.currencies.Currency'
    assert var_6.code is None
    assert var_6.name == '2=CJrub'
    assert var_6.decimals is None
    assert var_6.type == module_0.CurrencyType.MONEY
    assert var_6.quantizer is None
    assert var_6.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_7 = var_6.__eq__(var_3)
    assert var_7 is False
    var_8 = module_1.dataclass(var_3, unsafe_hash=var_3, match_args=var_3)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_9 = module_0.CurrencyRegistry()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_9) == 0
    var_10 = var_6.__repr__()
    assert var_10 == "Currency(code=None, name='2=CJrub', decimals=None, type=<CurrencyType.MONEY: 'Money'>, quantizer=None, hashcache=None)"
    var_11 = var_8.__le__(var_3)
    var_12 = var_8.__le__(var_2)
    var_3.__delattr__(var_10)

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
        var_0.__getitem__(var_0)

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
    var_5 = var_0.codenames

@pytest.mark.xfail(strict=True)
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
    var_7 = module_0.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    var_8 = var_5.__le__(var_3)
    var_9 = var_5.__lt__(var_8)
    var_10 = module_1.dataclass(var_2, kw_only=var_2, slots=var_2)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_11 = var_2.__le__(var_2)
    var_12 = var_5.__ge__(var_9)
    var_13 = module_0.CurrencyRegistry()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_13) == 0
    var_14 = 'sd'
    var_11.__getitem__(var_14)

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
    var_0 = None
    var_1 = '<2KOn#y[@D_='
    var_2 = False
    var_3 = module_0.CurrencyType.ALTERNATIVE
    var_4 = module_0.Currency(var_0, var_1, var_2, var_3, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.Currency'
    assert var_4.code is None
    assert var_4.name == '<2KOn#y[@D_='
    assert var_4.decimals is False
    assert var_4.type == module_0.CurrencyType.ALTERNATIVE
    assert var_4.quantizer is None
    assert var_4.hashcache is None
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_5 = var_4.__ge__(var_0)
    var_6 = var_5.__ge__(var_0)
    var_7 = module_0.CurrencyLookupError(var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_7.code is None
    var_8 = None
    var_9 = var_4.__ge__(var_0)
    var_10 = var_5.__gt__(var_6)
    var_11 = var_4.__lt__(var_8)
    var_12 = var_4.__hash__()
    var_13 = module_1.dataclass(order=var_0, slots=var_3)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_9.__exit__(var_10, var_0, var_8)

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
    var_1 = var_0.__contains__(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = '2=CJrub'
    var_4 = module_0.CurrencyType.MONEY
    var_5 = module_0.Currency(var_2, var_3, var_2, var_4, var_2, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.currencies.Currency'
    assert var_5.code is None
    assert var_5.name == '2=CJrub'
    assert var_5.decimals is None
    assert var_5.type == module_0.CurrencyType.MONEY
    assert var_5.quantizer is None
    assert var_5.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = module_1.dataclass(var_2, unsafe_hash=var_2, match_args=var_2)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_8 = var_7.__ge__(var_2)
    var_9 = var_0.__len__()
    assert var_9 == 0
    var_10 = None
    with pytest.raises(module_0.CurrencyLookupError):
        var_0.__getitem__(var_10)

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
    var_1 = var_0.__contains__(var_0)
    assert var_1 is False
    var_2 = None
    var_3 = None
    var_4 = '2=CJrub'
    var_5 = module_0.CurrencyType.MONEY
    var_6 = module_0.Currency(var_2, var_4, var_2, var_5, var_2, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.currencies.Currency'
    assert var_6.code is None
    assert var_6.name == '2=CJrub'
    assert var_6.decimals is None
    assert var_6.type == module_0.CurrencyType.MONEY
    assert var_6.quantizer is None
    assert var_6.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_0.__getitem__(var_6)