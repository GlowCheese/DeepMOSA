# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.currencies as module_0
import dataclasses as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Currency(var_0, var_0, var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.Currency'
    assert var_1.code is None
    assert var_1.name is None
    assert var_1.decimals is None
    assert var_1.type is None
    assert var_1.quantizer is None
    assert var_1.hashcache is None
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

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
    var_1 = '^f,mgI+-?P9TmxX~\x0b'
    with pytest.raises(module_0.CurrencyLookupError):
        var_0.__getitem__(var_1)

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
    var_1 = module_0.Currency(var_0, var_0, var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.currencies.Currency'
    assert var_1.code is None
    assert var_1.name is None
    assert var_1.decimals is None
    assert var_1.type is None
    assert var_1.quantizer is None
    assert var_1.hashcache is None
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.MaxPrecisionQuantizer).__module__}.{type(module_0.MaxPrecisionQuantizer).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 0
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = None
    var_4 = module_0.CurrencyLookupError(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.CurrencyLookupError'
    assert var_4.code is None
    var_5 = None
    var_1.quantize(var_5)

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
    var_10 = 'CO2YJp/UHDA'
    var_11 = var_0.get(var_10)
    var_12 = None
    var_13 = module_1.dataclass(var_12, kw_only=var_12, slots=var_12)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_14 = var_11.__le__(var_2)
    var_14.__enter__()

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

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    var_1 = var_0.__len__()
    assert var_1 == 0
    var_2 = 'h'
    var_3 = None
    var_4 = module_0.Currency(var_2, var_2, var_3, var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.currencies.Currency'
    assert var_4.code == 'h'
    assert var_4.name == 'h'
    assert var_4.decimals is None
    assert var_4.type is None
    assert var_4.quantizer is None
    assert var_4.hashcache == 0
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_5 = None
    var_6 = '8KpZ'
    var_7 = '"C]4m\\`2S'
    var_8 = var_4.__lt__(var_5)
    var_9 = module_0.Currency(var_6, var_7, var_1, var_3, var_8, var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.currencies.Currency'
    assert var_9.code == '8KpZ'
    assert var_9.name == '"C]4m\\`2S'
    assert var_9.decimals == 0
    assert var_9.type is None
    assert f'{type(var_9.quantizer).__module__}.{type(var_9.quantizer).__qualname__}' == 'builtins.NotImplementedType'
    assert var_9.hashcache is None
    var_10 = var_0.get(var_6, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.currencies.Currency'
    assert var_10.code == '8KpZ'
    assert var_10.name == '"C]4m\\`2S'
    assert var_10.decimals == 0
    assert var_10.type is None
    assert f'{type(var_10.quantizer).__module__}.{type(var_10.quantizer).__qualname__}' == 'builtins.NotImplementedType'
    assert var_10.hashcache is None
    var_11 = var_4.__repr__()
    assert var_11 == "Currency(code='h', name='h', decimals=None, type=None, quantizer=None, hashcache=0)"
    var_12 = '\tGr>k'
    var_13 = module_0.Currency(var_11, var_12, var_11, var_11, var_8, var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.currencies.Currency'
    assert var_13.code == "Currency(code='h', name='h', decimals=None, type=None, quantizer=None, hashcache=0)"
    assert var_13.name == '\tGr>k'
    assert var_13.decimals == "Currency(code='h', name='h', decimals=None, type=None, quantizer=None, hashcache=0)"
    assert var_13.type == "Currency(code='h', name='h', decimals=None, type=None, quantizer=None, hashcache=0)"
    assert f'{type(var_13.quantizer).__module__}.{type(var_13.quantizer).__qualname__}' == 'builtins.NotImplementedType'
    assert f'{type(var_13.hashcache).__module__}.{type(var_13.hashcache).__qualname__}' == 'builtins.NotImplementedType'
    var_14 = var_13.__eq__(var_10)
    assert var_14 is False
    var_15 = module_0.CurrencyRegistry()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_15) == 0
    var_16 = var_0.__len__()
    assert var_16 == 0
    var_8.__getitem__(var_5)

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
    var_1 = 'V">%\t)4XiV4t=M[t'
    var_2 = var_0.has(var_1)
    assert var_2 is False
    var_3 = var_0.__len__()
    assert var_3 == 0
    var_4 = None
    var_5 = None
    var_6 = var_0.get(var_5)
    var_7 = var_6.__repr__()
    assert var_7 == 'None'
    var_8 = True
    var_9 = 'xF\x0c6'
    var_10 = False
    var_11 = module_0.Currency(var_9, var_5, var_10, var_7, var_8, var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pypara.currencies.Currency'
    assert var_11.code == 'xF\x0c6'
    assert var_11.name is None
    assert var_11.decimals is False
    assert var_11.type == 'None'
    assert var_11.quantizer is True
    assert var_11.hashcache is None
    assert f'{type(module_0.Currency.of).__module__}.{type(module_0.Currency.of).__qualname__}' == 'builtins.method'
    var_12 = var_7.__eq__(var_5)
    var_13 = module_0.CurrencyRegistry()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_13) == 0
    var_14 = var_13.__len__()
    assert var_14 == 0
    var_13.__getitem__(var_11)

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
    var_1 = 'USD'
    var_2 = var_0.get(var_1)
    var_3 = module_0.CurrencyRegistry()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_3) == 0
    var_4 = 'NONEXISTENT'
    var_5 = var_3.get(var_4)
    assert var_5 is None
    var_6 = module_0.CurrencyRegistry()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_6) == 0
    var_7 = module_0.CurrencyRegistry()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_7) == 0
    var_8 = module_0.CurrencyRegistry()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(var_8) == 0
    var_9 = var_8.all
    var_10 = len(var_9)