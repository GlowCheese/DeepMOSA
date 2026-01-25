# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.dcc as module_0
import codecs as module_1
import datetime as module_2

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_0.find(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.dcfc_act_365_f(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_1.iterencode(var_0, var_0, var_0)
    assert module_1.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_1.BOM_LE == b'\xff\xfe'
    assert module_1.BOM_UTF16_LE == b'\xff\xfe'
    assert module_1.BOM_BE == b'\xfe\xff'
    assert module_1.BOM_UTF16_BE == b'\xfe\xff'
    assert module_1.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_1.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_1.BOM == b'\xff\xfe'
    assert module_1.BOM_UTF16 == b'\xff\xfe'
    assert module_1.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_1.BOM32_LE == b'\xff\xfe'
    assert module_1.BOM32_BE == b'\xfe\xff'
    assert module_1.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_1.BOM64_BE == b'\x00\x00\xfe\xff'
    module_0.dcfc_act_360(var_1, var_0, var_1)

def test_case_4():
    var_0 = 0
    var_1 = 1
    var_2 = 15
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_5():
    var_0 = 2023
    var_1 = 13
    var_2 = 15
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_6():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = 8
    var_2 = False
    with pytest.raises(ValueError):
        module_0._construct_date(var_1, var_2, var_0)

def test_case_7():
    var_0 = 573
    var_1 = -583
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_0, var_1)

def test_case_8():
    var_0 = 2023
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = 1
    var_2 = '\\fe~ZVWO1FYj`\t0^'
    var_3 = var_0.find(var_2)
    var_4 = 2081.0
    var_5 = True
    module_0._construct_date(var_4, var_1, var_5)