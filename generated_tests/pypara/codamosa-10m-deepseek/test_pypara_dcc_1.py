# Check out: https://github.com/GlowCheese/deepmosa
import codecs as module_1

import pypara.dcc as module_0
import pytest


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
    module_0.dcfc_nl_365(var_0, var_0, var_0)

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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.dcfc_act_365_f(var_0, var_0, var_0, var_0)