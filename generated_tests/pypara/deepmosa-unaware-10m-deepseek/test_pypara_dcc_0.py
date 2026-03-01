# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.dcc as module_0

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
    module_0.dcfc_act_act_icma(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.dcfc_act_360(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = "/X&w)1^}e^'\n"
    var_2 = var_0.find(var_1)
    module_0.dcfc_act_365_f(var_2, var_2, var_2)

def test_case_5():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = 'ACT/360'
    var_2 = 'Actual/360'
    var_3 = set()
    var_4 = '30/360'
    var_5 = 'Bond Basis'
    var_6 = set()
    var_7 = var_0.find(var_1)
    var_8 = var_0.find(var_2)
    var_9 = var_0.find(var_4)
    var_10 = var_0.find(var_5)
    var_11 = '  act/360  '
    var_12 = var_0.find(var_11)
    var_13 = '  actual/360  '
    var_14 = var_0.find(var_13)
    var_15 = 'act/360'
    var_16 = var_0.find(var_15)
    var_17 = 'NonExistent'
    var_18 = var_0.find(var_17)
    assert var_18 is None
    var_19 = ''
    var_20 = var_0.find(var_19)
    assert var_20 is None
    var_21 = var_0.registry
    var_22 = len(var_21)