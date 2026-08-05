# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.dcc as module_0
import enum as module_1
import datetime as module_2
import decimal as module_3

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
    var_1 = '4O"n\t'
    var_2 = var_0.find(var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.dcfc_act_360(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = '4O"n\t'
    var_2 = var_0.find(var_1)
    var_3 = module_1._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = var_3.__len__()
    assert var_4 == 0
    module_0.dcfc_act_365_f(var_3, var_3, var_3)

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
    var_1 = var_0.registry

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
    var_1 = bool(var_0.table == {})
    assert var_1 is True

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_2.date(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = module_0.dcfc_act_act(var_5, var_10, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = {}
    module_3.Decimal(*var_16, **var_17)

def test_case_7():
    var_0 = 2023
    var_1 = 9
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_2.date(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.date'
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = module_0.dcfc_act_act(var_4, var_7, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_8():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_2.date(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.date'
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = module_0.dcfc_act_act(var_4, var_7, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_12 = None
    var_13 = module_0.dcfc_30_360_isda(var_7, var_7, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'

def test_case_9():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = module_0._is_last_day_of_month(var_5)
    assert var_6 is True
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'

def test_case_10():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = '0.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_3.Decimal(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = 'Act/360'
    var_6 = set()
    var_7 = set()
    var_8 = []
    var_9 = 'name'
    var_10 = 'altnames'
    var_11 = 'currencies'
    var_12 = 'calculate_fraction_method'
    var_13 = {var_9: var_5, var_10: var_6, var_11: var_7, var_12: var_11}
    var_14 = module_0.DCC(*var_8, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_14) == 4
    assert f'{type(module_0.DCC.name).__module__}.{type(module_0.DCC.name).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.altnames).__module__}.{type(module_0.DCC.altnames).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.currencies).__module__}.{type(module_0.DCC.currencies).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.calculate_fraction_method).__module__}.{type(module_0.DCC.calculate_fraction_method).__qualname__}' == '_collections._tuplegetter'
    var_15 = set()
    var_16 = set()
    var_17 = []
    var_18 = 'name'
    var_19 = 'altnames'
    var_20 = 'currencies'
    var_21 = 'calculate_fraction_method'
    var_22 = {var_18: var_5, var_19: var_15, var_20: var_16, var_21: var_3}
    var_23 = module_0.DCC(*var_17, **var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_23) == 4
    var_24 = var_0.register(var_14)
    with pytest.raises(TypeError):
        var_0.register(var_23)

def test_case_11():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = '0.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_3.Decimal(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = 'ACT360'
    var_8 = 'ACT/360'
    var_9 = {var_7, var_8}
    var_10 = set()
    var_11 = []
    var_12 = 'name'
    var_13 = 'altnames'
    var_14 = 'currencies'
    var_15 = 'calculate_fraction_method'
    var_16 = {var_12: var_6, var_13: var_9, var_14: var_10, var_15: var_5}
    var_17 = module_0.DCC(*var_11, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_17) == 4
    assert f'{type(module_0.DCC.name).__module__}.{type(module_0.DCC.name).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.altnames).__module__}.{type(module_0.DCC.altnames).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.currencies).__module__}.{type(module_0.DCC.currencies).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.calculate_fraction_method).__module__}.{type(module_0.DCC.calculate_fraction_method).__qualname__}' == '_collections._tuplegetter'
    var_18 = var_0.register(var_17)
    var_19 = var_0.find(var_6)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_19) == 4
    var_20 = bool(var_19 == var_17)
    assert var_20 is True
    var_21 = var_0.find(var_7)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_21) == 4
    var_22 = bool(var_21 == var_17)
    assert var_22 is True
    var_23 = 'act/360'
    var_24 = var_0.find(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_24) == 4
    var_25 = bool(var_24 == var_17)
    assert var_25 is True
    var_26 = var_0.registry
    var_27 = len(var_26)
    assert var_27 == 1

def test_case_12():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = '0.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_3.Decimal(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = set()
    var_8 = set()
    var_9 = []
    var_10 = 'name'
    var_11 = 'altnames'
    var_12 = 'currencies'
    var_13 = 'calculate_fraction_method'
    var_14 = {var_10: var_6, var_11: var_7, var_12: var_8, var_13: var_5}
    var_15 = module_0.DCC(*var_9, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_15) == 4
    assert f'{type(module_0.DCC.name).__module__}.{type(module_0.DCC.name).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.altnames).__module__}.{type(module_0.DCC.altnames).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.currencies).__module__}.{type(module_0.DCC.currencies).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.calculate_fraction_method).__module__}.{type(module_0.DCC.calculate_fraction_method).__qualname__}' == '_collections._tuplegetter'
    var_16 = {var_6}
    var_17 = set()
    var_18 = []
    var_19 = 'name'
    var_20 = 'altnames'
    var_21 = 'currencies'
    var_22 = 'calculate_fraction_method'
    var_23 = {var_19: var_11, var_20: var_16, var_21: var_17, var_22: var_5}
    var_24 = module_0.DCC(*var_18, **var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_24) == 4
    var_25 = var_0.register(var_15)
    with pytest.raises(TypeError):
        var_0.register(var_24)

def test_case_13():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 3
    var_6 = [var_0, var_5, var_1]
    var_7 = {}
    var_8 = module_2.date(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.date'
    var_9 = [var_0, var_5, var_1]
    var_10 = {}
    var_11 = module_2.date(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
    var_12 = '60'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_3.Decimal(*var_13, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_16 = {}
    var_17 = module_3.Decimal(*var_13, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'decimal.Decimal'
    var_18 = var_15 / var_17
    var_19 = module_0.dcfc_act_365_a(var_4, var_8, var_11)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_20 = bool(var_19 == var_18)

def test_case_14():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 10
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_2.date(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.date'
    var_9 = [var_0, var_1, var_5]
    var_10 = {}
    var_11 = module_2.date(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
    var_12 = '9'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_3.Decimal(*var_13, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_16 = var_15 / var_15
    var_17 = module_0.dcfc_act_365_a(var_4, var_8, var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_18 = bool(var_17 == var_16)

def test_case_15():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_2.date(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = '62'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_3.Decimal(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_18 = '365'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_3.Decimal(*var_19, **var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'decimal.Decimal'
    var_22 = var_17 / var_21
    var_23 = module_0.dcfc_act_365_a(var_5, var_10, var_13)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_24 = bool(var_9 == var_21)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 12
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_2.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.DCCRegistryMachinery()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_5 = {}
    var_6 = None
    var_7 = module_0.dcfc_30_360_german(var_3, var_3, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_8 = module_0.dcfc_nl_365(var_3, var_3, var_3, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    module_2.date(*var_0, **var_5)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 1969
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2
    var_7 = None
    var_8 = module_0.dcfc_30_360_us(var_5, var_5, var_5, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_9 = None
    var_10 = module_0.dcfc_30_360_german(var_5, var_5, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_11 = [var_0, var_6, var_2]
    var_12 = {}
    var_13 = module_2.date(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = ''
    var_15 = [var_14]
    var_16 = {}
    module_3.Decimal(*var_15, **var_16)

def test_case_18():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_2.date(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_2.date(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'datetime.date'
    var_15 = module_0.dcfc_30_e_360(var_5, var_11, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_16 = '390'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_3.Decimal(*var_17, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_20 = '360'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_3.Decimal(*var_21, **var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'decimal.Decimal'
    var_24 = bool(var_15 == var_5)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 1978
    var_1 = 28
    var_2 = 2
    var_3 = None
    var_4 = None
    var_5 = [var_0, var_2, var_1]
    var_6 = {}
    var_7 = module_2.date(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    module_0.dcfc_30_e_360(var_7, var_3, var_4, var_3)

def test_case_20():
    var_0 = 8
    var_1 = 31
    var_2 = [var_1, var_0, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = [var_1, var_0, var_1]
    var_6 = module_2.date(*var_5, **var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.date'
    var_7 = module_0.dcfc_30_360_us(var_4, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_8 = bool(var_7 == var_7)
    assert var_8 is True

def test_case_21():
    var_0 = 2023
    var_1 = 8
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_2.date(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.date'
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = module_0.dcfc_30_360_us(var_4, var_7, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_12 = '0'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_3.Decimal(*var_13, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_16 = bool(var_11 == var_15)
    assert var_16 is True

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 1969
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = None
    var_7 = module_0.dcfc_30_360_german(var_5, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_8 = module_0.dcfc_nl_365(var_5, var_5, var_5, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_9 = [var_1, var_0, var_0, var_2]
    module_2.date(*var_9, **var_4)

def test_case_23():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 5
    var_7 = 31
    var_8 = [var_2, var_6, var_7]
    var_9 = module_2.date(*var_8, **var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.date'
    var_10 = [var_7, var_6, var_7]
    var_11 = {}
    var_12 = module_2.date(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.date'
    var_13 = module_0.dcfc_30_360_us(var_5, var_9, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_14 = 14
    var_15 = round(var_13, var_14)
    var_16 = '1.33333333333333'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_3.Decimal(*var_17, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_20 = bool(var_15 == var_19)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = module_0.dcfc_act_365_l(var_5, var_10, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_12 = 14
    var_13 = round(var_11, var_12)
    var_14 = {}
    module_3.Decimal(*var_8, **var_14)

def test_case_25():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = '0'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_3.Decimal(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_9 = module_0.dcfc_act_365_l(var_4, var_4, var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

def test_case_26():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 3
    var_7 = 31
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = [var_0, var_6, var_7]
    var_12 = {}
    var_13 = module_2.date(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = module_0.dcfc_30_360_german(var_5, var_10, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_15 = '360'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_3.Decimal(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'

def test_case_27():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = module_2.date(*var_9, **var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = [var_6, var_7, var_8]
    var_12 = {}
    var_13 = module_2.date(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = module_0.dcfc_30_360_german(var_5, var_10, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '1.08333333333333'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_3.Decimal(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

def test_case_28():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_2.date(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = module_0.dcfc_30_360_german(var_5, var_10, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '0.16666666666667'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_3.Decimal(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

def test_case_29():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2
    var_7 = 28
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_2.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = [var_0, var_6, var_7]
    var_12 = module_2.date(*var_11, **var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.date'
    var_13 = module_0.dcfc_30_360_german(var_5, var_10, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_14 = '28'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_3.Decimal(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_18 = '360'
    var_19 = [var_18]
    var_20 = module_3.Decimal(*var_19, **var_16)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'decimal.Decimal'
    var_21 = var_17 / var_20
    var_22 = bool(var_13 == var_21)
    assert var_22 is True

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 28
    var_1 = 2008
    var_2 = 2
    var_3 = [var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = [var_1, var_2, var_0]
    var_7 = {}
    var_8 = module_2.date(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.date'
    var_9 = module_0.dcfc_30_360_german(var_5, var_5, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_10 = 14
    var_11 = round(var_9, var_10)
    module_0.dcfc_act_365_l(var_5, var_7, var_11)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_2.date(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
    var_12 = [var_6, var_7, var_8]
    var_13 = None
    var_14 = module_0.dcfc_nl_365(var_11, var_11, var_13, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_15 = {}
    var_16 = module_2.date(*var_12, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'datetime.date'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'
    var_17 = module_0.dcfc_30_360_us(var_5, var_11, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'decimal.Decimal'
    var_18 = 14
    var_19 = round(var_17, var_18)
    var_20 = '0.16944444444444'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_3.Decimal(*var_21, **var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'decimal.Decimal'
    var_24 = bool(var_19 == var_23)
    assert var_24 is True
    module_0.dcfc_act_365_f(var_11, var_19, var_11, var_14)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = None
    module_0.dcfc_30_e_plus_360(var_5, var_6, var_6, var_6)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 8
    var_1 = 31
    var_2 = [var_1, var_0, var_1]
    var_3 = {}
    var_4 = module_2.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = [var_0, var_0, var_1]
    var_6 = {}
    var_7 = module_2.date(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.date'
    var_8 = {}
    var_9 = module_2.date(*var_2, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.date'
    var_10 = module_0.dcfc_30_360_us(var_4, var_7, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_11 = None
    module_0.dcfc_30_360_isda(var_4, var_11, var_0, var_11)

def test_case_34():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2
    var_7 = [var_0, var_6, var_1]
    var_8 = {}
    var_9 = module_2.date(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.date'
    var_10 = [var_0, var_6, var_1]
    var_11 = {}
    var_12 = module_2.date(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.date'
    var_13 = module_0.dcfc_30_360_us(var_5, var_9, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_35():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_2.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_2.date(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_2.date(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'datetime.date'
    var_15 = module_0.dcfc_30_e_360(var_5, var_11, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_16 = '360'
    var_17 = [var_16]
    var_18 = module_3.Decimal(*var_17, **var_10)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.DefaultContext).__module__}.{type(module_3.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_3.HAVE_CONTEXTVAR is True
    assert module_3.HAVE_THREADS is True
    assert f'{type(module_3.BasicContext).__module__}.{type(module_3.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_3.ExtendedContext).__module__}.{type(module_3.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_3.MAX_PREC == 999999999999999999
    assert module_3.MAX_EMAX == 999999999999999999
    assert module_3.MIN_EMIN == -999999999999999999
    assert module_3.MIN_ETINY == -1999999999999999997
    assert module_3.ROUND_UP == 'ROUND_UP'
    assert module_3.ROUND_DOWN == 'ROUND_DOWN'
    assert module_3.ROUND_CEILING == 'ROUND_CEILING'
    assert module_3.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_3.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_3.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_3.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_3.ROUND_05UP == 'ROUND_05UP'