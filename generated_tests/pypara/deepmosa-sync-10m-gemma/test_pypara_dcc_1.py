# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.dcc as module_0
import datetime as module_1
import enum as module_2
import decimal as module_3
import _locale as module_4

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.DCCRegistryMachinery()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1.find(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.dcfc_act_365_l(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 2023
    var_1 = 4
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = None
    module_0.dcfc_act_360(var_3, var_1, var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = module_2._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = None
    var_6 = module_0.dcfc_nl_365(var_4, var_4, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    module_0.dcfc_act_365_f(var_5, var_4, var_4)

def test_case_5():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.date'
    var_9 = module_0._get_date_range(var_4, var_8)
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_10 = list(var_9)
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_1.date(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = [var_13]
    var_15 = bool(var_10 == var_14)
    assert var_15 is True

def test_case_6():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_1.date(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.date'
    var_8 = module_0._get_date_range(var_4, var_7)
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

def test_case_7():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.dcfc_nl_365(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_8():
    var_0 = 2
    var_1 = 93
    var_2 = module_0._construct_date(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_0.dcfc_30_360_isda(var_2, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_0.dcfc_30_360_us(var_2, var_2, var_2)
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

def test_case_9():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0._last_payment_date(var_3, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_5 = module_0.dcfc_30_e_plus_360(var_4, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_6 = module_0.dcfc_30_360_us(var_4, var_4, var_3, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
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
    var_1 = var_0.registry
    var_2 = bool(var_0.registry == [])
    assert var_2 is True

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
    var_1 = var_0.table
    var_2 = bool(var_0.table == {})
    assert var_2 is True

def test_case_12():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.date(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
    var_12 = module_0.dcfc_act_act(var_5, var_11, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_13 = 14
    var_14 = round(var_12, var_13)
    var_15 = '1.32625945055768'
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
    var_19 = bool(var_14 == var_18)
    assert var_19 is True

def test_case_13():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = [var_0, var_0, var_0]
    var_5 = {}
    var_6 = module_1.date(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.date'
    var_7 = module_0.dcfc_act_act(var_3, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_8 = '0'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_3.Decimal(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
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
    var_12 = bool(var_7 == var_11)
    assert var_12 is True

def test_case_14():
    var_0 = 2
    var_1 = 29
    var_2 = module_0._construct_date(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_15():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0._last_payment_date(var_3, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_5 = module_0.dcfc_30_360_german(var_4, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_16():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2015
    var_6 = 12
    var_7 = 31
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = module_0._last_payment_date(var_4, var_10, var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_12 = [var_5, var_1, var_1]
    var_13 = {}
    var_14 = module_1.date(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'datetime.date'
    var_15 = bool(var_11 == var_14)
    assert var_15 is True
    var_16 = [var_5, var_1, var_1]
    var_17 = {}
    var_18 = module_1.date(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'datetime.date'
    var_19 = [var_5, var_6, var_7]
    var_20 = {}
    var_21 = module_1.date(*var_19, **var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'datetime.date'
    var_22 = module_0._last_payment_date(var_18, var_21, var_1)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'datetime.date'
    var_23 = [var_5, var_1, var_1]
    var_24 = {}
    var_25 = module_1.date(*var_23, **var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'datetime.date'
    var_26 = bool(var_22 == var_25)
    assert var_26 is True
    var_27 = [var_0, var_1, var_1]
    var_28 = {}
    var_29 = module_1.date(*var_27, **var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'datetime.date'
    var_30 = [var_5, var_6, var_7]
    var_31 = {}
    var_32 = module_1.date(*var_30, **var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'datetime.date'
    var_33 = 2
    var_34 = module_0._last_payment_date(var_29, var_32, var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'datetime.date'
    var_35 = 7
    var_36 = [var_5, var_35, var_1]
    var_37 = {}
    var_38 = module_1.date(*var_36, **var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'datetime.date'
    var_39 = bool(var_34 == var_38)
    assert var_39 is True
    var_40 = [var_0, var_1, var_1]
    var_41 = {}
    var_42 = module_1.date(*var_40, **var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'datetime.date'
    var_43 = 8
    var_44 = [var_5, var_43, var_7]
    var_45 = {}
    var_46 = module_1.date(*var_44, **var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'datetime.date'
    var_47 = module_0._last_payment_date(var_42, var_46, var_33)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'datetime.date'
    var_48 = [var_5, var_35, var_1]
    var_49 = {}
    var_50 = module_1.date(*var_48, **var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'datetime.date'
    var_51 = bool(var_47 == var_50)
    assert var_51 is True
    var_52 = [var_0, var_1, var_1]
    var_53 = {}
    var_54 = module_1.date(*var_52, **var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'datetime.date'
    var_55 = 4
    var_56 = 30
    var_57 = [var_5, var_55, var_56]
    var_58 = {}
    var_59 = module_1.date(*var_57, **var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'datetime.date'
    var_60 = module_0._last_payment_date(var_54, var_59, var_33)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'datetime.date'
    var_61 = [var_5, var_1, var_1]
    var_62 = {}
    var_63 = module_1.date(*var_61, **var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'datetime.date'
    var_64 = bool(var_60 == var_63)
    assert var_64 is True
    var_65 = 6
    var_66 = [var_0, var_65, var_1]
    var_67 = {}
    var_68 = module_1.date(*var_66, **var_67)
    assert f'{type(var_68).__module__}.{type(var_68).__qualname__}' == 'datetime.date'
    var_69 = [var_5, var_55, var_56]
    var_70 = {}
    var_71 = module_1.date(*var_69, **var_70)
    assert f'{type(var_71).__module__}.{type(var_71).__qualname__}' == 'datetime.date'
    var_72 = module_0._last_payment_date(var_68, var_71, var_1)
    assert f'{type(var_72).__module__}.{type(var_72).__qualname__}' == 'datetime.date'
    var_73 = [var_0, var_65, var_1]
    var_74 = {}
    var_75 = module_1.date(*var_73, **var_74)
    assert f'{type(var_75).__module__}.{type(var_75).__qualname__}' == 'datetime.date'
    var_76 = bool(var_72 == var_75)
    assert var_76 is True
    var_77 = 2008
    var_78 = [var_77, var_35, var_35]
    var_79 = {}
    var_80 = module_1.date(*var_78, **var_79)
    assert f'{type(var_80).__module__}.{type(var_80).__qualname__}' == 'datetime.date'
    var_81 = 10
    var_82 = [var_5, var_81, var_65]
    var_83 = {}
    var_84 = module_1.date(*var_82, **var_83)
    assert f'{type(var_84).__module__}.{type(var_84).__qualname__}' == 'datetime.date'
    var_85 = module_0._last_payment_date(var_80, var_84, var_55)
    assert f'{type(var_85).__module__}.{type(var_85).__qualname__}' == 'datetime.date'
    var_86 = [var_5, var_35, var_35]
    var_87 = {}
    var_88 = module_1.date(*var_86, **var_87)
    assert f'{type(var_88).__module__}.{type(var_88).__qualname__}' == 'datetime.date'
    var_89 = bool(var_85 == var_88)
    assert var_89 is True
    var_90 = 9
    var_91 = [var_0, var_6, var_90]
    var_92 = {}
    var_93 = module_1.date(*var_91, **var_92)
    assert f'{type(var_93).__module__}.{type(var_93).__qualname__}' == 'datetime.date'
    var_94 = [var_5, var_6, var_55]
    var_95 = {}
    var_96 = module_1.date(*var_94, **var_95)
    assert f'{type(var_96).__module__}.{type(var_96).__qualname__}' == 'datetime.date'
    var_97 = module_0._last_payment_date(var_93, var_96, var_1)
    assert f'{type(var_97).__module__}.{type(var_97).__qualname__}' == 'datetime.date'
    var_98 = [var_0, var_6, var_90]
    var_99 = {}
    var_100 = module_1.date(*var_98, **var_99)
    assert f'{type(var_100).__module__}.{type(var_100).__qualname__}' == 'datetime.date'
    var_101 = bool(var_97 == var_100)
    assert var_101 is True
    var_102 = 2012
    var_103 = 15
    var_104 = [var_102, var_6, var_103]
    var_105 = {}
    var_106 = module_1.date(*var_104, **var_105)
    assert f'{type(var_106).__module__}.{type(var_106).__qualname__}' == 'datetime.date'
    var_107 = 2016
    var_108 = [var_107, var_1, var_65]
    var_109 = {}
    var_110 = module_1.date(*var_108, **var_109)
    assert f'{type(var_110).__module__}.{type(var_110).__qualname__}' == 'datetime.date'
    var_111 = module_0._last_payment_date(var_106, var_110, var_33)
    assert f'{type(var_111).__module__}.{type(var_111).__qualname__}' == 'datetime.date'
    var_112 = [var_5, var_6, var_103]
    var_113 = {}
    var_114 = module_1.date(*var_112, **var_113)
    assert f'{type(var_114).__module__}.{type(var_114).__qualname__}' == 'datetime.date'
    var_115 = bool(var_111 == var_114)
    assert var_115 is True
    var_116 = [var_102, var_6, var_103]
    var_117 = {}
    var_118 = module_1.date(*var_116, **var_117)
    assert f'{type(var_118).__module__}.{type(var_118).__qualname__}' == 'datetime.date'
    var_119 = [var_5, var_6, var_7]
    var_120 = {}
    var_121 = module_1.date(*var_119, **var_120)
    assert f'{type(var_121).__module__}.{type(var_121).__qualname__}' == 'datetime.date'
    var_122 = module_0._last_payment_date(var_118, var_121, var_33)
    assert f'{type(var_122).__module__}.{type(var_122).__qualname__}' == 'datetime.date'
    var_123 = [var_5, var_6, var_103]
    var_124 = {}
    var_125 = module_1.date(*var_123, **var_124)
    assert f'{type(var_125).__module__}.{type(var_125).__qualname__}' == 'datetime.date'
    var_126 = bool(var_122 == var_125)
    assert var_126 is True

def test_case_17():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = [var_0, var_0, var_0]
    var_5 = {}
    var_6 = module_1.date(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.date'
    var_7 = 1
    var_8 = 1
    var_9 = module_0._last_payment_date(var_3, var_6, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_10 = bool(var_9 == var_3)
    assert var_10 is True

def test_case_18():
    var_0 = 2023
    var_1 = -1
    var_2 = 15
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_19():
    var_0 = 10
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0._last_payment_date(var_3, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_5 = module_0.dcfc_30_360_german(var_4, var_3, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_20():
    var_0 = 2023
    var_1 = 5
    var_2 = -1
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_21():
    var_0 = 2023
    var_1 = 13
    var_2 = 1
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_22():
    var_0 = 0
    var_1 = 5
    var_2 = 15
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_23():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.dcfc_30_360_german(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_24():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = 'Act/360'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_3.Decimal(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
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
    var_8 = lambda s, a, e, f: var_7
    var_9 = []
    var_10 = 'name'
    var_11 = 'altnames'
    var_12 = 'currencies'
    var_13 = 'calculate_fraction_method'
    var_14 = {var_10: var_1, var_11: var_2, var_12: var_3, var_13: var_8}
    var_15 = module_0.DCC(*var_9, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_15) == 4
    assert f'{type(module_0.DCC.name).__module__}.{type(module_0.DCC.name).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.altnames).__module__}.{type(module_0.DCC.altnames).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.currencies).__module__}.{type(module_0.DCC.currencies).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.calculate_fraction_method).__module__}.{type(module_0.DCC.calculate_fraction_method).__qualname__}' == '_collections._tuplegetter'
    var_16 = set()
    var_17 = set()
    var_18 = '0.1'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_3.Decimal(*var_19, **var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'decimal.Decimal'
    var_22 = lambda s, a, e, f: var_21
    var_23 = []
    var_24 = 'name'
    var_25 = 'altnames'
    var_26 = 'currencies'
    var_27 = 'calculate_fraction_method'
    var_28 = {var_24: var_1, var_25: var_16, var_26: var_17, var_27: var_22}
    var_29 = module_0.DCC(*var_23, **var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_29) == 4
    var_30 = var_0.register(var_15)
    with pytest.raises(TypeError):
        var_0.register(var_29)

def test_case_25():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = 'Act/360'
    var_2 = 'ACT/360'
    var_3 = 'Actual/360'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_3.Decimal(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
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
    var_10 = lambda s, a, e, f: var_9
    var_11 = []
    var_12 = 'name'
    var_13 = 'altnames'
    var_14 = 'currencies'
    var_15 = 'calculate_fraction_method'
    var_16 = {var_12: var_1, var_13: var_4, var_14: var_5, var_15: var_10}
    var_17 = module_0.DCC(*var_11, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_17) == 4
    assert f'{type(module_0.DCC.name).__module__}.{type(module_0.DCC.name).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.altnames).__module__}.{type(module_0.DCC.altnames).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.currencies).__module__}.{type(module_0.DCC.currencies).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.calculate_fraction_method).__module__}.{type(module_0.DCC.calculate_fraction_method).__qualname__}' == '_collections._tuplegetter'
    var_18 = var_0.register(var_17)
    var_19 = var_0.find(var_1)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_19) == 4
    var_20 = bool(var_19 == var_17)
    assert var_20 is True
    var_21 = var_0.find(var_2)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_21) == 4
    var_22 = bool(var_21 == var_17)
    assert var_22 is True
    var_23 = var_0.find(var_3)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_23) == 4
    var_24 = bool(var_23 == var_17)
    assert var_24 is True
    var_25 = var_0.registry
    var_26 = len(var_25)
    assert var_26 == 1

def test_case_26():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = 'Act/360'
    var_2 = 'ACT/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
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
    var_9 = lambda s, a, e, f: var_8
    var_10 = []
    var_11 = 'name'
    var_12 = 'altnames'
    var_13 = 'currencies'
    var_14 = 'calculate_fraction_method'
    var_15 = {var_11: var_1, var_12: var_3, var_13: var_4, var_14: var_9}
    var_16 = module_0.DCC(*var_10, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_16) == 4
    assert f'{type(module_0.DCC.name).__module__}.{type(module_0.DCC.name).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.altnames).__module__}.{type(module_0.DCC.altnames).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.currencies).__module__}.{type(module_0.DCC.currencies).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.DCC.calculate_fraction_method).__module__}.{type(module_0.DCC.calculate_fraction_method).__qualname__}' == '_collections._tuplegetter'
    var_17 = 'Other'
    var_18 = {var_2}
    var_19 = set()
    var_20 = '0.1'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_3.Decimal(*var_21, **var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'decimal.Decimal'
    var_24 = lambda s, a, e, f: var_23
    var_25 = []
    var_26 = 'name'
    var_27 = 'altnames'
    var_28 = 'currencies'
    var_29 = 'calculate_fraction_method'
    var_30 = {var_26: var_17, var_27: var_18, var_28: var_19, var_29: var_24}
    var_31 = module_0.DCC(*var_25, **var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'pypara.dcc.DCC'
    assert len(var_31) == 4
    var_32 = var_0.register(var_16)
    with pytest.raises(TypeError):
        var_0.register(var_31)

def test_case_27():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 3
    var_6 = 15
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = None
    var_10 = module_0.dcfc_30_e_360(var_4, var_4, var_4, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_11 = module_1.date(*var_7, **var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.date'
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
    var_12 = 31
    var_13 = module_0._last_payment_date(var_4, var_11, var_1, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.date'
    var_14 = 2
    var_15 = 29
    var_16 = [var_0, var_14, var_15]
    var_17 = {}
    var_18 = module_1.date(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'datetime.date'
    var_19 = bool(var_13 == var_18)

def test_case_28():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = module_1.date(*var_9, **var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = module_0.dcfc_30_360_us(var_5, var_10, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_12 = '1.33333333333333'
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
    var_16 = bool(var_12 == var_15)

def test_case_29():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 9
    var_7 = 10
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = 2020
    var_12 = [var_0, var_1, var_2]
    var_13 = {}
    var_14 = module_1.date(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'datetime.date'
    var_15 = [var_0, var_6, var_7]
    var_16 = {}
    var_17 = module_1.date(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'datetime.date'
    var_18 = [var_11, var_1, var_2]
    var_19 = {}
    var_20 = module_1.date(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'datetime.date'
    var_21 = '1'
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_3.Decimal(*var_22, **var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'decimal.Decimal'
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
    var_25 = module_0.dcfc_act_act_icma(var_14, var_17, var_20, var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'

def test_case_30():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 11
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.date'
    var_9 = 31
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_1.date(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.date'
    var_13 = None
    var_14 = module_0.dcfc_act_act_icma(var_4, var_8, var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_15 = '10'
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
    var_19 = '30'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_3.Decimal(*var_20, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'decimal.Decimal'
    var_23 = var_18 / var_22
    var_24 = '1'
    var_25 = [var_24]
    var_26 = {}
    var_27 = module_3.Decimal(*var_25, **var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'decimal.Decimal'
    var_28 = var_23 / var_27
    var_29 = bool(var_14 == var_28)
    assert var_29 is True

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.date(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.date'
    var_10 = [var_4, var_5, var_6]
    var_11 = {}
    var_12 = module_1.date(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.date'
    var_13 = module_0.dcfc_30_e_360(var_12, var_12, var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_14 = module_0.dcfc_30_360_us(var_3, var_9, var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
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
    var_15 = round(var_14, var_5)
    var_15.readline()

def test_case_32():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = module_1.date(*var_9, **var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = None
    var_12 = module_0.dcfc_30_e_360(var_10, var_5, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_13 = module_0.dcfc_30_360_us(var_5, var_10, var_5)
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
    var_14 = '1.33333333333333'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_3.Decimal(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'decimal.Decimal'
    var_18 = bool(var_14 == var_17)

def test_case_33():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_1.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = 2009
    var_7 = 31
    var_8 = [var_6, var_2, var_7]
    var_9 = module_1.date(*var_8, **var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.date'
    var_10 = module_0.dcfc_30_360_us(var_5, var_9, var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_11 = '1.33333333333333'
    var_12 = [var_11]
    var_13 = module_0.dcfc_act_365_l(var_5, var_5, var_9)
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
    var_14 = {}
    var_15 = module_3.Decimal(*var_12, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'decimal.Decimal'
    var_16 = bool(var_11 == var_15)

def test_case_34():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.DCCRegistryMachinery()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_3 = {}
    var_4 = module_1.date(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = module_0._last_payment_date(var_4, var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    var_6 = None
    var_7 = module_0.dcfc_act_365_l(var_4, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_8 = module_0.dcfc_30_360_german(var_5, var_4, var_4)
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

def test_case_35():
    var_0 = 2032
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = None
    var_6 = module_0.dcfc_nl_365(var_4, var_4, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_7 = list(var_3)

def test_case_36():
    var_0 = 10
    var_1 = 31
    var_2 = [var_0, var_0, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2008
    var_6 = 11
    var_7 = 17
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.date'
    var_11 = module_0.dcfc_30_360_us(var_4, var_10, var_0)
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
    var_14 = module_0.dcfc_nl_365(var_10, var_10, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'decimal.Decimal'
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

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = 2
    var_1 = 29
    var_2 = module_0._construct_date(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    module_0.dcfc_30_360_german(var_2, var_4, var_2)

def test_case_38():
    var_0 = 2100
    var_1 = 2
    var_2 = module_0._construct_date(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    var_5 = module_0.dcfc_30_360_german(var_2, var_2, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
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

def test_case_39():
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
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.dcfc_30_e_plus_360(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = None
    var_6 = module_0.dcfc_30_360_german(var_3, var_3, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
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

def test_case_40():
    var_0 = 10
    var_1 = 31
    var_2 = [var_0, var_0, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = module_0.dcfc_30_360_us(var_4, var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_6 = 14
    var_7 = round(var_5, var_6)
    var_8 = module_0.dcfc_30_360_german(var_4, var_4, var_7)
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

def test_case_41():
    var_0 = 2007
    var_1 = 2
    var_2 = 34
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.dcfc_30_e_plus_360(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_30_360_german(var_3, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
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

def test_case_42():
    var_0 = 8
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_1.date(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0._last_payment_date(var_3, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_5 = module_0.dcfc_30_360_german(var_4, var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_6 = None
    var_7 = module_0.dcfc_nl_365(var_4, var_4, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
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

def test_case_43():
    var_0 = 2068
    var_1 = 11
    var_2 = 93
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.dcfc_30_360_isda(var_3, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_30_e_plus_360(var_3, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
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

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 2068
    var_1 = 2
    var_2 = 93
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.dcfc_30_360_isda(var_3, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_30_360_us(var_3, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
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
    var_6 = None
    var_7 = module_0.dcfc_nl_365(var_3, var_3, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    module_0.dcfc_30_360_isda(var_3, var_6, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = 1
    var_1 = 366
    var_2 = module_0._construct_date(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_0.dcfc_30_360_isda(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    var_5 = module_4.localeconv()
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
    assert module_4.LC_CTYPE == 0
    assert module_4.LC_TIME == 2
    assert module_4.LC_COLLATE == 3
    assert module_4.LC_MONETARY == 4
    assert module_4.LC_MESSAGES == 5
    assert module_4.LC_NUMERIC == 1
    assert module_4.LC_ALL == 6
    assert module_4.CHAR_MAX == 127
    assert module_4.DAY_1 == 131079
    assert module_4.DAY_2 == 131080
    assert module_4.DAY_3 == 131081
    assert module_4.DAY_4 == 131082
    assert module_4.DAY_5 == 131083
    assert module_4.DAY_6 == 131084
    assert module_4.DAY_7 == 131085
    assert module_4.ABDAY_1 == 131072
    assert module_4.ABDAY_2 == 131073
    assert module_4.ABDAY_3 == 131074
    assert module_4.ABDAY_4 == 131075
    assert module_4.ABDAY_5 == 131076
    assert module_4.ABDAY_6 == 131077
    assert module_4.ABDAY_7 == 131078
    assert module_4.MON_1 == 131098
    assert module_4.MON_2 == 131099
    assert module_4.MON_3 == 131100
    assert module_4.MON_4 == 131101
    assert module_4.MON_5 == 131102
    assert module_4.MON_6 == 131103
    assert module_4.MON_7 == 131104
    assert module_4.MON_8 == 131105
    assert module_4.MON_9 == 131106
    assert module_4.MON_10 == 131107
    assert module_4.MON_11 == 131108
    assert module_4.MON_12 == 131109
    assert module_4.ABMON_1 == 131086
    assert module_4.ABMON_2 == 131087
    assert module_4.ABMON_3 == 131088
    assert module_4.ABMON_4 == 131089
    assert module_4.ABMON_5 == 131090
    assert module_4.ABMON_6 == 131091
    assert module_4.ABMON_7 == 131092
    assert module_4.ABMON_8 == 131093
    assert module_4.ABMON_9 == 131094
    assert module_4.ABMON_10 == 131095
    assert module_4.ABMON_11 == 131096
    assert module_4.ABMON_12 == 131097
    assert module_4.RADIXCHAR == 65536
    assert module_4.THOUSEP == 65537
    assert module_4.CRNCYSTR == 262159
    assert module_4.D_T_FMT == 131112
    assert module_4.D_FMT == 131113
    assert module_4.T_FMT == 131114
    assert module_4.AM_STR == 131110
    assert module_4.PM_STR == 131111
    assert module_4.CODESET == 14
    assert module_4.T_FMT_AMPM == 131115
    assert module_4.ERA == 131116
    assert module_4.ERA_D_FMT == 131118
    assert module_4.ERA_D_T_FMT == 131120
    assert module_4.ERA_T_FMT == 131121
    assert module_4.ALT_DIGITS == 131119
    assert module_4.YESEXPR == 327680
    assert module_4.NOEXPR == 327681
    module_0.dcfc_nl_365(var_4, var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = 2032
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_1.date(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = None
    var_6 = module_0.dcfc_nl_365(var_4, var_4, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_7 = list(var_3)
    var_8 = module_0.dcfc_act_365_a(var_4, var_4, var_7)
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
    module_0.dcfc_act_act_icma(var_7, var_0, var_7)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = 2068
    var_1 = 2
    var_2 = 118
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = module_0.dcfc_30_360_isda(var_3, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.Decimal.real).__module__}.{type(module_3.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.Decimal.imag).__module__}.{type(module_3.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_30_360_us(var_3, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
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
    var_6 = module_0.dcfc_act_365_a(var_3, var_3, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_6.__iter__()