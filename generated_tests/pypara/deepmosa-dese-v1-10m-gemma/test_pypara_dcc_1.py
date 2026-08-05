# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.dcc as module_0
import datetime as module_1
import decimal as module_2
import re as module_3
import enum as module_4
import stringprep as module_5

def test_case_0():
    pass

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
    module_0.dcfc_act_365_l(var_0, var_0, var_0, var_0)

def test_case_3():
    var_0 = 1
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_act_360(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.dcfc_30_360_us(var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_4 = module_0.dcfc_act_act(var_1, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 11
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = None
    var_3 = module_0.dcfc_act_365_f(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    var_5 = module_0.dcfc_30_360_us(var_1, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_6 = module_0.dcfc_act_365_l(var_1, var_1, var_1, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.dcfc_30_360_german(var_1, var_1, var_4, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = None
    var_9 = module_0.dcfc_30_e_360(var_1, var_1, var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    module_0.dcfc_act_365_a(var_8, var_1, var_8, var_9)

def test_case_5():
    var_0 = 0
    var_1 = 1
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_1)

def test_case_6():
    var_0 = 11
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_act_365_a(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 1
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = None
    var_3 = module_0.dcfc_30_e_plus_360(var_1, var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_3.purge()
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    assert module_3.ASCII == module_3.RegexFlag.ASCII
    assert module_3.A == module_3.RegexFlag.ASCII
    assert module_3.IGNORECASE == module_3.RegexFlag.IGNORECASE
    assert module_3.I == module_3.RegexFlag.IGNORECASE
    assert module_3.LOCALE == module_3.RegexFlag.LOCALE
    assert module_3.L == module_3.RegexFlag.LOCALE
    assert module_3.UNICODE == module_3.RegexFlag.UNICODE
    assert module_3.U == module_3.RegexFlag.UNICODE
    assert module_3.MULTILINE == module_3.RegexFlag.MULTILINE
    assert module_3.M == module_3.RegexFlag.MULTILINE
    assert module_3.DOTALL == module_3.RegexFlag.DOTALL
    assert module_3.S == module_3.RegexFlag.DOTALL
    assert module_3.VERBOSE == module_3.RegexFlag.VERBOSE
    assert module_3.X == module_3.RegexFlag.VERBOSE
    assert module_3.TEMPLATE == module_3.RegexFlag.TEMPLATE
    assert module_3.T == module_3.RegexFlag.TEMPLATE
    assert module_3.DEBUG == module_3.RegexFlag.DEBUG
    var_4.encode(var_4, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 1
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_act_365_l(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = None
    var_4 = True
    module_0._construct_date(var_4, var_4, var_3)

def test_case_9():
    var_0 = 2
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_30_360_german(var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = False
    with pytest.raises(ValueError):
        module_0._construct_date(var_3, var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 1
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = None
    module_0.dcfc_30_360_german(var_1, var_2, var_1)

def test_case_11():
    var_0 = 5
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = None
    var_3 = module_0.dcfc_30_360_us(var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_0.dcfc_30_360_german(var_1, var_1, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_5 = module_0.DCCRegistryMachinery()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'

def test_case_12():
    var_0 = 11
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = None
    var_3 = module_0.dcfc_act_365_a(var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_0.dcfc_30_360_us(var_1, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_5 = module_0.dcfc_act_365_l(var_1, var_1, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    var_6 = module_0.dcfc_30_360_german(var_1, var_1, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.dcfc_30_e_360(var_1, var_1, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = True
    var_9 = -334
    with pytest.raises(ValueError):
        module_0._construct_date(var_8, var_6, var_9)

def test_case_13():
    var_0 = 42
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_0, var_0)

def test_case_14():
    var_0 = 1
    var_1 = True
    var_2 = -65
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.DCCRegistryMachinery()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.ONE).__module__}.{type(module_0.ONE).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.ZERO).__module__}.{type(module_0.ZERO).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_0.Currencies).__module__}.{type(module_0.Currencies).__qualname__}' == 'pypara.currencies.CurrencyRegistry'
    assert len(module_0.Currencies) == 188
    assert f'{type(module_0.DCCRegistry).__module__}.{type(module_0.DCCRegistry).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_1 = 1813
    var_2 = True
    module_0._construct_date(var_2, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 12
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_30_360_isda(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    module_4.unique(var_1)

def test_case_17():
    var_0 = 11
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_act_365_a(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.dcfc_30_360_us(var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_4 = module_0.DCCRegistryMachinery()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'

def test_case_18():
    var_0 = 12
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_30_e_360(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_19():
    var_0 = 12
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_nl_365(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_20():
    var_0 = 12
    var_1 = module_0._construct_date(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.date'
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
    var_2 = module_0.dcfc_30_360_german(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_3 = module_0.dcfc_30_e_360(var_1, var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_4 = module_0.dcfc_act_365_l(var_1, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    var_5 = module_0.DCCRegistryMachinery()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 12
    var_1 = 40
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
    var_3 = module_0.dcfc_30_360_german(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    var_5 = module_0.dcfc_30_e_360(var_2, var_2, var_2, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_6 = module_0.dcfc_30_360_us(var_2, var_2, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.dcfc_30_360_isda(var_2, var_2, var_2, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.dcfc_act_365_a(var_2, var_2, var_4, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.dcfc_nl_365(var_2, var_2, var_2, var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    module_5.in_table_c21(var_9)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 12
    var_1 = 40
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
    var_3 = module_0.dcfc_30_360_german(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    var_5 = module_0.dcfc_30_e_360(var_2, var_2, var_2, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_6 = module_0.dcfc_30_360_us(var_2, var_2, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.dcfc_30_360_isda(var_2, var_2, var_2, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.dcfc_act_365_a(var_2, var_2, var_4, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.DCCRegistryMachinery()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_10 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_4, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'
    var_9.register(var_4)

def test_case_23():
    var_0 = 4
    var_1 = 31
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
    var_3 = module_0.dcfc_30_360_german(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    var_5 = module_0.dcfc_30_e_360(var_2, var_2, var_2, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    var_6 = module_0.dcfc_30_360_isda(var_2, var_2, var_2, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.dcfc_act_365_a(var_2, var_2, var_4, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.dcfc_nl_365(var_2, var_2, var_2, var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.DCCRegistryMachinery()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_10 = module_0.DCCRegistryMachinery()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_11 = None
    var_12 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_11, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'decimal.Decimal'
    var_13 = ')H4}QvW+$n[}t'
    var_14 = var_9.find(var_13)
    var_15 = True
    var_16 = module_0.dcfc_act_act(var_2, var_2, var_2, var_7)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'decimal.Decimal'
    var_17 = 875
    var_18 = module_0._construct_date(var_15, var_15, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'datetime.date'

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 2
    var_1 = 31
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
    var_3 = module_0.dcfc_30_360_german(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = None
    var_5 = module_0.dcfc_act_365_l(var_2, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert f'{type(module_2.DefaultContext).__module__}.{type(module_2.DefaultContext).__qualname__}' == 'decimal.Context'
    assert module_2.HAVE_CONTEXTVAR is True
    assert module_2.HAVE_THREADS is True
    assert f'{type(module_2.BasicContext).__module__}.{type(module_2.BasicContext).__qualname__}' == 'decimal.Context'
    assert f'{type(module_2.ExtendedContext).__module__}.{type(module_2.ExtendedContext).__qualname__}' == 'decimal.Context'
    assert module_2.MAX_PREC == 999999999999999999
    assert module_2.MAX_EMAX == 999999999999999999
    assert module_2.MIN_EMIN == -999999999999999999
    assert module_2.MIN_ETINY == -1999999999999999997
    assert module_2.ROUND_UP == 'ROUND_UP'
    assert module_2.ROUND_DOWN == 'ROUND_DOWN'
    assert module_2.ROUND_CEILING == 'ROUND_CEILING'
    assert module_2.ROUND_FLOOR == 'ROUND_FLOOR'
    assert module_2.ROUND_HALF_UP == 'ROUND_HALF_UP'
    assert module_2.ROUND_HALF_DOWN == 'ROUND_HALF_DOWN'
    assert module_2.ROUND_HALF_EVEN == 'ROUND_HALF_EVEN'
    assert module_2.ROUND_05UP == 'ROUND_05UP'
    module_0.dcfc_act_365_l(var_4, var_4, var_2, var_4)