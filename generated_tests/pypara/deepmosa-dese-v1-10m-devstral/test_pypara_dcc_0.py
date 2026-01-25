# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pypara.dcc as module_0
import datetime as module_1
import decimal as module_2
import encodings.idna as module_3
import locale as module_4
import stringprep as module_5
import codecs as module_6

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
    var_0 = 2014
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
    var_3 = None
    var_4 = module_0.dcfc_act_365_l(var_2, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_act_365_a(var_2, var_2, var_3)
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
    var_6 = module_0.dcfc_act_365_f(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    module_0.dcfc_act_365_a(var_6, var_6, var_3)

def test_case_5():
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
    var_2 = module_0.dcfc_act_365_a(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_6():
    var_0 = -2023
    var_1 = 5
    var_2 = 15
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_7():
    var_0 = 30
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_0, var_0)

def test_case_8():
    var_0 = 2023
    var_1 = 2
    var_2 = 30
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

def test_case_9():
    var_0 = 2023
    var_1 = False
    var_2 = None
    with pytest.raises(ValueError):
        module_0._construct_date(var_0, var_1, var_2)

def test_case_10():
    var_0 = 2
    var_1 = True
    var_2 = -3934
    with pytest.raises(ValueError):
        module_0._construct_date(var_1, var_0, var_2)

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
    var_1 = 2020
    var_2 = 2
    var_3 = module_0._construct_date(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = None
    var_5 = module_0.dcfc_act_365_a(var_3, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

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
    var_1 = 2020
    var_2 = 2
    var_3 = 30
    var_4 = module_0._construct_date(var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.year).__module__}.{type(module_1.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.month).__module__}.{type(module_1.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.day).__module__}.{type(module_1.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_1.date.min).__module__}.{type(module_1.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.max).__module__}.{type(module_1.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_1.date.resolution).__module__}.{type(module_1.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = None
    var_6 = module_0.dcfc_act_365_a(var_4, var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 2023
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
    var_3 = None
    var_4 = module_0.dcfc_act_365_l(var_2, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_2.formatweekheader(var_3)

def test_case_14():
    var_0 = 2
    var_1 = 52
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    var_3 = None
    var_4 = module_0.dcfc_30_360_german(var_2, var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_3)
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

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_3.getregentry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'codecs.CodecInfo'
    assert len(var_0) == 4
    assert f'{type(module_3.unicodedata).__module__}.{type(module_3.unicodedata).__qualname__}' == 'unicodedata.UCD'
    assert f'{type(module_3.dots).__module__}.{type(module_3.dots).__qualname__}' == 're.Pattern'
    assert module_3.ace_prefix == b'xn--'
    assert module_3.sace_prefix == 'xn--'
    var_1 = 670
    var_2 = 2
    var_3 = 30
    var_4 = module_0._construct_date(var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
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
    module_0.dcfc_30_e_plus_360(var_4, var_0, var_4)

def test_case_16():
    var_0 = 2
    var_1 = 52
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    var_3 = None
    var_4 = module_0.dcfc_act_365_l(var_2, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = None
    var_6 = module_0.dcfc_30_360_german(var_2, var_2, var_3, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
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
    var_7 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'

def test_case_17():
    var_0 = 2
    var_1 = 30
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    var_3 = module_0.dcfc_30_360_us(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

def test_case_18():
    var_0 = 2
    var_1 = 30
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
    var_3 = module_0.dcfc_30_360_german(var_2, var_2, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_19():
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
    var_2 = module_0.dcfc_30_360_us(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    module_4.atoi(var_0)

def test_case_20():
    var_0 = 2023
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
    var_3 = None
    var_4 = None
    var_5 = module_0.dcfc_30_360_german(var_2, var_2, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_6 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
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
    var_7 = module_0.dcfc_nl_365(var_2, var_2, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'

def test_case_21():
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
    var_2 = None
    var_3 = module_0.dcfc_30_360_german(var_1, var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_0.dcfc_act_act(var_1, var_1, var_2, var_2)
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
    var_5 = module_0.dcfc_30_e_plus_360(var_1, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_22():
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
    var_2 = None
    var_3 = module_0.dcfc_30_360_german(var_1, var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_0.dcfc_act_act(var_1, var_1, var_2, var_2)
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
    module_0.dcfc_30_360_isda(var_1, var_3, var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = 2023
    var_1 = 2
    var_2 = 30
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
    var_5 = None
    var_6 = module_0.dcfc_30_360_german(var_3, var_3, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_7 = module_0.dcfc_30_e_plus_360(var_3, var_3, var_3, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
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
    module_5.in_table_c11_c12(var_4)

def test_case_24():
    var_0 = 5
    var_1 = 52
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    var_3 = None
    var_4 = module_0.dcfc_30_360_german(var_2, var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_3)
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

def test_case_25():
    var_0 = 10
    var_1 = 52
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_0.DCCRegistryMachinery()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
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
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = 2
    var_1 = 52
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    var_3 = None
    var_4 = module_0.dcfc_act_365_l(var_2, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_3)
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
    var_6 = module_0.dcfc_30_360_isda(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'decimal.Decimal'
    var_7 = module_0.dcfc_nl_365(var_2, var_2, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
    var_8 = module_0.DCCRegistryMachinery()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_9 = ''
    var_10 = var_8.find(var_9)
    module_0.dcfc_30_e_plus_360(var_3, var_3, var_2)

def test_case_27():
    var_0 = 2
    var_1 = 52
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    var_3 = module_0.dcfc_30_e_360(var_2, var_2, var_2)
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
    with pytest.raises(ValueError):
        module_4.currency(var_4)

def test_case_28():
    var_0 = 10
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
    var_3 = module_0.dcfc_30_360_german(var_1, var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_0.dcfc_30_e_plus_360(var_1, var_1, var_2)
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
    var_5 = module_0.dcfc_30_360_isda(var_1, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'decimal.Decimal'
    var_6 = module_0.DCCRegistryMachinery()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_7 = module_0.DCCRegistryMachinery()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    var_8 = '5Pn'
    var_9 = var_7.find(var_8)
    var_10 = module_0.dcfc_30_360_isda(var_1, var_1, var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'decimal.Decimal'

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 10
    var_1 = 51
    var_2 = module_0._construct_date(var_1, var_0, var_1)
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
    var_3 = None
    var_4 = module_0.dcfc_30_e_360(var_2, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'decimal.Decimal'
    assert module_1.MINYEAR == 1
    assert module_1.MAXYEAR == 9999
    assert f'{type(module_1.datetime_CAPI).__module__}.{type(module_1.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.Decimal.real).__module__}.{type(module_2.Decimal.real).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.Decimal.imag).__module__}.{type(module_2.Decimal.imag).__qualname__}' == 'builtins.getset_descriptor'
    var_5 = None
    var_6 = None
    var_7 = module_0.dcfc_30_360_german(var_2, var_2, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'decimal.Decimal'
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
    var_8 = module_0.dcfc_30_e_plus_360(var_2, var_2, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'decimal.Decimal'
    var_9 = module_0.dcfc_30_360_isda(var_2, var_2, var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'decimal.Decimal'
    var_10 = module_0.DCCRegistryMachinery()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pypara.dcc.DCCRegistryMachinery'
    assert f'{type(module_0.DCCRegistryMachinery.registry).__module__}.{type(module_0.DCCRegistryMachinery.registry).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.DCCRegistryMachinery.table).__module__}.{type(module_0.DCCRegistryMachinery.table).__qualname__}' == 'builtins.property'
    var_11 = ''
    var_12 = var_10.find(var_11)
    module_6.getwriter(var_4)