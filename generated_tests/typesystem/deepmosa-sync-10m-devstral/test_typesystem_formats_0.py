# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.formats as module_0
import re as module_1
import uuid as module_2
import platform as module_3
import datetime as module_4
import ipaddress as module_5

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

def test_case_1():
    var_0 = module_0.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1 = '2023-02-30'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1 = module_0.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = None
    var_3 = module_0.DateFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.DateFormat'
    var_4 = None
    var_5 = var_1.is_native_type(var_2)
    assert var_5 is False
    var_6 = var_0.serialize(var_4)
    module_1.sub(var_2, var_4, var_6, var_2)

def test_case_4():
    var_0 = module_0.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1 = module_0.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = None
    var_3 = var_0.serialize(var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1 = '2023-02-30'
    var_2 = None
    var_3 = var_0.is_native_type(var_2)
    assert var_3 is False
    var_0.validate(var_1)

def test_case_6():
    var_0 = module_0.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)

def test_case_7():
    var_0 = module_0.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = var_0.is_native_type(var_2)
    assert var_3 is False

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.UUIDFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = var_1.serialize(var_0)
    var_3 = var_1.is_native_type(var_1)
    assert var_3 is False
    module_2.uuid5(var_2, var_2)

def test_case_9():
    var_0 = module_0.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1 = module_0.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = None
    var_3 = module_0.DateFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.DateFormat'
    var_4 = var_0.serialize(var_2)
    var_5 = var_1.serialize(var_4)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_0.serialize(var_0)

def test_case_11():
    var_0 = module_0.IPAddressFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)

def test_case_12():
    var_0 = module_0.BaseFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.BaseFormat.errors == {}
    with pytest.raises(NotImplementedError):
        var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = module_0.TimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.IPAddressFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_1 = module_0.IPAddressFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    var_2 = None
    var_3 = var_1.is_native_type(var_2)
    assert var_3 is False
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1 = '2023-02-30'
    var_0.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_0.URLFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1 = var_0.serialize(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.URLFormat'
    var_2 = module_0.DateFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_3 = module_3.win32_ver()
    var_3.__pos__()

def test_case_17():
    var_0 = module_0.BaseFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.BaseFormat.errors == {}
    var_1 = None
    with pytest.raises(NotImplementedError):
        var_0.validate(var_1)

def test_case_18():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

def test_case_19():
    var_0 = module_0.IPAddressFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_1 = module_2.getnode()
    assert var_1 == 42623999026922
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    with pytest.raises(AssertionError):
        var_0.serialize(var_1)

def test_case_20():
    var_0 = module_0.BaseFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.BaseFormat.errors == {}
    var_1 = module_0.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    with pytest.raises(NotImplementedError):
        var_0.is_native_type(var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_0.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = module_3.python_version()
    assert var_1 == '3.10.19'
    var_0.validate(var_1)

def test_case_22():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_4.time(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.time'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.time.hour).__module__}.{type(module_4.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.minute).__module__}.{type(module_4.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.second).__module__}.{type(module_4.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.microsecond).__module__}.{type(module_4.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.tzinfo).__module__}.{type(module_4.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.fold).__module__}.{type(module_4.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.min).__module__}.{type(module_4.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_4.time.max).__module__}.{type(module_4.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_4.time.resolution).__module__}.{type(module_4.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = []
    var_7 = {}
    var_8 = module_0.TimeFormat(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_9 = var_8.serialize(var_5)
    assert var_9 == '12:30:45'

def test_case_23():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = 'test@example.com'
    var_4 = var_2.serialize(var_3)
    assert var_4 == 'test@example.com'
    var_5 = bool(var_4 == var_3)
    assert var_5 is True

def test_case_24():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    with pytest.raises(AssertionError):
        var_2.serialize(var_2)

def test_case_25():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_2.UUID(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'uuid.UUID'
    assert module_2.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_2.RFC_4122 == 'specified in RFC 4122'
    assert module_2.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_2.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_2.NAMESPACE_DNS).__module__}.{type(module_2.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_URL).__module__}.{type(module_2.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_OID).__module__}.{type(module_2.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.NAMESPACE_X500).__module__}.{type(module_2.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_2.UUID.bytes).__module__}.{type(module_2.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.bytes_le).__module__}.{type(module_2.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.fields).__module__}.{type(module_2.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_low).__module__}.{type(module_2.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_mid).__module__}.{type(module_2.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time_hi_version).__module__}.{type(module_2.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq_hi_variant).__module__}.{type(module_2.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq_low).__module__}.{type(module_2.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.time).__module__}.{type(module_2.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.clock_seq).__module__}.{type(module_2.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.node).__module__}.{type(module_2.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.hex).__module__}.{type(module_2.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.urn).__module__}.{type(module_2.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.variant).__module__}.{type(module_2.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.version).__module__}.{type(module_2.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.UUID.int).__module__}.{type(module_2.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.UUID.is_safe).__module__}.{type(module_2.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    var_2 = []
    var_3 = {}
    var_4 = module_0.UUIDFormat(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_5 = var_4.serialize(var_1)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

def test_case_26():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_3 = '192.168.1.1'
    var_4 = module_5.IPv4Address(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ipaddress.IPv4Address'
    assert module_5.IPV4LENGTH == 32
    assert module_5.IPV6LENGTH == 128
    assert f'{type(module_5.IPv4Address.packed).__module__}.{type(module_5.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_reserved).__module__}.{type(module_5.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_private).__module__}.{type(module_5.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_global).__module__}.{type(module_5.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_multicast).__module__}.{type(module_5.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_unspecified).__module__}.{type(module_5.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_loopback).__module__}.{type(module_5.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_link_local).__module__}.{type(module_5.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.168.1.1'

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_3 = '31-12-2023'
    var_2.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_3 = 'www.example.com'
    var_2.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.URLFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1 = module_0.DateFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_2 = None
    var_3 = var_0.is_native_type(var_2)
    assert var_3 is False
    var_4 = module_0.DateTimeFormat()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    module_2.uuid5(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = 'invalid-email'
    var_2.validate(var_3)

def test_case_31():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_4.date(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.date'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.date.year).__module__}.{type(module_4.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.month).__module__}.{type(module_4.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.day).__module__}.{type(module_4.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.min).__module__}.{type(module_4.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_4.date.max).__module__}.{type(module_4.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_4.date.resolution).__module__}.{type(module_4.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = []
    var_7 = {}
    var_8 = module_0.DateFormat(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_9 = var_8.serialize(var_5)
    assert var_9 == '2023-01-05'

def test_case_32():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_3 = 'https://www.example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://www.example.com'

def test_case_33():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = module_0.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = '2023-02-30T12:00:00'
    var_0.validate(var_1)

def test_case_35():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = '2023-01-01T12:00:00.123456'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.hour).__module__}.{type(module_4.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.minute).__module__}.{type(module_4.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.second).__module__}.{type(module_4.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.microsecond).__module__}.{type(module_4.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.tzinfo).__module__}.{type(module_4.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.fold).__module__}.{type(module_4.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.min).__module__}.{type(module_4.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.max).__module__}.{type(module_4.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.resolution).__module__}.{type(module_4.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456
    var_10 = [var_5, var_6, var_6, var_7, var_8, var_8, var_9]
    var_11 = module_4.datetime(*var_10, **var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.datetime'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = '2023-02-30T12:00:00Z'
    var_2.validate(var_3)

def test_case_37():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = '2023-01-01T12:00:00+02:00'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.hour).__module__}.{type(module_4.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.minute).__module__}.{type(module_4.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.second).__module__}.{type(module_4.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.microsecond).__module__}.{type(module_4.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.tzinfo).__module__}.{type(module_4.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.fold).__module__}.{type(module_4.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.min).__module__}.{type(module_4.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.max).__module__}.{type(module_4.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.resolution).__module__}.{type(module_4.datetime.resolution).__qualname__}' == 'datetime.timedelta'

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.URLFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1 = None
    var_2 = module_0.URLFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.URLFormat'
    var_3 = var_2.is_native_type(var_1)
    assert var_3 is False
    var_4 = var_2.serialize(var_1)
    var_5 = module_3.platform(terse=var_1)
    assert var_5 == 'Linux-6.17.9-76061709-generic-x86_64-with-glibc2.41'
    var_6 = module_3.platform(terse=var_1)
    assert var_6 == 'Linux-6.17.9-76061709-generic-x86_64-with-glibc2.41'
    var_6.validate(var_6)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = '{12345678-1234-5678-1234-567812345678}'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_3.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_3 = '25:00:00'
    var_2.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_3 = 'invalid'
    var_2.validate(var_3)

def test_case_42():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_3 = '12:34:56.78'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.time'
    assert f'{type(module_4.time.hour).__module__}.{type(module_4.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.minute).__module__}.{type(module_4.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.second).__module__}.{type(module_4.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.microsecond).__module__}.{type(module_4.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.tzinfo).__module__}.{type(module_4.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.fold).__module__}.{type(module_4.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.time.min).__module__}.{type(module_4.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_4.time.max).__module__}.{type(module_4.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_4.time.resolution).__module__}.{type(module_4.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = var_4.microsecond
    assert var_5 == 780000

def test_case_43():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = '2023-01-01T12:00:00-03:00'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.hour).__module__}.{type(module_4.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.minute).__module__}.{type(module_4.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.second).__module__}.{type(module_4.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.microsecond).__module__}.{type(module_4.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.tzinfo).__module__}.{type(module_4.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.fold).__module__}.{type(module_4.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.min).__module__}.{type(module_4.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.max).__module__}.{type(module_4.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.resolution).__module__}.{type(module_4.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 'hours'
    var_10 = module_4.timedelta(*var_0, **var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.timedelta'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.timedelta.days).__module__}.{type(module_4.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.seconds).__module__}.{type(module_4.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.microseconds).__module__}.{type(module_4.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.resolution).__module__}.{type(module_4.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.min).__module__}.{type(module_4.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.max).__module__}.{type(module_4.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_4.timezone(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.utc).__module__}.{type(module_4.timezone.utc).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.min).__module__}.{type(module_4.timezone.min).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.max).__module__}.{type(module_4.timezone.max).__qualname__}' == 'datetime.timezone'
    var_14 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_15 = 'tzinfo'
    var_16 = {var_15: var_13}
    var_17 = module_4.datetime(*var_14, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'datetime.datetime'
    var_18 = bool(var_9 == var_17)

def test_case_44():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = 1
    var_4 = 12
    var_5 = 0
    var_6 = [var_4, var_3, var_3, var_4, var_5, var_5]
    var_7 = {}
    var_8 = module_4.datetime(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.datetime'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.datetime.hour).__module__}.{type(module_4.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.minute).__module__}.{type(module_4.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.second).__module__}.{type(module_4.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.microsecond).__module__}.{type(module_4.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.tzinfo).__module__}.{type(module_4.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.fold).__module__}.{type(module_4.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.min).__module__}.{type(module_4.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.max).__module__}.{type(module_4.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.resolution).__module__}.{type(module_4.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_9 = var_2.serialize(var_8)
    assert var_9 == '0012-01-01T12:00:00'

def test_case_45():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = []
    var_8 = {}
    var_9 = module_4.timedelta(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.timedelta'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.timedelta.days).__module__}.{type(module_4.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.seconds).__module__}.{type(module_4.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.microseconds).__module__}.{type(module_4.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.resolution).__module__}.{type(module_4.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.min).__module__}.{type(module_4.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.max).__module__}.{type(module_4.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_4.timezone(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.utc).__module__}.{type(module_4.timezone.utc).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.min).__module__}.{type(module_4.timezone.min).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.max).__module__}.{type(module_4.timezone.max).__qualname__}' == 'datetime.timezone'
    var_13 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_14 = 'tzinfo'
    var_15 = {var_14: var_12}
    var_16 = module_4.datetime(*var_13, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.hour).__module__}.{type(module_4.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.minute).__module__}.{type(module_4.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.second).__module__}.{type(module_4.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.microsecond).__module__}.{type(module_4.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.tzinfo).__module__}.{type(module_4.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.fold).__module__}.{type(module_4.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.min).__module__}.{type(module_4.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.max).__module__}.{type(module_4.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.resolution).__module__}.{type(module_4.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_17 = var_2.serialize(var_16)
    assert var_17 == '2023-01-01T12:00:00Z'

def test_case_46():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = 'test@example.com'
    var_4 = var_2.is_native_type(var_3)
    assert var_4 is False

def test_case_47():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ipaddress.IPv4Address'
    assert f'{type(module_5.IPv4Address.packed).__module__}.{type(module_5.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_reserved).__module__}.{type(module_5.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_private).__module__}.{type(module_5.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_global).__module__}.{type(module_5.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_multicast).__module__}.{type(module_5.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_unspecified).__module__}.{type(module_5.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_loopback).__module__}.{type(module_5.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv4Address.is_link_local).__module__}.{type(module_5.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_5 = str(var_4)
    assert var_5 == '192.168.1.1'

@pytest.mark.xfail(strict=True)
def test_case_48():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_3 = 'invalid_ip'
    var_2.validate(var_3)

def test_case_49():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ipaddress.IPv6Address'
    assert f'{type(module_5.IPv6Address.scope_id).__module__}.{type(module_5.IPv6Address.scope_id).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.packed).__module__}.{type(module_5.IPv6Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_multicast).__module__}.{type(module_5.IPv6Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_reserved).__module__}.{type(module_5.IPv6Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_link_local).__module__}.{type(module_5.IPv6Address.is_link_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_site_local).__module__}.{type(module_5.IPv6Address.is_site_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_private).__module__}.{type(module_5.IPv6Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_global).__module__}.{type(module_5.IPv6Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_unspecified).__module__}.{type(module_5.IPv6Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.is_loopback).__module__}.{type(module_5.IPv6Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.ipv4_mapped).__module__}.{type(module_5.IPv6Address.ipv4_mapped).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.teredo).__module__}.{type(module_5.IPv6Address.teredo).__qualname__}' == 'builtins.property'
    assert f'{type(module_5.IPv6Address.sixtofour).__module__}.{type(module_5.IPv6Address.sixtofour).__qualname__}' == 'builtins.property'