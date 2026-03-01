# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import platform as module_0
import typesystem.formats as module_1
import datetime as module_2
import uuid as module_3
import ipaddress as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.processor()
    assert var_0 == ''
    var_1 = module_1.UUIDFormat(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = module_1.DateTimeFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = None
    var_4 = var_1.serialize(var_3)
    var_0.validate(var_3)

def test_case_1():
    var_0 = module_1.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = module_0.python_revision()
    assert var_1 == ''
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_1.TimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False
    var_0.serialize(var_0)

def test_case_4():
    var_0 = module_1.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1 = module_1.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = None
    var_3 = var_0.serialize(var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_1.TimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_1 = module_1.DateFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_2 = None
    var_3 = var_1.is_native_type(var_2)
    assert var_3 is False
    var_0.serialize(var_0)

def test_case_6():
    var_0 = module_1.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)

def test_case_7():
    var_0 = module_1.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.processor()
    assert var_0 == ''
    var_1 = module_1.UUIDFormat(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = module_1.DateTimeFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = var_1.is_native_type(var_0)
    assert var_3 is False
    var_4 = None
    var_5 = var_1.serialize(var_4)
    var_6 = module_0.processor()
    assert var_6 == ''
    var_2.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_1.TimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_1.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_0.serialize(var_0)

def test_case_11():
    var_0 = module_1.IPAddressFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)

def test_case_12():
    var_0 = module_1.BaseFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.BaseFormat.errors == {}
    with pytest.raises(NotImplementedError):
        var_0.serialize(var_0)

def test_case_13():
    var_0 = module_1.TimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

def test_case_14():
    var_0 = module_1.IPAddressFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_1.DateFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_1.URLFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1 = var_0.serialize(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.URLFormat'
    var_2 = module_1.DateFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_3 = module_0.win32_ver()
    var_3.validate(var_3)

def test_case_17():
    var_0 = module_1.BaseFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.BaseFormat.errors == {}
    var_1 = None
    with pytest.raises(NotImplementedError):
        var_0.validate(var_1)

def test_case_18():
    var_0 = module_1.IPAddressFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_1.EmailFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_1 = '4BBj;qk3Hd`r\x0co'
    var_2 = None
    var_3 = var_0.is_native_type(var_2)
    assert var_3 is False
    var_4 = var_0.serialize(var_1)
    assert var_4 == '4BBj;qk3Hd`r\x0co'
    var_5 = None
    var_6 = var_0.serialize(var_5)
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_1.TimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_1 = module_0.python_implementation()
    assert var_1 == 'CPython'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_1.URLFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1 = None
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_0.processor()
    assert var_0 == ''
    var_1 = module_1.DateFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_1.URLFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = module_1.IPAddressFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_4 = 'Bn*O*~r,ZXk[/'
    var_3.validate(var_4)

def test_case_24():
    var_0 = module_1.BaseFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.BaseFormat.errors == {}
    var_1 = None
    with pytest.raises(NotImplementedError):
        var_0.is_native_type(var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_1.EmailFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_1 = '4BBj;qk3Hd`r\x0co'
    var_2 = None
    var_3 = var_0.serialize(var_2)
    var_0.validate(var_1)

def test_case_26():
    var_0 = []
    var_1 = {}
    var_2 = module_1.DateFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_3 = 1
    var_4 = [var_3, var_3, var_3]
    var_5 = {}
    var_6 = module_2.date(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.date'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_7 = var_2.serialize(var_6)
    assert var_7 == '0001-01-01'

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_1.EmailFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_1 = '4BBj;qk3Hd`r\x0co'
    var_2 = var_0.serialize(var_1)
    assert var_2 == '4BBj;qk3Hd`r\x0co'
    var_3 = None
    var_4 = var_0.serialize(var_3)
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_1.URLFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1 = None
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = []
    var_1 = module_1.UUIDFormat(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = module_1.DateFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_1.serialize(var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = []
    var_1 = module_1.UUIDFormat(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = module_0.python_version()
    assert var_2 == '3.10.19'
    var_1.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = module_1.DateTimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1 = module_1.UUIDFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = module_3.uuid4()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_1.serialize(var_2)
    var_4 = module_1.DateTimeFormat()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    var_5 = None
    var_6 = var_4.serialize(var_5)
    var_7 = module_0.processor()
    assert var_7 == ''
    var_7.is_native_type(var_5)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = module_1.TimeFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_1 = '14:30:60'
    var_0.validate(var_1)

def test_case_33():
    var_0 = []
    var_1 = {}
    var_2 = module_1.TimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_3 = '14:30:45.123456'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.hour).__module__}.{type(module_2.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.minute).__module__}.{type(module_2.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.second).__module__}.{type(module_2.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.microsecond).__module__}.{type(module_2.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.tzinfo).__module__}.{type(module_2.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.fold).__module__}.{type(module_2.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.min).__module__}.{type(module_2.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.max).__module__}.{type(module_2.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.resolution).__module__}.{type(module_2.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 14
    var_6 = 30
    var_7 = 45
    var_8 = 123456
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_2.time(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.time'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = module_1.EmailFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_1 = '4BBj;qk3Hd`r\x0co'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.processor()
    assert var_0 == ''
    var_1 = module_1.IPAddressFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = {}
    var_1 = module_1.DateFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_2 = '2023-02-30'
    var_1.validate(var_2)

def test_case_37():
    var_0 = {}
    var_1 = module_1.DateTimeFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_2 = 2
    var_3 = [var_2, var_2, var_2, var_2, var_2, var_2, var_2]
    var_4 = module_2.datetime(*var_3, **var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = var_1.serialize(var_4)
    assert var_5 == '0002-02-02T02:02:02.000002'

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = {}
    var_1 = module_1.DateTimeFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_2 = '2023-13-45T25:61:61'
    var_1.validate(var_2)

def test_case_39():
    var_0 = []
    var_1 = {}
    var_2 = module_1.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = 2
    var_4 = [var_3, var_3, var_3, var_3, var_3, var_3, var_3]
    var_5 = module_2.datetime(*var_4, **var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.datetime'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = var_2.serialize(var_5)
    assert var_6 == '0002-02-02T02:02:02.000002'
    var_7 = var_2.validate(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.datetime'

def test_case_40():
    var_0 = 'fec0::1'
    var_1 = module_4.IPv6Address(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ipaddress.IPv6Address'
    assert module_4.IPV4LENGTH == 32
    assert module_4.IPV6LENGTH == 128
    assert f'{type(module_4.IPv6Address.scope_id).__module__}.{type(module_4.IPv6Address.scope_id).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.packed).__module__}.{type(module_4.IPv6Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_multicast).__module__}.{type(module_4.IPv6Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_reserved).__module__}.{type(module_4.IPv6Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_link_local).__module__}.{type(module_4.IPv6Address.is_link_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_site_local).__module__}.{type(module_4.IPv6Address.is_site_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_private).__module__}.{type(module_4.IPv6Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_global).__module__}.{type(module_4.IPv6Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_unspecified).__module__}.{type(module_4.IPv6Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_loopback).__module__}.{type(module_4.IPv6Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.ipv4_mapped).__module__}.{type(module_4.IPv6Address.ipv4_mapped).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.teredo).__module__}.{type(module_4.IPv6Address.teredo).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.sixtofour).__module__}.{type(module_4.IPv6Address.sixtofour).__qualname__}' == 'builtins.property'
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_5 = var_4.serialize(var_1)
    assert var_5 == 'fec0::1'

def test_case_41():
    var_0 = []
    var_1 = {}
    var_2 = module_1.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'

def test_case_42():
    var_0 = []
    var_1 = {}
    var_2 = module_1.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = '2023-01-15T14:30:45-03:45'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = -3
    var_6 = -45
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_2.timedelta(*var_7, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.timedelta'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'

def test_case_43():
    var_0 = []
    var_1 = {}
    var_2 = module_1.DateTimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = '2023-01-15T14:30:45+02'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_2.timedelta(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.timedelta'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_2.timezone(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_2.timezone.utc).__module__}.{type(module_2.timezone.utc).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_2.timezone.min).__module__}.{type(module_2.timezone.min).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_2.timezone.max).__module__}.{type(module_2.timezone.max).__qualname__}' == 'datetime.timezone'
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_2.datetime(*var_19, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'datetime.datetime'
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

def test_case_44():
    var_0 = []
    var_1 = {}
    var_2 = module_1.URLFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_3 = 'ftp://example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'ftp://example.com'

def test_case_45():
    var_0 = module_0.python_branch()
    assert var_0 == ''
    var_1 = {}
    var_2 = module_1.IPAddressFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ipaddress.IPv4Address'
    assert f'{type(module_4.IPv4Address.packed).__module__}.{type(module_4.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_reserved).__module__}.{type(module_4.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_private).__module__}.{type(module_4.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_global).__module__}.{type(module_4.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_multicast).__module__}.{type(module_4.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_unspecified).__module__}.{type(module_4.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_loopback).__module__}.{type(module_4.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_link_local).__module__}.{type(module_4.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_5 = str(var_4)
    assert var_5 == '192.168.1.1'

def test_case_46():
    var_0 = module_1.IPAddressFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_1 = '2001:0db8:85a3:0000:0000:8ae:370:7334'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ipaddress.IPv6Address'
    assert f'{type(module_4.IPv6Address.scope_id).__module__}.{type(module_4.IPv6Address.scope_id).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.packed).__module__}.{type(module_4.IPv6Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_multicast).__module__}.{type(module_4.IPv6Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_reserved).__module__}.{type(module_4.IPv6Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_link_local).__module__}.{type(module_4.IPv6Address.is_link_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_site_local).__module__}.{type(module_4.IPv6Address.is_site_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_private).__module__}.{type(module_4.IPv6Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_global).__module__}.{type(module_4.IPv6Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_unspecified).__module__}.{type(module_4.IPv6Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.is_loopback).__module__}.{type(module_4.IPv6Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.ipv4_mapped).__module__}.{type(module_4.IPv6Address.ipv4_mapped).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.teredo).__module__}.{type(module_4.IPv6Address.teredo).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv6Address.sixtofour).__module__}.{type(module_4.IPv6Address.sixtofour).__qualname__}' == 'builtins.property'

def test_case_47():
    var_0 = []
    var_1 = {}
    var_2 = module_1.TimeFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_3 = 0
    var_4 = [var_3, var_3, var_3]
    var_5 = {}
    var_6 = module_2.time(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.time'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.time.hour).__module__}.{type(module_2.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.minute).__module__}.{type(module_2.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.second).__module__}.{type(module_2.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.microsecond).__module__}.{type(module_2.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.tzinfo).__module__}.{type(module_2.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.fold).__module__}.{type(module_2.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.min).__module__}.{type(module_2.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.max).__module__}.{type(module_2.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.resolution).__module__}.{type(module_2.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_7 = var_2.serialize(var_6)
    assert var_7 == '00:00:00'

def test_case_48():
    var_0 = []
    var_1 = {}
    var_2 = module_1.EmailFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = 'user@sub.example.co.uk'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user@sub.example.co.uk'

def test_case_49():
    var_0 = []
    var_1 = {}
    var_2 = module_1.UUIDFormat(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_3 = 'c232ab00-9414-11ec-b3c8-9f6b385d64be'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.UUID.bytes).__module__}.{type(module_3.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.bytes_le).__module__}.{type(module_3.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.fields).__module__}.{type(module_3.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_low).__module__}.{type(module_3.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_mid).__module__}.{type(module_3.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time_hi_version).__module__}.{type(module_3.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_hi_variant).__module__}.{type(module_3.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq_low).__module__}.{type(module_3.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.time).__module__}.{type(module_3.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.clock_seq).__module__}.{type(module_3.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.node).__module__}.{type(module_3.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.hex).__module__}.{type(module_3.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.urn).__module__}.{type(module_3.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.variant).__module__}.{type(module_3.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.version).__module__}.{type(module_3.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.UUID.int).__module__}.{type(module_3.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.UUID.is_safe).__module__}.{type(module_3.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'