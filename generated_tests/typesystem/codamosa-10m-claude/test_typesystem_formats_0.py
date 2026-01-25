# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.formats as module_0
import platform as module_1
import datetime as module_2
import uuid as module_3
import ipaddress as module_4
import re as module_5
import collections as module_6

@pytest.mark.xfail(strict=True)
def test_case_0():
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
    var_1 = module_0.URLFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.URLFormat'
    assert module_0.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_2 = None
    var_3 = var_1.serialize(var_2)
    var_4 = '2023-22-5T10:30:45+0200'
    var_0.validate(var_4)

def test_case_1():
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

@pytest.mark.xfail(strict=True)
def test_case_2():
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
    var_1 = '25:3'
    var_2 = var_0.is_native_type(var_0)
    assert var_2 is False
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
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
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_1 = module_0.DateTimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_2 = '2023-22-2T10:30:45+0200'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '2023-22-2T10:30:45+0200'
    var_1.validate(var_2)

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
    var_1 = None
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    var_3 = '2023-12-85'
    var_0.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_6():
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
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    var_3 = None
    var_4 = var_0.serialize(var_3)
    var_0.validate(var_3)

def test_case_7():
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
        var_0.is_native_type(var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_1 = module_1.python_branch()
    assert var_1 == ''
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.EmailFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_1 = module_1.python_branch()
    assert var_1 == ''
    var_2 = ';^<.Ruw~B\rfIW >L'
    var_0.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False
    var_2 = '0.0.0.0c'
    var_0.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    var_1 = '25:30'
    var_2 = None
    var_3 = var_0.serialize(var_2)
    var_0.validate(var_1)

def test_case_12():
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
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

def test_case_13():
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
        var_0.serialize(var_1)

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
    var_1 = '0.0.0.0c'
    var_2 = None
    var_3 = var_0.serialize(var_2)
    var_0.validate(var_1)

def test_case_15():
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
    var_1 = module_1.uname()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'platform.uname_result'
    assert len(var_1) == 6
    assert f'{type(module_1.uname_result.processor).__module__}.{type(module_1.uname_result.processor).__qualname__}' == 'functools.cached_property'
    assert module_1.uname_result.processor.attrname == 'processor'
    assert f'{type(module_1.uname_result.processor.lock).__module__}.{type(module_1.uname_result.processor.lock).__qualname__}' == '_thread.RLock'
    with pytest.raises(AssertionError):
        var_0.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
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
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = '2023-12-85'
    var_0.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_17():
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
    var_1 = '12:30:45'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.hour).__module__}.{type(module_2.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.minute).__module__}.{type(module_2.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.second).__module__}.{type(module_2.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.microsecond).__module__}.{type(module_2.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.tzinfo).__module__}.{type(module_2.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.fold).__module__}.{type(module_2.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.min).__module__}.{type(module_2.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.max).__module__}.{type(module_2.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.resolution).__module__}.{type(module_2.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = '12:30:45.123456'
    var_4 = var_0.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.time'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = var_0.validate(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.time'
    var_6 = '25:30'
    var_0.validate(var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
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
    var_1 = '25:30'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
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
    var_3 = module_1.python_branch()
    assert var_3 == ''
    var_0.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_0.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1 = '550e8400-e29b-41d4-a716-446655440000'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'uuid.UUID'
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
    var_3 = str(var_2)
    var_4 = '6ba7b810-9dad-11d1-80b4-00c04fd430c8'
    var_5 = var_0.validate(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'uuid.UUID'
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_6 = str(var_5)
    var_7 = 'invalid-uuid-format'
    var_0.validate(var_7)

@pytest.mark.xfail(strict=True)
def test_case_21():
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
    var_1 = '192.168.1.1'
    var_2 = module_1.machine()
    assert var_2 == 'x86_64'
    var_3 = module_4.IPv4Address(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ipaddress.IPv4Address'
    assert module_4.IPV4LENGTH == 32
    assert module_4.IPV6LENGTH == 128
    assert f'{type(module_4.IPv4Address.packed).__module__}.{type(module_4.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_reserved).__module__}.{type(module_4.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_private).__module__}.{type(module_4.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_global).__module__}.{type(module_4.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_multicast).__module__}.{type(module_4.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_unspecified).__module__}.{type(module_4.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_loopback).__module__}.{type(module_4.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_link_local).__module__}.{type(module_4.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_4 = '0.0.0.0'
    var_5 = var_0.validate(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ipaddress.IPv4Address'
    var_6 = module_4.IPv4Address(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ipaddress.IPv4Address'
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ipaddress.IPv4Address'
    var_9 = module_4.IPv4Address(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ipaddress.IPv4Address'
    var_10 = '10.0.0.1'
    var_11 = var_0.validate(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ipaddress.IPv4Address'
    var_12 = module_4.IPv4Address(var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ipaddress.IPv4Address'
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ipaddress.IPv6Address'
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
    var_15 = module_4.IPv6Address(var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ipaddress.IPv6Address'
    module_2.timedelta(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_22():
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
    var_1 = module_1.python_version()
    assert var_1 == '3.10.19'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_1 = '0.0.0.0c'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
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
    var_1 = '2023-22-25T10:30:45+0200'
    var_2 = var_0.is_native_type(var_0)
    assert var_2 is False
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
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
    var_1 = '12:30:45'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.hour).__module__}.{type(module_2.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.minute).__module__}.{type(module_2.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.second).__module__}.{type(module_2.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.microsecond).__module__}.{type(module_2.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.tzinfo).__module__}.{type(module_2.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.fold).__module__}.{type(module_2.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.min).__module__}.{type(module_2.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.max).__module__}.{type(module_2.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.resolution).__module__}.{type(module_2.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '12:30:45'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_4 = '25:30'
    var_0.validate(var_4)

def test_case_26():
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
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

def test_case_27():
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
    var_1 = '00:00'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.hour).__module__}.{type(module_2.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.minute).__module__}.{type(module_2.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.second).__module__}.{type(module_2.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.microsecond).__module__}.{type(module_2.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.tzinfo).__module__}.{type(module_2.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.fold).__module__}.{type(module_2.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.time.min).__module__}.{type(module_2.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.max).__module__}.{type(module_2.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_2.time.resolution).__module__}.{type(module_2.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = '12:30:45'
    var_4 = var_0.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.time'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = '12:30:45.123456'
    var_6 = module_0.UUIDFormat()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_7 = var_0.validate(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.time'
    var_8 = var_0.validate(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.time'
    var_9 = module_5.purge()
    assert module_5.ASCII == module_5.RegexFlag.ASCII
    assert module_5.A == module_5.RegexFlag.ASCII
    assert module_5.IGNORECASE == module_5.RegexFlag.IGNORECASE
    assert module_5.I == module_5.RegexFlag.IGNORECASE
    assert module_5.LOCALE == module_5.RegexFlag.LOCALE
    assert module_5.L == module_5.RegexFlag.LOCALE
    assert module_5.UNICODE == module_5.RegexFlag.UNICODE
    assert module_5.U == module_5.RegexFlag.UNICODE
    assert module_5.MULTILINE == module_5.RegexFlag.MULTILINE
    assert module_5.M == module_5.RegexFlag.MULTILINE
    assert module_5.DOTALL == module_5.RegexFlag.DOTALL
    assert module_5.S == module_5.RegexFlag.DOTALL
    assert module_5.VERBOSE == module_5.RegexFlag.VERBOSE
    assert module_5.X == module_5.RegexFlag.VERBOSE
    assert module_5.TEMPLATE == module_5.RegexFlag.TEMPLATE
    assert module_5.T == module_5.RegexFlag.TEMPLATE
    assert module_5.DEBUG == module_5.RegexFlag.DEBUG
    var_10 = var_6.is_native_type(var_9)
    assert var_10 is False
    var_11 = module_3.uuid4()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'uuid.UUID'
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
    var_12 = module_2.timedelta()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_0.EmailFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_1 = None
    var_2 = None
    var_3 = var_0.serialize(var_2)
    module_5.search(var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_1 = module_0.EmailFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_2 = '25:73n<Hu'
    var_3 = None
    var_4 = var_1.is_native_type(var_3)
    assert var_4 is False
    var_0.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
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
    var_1 = module_0.EmailFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_2 = var_1.serialize(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    module_5.search(var_2, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_31():
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
    var_1 = module_1.python_branch()
    assert var_1 == ''
    var_0.validate(var_1)

def test_case_32():
    var_0 = None
    var_1 = module_0.BaseFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.BaseFormat.errors == {}
    with pytest.raises(NotImplementedError):
        var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_33():
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
    var_1 = module_0.DateFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_2 = module_1.node()
    assert var_2 == 'e8b6bb68b8dd'
    var_1.validate(var_2)

def test_case_34():
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
    var_1 = module_1.python_branch()
    assert var_1 == ''
    with pytest.raises(AssertionError):
        var_0.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = var_0.serialize(var_1)
    var_4 = module_1.python_branch()
    assert var_4 == ''
    var_4.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_36():
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
    var_1 = '2023-12-85'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_37():
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
    var_1 = '2023-22-25T10:30:45+0200'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_38():
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
    var_1 = '2023-12-25T10:30:45+02:00'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = 1287
    var_4 = module_2.timedelta()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.timedelta'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_5 = '2023-12-25T10:30:45-05:00'
    var_6 = var_0.validate(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.datetime'
    var_7 = -5
    var_8 = module_2.timedelta()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.timedelta'
    var_9 = module_1.python_revision()
    assert var_9 == ''
    var_10 = module_2.timedelta()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.timedelta'
    var_11 = '2023-12-25T10:30:45'
    var_12 = var_0.validate(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.datetime'
    var_13 = '2023-12-25 10:30:45'
    var_14 = var_0.validate(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'datetime.datetime'
    var_0.validate(var_9)

@pytest.mark.xfail(strict=True)
def test_case_39():
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
    var_1 = '2023-12-25T10:30:45Z'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = '2023-12-25T10:30:45+02:00'
    var_4 = var_0.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = '2023-12-25T10:30:45-05:00'
    var_6 = var_0.validate(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.datetime'
    var_7 = -5
    var_8 = module_2.timedelta()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_9 = '2023-12-25T10:30:45+05:30'
    var_10 = module_1.python_revision()
    assert var_10 == ''
    var_11 = module_2.timedelta()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.timedelta'
    var_12 = '2023-12-25T10:30:45'
    var_13 = var_0.validate(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.datetime'
    var_14 = '2023-12-25 10:30:45'
    var_15 = var_0.validate(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'datetime.datetime'
    var_16 = '2023-12-25T10:30:45.123456'
    var_17 = var_0.validate(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'datetime.datetime'
    var_18 = var_0.validate(var_9)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'datetime.datetime'
    var_19 = var_0.validate(var_16)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'datetime.datetime'
    var_10.serialize(var_13)

@pytest.mark.xfail(strict=True)
def test_case_40():
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
    var_1 = '2023-12-25T10:30:45Z'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = '2023-12-25T10:30:45+02:00'
    var_4 = var_0.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = -26
    var_6 = module_2.timedelta()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_7 = '2023-12-25T10:30:45-05:00'
    var_8 = var_0.validate(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.datetime'
    var_9 = var_0.serialize(var_2)
    assert var_9 == '2023-12-25T10:30:45Z'
    var_10 = module_1.python_branch()
    assert var_10 == ''
    var_10.serialize(var_10)

@pytest.mark.xfail(strict=True)
def test_case_41():
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
    var_1 = '2023-12-25T10:30:45Z'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = None
    var_4 = var_0.serialize(var_3)
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_0.validate(var_3)

def test_case_42():
    var_0 = module_0.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = '550e8400-e29b-41d4-a716-446655440000'
    var_4 = module_3.UUID(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'uuid.UUID'
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
    var_5 = var_0.serialize(var_4)
    assert var_5 == '550e8400-e29b-41d4-a716-446655440000'
    with pytest.raises(TypeError):
        module_3.UUID(int=var_1)

def test_case_43():
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
    var_1 = 'http://example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'http://example.com'
    var_3 = 'https://www.example.com'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'https://www.example.com'
    var_5 = None
    with pytest.raises(TypeError):
        module_6.namedtuple(var_5, var_5, rename=var_5, module=var_5)

@pytest.mark.xfail(strict=True)
def test_case_44():
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
    var_1 = '2023-12-25T10:30:45Z'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = '2023-12-25T10:30:45+02:00'
    var_4 = var_0.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = 2
    var_6 = module_2.timedelta()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_7 = var_0.serialize(var_2)
    assert var_7 == '2023-12-25T10:30:45Z'
    var_8 = None
    module_3.uuid3(var_8, var_7)

def test_case_45():
    var_0 = module_0.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)
    assert var_1 is None

@pytest.mark.xfail(strict=True)
def test_case_46():
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
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.year).__module__}.{type(module_2.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.month).__module__}.{type(module_2.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.day).__module__}.{type(module_2.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.date.min).__module__}.{type(module_2.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.max).__module__}.{type(module_2.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_2.date.resolution).__module__}.{type(module_2.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '2023-12-25'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_4 = module_1.version()
    assert var_4 == '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D'
    var_5 = '2023-02-30'
    var_0.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_47():
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
    var_1 = '2023-12-25T10:30:45Z'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.hour).__module__}.{type(module_2.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.minute).__module__}.{type(module_2.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.second).__module__}.{type(module_2.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.microsecond).__module__}.{type(module_2.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.tzinfo).__module__}.{type(module_2.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.fold).__module__}.{type(module_2.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_2.datetime.min).__module__}.{type(module_2.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.max).__module__}.{type(module_2.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_2.datetime.resolution).__module__}.{type(module_2.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = '2023-12-25T10:30:45+02:00'
    var_4 = var_0.serialize(var_2)
    assert var_4 == '2023-12-25T10:30:45Z'
    assert module_2.MINYEAR == 1
    assert module_2.MAXYEAR == 9999
    assert f'{type(module_2.datetime_CAPI).__module__}.{type(module_2.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = var_0.validate(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.datetime'
    var_6 = var_0.serialize(var_5)
    assert var_6 == '2023-12-25T10:30:45+02:00'
    var_7 = 2
    var_8 = module_2.timedelta()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.days).__module__}.{type(module_2.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.seconds).__module__}.{type(module_2.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.microseconds).__module__}.{type(module_2.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.timedelta.resolution).__module__}.{type(module_2.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.min).__module__}.{type(module_2.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_2.timedelta.max).__module__}.{type(module_2.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_9 = var_0.serialize(var_2)
    assert var_9 == '2023-12-25T10:30:45Z'
    var_10 = None
    module_3.uuid3(var_10, var_9)

def test_case_48():
    var_0 = module_0.EmailFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_1 = 'user@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user@example.com'
    var_3 = None
    with pytest.raises(module_4.AddressValueError):
        module_4.IPv4Address(var_3)

def test_case_49():
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
    var_1 = '192.168.1.1'
    var_2 = module_4.IPv4Address(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ipaddress.IPv4Address'
    assert module_4.IPV4LENGTH == 32
    assert module_4.IPV6LENGTH == 128
    assert f'{type(module_4.IPv4Address.packed).__module__}.{type(module_4.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_reserved).__module__}.{type(module_4.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_private).__module__}.{type(module_4.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_global).__module__}.{type(module_4.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_multicast).__module__}.{type(module_4.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_unspecified).__module__}.{type(module_4.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_loopback).__module__}.{type(module_4.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.IPv4Address.is_link_local).__module__}.{type(module_4.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '192.168.1.1'
    var_4 = '2001:db8::8a2e:370:7334'
    var_5 = module_4.IPv6Address(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ipaddress.IPv6Address'
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
    var_6 = var_0.serialize(var_5)
    assert var_6 == '2001:db8::8a2e:370:7334'
    var_7 = None
    var_8 = var_0.serialize(var_7)
    assert var_8 is None
    with pytest.raises(module_4.AddressValueError):
        module_4.IPv4Address(var_6)

@pytest.mark.xfail(strict=True)
def test_case_50():
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
    var_1 = '2023-22-25T10:30:5+02'
    var_0.validate(var_1)