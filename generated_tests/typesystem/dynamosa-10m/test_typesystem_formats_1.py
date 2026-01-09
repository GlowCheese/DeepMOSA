# Check out: https://github.com/GlowCheese/deepmosa
import datetime as module_3
import platform as module_0
import uuid as module_2

import pytest
import typesystem.formats as module_1


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.python_revision()
    assert var_0 == ''
    var_1 = module_1.URLFormat(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.URLFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_1.validate(var_0)

def test_case_1():
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

@pytest.mark.xfail(strict=True)
def test_case_2():
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
    var_0.validation_error(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
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

def test_case_4():
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

def test_case_5():
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

def test_case_6():
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
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False

def test_case_7():
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

def test_case_8():
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
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False

def test_case_9():
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
    var_1 = None
    var_2 = var_0.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_1 = '$o'
    var_0.validate(var_1)

def test_case_11():
    var_0 = None
    var_1 = module_1.BaseFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.BaseFormat.errors == {}
    with pytest.raises(NotImplementedError):
        var_1.is_native_type(var_0)

def test_case_12():
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
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False

def test_case_13():
    var_0 = module_1.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1 = None
    var_2 = var_0.serialize(var_1)

def test_case_14():
    var_0 = module_1.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False

def test_case_15():
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

def test_case_16():
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

def test_case_17():
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
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False

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
    var_1 = None
    var_2 = var_0.serialize(var_1)

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
    var_1 = None
    var_2 = var_0.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_1.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_0.serialize(var_0)

def test_case_21():
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

def test_case_22():
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
def test_case_23():
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
    var_0.serialize(var_0)

def test_case_24():
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

@pytest.mark.xfail(strict=True)
def test_case_25():
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

@pytest.mark.xfail(strict=True)
def test_case_26():
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
    var_1 = module_0.release()
    assert var_1 == '6.17.9-76061709-generic'
    var_0.validate(var_1)

def test_case_27():
    var_0 = module_1.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1 = module_2.uuid4()
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
    var_2 = var_0.serialize(var_1)

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
    var_1 = var_0.serialize(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.URLFormat'

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_1 = module_0.python_revision()
    assert var_1 == ''
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
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
    var_1 = module_0.python_revision()
    assert var_1 == ''
    var_0.validate(var_1)

def test_case_31():
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
    var_1 = var_0.serialize(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.EmailFormat'

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = module_0.python_implementation()
    assert var_0 == 'CPython'
    var_1 = module_1.UUIDFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_33():
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
    var_1 = module_0.processor()
    assert var_1 == ''
    var_0.validate(var_1)

def test_case_34():
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
    var_1 = module_3.time()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.time'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.time.hour).__module__}.{type(module_3.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.minute).__module__}.{type(module_3.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.second).__module__}.{type(module_3.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.microsecond).__module__}.{type(module_3.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.tzinfo).__module__}.{type(module_3.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.fold).__module__}.{type(module_3.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.min).__module__}.{type(module_3.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.max).__module__}.{type(module_3.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.resolution).__module__}.{type(module_3.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_2 = var_0.serialize(var_1)
    assert var_2 == '00:00:00'

def test_case_35():
    var_0 = module_1.UUIDFormat()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_1 = module_2.uuid4()
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
    var_2 = var_0.serialize(var_1)
    var_3 = var_0.validate(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'uuid.UUID'

def test_case_36():
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
    var_1 = module_3.time()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'datetime.time'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.time.hour).__module__}.{type(module_3.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.minute).__module__}.{type(module_3.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.second).__module__}.{type(module_3.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.microsecond).__module__}.{type(module_3.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.tzinfo).__module__}.{type(module_3.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.fold).__module__}.{type(module_3.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.min).__module__}.{type(module_3.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.max).__module__}.{type(module_3.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.resolution).__module__}.{type(module_3.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_2 = var_0.serialize(var_1)
    assert var_2 == '00:00:00'
    var_3 = var_0.validate(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.time'