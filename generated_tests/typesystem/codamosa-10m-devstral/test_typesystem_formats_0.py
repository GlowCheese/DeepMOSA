# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.formats as module_0
import platform as module_1
import ipaddress as module_2
import datetime as module_3
import uuid as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
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
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = '25:00'
    var_0.validate(var_3)

@pytest.mark.xfail(strict=True)
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
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = module_1.system_alias(var_1, var_1, var_1)
    var_4 = var_3.__iter__()
    var_4.validate(var_2)

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

def test_case_3():
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
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

def test_case_4():
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

@pytest.mark.xfail(strict=True)
def test_case_5():
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
    var_2 = var_0.serialize(var_1)
    var_3 = var_0.is_native_type(var_2)
    assert var_3 is False
    with pytest.raises(ValueError):
        module_2.ip_address(var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.DateFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_2 = var_1.is_native_type(var_0)
    assert var_2 is False
    var_3 = []
    var_4 = ":vf'|K#\x0cHm{@c"
    var_5 = "]'!Anhwq"
    var_6 = {var_4: var_0, var_4: var_0, var_5: var_0}
    module_0.TimeFormat(*var_3, **var_6)

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_1 = module_0.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = '25:00'
    var_3 = None
    var_4 = var_0.serialize(var_3)
    var_5 = module_1.python_revision()
    assert var_5 == ''
    var_1.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    var_1 = '2023-05-25T12:34:56.789123+02:00'
    var_2 = None
    var_3 = var_0.is_native_type(var_2)
    assert var_3 is False
    var_4 = var_0.validate(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_0.validate(var_4)

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
    var_1 = module_0.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = None
    var_3 = var_0.is_native_type(var_2)
    assert var_3 is False
    var_4 = '25:00'
    var_5 = None
    var_6 = var_0.is_native_type(var_5)
    assert var_6 is False
    var_1.validate(var_4)

def test_case_11():
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
    var_1 = '2023-05-25T12:34:56.789123+02:00'
    with pytest.raises(AssertionError):
        var_0.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
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
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = None
    var_3 = var_0.serialize(var_2)
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
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
    var_0.serialize(var_0)

def test_case_14():
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
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = module_4.UUID(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'uuid.UUID'
    assert module_4.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_4.RFC_4122 == 'specified in RFC 4122'
    assert module_4.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_4.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_4.NAMESPACE_DNS).__module__}.{type(module_4.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.NAMESPACE_URL).__module__}.{type(module_4.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.NAMESPACE_OID).__module__}.{type(module_4.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.NAMESPACE_X500).__module__}.{type(module_4.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.UUID.bytes).__module__}.{type(module_4.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.bytes_le).__module__}.{type(module_4.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.fields).__module__}.{type(module_4.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time_low).__module__}.{type(module_4.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time_mid).__module__}.{type(module_4.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time_hi_version).__module__}.{type(module_4.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.clock_seq_hi_variant).__module__}.{type(module_4.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.clock_seq_low).__module__}.{type(module_4.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time).__module__}.{type(module_4.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.clock_seq).__module__}.{type(module_4.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.node).__module__}.{type(module_4.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.hex).__module__}.{type(module_4.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.urn).__module__}.{type(module_4.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.variant).__module__}.{type(module_4.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.version).__module__}.{type(module_4.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.int).__module__}.{type(module_4.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.UUID.is_safe).__module__}.{type(module_4.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    var_5 = var_0.serialize(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'
    var_6 = 'ffffffff-ffff-ffff-ffff-ffffffffffff'
    var_7 = module_4.UUID(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'uuid.UUID'
    var_8 = var_0.serialize(var_7)
    assert var_8 == 'ffffffff-ffff-ffff-ffff-ffffffffffff'

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_2 = var_0.serialize(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.URLFormat'
    var_3 = var_0.serialize(var_1)
    var_4 = module_1.system_alias(var_1, var_1, var_1)
    var_2.validate(var_3)

def test_case_16():
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
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 'test@example.com'
    var_4 = var_0.serialize(var_3)
    assert var_4 == 'test@example.com'
    var_5 = 'user.name+tag@example.org'
    var_6 = var_0.serialize(var_5)
    assert var_6 == 'user.name+tag@example.org'

@pytest.mark.xfail(strict=True)
def test_case_17():
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
    var_1 = '2023-05-25T12:34:56.789123+02:00'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_0.TimeFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_4 = module_0.EmailFormat()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_5 = '25:00'
    var_6 = None
    var_7 = var_0.serialize(var_6)
    var_8 = var_4.is_native_type(var_7)
    assert var_8 is False
    var_9 = module_1.python_revision()
    assert var_9 == ''
    var_3.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_18():
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
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    var_3 = None
    var_4 = var_0.serialize(var_3)
    var_5 = None
    var_0.validate(var_5)

def test_case_19():
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
    assert var_2 is None

@pytest.mark.xfail(strict=True)
def test_case_20():
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
    var_1 = '12:34:56.789'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.hour).__module__}.{type(module_3.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.minute).__module__}.{type(module_3.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.second).__module__}.{type(module_3.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.microsecond).__module__}.{type(module_3.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.tzinfo).__module__}.{type(module_3.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.fold).__module__}.{type(module_3.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.min).__module__}.{type(module_3.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.max).__module__}.{type(module_3.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.resolution).__module__}.{type(module_3.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_0.TimeFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_4 = module_0.TimeFormat()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.TimeFormat'
    var_5 = '25:00'
    var_4.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
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
    var_1 = '25:00'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_1.machine()
    assert var_0 == 'x86_64'
    var_1 = module_0.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.hour).__module__}.{type(module_3.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.minute).__module__}.{type(module_3.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.second).__module__}.{type(module_3.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.microsecond).__module__}.{type(module_3.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.tzinfo).__module__}.{type(module_3.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.fold).__module__}.{type(module_3.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.time.min).__module__}.{type(module_3.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.max).__module__}.{type(module_3.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_3.time.resolution).__module__}.{type(module_3.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '12:34:00'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_4 = module_0.TimeFormat()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.TimeFormat'
    var_5 = '12:34:56'
    var_6 = var_4.validate(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.time'
    var_7 = module_0.TimeFormat()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.formats.TimeFormat'
    var_8 = '12:34:56.789'
    var_9 = var_7.validate(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.time'
    var_10 = module_0.TimeFormat()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.formats.TimeFormat'
    var_11 = '12:34:56.789123'
    var_12 = var_10.validate(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.time'
    var_13 = module_0.TimeFormat()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.formats.TimeFormat'
    var_14 = module_0.TimeFormat()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.formats.TimeFormat'
    var_15 = '12:60'
    var_14.validate(var_15)

@pytest.mark.xfail(strict=True)
def test_case_24():
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
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
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
    var_1 = module_1.node()
    assert var_1 == '1f6dba324bc9'
    var_0.validate(var_1)

def test_case_26():
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

def test_case_27():
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
    var_1 = '2023-05-25T12:34:56.789123+02:00'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'

@pytest.mark.xfail(strict=True)
def test_case_28():
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
    var_1 = 5
    var_2 = 30
    var_3 = module_3.timedelta()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.timedelta'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_3.timedelta.days).__module__}.{type(module_3.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.timedelta.seconds).__module__}.{type(module_3.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.timedelta.microseconds).__module__}.{type(module_3.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.timedelta.resolution).__module__}.{type(module_3.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_3.timedelta.min).__module__}.{type(module_3.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_3.timedelta.max).__module__}.{type(module_3.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_4 = '2023-01-01T12:00:00-0300'
    var_5 = var_0.validate(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_6 = '2023-01-01 12:00:00'
    var_7 = var_0.validate(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.datetime'
    var_8 = '2023-01-01T12:00:00.123456Z'
    var_9 = var_0.validate(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.datetime'
    var_10 = '2023-02-30T12:00:00Z'
    var_0.validate(var_10)

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_1 = '2023-05-25T12:34:56.789123+02:00'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = module_1.node()
    assert var_3 == '1f6dba324bc9'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_0.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_30():
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
    var_1 = '2023-01-01T12:00:00-0300'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_2.validate(var_2)

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
    var_1 = '2023-01-01T12:00:00Z'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = '2023-01-01 12:00:00+05:30'
    var_4 = var_0.serialize(var_2)
    assert var_4 == '2023-01-01T12:00:00Z'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = var_0.validate(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'datetime.datetime'
    var_6 = 30
    var_7 = module_3.timedelta()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_3.timedelta.days).__module__}.{type(module_3.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.timedelta.seconds).__module__}.{type(module_3.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.timedelta.microseconds).__module__}.{type(module_3.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.timedelta.resolution).__module__}.{type(module_3.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_3.timedelta.min).__module__}.{type(module_3.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_3.timedelta.max).__module__}.{type(module_3.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_8 = '2023-01-01T12:00:00-0300'
    var_9 = var_0.validate(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.datetime'
    var_10 = -3
    var_11 = module_3.timedelta()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.timedelta'
    var_7.setdefault(var_4)

@pytest.mark.xfail(strict=True)
def test_case_32():
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
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_0.validate(var_1)

def test_case_33():
    var_0 = '192.168.1.1'
    var_1 = module_0.IPAddressFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_2 = '0.0.0.0'
    var_3 = var_1.validate(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ipaddress.IPv4Address'
    assert f'{type(module_2.IPv4Address.packed).__module__}.{type(module_2.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_reserved).__module__}.{type(module_2.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_private).__module__}.{type(module_2.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_global).__module__}.{type(module_2.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_multicast).__module__}.{type(module_2.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_unspecified).__module__}.{type(module_2.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_loopback).__module__}.{type(module_2.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_link_local).__module__}.{type(module_2.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_4 = module_2.IPv4Address(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ipaddress.IPv4Address'
    assert module_2.IPV4LENGTH == 32
    assert module_2.IPV6LENGTH == 128
    var_5 = module_0.IPAddressFormat()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    var_6 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_7 = var_5.validate(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ipaddress.IPv6Address'
    assert f'{type(module_2.IPv6Address.scope_id).__module__}.{type(module_2.IPv6Address.scope_id).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.packed).__module__}.{type(module_2.IPv6Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_multicast).__module__}.{type(module_2.IPv6Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_reserved).__module__}.{type(module_2.IPv6Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_link_local).__module__}.{type(module_2.IPv6Address.is_link_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_site_local).__module__}.{type(module_2.IPv6Address.is_site_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_private).__module__}.{type(module_2.IPv6Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_global).__module__}.{type(module_2.IPv6Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_unspecified).__module__}.{type(module_2.IPv6Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_loopback).__module__}.{type(module_2.IPv6Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.ipv4_mapped).__module__}.{type(module_2.IPv6Address.ipv4_mapped).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.teredo).__module__}.{type(module_2.IPv6Address.teredo).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.sixtofour).__module__}.{type(module_2.IPv6Address.sixtofour).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_34():
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
    var_1 = '0b0.0.0/'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_1 = '0.0.0.0G'
    var_0.validate(var_1)

def test_case_36():
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
    var_2 = module_2.IPv4Address(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ipaddress.IPv4Address'
    assert module_2.IPV4LENGTH == 32
    assert module_2.IPV6LENGTH == 128
    assert f'{type(module_2.IPv4Address.packed).__module__}.{type(module_2.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_reserved).__module__}.{type(module_2.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_private).__module__}.{type(module_2.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_global).__module__}.{type(module_2.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_multicast).__module__}.{type(module_2.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_unspecified).__module__}.{type(module_2.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_loopback).__module__}.{type(module_2.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_link_local).__module__}.{type(module_2.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '192.168.1.1'
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = module_2.IPv6Address(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ipaddress.IPv6Address'
    assert f'{type(module_2.IPv6Address.scope_id).__module__}.{type(module_2.IPv6Address.scope_id).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.packed).__module__}.{type(module_2.IPv6Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_multicast).__module__}.{type(module_2.IPv6Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_reserved).__module__}.{type(module_2.IPv6Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_link_local).__module__}.{type(module_2.IPv6Address.is_link_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_site_local).__module__}.{type(module_2.IPv6Address.is_site_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_private).__module__}.{type(module_2.IPv6Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_global).__module__}.{type(module_2.IPv6Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_unspecified).__module__}.{type(module_2.IPv6Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_loopback).__module__}.{type(module_2.IPv6Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.ipv4_mapped).__module__}.{type(module_2.IPv6Address.ipv4_mapped).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.teredo).__module__}.{type(module_2.IPv6Address.teredo).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.sixtofour).__module__}.{type(module_2.IPv6Address.sixtofour).__qualname__}' == 'builtins.property'
    var_6 = var_0.serialize(var_5)
    assert var_6 == '2001:db8:85a3::8a2e:370:7334'
    var_7 = None
    var_8 = var_0.serialize(var_7)

def test_case_37():
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
    var_2 = module_2.IPv4Address(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ipaddress.IPv4Address'
    assert module_2.IPV4LENGTH == 32
    assert module_2.IPV6LENGTH == 128
    assert f'{type(module_2.IPv4Address.packed).__module__}.{type(module_2.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_reserved).__module__}.{type(module_2.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_private).__module__}.{type(module_2.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_global).__module__}.{type(module_2.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_multicast).__module__}.{type(module_2.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_unspecified).__module__}.{type(module_2.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_loopback).__module__}.{type(module_2.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_link_local).__module__}.{type(module_2.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '192.168.1.1'
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = module_2.IPv6Address(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ipaddress.IPv6Address'
    assert f'{type(module_2.IPv6Address.scope_id).__module__}.{type(module_2.IPv6Address.scope_id).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.packed).__module__}.{type(module_2.IPv6Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_multicast).__module__}.{type(module_2.IPv6Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_reserved).__module__}.{type(module_2.IPv6Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_link_local).__module__}.{type(module_2.IPv6Address.is_link_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_site_local).__module__}.{type(module_2.IPv6Address.is_site_local).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_private).__module__}.{type(module_2.IPv6Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_global).__module__}.{type(module_2.IPv6Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_unspecified).__module__}.{type(module_2.IPv6Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.is_loopback).__module__}.{type(module_2.IPv6Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.ipv4_mapped).__module__}.{type(module_2.IPv6Address.ipv4_mapped).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.teredo).__module__}.{type(module_2.IPv6Address.teredo).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv6Address.sixtofour).__module__}.{type(module_2.IPv6Address.sixtofour).__qualname__}' == 'builtins.property'
    with pytest.raises(AssertionError):
        var_0.serialize(var_3)

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
    var_1 = '2023-05-25T12:34:56.789123+02:00'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_3 = var_0.serialize(var_2)
    assert var_3 == '2023-05-25T12:34:56.789123+02:00'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'

@pytest.mark.xfail(strict=True)
def test_case_39():
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
    var_1.is_native_type(var_0)

def test_case_40():
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
def test_case_41():
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
    var_1 = '2023-1-0'
    var_0.validate(var_1)

def test_case_42():
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
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test@example.com'
    var_3 = 'user.name+tag@example.org'
    var_4 = None
    var_5 = var_0.serialize(var_4)
    var_6 = var_0.serialize(var_4)
    var_7 = var_0.serialize(var_3)
    assert var_7 == 'user.name+tag@example.org'

@pytest.mark.xfail(strict=True)
def test_case_43():
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
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = "2'fgJ"
    var_0.validate(var_3)

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
    var_1 = '2023-05-25T1:34:56.789123+02:00'
    var_2 = module_0.BaseFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert module_0.BaseFormat.errors == {}
    var_3 = var_0.validate(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.hour).__module__}.{type(module_3.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.minute).__module__}.{type(module_3.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.second).__module__}.{type(module_3.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.microsecond).__module__}.{type(module_3.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.tzinfo).__module__}.{type(module_3.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.fold).__module__}.{type(module_3.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.datetime.min).__module__}.{type(module_3.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.max).__module__}.{type(module_3.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_3.datetime.resolution).__module__}.{type(module_3.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_4 = var_0.serialize(var_3)
    assert var_4 == '2023-05-25T01:34:56.789123+02:00'
    assert module_3.MINYEAR == 1
    assert module_3.MAXYEAR == 9999
    assert f'{type(module_3.datetime_CAPI).__module__}.{type(module_3.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_5 = module_0.DateFormat()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_6 = var_5.serialize(var_3)
    assert var_6 == '2023-05-25T01:34:56.789123+02:00'
    var_7 = None
    with pytest.raises(TypeError):
        var_5.__setattr__(var_7, var_7)

@pytest.mark.xfail(strict=True)
def test_case_45():
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
    var_1 = '2023-95-25T12:34:56.789123+02:00'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_46():
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
    var_1 = '123e4567-e89b-12d3-a456-426614174000'
    var_2 = var_0.validate(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.UUID.bytes).__module__}.{type(module_4.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.bytes_le).__module__}.{type(module_4.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.fields).__module__}.{type(module_4.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time_low).__module__}.{type(module_4.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time_mid).__module__}.{type(module_4.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time_hi_version).__module__}.{type(module_4.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.clock_seq_hi_variant).__module__}.{type(module_4.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.clock_seq_low).__module__}.{type(module_4.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.time).__module__}.{type(module_4.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.clock_seq).__module__}.{type(module_4.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.node).__module__}.{type(module_4.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.hex).__module__}.{type(module_4.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.urn).__module__}.{type(module_4.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.variant).__module__}.{type(module_4.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.version).__module__}.{type(module_4.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.UUID.int).__module__}.{type(module_4.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.UUID.is_safe).__module__}.{type(module_4.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_4.UUID(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'uuid.UUID'
    assert module_4.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_4.RFC_4122 == 'specified in RFC 4122'
    assert module_4.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_4.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_4.NAMESPACE_DNS).__module__}.{type(module_4.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.NAMESPACE_URL).__module__}.{type(module_4.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.NAMESPACE_OID).__module__}.{type(module_4.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_4.NAMESPACE_X500).__module__}.{type(module_4.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_4 = 'not-a-uuid'
    var_0.validate(var_4)

@pytest.mark.xfail(strict=True)
def test_case_47():
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
    var_3 = None
    var_4 = None
    var_5 = module_1.win32_ver(var_4, csd=var_4)
    var_5.__getitem__(var_3)