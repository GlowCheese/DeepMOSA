# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.formats as module_0
import uuid as module_1
import platform as module_2
import re as module_3
import ipaddress as module_4
import datetime as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
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

def test_case_1():
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
    var_1 = 'bh\\i>\\R#7wvxv?<<M'
    var_0.validation_error(var_1)

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
    with pytest.raises(AssertionError):
        var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
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
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = 'phpJUED`,W'
    var_0.validate(var_3)

def test_case_5():
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

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
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
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_3 = var_2.is_native_type(var_2)
    assert var_3 is False
    var_4 = '12345678123456781234567812345678'
    var_2.validate(var_4)

def test_case_9():
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
    var_1 = module_1.getnode()
    assert var_1 == 16387873806476
    assert module_1.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_1.RFC_4122 == 'specified in RFC 4122'
    assert module_1.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_1.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_1.NAMESPACE_DNS).__module__}.{type(module_1.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.NAMESPACE_URL).__module__}.{type(module_1.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.NAMESPACE_OID).__module__}.{type(module_1.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.NAMESPACE_X500).__module__}.{type(module_1.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    with pytest.raises(NotImplementedError):
        var_0.is_native_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_1 = var_0.is_native_type(var_0)
    assert var_1 is False
    var_2 = '2023-05-0'
    var_0.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    var_3 = '2023-05-0'
    var_0.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
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
    var_1 = 'Y2823-5-0'
    var_0.validate(var_1)

def test_case_13():
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
    var_3 = var_1.__le__(var_0)
    with pytest.raises(AssertionError):
        var_1.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
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
    var_3 = module_2.version()
    assert var_3 == '#202511241048~1764607909~24.04~df6b2b6 SMP PREEMPT_DYNAMIC Mon D'
    var_0.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_0.serialize(var_1)

def test_case_16():
    var_0 = {}
    var_1 = module_0.BaseFormat(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.BaseFormat.errors == {}
    var_2 = module_0.BaseFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.BaseFormat'
    var_3 = module_0.DateTimeFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_4 = None
    with pytest.raises(NotImplementedError):
        var_1.serialize(var_4)

@pytest.mark.xfail(strict=True)
def test_case_17():
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
    var_3 = None
    var_4 = module_0.BaseFormat()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert module_0.BaseFormat.errors == {}
    module_3.escape(var_3)

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
    var_1 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_2 = module_0.EmailFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = None
    var_4 = var_2.is_native_type(var_3)
    assert var_4 is False
    var_0.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
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
    var_1 = 'h#jN;R,KX,$Bp.='
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_20():
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

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = module_0.IPAddressFormat(*var_0, **var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_4 = module_0.TimeFormat()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_5 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_6 = module_0.DateFormat(*var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_7 = var_3.validate(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ipaddress.IPv6Address'
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
    var_8 = module_0.EmailFormat(*var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_4.IPV4LENGTH == 32
    assert module_4.IPV6LENGTH == 128
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_9 = var_8.serialize(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.formats.DateFormat'
    var_2.serialize(var_3)

@pytest.mark.xfail(strict=True)
def test_case_22():
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
def test_case_23():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_3 = module_0.URLFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.URLFormat'
    assert module_0.URLFormat.errors == {'invalid': 'Must be a real URL.'}
    var_4 = None
    var_5 = var_2.serialize(var_4)
    var_6 = var_2.serialize(var_4)
    var_7 = module_0.TimeFormat()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_8 = None
    var_9 = var_7.serialize(var_8)
    var_10 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_11 = module_0.DateFormat(*var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.formats.DateFormat'
    assert module_0.DateFormat.errors == {'format': 'Must be a valid date format.', 'invalid': 'Must be a real date.'}
    var_12 = var_3.serialize(var_10)
    assert var_12 == '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_2.validate(var_10)

@pytest.mark.xfail(strict=True)
def test_case_24():
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
    var_2 = var_0.is_native_type(var_0)
    assert var_2 is False
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
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
    var_2.serialize(var_2)

def test_case_26():
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
    var_2 = var_1.__le__(var_0)
    with pytest.raises(AssertionError):
        var_1.serialize(var_1)

@pytest.mark.xfail(strict=True)
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
    var_1 = module_2.python_revision()
    assert var_1 == ''
    var_0.validate(var_1)

def test_case_28():
    var_0 = None
    var_1 = []
    var_2 = {}
    var_3 = module_0.IPAddressFormat(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_4 = var_3.is_native_type(var_0)
    assert var_4 is False

def test_case_29():
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
    var_3 = 'user@sub.example.co.uk'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user@sub.example.co.uk'

@pytest.mark.xfail(strict=True)
def test_case_30():
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
    var_1 = 'p}pJ*ED`,sWb'
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_31():
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
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
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
    var_1 = '2023-05-00'
    var_0.validate(var_1)

def test_case_33():
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
    var_3 = 1
    var_4 = [var_3, var_3, var_3]
    var_5 = {}
    var_6 = module_5.date(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.date'
    assert module_5.MINYEAR == 1
    assert module_5.MAXYEAR == 9999
    assert f'{type(module_5.datetime_CAPI).__module__}.{type(module_5.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_5.date.year).__module__}.{type(module_5.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.date.month).__module__}.{type(module_5.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.date.day).__module__}.{type(module_5.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.date.min).__module__}.{type(module_5.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_5.date.max).__module__}.{type(module_5.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_5.date.resolution).__module__}.{type(module_5.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_7 = var_2.serialize(var_6)
    assert var_7 == '0001-01-01'

def test_case_34():
    var_0 = 'ABCDEFAB-1234-5678-9ABC-DEF123456789'
    var_1 = module_1.UUID(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'uuid.UUID'
    assert module_1.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_1.RFC_4122 == 'specified in RFC 4122'
    assert module_1.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_1.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_1.NAMESPACE_DNS).__module__}.{type(module_1.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.NAMESPACE_URL).__module__}.{type(module_1.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.NAMESPACE_OID).__module__}.{type(module_1.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.NAMESPACE_X500).__module__}.{type(module_1.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.UUID.bytes).__module__}.{type(module_1.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.bytes_le).__module__}.{type(module_1.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.fields).__module__}.{type(module_1.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time_low).__module__}.{type(module_1.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time_mid).__module__}.{type(module_1.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time_hi_version).__module__}.{type(module_1.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.clock_seq_hi_variant).__module__}.{type(module_1.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.clock_seq_low).__module__}.{type(module_1.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time).__module__}.{type(module_1.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.clock_seq).__module__}.{type(module_1.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.node).__module__}.{type(module_1.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.hex).__module__}.{type(module_1.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.urn).__module__}.{type(module_1.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.variant).__module__}.{type(module_1.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.version).__module__}.{type(module_1.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.int).__module__}.{type(module_1.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.UUID.is_safe).__module__}.{type(module_1.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
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
    assert var_5 == 'abcdefab-1234-5678-9abc-def123456789'
    var_6 = 'abcdefab-1234-678-9abc-def123456789'
    var_7 = bool(var_5 == var_6)

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
    var_1 = 'S[V`pG'
    var_0.validate(var_1)

def test_case_36():
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
    var_3 = 'https://example.com?query=value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com?query=value'

def test_case_37():
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

def test_case_38():
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
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = {}
    var_1 = module_0.IPAddressFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_2 = '2001:0db8:85a3:0000:000:a2e:0370:733m'
    var_1.validate(var_2)

def test_case_40():
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
    var_3 = module_0.TimeFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = module_0.EmailFormat()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_0.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_6 = var_2.validate(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ipaddress.IPv6Address'
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
    var_7 = str(var_6)
    assert var_7 == '2001:db8:85a3::8a2e:370:7334'
    var_8 = var_5.is_native_type(var_7)
    assert var_8 is False
    assert module_4.IPV4LENGTH == 32
    assert module_4.IPV6LENGTH == 128
    var_9 = var_3.is_native_type(var_7)
    assert var_9 is False
    var_10 = var_2.serialize(var_6)
    assert var_10 == '2001:db8:85a3::8a2e:370:7334'

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = {}
    var_1 = module_0.TimeFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = '12:30:60'
    var_1.validate(var_2)

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
    var_3 = '12:34:56.1000000'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.time'
    assert f'{type(module_5.time.hour).__module__}.{type(module_5.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.minute).__module__}.{type(module_5.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.second).__module__}.{type(module_5.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.microsecond).__module__}.{type(module_5.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.tzinfo).__module__}.{type(module_5.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.fold).__module__}.{type(module_5.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.min).__module__}.{type(module_5.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_5.time.max).__module__}.{type(module_5.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_5.time.resolution).__module__}.{type(module_5.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = bool(False)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = {}
    var_1 = module_0.DateTimeFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_2 = '2023-02-30T12:3:4'
    var_1.validate(var_2)

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
    var_3 = '2023-04-01T12:30:45.123'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.hour).__module__}.{type(module_5.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.minute).__module__}.{type(module_5.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.second).__module__}.{type(module_5.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.microsecond).__module__}.{type(module_5.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.tzinfo).__module__}.{type(module_5.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.fold).__module__}.{type(module_5.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.min).__module__}.{type(module_5.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.max).__module__}.{type(module_5.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.resolution).__module__}.{type(module_5.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2023
    var_6 = 4
    var_7 = 1
    var_8 = 12
    var_9 = 30
    var_10 = 45
    var_11 = 123000
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_5.datetime(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'datetime.datetime'
    assert module_5.MINYEAR == 1
    assert module_5.MAXYEAR == 9999
    assert f'{type(module_5.datetime_CAPI).__module__}.{type(module_5.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

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
    var_3 = '2023-04-01T12:30:45Z'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.hour).__module__}.{type(module_5.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.minute).__module__}.{type(module_5.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.second).__module__}.{type(module_5.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.microsecond).__module__}.{type(module_5.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.tzinfo).__module__}.{type(module_5.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.fold).__module__}.{type(module_5.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.min).__module__}.{type(module_5.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.max).__module__}.{type(module_5.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.resolution).__module__}.{type(module_5.datetime.resolution).__qualname__}' == 'datetime.timedelta'

def test_case_46():
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
    var_3 = '2023-04-01T12:30:45+02'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.hour).__module__}.{type(module_5.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.minute).__module__}.{type(module_5.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.second).__module__}.{type(module_5.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.microsecond).__module__}.{type(module_5.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.tzinfo).__module__}.{type(module_5.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.fold).__module__}.{type(module_5.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.min).__module__}.{type(module_5.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.max).__module__}.{type(module_5.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.resolution).__module__}.{type(module_5.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = 2
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_5.timedelta(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.timedelta'
    assert module_5.MINYEAR == 1
    assert module_5.MAXYEAR == 9999
    assert f'{type(module_5.datetime_CAPI).__module__}.{type(module_5.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_5.timedelta.days).__module__}.{type(module_5.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.seconds).__module__}.{type(module_5.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.microseconds).__module__}.{type(module_5.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.resolution).__module__}.{type(module_5.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_5.timedelta.min).__module__}.{type(module_5.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_5.timedelta.max).__module__}.{type(module_5.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_5.timezone(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.utc).__module__}.{type(module_5.timezone.utc).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.min).__module__}.{type(module_5.timezone.min).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.max).__module__}.{type(module_5.timezone.max).__qualname__}' == 'datetime.timezone'
    var_13 = 2023
    var_14 = 4
    var_15 = 1
    var_16 = 12
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_5.datetime(*var_19, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'datetime.datetime'
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

def test_case_47():
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
    var_3 = '2023-04-01T12:30:45-08:00'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.hour).__module__}.{type(module_5.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.minute).__module__}.{type(module_5.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.second).__module__}.{type(module_5.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.microsecond).__module__}.{type(module_5.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.tzinfo).__module__}.{type(module_5.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.fold).__module__}.{type(module_5.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.min).__module__}.{type(module_5.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.max).__module__}.{type(module_5.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.resolution).__module__}.{type(module_5.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_5 = -8
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_5.timedelta(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.timedelta'
    assert module_5.MINYEAR == 1
    assert module_5.MAXYEAR == 9999
    assert f'{type(module_5.datetime_CAPI).__module__}.{type(module_5.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_5.timedelta.days).__module__}.{type(module_5.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.seconds).__module__}.{type(module_5.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.microseconds).__module__}.{type(module_5.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.resolution).__module__}.{type(module_5.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_5.timedelta.min).__module__}.{type(module_5.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_5.timedelta.max).__module__}.{type(module_5.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_5.timezone(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.utc).__module__}.{type(module_5.timezone.utc).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.min).__module__}.{type(module_5.timezone.min).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.max).__module__}.{type(module_5.timezone.max).__qualname__}' == 'datetime.timezone'
    var_13 = 2023
    var_14 = 4
    var_15 = 1
    var_16 = 12
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_5.datetime(*var_19, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'datetime.datetime'
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

def test_case_48():
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
    var_3 = 5
    var_4 = 17
    var_5 = 14
    var_6 = 30
    var_7 = 45
    var_8 = [var_4, var_3, var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_5.datetime(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.datetime'
    assert module_5.MINYEAR == 1
    assert module_5.MAXYEAR == 9999
    assert f'{type(module_5.datetime_CAPI).__module__}.{type(module_5.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_5.datetime.hour).__module__}.{type(module_5.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.minute).__module__}.{type(module_5.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.second).__module__}.{type(module_5.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.microsecond).__module__}.{type(module_5.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.tzinfo).__module__}.{type(module_5.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.fold).__module__}.{type(module_5.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.min).__module__}.{type(module_5.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.max).__module__}.{type(module_5.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.resolution).__module__}.{type(module_5.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_11 = var_2.serialize(var_10)
    assert var_11 == '0017-05-17T14:30:45'
    var_12 = '2023-05-17T14:30:45'
    var_13 = bool(var_11 == var_12)

def test_case_49():
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
    var_3 = 0
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_5.timedelta(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'datetime.timedelta'
    assert module_5.MINYEAR == 1
    assert module_5.MAXYEAR == 9999
    assert f'{type(module_5.datetime_CAPI).__module__}.{type(module_5.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_5.timedelta.days).__module__}.{type(module_5.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.seconds).__module__}.{type(module_5.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.microseconds).__module__}.{type(module_5.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_5.timedelta.resolution).__module__}.{type(module_5.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_5.timedelta.min).__module__}.{type(module_5.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_5.timedelta.max).__module__}.{type(module_5.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_5.timezone(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.utc).__module__}.{type(module_5.timezone.utc).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.min).__module__}.{type(module_5.timezone.min).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_5.timezone.max).__module__}.{type(module_5.timezone.max).__qualname__}' == 'datetime.timezone'
    var_11 = 2023
    var_12 = 5
    var_13 = 17
    var_14 = 14
    var_15 = 30
    var_16 = 45
    var_17 = 123456
    var_18 = [var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_10}
    var_21 = module_5.datetime(*var_18, **var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.hour).__module__}.{type(module_5.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.minute).__module__}.{type(module_5.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.second).__module__}.{type(module_5.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.microsecond).__module__}.{type(module_5.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.tzinfo).__module__}.{type(module_5.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.fold).__module__}.{type(module_5.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.datetime.min).__module__}.{type(module_5.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.max).__module__}.{type(module_5.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_5.datetime.resolution).__module__}.{type(module_5.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_22 = var_2.serialize(var_21)
    assert var_22 == '2023-05-17T14:30:45.123456Z'
    var_23 = '2023-05-17T14:30:45.123456Z'
    var_24 = bool(var_22 == var_23)
    assert var_24 is True

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = {}
    var_1 = module_0.UUIDFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_0.DATE_REGEX).__module__}.{type(module_0.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.TIME_REGEX).__module__}.{type(module_0.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.DATETIME_REGEX).__module__}.{type(module_0.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.UUID_REGEX).__module__}.{type(module_0.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.EMAIL_REGEX).__module__}.{type(module_0.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV4_REGEX).__module__}.{type(module_0.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.IPV6_REGEX).__module__}.{type(module_0.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_0.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = '{12345678-1234-5678-1234-567812345678}'
    var_1.validate(var_2)

def test_case_51():
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
    var_3 = '74738ff5-5367-5958-9aee-98fffdcd1876'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_1.UUID.bytes).__module__}.{type(module_1.UUID.bytes).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.bytes_le).__module__}.{type(module_1.UUID.bytes_le).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.fields).__module__}.{type(module_1.UUID.fields).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time_low).__module__}.{type(module_1.UUID.time_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time_mid).__module__}.{type(module_1.UUID.time_mid).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time_hi_version).__module__}.{type(module_1.UUID.time_hi_version).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.clock_seq_hi_variant).__module__}.{type(module_1.UUID.clock_seq_hi_variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.clock_seq_low).__module__}.{type(module_1.UUID.clock_seq_low).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.time).__module__}.{type(module_1.UUID.time).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.clock_seq).__module__}.{type(module_1.UUID.clock_seq).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.node).__module__}.{type(module_1.UUID.node).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.hex).__module__}.{type(module_1.UUID.hex).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.urn).__module__}.{type(module_1.UUID.urn).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.variant).__module__}.{type(module_1.UUID.variant).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.version).__module__}.{type(module_1.UUID.version).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.UUID.int).__module__}.{type(module_1.UUID.int).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.UUID.is_safe).__module__}.{type(module_1.UUID.is_safe).__qualname__}' == 'builtins.member_descriptor'
    var_5 = str(var_4)
    assert var_5 == '74738ff5-5367-5958-9aee-98fffdcd1876'

def test_case_52():
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
    var_3 = 0
    var_4 = [var_3, var_3, var_3]
    var_5 = {}
    var_6 = module_5.time(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.time'
    assert module_5.MINYEAR == 1
    assert module_5.MAXYEAR == 9999
    assert f'{type(module_5.datetime_CAPI).__module__}.{type(module_5.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_5.time.hour).__module__}.{type(module_5.time.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.minute).__module__}.{type(module_5.time.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.second).__module__}.{type(module_5.time.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.microsecond).__module__}.{type(module_5.time.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.tzinfo).__module__}.{type(module_5.time.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.fold).__module__}.{type(module_5.time.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_5.time.min).__module__}.{type(module_5.time.min).__qualname__}' == 'datetime.time'
    assert f'{type(module_5.time.max).__module__}.{type(module_5.time.max).__qualname__}' == 'datetime.time'
    assert f'{type(module_5.time.resolution).__module__}.{type(module_5.time.resolution).__qualname__}' == 'datetime.timedelta'
    var_7 = var_2.serialize(var_6)
    assert var_7 == '00:00:00'
    var_8 = '00:00:00'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True