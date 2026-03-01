# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import platform as module_0
import typesystem.formats as module_1
import ipaddress as module_2
import uuid as module_3
import datetime as module_4
import enum as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.node()
    assert var_0 == 'baf607c0cad0'
    var_1 = module_1.URLFormat()
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

@pytest.mark.xfail(strict=True)
def test_case_2():
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
    var_1 = 'bh\\i>\\R#7wvxv?<<M'
    var_0.validation_error(var_1)

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
    var_1 = None
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    var_0.serialize(var_2)

def test_case_5():
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
    var_3 = var_0.is_native_type(var_1)
    assert var_3 is False

@pytest.mark.xfail(strict=True)
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
    var_1 = None
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    var_0.validate(var_1)

def test_case_7():
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
    var_2 = var_0.is_native_type(var_1)
    assert var_2 is False
    with pytest.raises(AssertionError):
        var_0.serialize(var_2)

@pytest.mark.xfail(strict=True)
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
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
    var_1 = []
    var_2 = {}
    var_3 = module_1.IPAddressFormat(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.IPAddressFormat'
    assert module_1.IPAddressFormat.errors == {'format': 'Must be a valid IP format.', 'invalid': 'Must be a real IP.'}
    var_4 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_5 = None
    var_6 = var_0.serialize(var_5)
    var_7 = var_3.validate(var_4)
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
    var_8 = module_1.DateTimeFormat(*var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_2.IPV4LENGTH == 32
    assert module_2.IPV6LENGTH == 128
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_9 = None
    var_10 = var_3.serialize(var_9)
    var_11 = var_7.__add__(var_9)
    var_11.__iter__()

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
    var_1 = 'q!L:/y9vUZD=]'
    var_0.validate(var_1)

def test_case_11():
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

@pytest.mark.xfail(strict=True)
def test_case_12():
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
    var_0.serialize(var_0)

def test_case_13():
    var_0 = None
    var_1 = module_1.EmailFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_2 = var_1.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'F!E:y9vUP=]'
    var_1 = module_1.DateTimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
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
    var_0.serialize(var_0)

@pytest.mark.xfail(strict=True)
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
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_0.serialize(var_3)

def test_case_19():
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
    var_1 = 'Op&WDt:T]'
    var_2 = var_0.serialize(var_1)
    assert var_2 == 'Op&WDt:T]'
    with pytest.raises(TypeError):
        module_3.UUID(bytes_le=var_2, fields=var_2, version=var_2)

def test_case_20():
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

def test_case_21():
    var_0 = module_3.getnode()
    assert var_0 == 248585874689897
    assert module_3.RESERVED_NCS == 'reserved for NCS compatibility'
    assert module_3.RFC_4122 == 'specified in RFC 4122'
    assert module_3.RESERVED_MICROSOFT == 'reserved for Microsoft compatibility'
    assert module_3.RESERVED_FUTURE == 'reserved for future definition'
    assert f'{type(module_3.NAMESPACE_DNS).__module__}.{type(module_3.NAMESPACE_DNS).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_URL).__module__}.{type(module_3.NAMESPACE_URL).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_OID).__module__}.{type(module_3.NAMESPACE_OID).__qualname__}' == 'uuid.UUID'
    assert f'{type(module_3.NAMESPACE_X500).__module__}.{type(module_3.NAMESPACE_X500).__qualname__}' == 'uuid.UUID'
    var_1 = None
    var_2 = module_1.TimeFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_3 = var_2.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'ns'
    var_1 = module_1.TimeFormat()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = None
    var_3 = module_1.DateTimeFormat()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_1.DateTimeFormat.errors == {'format': 'Must be a valid datetime format.', 'invalid': 'Must be a real datetime.'}
    var_4 = var_1.serialize(var_2)
    var_5 = None
    var_6 = var_1.serialize(var_5)
    var_7 = var_1.is_native_type(var_5)
    assert var_7 is False
    var_8 = var_1.serialize(var_5)
    var_9 = {var_0: var_5, var_0: var_5, var_0: var_5, var_0: var_5}
    module_1.TimeFormat(**var_9)

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_1 = module_0.python_branch()
    assert var_1 == ''
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
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

def test_case_25():
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
    var_1 = module_0.win32_edition()
    var_2 = module_1.BaseFormat()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.BaseFormat'
    assert module_1.BaseFormat.errors == {}
    with pytest.raises(NotImplementedError):
        var_2.serialize(var_1)

def test_case_26():
    var_0 = None
    var_1 = []
    var_2 = module_1.EmailFormat(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_3 = var_2.is_native_type(var_0)
    assert var_3 is False
    var_4 = var_2.is_native_type(var_0)
    assert var_4 is False
    var_5 = var_2.serialize(var_0)

def test_case_27():
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

def test_case_28():
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
    var_3 = '2023-2-5'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'datetime.date'
    assert f'{type(module_4.date.year).__module__}.{type(module_4.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.month).__module__}.{type(module_4.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.day).__module__}.{type(module_4.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.min).__module__}.{type(module_4.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_4.date.max).__module__}.{type(module_4.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_4.date.resolution).__module__}.{type(module_4.date.resolution).__qualname__}' == 'datetime.timedelta'

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_2 = '2023-13-01'
    var_1.validate(var_2)

def test_case_30():
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
    var_6 = module_4.date(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.date'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.date.year).__module__}.{type(module_4.date.year).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.month).__module__}.{type(module_4.date.month).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.day).__module__}.{type(module_4.date.day).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.date.min).__module__}.{type(module_4.date.min).__qualname__}' == 'datetime.date'
    assert f'{type(module_4.date.max).__module__}.{type(module_4.date.max).__qualname__}' == 'datetime.date'
    assert f'{type(module_4.date.resolution).__module__}.{type(module_4.date.resolution).__qualname__}' == 'datetime.timedelta'
    var_7 = var_2.serialize(var_6)
    assert var_7 == '0001-01-01'

def test_case_31():
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
    var_6 = module_4.time(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.time'
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
    var_7 = var_2.serialize(var_6)
    assert var_7 == '00:00:00'

def test_case_32():
    var_0 = []
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
    assert f'{type(module_2.IPv4Address.packed).__module__}.{type(module_2.IPv4Address.packed).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_reserved).__module__}.{type(module_2.IPv4Address.is_reserved).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_private).__module__}.{type(module_2.IPv4Address.is_private).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_global).__module__}.{type(module_2.IPv4Address.is_global).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_multicast).__module__}.{type(module_2.IPv4Address.is_multicast).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_unspecified).__module__}.{type(module_2.IPv4Address.is_unspecified).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_loopback).__module__}.{type(module_2.IPv4Address.is_loopback).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.IPv4Address.is_link_local).__module__}.{type(module_2.IPv4Address.is_link_local).__qualname__}' == 'builtins.property'
    var_5 = str(var_4)
    assert var_5 == '192.168.1.1'

def test_case_33():
    var_0 = []
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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ipaddress.IPv6Address'
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
    var_0 = {}
    var_1 = module_1.TimeFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.TimeFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.TimeFormat.errors == {'format': 'Must be a valid time format.', 'invalid': 'Must be a real time.'}
    var_2 = '12:34:60'
    var_1.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_3 = 'invalid'
    var_2.validate(var_3)

def test_case_36():
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
    var_3 = '12:34:56.1234567'
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
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123456
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_4.time(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.time'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

def test_case_37():
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
    var_3 = module_0.release()
    assert var_3 == '6.17.9-76061709-generic'
    with pytest.raises(AssertionError):
        var_2.serialize(var_3)
    assert var_4 == '00:00:00'

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = []
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
    var_3 = module_1.EmailFormat(**var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_4 = '#(VQ$s'
    var_5 = var_3.serialize(var_4)
    assert var_5 == '#(VQ$s'
    var_2.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = []
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
    var_3 = module_1.EmailFormat(**var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.formats.EmailFormat'
    assert module_1.EmailFormat.errors == {'format': 'Must be a valid email format.'}
    var_4 = '#(VQ$s'
    var_5 = var_3.serialize(var_4)
    assert var_5 == '#(VQ$s'
    var_3.validate(var_5)

def test_case_40():
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
    var_3 = 'ABCDEFAB-1234-5678-9ABC-DEF123456789'
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
    var_5 = var_2.serialize(var_4)
    assert var_5 == 'abcdefab-1234-5678-9abc-def123456789'

def test_case_41():
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
    var_3 = 'http://example.com?query=value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'http://example.com?query=value'

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_5._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
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
    var_2 = '2023-02-30T14:30:45'
    var_1.validate(var_2)

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
    var_3 = '2023-01-15T14:30:00.123'
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
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 0
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10, var_10]
    var_12 = {}
    var_13 = module_4.datetime(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'datetime.datetime'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_14 = bool(var_4 == var_13)

def test_case_44():
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
    var_3 = '2023-01-15T14:30:00Z'
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

def test_case_45():
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
    var_3 = '2023-04-15T12:30:45+05'
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
    var_5 = var_4.tzinfo
    var_6 = bool(var_4.tzinfo is not None)
    assert var_6 is True
    var_7 = 5
    var_8 = []
    var_9 = 'hours'
    var_10 = {var_9: var_7}
    var_11 = module_4.timedelta(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.timedelta'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.timedelta.days).__module__}.{type(module_4.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.seconds).__module__}.{type(module_4.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.microseconds).__module__}.{type(module_4.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.resolution).__module__}.{type(module_4.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.min).__module__}.{type(module_4.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.max).__module__}.{type(module_4.timedelta.max).__qualname__}' == 'datetime.timedelta'

def test_case_46():
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
    var_3 = '2023-04-15T12:30:45-08:00'
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
    var_5 = var_4.tzinfo
    var_6 = bool(var_4.tzinfo is not None)
    assert var_6 is True
    var_7 = -8
    var_8 = []
    var_9 = 'hours'
    var_10 = {var_9: var_7}
    var_11 = module_4.timedelta(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'datetime.timedelta'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.timedelta.days).__module__}.{type(module_4.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.seconds).__module__}.{type(module_4.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.microseconds).__module__}.{type(module_4.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.resolution).__module__}.{type(module_4.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.min).__module__}.{type(module_4.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.max).__module__}.{type(module_4.timedelta.max).__qualname__}' == 'datetime.timedelta'

def test_case_47():
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
    var_3 = '2023-12-31T23:59:59.999999-11:30'
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
    var_5 = module_1.DateTimeFormat(**var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.formats.DateTimeFormat'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    var_6 = var_2.serialize(var_4)
    assert var_6 == '2023-12-31T23:59:59.999999-11:30'

def test_case_48():
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
    var_3 = 5
    var_4 = 30
    var_5 = []
    var_6 = module_4.timedelta(*var_5, **var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'datetime.timedelta'
    assert module_4.MINYEAR == 1
    assert module_4.MAXYEAR == 9999
    assert f'{type(module_4.datetime_CAPI).__module__}.{type(module_4.datetime_CAPI).__qualname__}' == 'builtins.PyCapsule'
    assert f'{type(module_4.timedelta.days).__module__}.{type(module_4.timedelta.days).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.seconds).__module__}.{type(module_4.timedelta.seconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.microseconds).__module__}.{type(module_4.timedelta.microseconds).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_4.timedelta.resolution).__module__}.{type(module_4.timedelta.resolution).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.min).__module__}.{type(module_4.timedelta.min).__qualname__}' == 'datetime.timedelta'
    assert f'{type(module_4.timedelta.max).__module__}.{type(module_4.timedelta.max).__qualname__}' == 'datetime.timedelta'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_4.timezone(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.utc).__module__}.{type(module_4.timezone.utc).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.min).__module__}.{type(module_4.timezone.min).__qualname__}' == 'datetime.timezone'
    assert f'{type(module_4.timezone.max).__module__}.{type(module_4.timezone.max).__qualname__}' == 'datetime.timezone'
    var_10 = 2023
    var_11 = 17
    var_12 = 14
    var_13 = 45
    var_14 = 123456
    var_15 = [var_10, var_3, var_11, var_12, var_4, var_13, var_14]
    var_16 = 'tzinfo'
    var_17 = {var_16: var_9}
    var_18 = module_4.datetime(*var_15, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.hour).__module__}.{type(module_4.datetime.hour).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.minute).__module__}.{type(module_4.datetime.minute).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.second).__module__}.{type(module_4.datetime.second).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.microsecond).__module__}.{type(module_4.datetime.microsecond).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.tzinfo).__module__}.{type(module_4.datetime.tzinfo).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.fold).__module__}.{type(module_4.datetime.fold).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_4.datetime.min).__module__}.{type(module_4.datetime.min).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.max).__module__}.{type(module_4.datetime.max).__qualname__}' == 'datetime.datetime'
    assert f'{type(module_4.datetime.resolution).__module__}.{type(module_4.datetime.resolution).__qualname__}' == 'datetime.timedelta'
    var_19 = var_2.serialize(var_18)
    assert var_19 == '2023-05-17T14:30:45.123456Z'
    var_20 = '2023-05-17T14:30:45.123456+05:30'
    var_21 = bool(var_19 == var_20)

def test_case_49():
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
    var_3 = 'first.last@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'first.last@example.com'

def test_case_50():
    var_0 = '192.168.1.1'
    var_1 = module_2.IPv4Address(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ipaddress.IPv4Address'
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
    assert var_5 == '192.168.1.1'
    var_6 = '192.168.1.1'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

@pytest.mark.xfail(strict=True)
def test_case_51():
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
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_2.validate(var_3)

def test_case_52():
    var_0 = {}
    var_1 = module_1.UUIDFormat(*var_0, **var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.formats.UUIDFormat'
    assert f'{type(module_1.DATE_REGEX).__module__}.{type(module_1.DATE_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.TIME_REGEX).__module__}.{type(module_1.TIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.DATETIME_REGEX).__module__}.{type(module_1.DATETIME_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.UUID_REGEX).__module__}.{type(module_1.UUID_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.EMAIL_REGEX).__module__}.{type(module_1.EMAIL_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV4_REGEX).__module__}.{type(module_1.IPV4_REGEX).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.IPV6_REGEX).__module__}.{type(module_1.IPV6_REGEX).__qualname__}' == 're.Pattern'
    assert module_1.UUIDFormat.errors == {'format': 'Must be a valid UUID format.'}
    var_2 = 'c232ab00-9414-11ec-b3c8-9a6bdfc4b925'
    var_3 = var_1.validate(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'uuid.UUID'
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
    var_4 = str(var_3)
    assert var_4 == 'c232ab00-9414-11ec-b3c8-9a6bdfc4b925'