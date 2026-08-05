####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 12
    var_4 = 30
    var_5 = 45
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '12:30:45'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 12
    var_4 = 30
    var_5 = 45
    var_6 = 123456
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '12:30:45.123456'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'not a time object'
    var_4 = var_2.serialize(var_3)
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.0.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.168.0.1'

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::1'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::ffff:192.168.0.1'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.168.0.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not an address object'
    var_4 = var_2.serialize(var_3)
    var_5 = 'Should have raised AssertionError for invalid type'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_valid_date. Retrieved 7/9 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-10-25'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 10
    var_7 = 25
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '25/10/2023'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid date format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 'not-a-date'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid date format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real date'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-13-01'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real date'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-32'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real date'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #4
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 30
    var_7 = []
    var_8 = 'hour'
    var_9 = 'minute'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.time(*var_7, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 30
    var_7 = 45
    var_8 = []
    var_9 = 'hour'
    var_10 = 'minute'
    var_11 = 'second'
    var_12 = {var_9: var_5, var_10: var_6, var_11: var_7}
    var_13 = module_1.time(*var_8, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 123000
    var_7 = []
    var_8 = 'hour'
    var_9 = 'microsecond'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.time(*var_7, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'not-a-time'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid time format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:60'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:00:61'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.0.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '192.168.0.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not-an-ip'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_utc_z. Retrieved 5/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 2
    var_8 = []
    var_9 = 'hours'
    var_10 = {var_9: var_7}
    var_11 = module_1.timedelta(*var_8, **var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.timezone(*var_12, **var_13)
    var_15 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_16 = 'tzinfo'
    var_17 = {var_16: var_14}
    var_18 = module_1.datetime(*var_15, **var_17)
    var_19 = var_2.serialize(var_18)
    assert var_19 == '2023-01-01T12:00:00+02:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = -5
    var_8 = -30
    var_9 = []
    var_10 = 'hours'
    var_11 = 'minutes'
    var_12 = {var_10: var_7, var_11: var_8}
    var_13 = module_1.timedelta(*var_9, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00-05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.datetime(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '2023-01-01T12:00:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 123456
    var_8 = [var_3, var_4, var_4, var_5, var_6, var_6, var_7]
    var_9 = {}
    var_10 = module_1.datetime(*var_8, **var_9)
    var_11 = var_2.serialize(var_10)
    assert var_11 == '2023-01-01T12:00:00.123456'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'not a datetime'
    var_4 = var_2.serialize(var_3)



# Parsed testcases at query #7
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '2023-10-05'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = 12
    var_7 = 0
    var_8 = [var_3, var_4, var_5, var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_1.datetime(*var_8, **var_9)
    var_11 = var_2.serialize(var_10)
    var_12 = 'Expected AssertionError because datetime is not date'
    var_13 = AssertionError(var_12)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-10-05'
    var_4 = var_2.serialize(var_3)
    var_5 = 'Expected AssertionError for string input'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_uuid_format_validate_valid_hex. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_urn. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_braces. Retrieved 4/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_3)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #9
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.serialize(var_3)
    assert var_4 == 'test@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.serialize(var_3)
    assert var_4 == 'abc'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_success. Retrieved 7/9 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-10-25'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 10
    var_7 = 25
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '25/10/2023'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid date format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real date'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid date format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid date format'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_raises_value_error_on_invalid_ip_format. Retrieved 2/26 statements.
# Partially parsed test_validate_triggers_value_error_exception_block. Retrieved 1/18 statements.


def test_case_0():
    var_0 = '256.256.256.256'
    var_1 = '999.999.999.999'

def test_case_0():
    var_0 = '999.999.999.999'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serialize_with_utc_timezone_returns_z_suffix. Retrieved 5/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0



# Parsed testcases at query #13
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #14
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.is_native_type(var_3)
    assert var_4 is False
    var_5 = None
    var_6 = var_2.is_native_type(var_5)
    assert var_6 is False
    var_7 = 123
    var_8 = var_2.is_native_type(var_7)
    assert var_8 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_replaces_plus_zero_zero_with_z_for_utc. Retrieved 7/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = [var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = var_2.serialize(var_11)
    assert var_12 == '2023-10-27T15:30:45'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 2023
    var_12 = 10
    var_13 = 27
    var_14 = 15
    var_15 = 30
    var_16 = 45
    var_17 = [var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_10}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-10-27T15:30:45+02:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -5
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 2023
    var_12 = 10
    var_13 = 27
    var_14 = 15
    var_15 = 30
    var_16 = 45
    var_17 = [var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_10}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-10-27T15:30:45-05:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = var_2.serialize(var_12)
    assert var_13 == '2023-10-27T15:30:45.123456'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_success_utc. Retrieved 9/11 statements.
# Partially parsed test_validate_success_offset. Retrieved 8/9 statements.
# Partially parsed test_validate_negative_offset. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T15:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 10
    var_7 = 27
    var_8 = 15
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '202lag-10-27T15:30:45+02:00'
    var_4 = var_2.validate(var_3)
    var_5 = '2023-10-27T15:30:45+02:00'
    var_6 = var_2.validate(var_5)
    var_7 = None
    var_8 = 2
    var_9 = []
    var_10 = 'hours'
    var_11 = {var_10: var_8}
    var_12 = module_1.timedelta(*var_9, **var_11)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T15:30:45.123Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'not-a-date'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid datetime format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-32T15:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real datetime'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T15:30:45-05:00'
    var_4 = var_2.validate(var_3)
    var_5 = None
    var_6 = -5
    var_7 = []
    var_8 = 'hours'
    var_9 = {var_8: var_6}
    var_10 = module_1.timedelta(*var_7, **var_9)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T15:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    assert var_5 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_uuid_format_serialize_returns_string_for_valid_uuid. Retrieved 6/15 statements.
# Partially parsed test_uuid_format_serialize_returns_none_for_none. Retrieved 1/8 statements.
# Partially parsed test_uuid_format_serialize_raises_assertion_error_on_invalid_type. Retrieved 3/12 statements.


import uuid as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUID(var_0)
    var_2 = '12lag45678-1234-5678-1234-567812345678'
    var_3 = 'l'
    var_4 = ''
    var_5 = 'a'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 'not-a-uuid-object'
    var_1 = 'Should have raised AssertionError for non-UUID type'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #18
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 30
    var_7 = []
    var_8 = 'hour'
    var_9 = 'minute'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.time(*var_7, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 30
    var_7 = 45
    var_8 = []
    var_9 = 'hour'
    var_10 = 'minute'
    var_11 = 'second'
    var_12 = {var_9: var_5, var_10: var_6, var_11: var_7}
    var_13 = module_1.time(*var_8, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 123000
    var_7 = []
    var_8 = 'hour'
    var_9 = 'microsecond'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.time(*var_7, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'not-a-time'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid time format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid time format'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_tzinfo_is_Z. Retrieved 1/34 statements.


def test_case_0():
    var_0 = '2023-10-27T12:00:00Z'



# Parsed testcases at query #20
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'testexample.com'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user@mail.subdomain.org'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user@mail.subdomain.org'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_with_microseconds_present. Retrieved 5/26 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = 'invalid'
    var_2 = 'Must be a valid datetime format.'
    var_3 = 'Must be a real datetime.'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'typesystem.formats'
    var_6 = []
    var_7 = {}
    var_8 = module_0.DateTimeFormat(*var_6, **var_7)
    var_9 = '2023-10-27 10:30:05.123'
    var_10 = var_8.validate(var_9)
    var_11 = var_10.microsecond
    assert var_11 == 123000
    var_12 = var_10.year
    assert var_12 == 2023



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_raises_invalid_error_on_unparseable_regex_match. Retrieved 5/8 statements.


import typesystem.formats as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.ValueError(*var_4, **var_5)
    var_7 = '256.256.266.266'
    var_8 = var_2.validate(var_7)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_invalid_url_missing_scheme. Retrieved 4/7 statements.
# Partially parsed test_validate_invalid_url_missing_netloc. Retrieved 4/7 statements.
# Partially parsed test_validate_invalid_url_empty_string. Retrieved 4/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://localhost:8080/api/v1'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'http://localhost:8080/api/v1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'example.com'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'Must be a real URL.'
    var_7 = bool('Must be a real URL.' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https:///path/only'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'Must be a real URL.'
    var_7 = bool('Must be a real URL.' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'Must be a real URL.'
    var_7 = bool('Must be a real URL.' in var_5)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_raises_format_error_for_non_ip_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not-an-ip-address'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = 'Must be a valid IP format.'
    var_7 = bool('Must be a valid IP format.' in var_5)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_raises_invalid_on_out_of_range_values. Retrieved 5/31 statements.


import re as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Must be a real time.'
    var_2 = {var_0: var_1}
    var_3 = '(?P<hour>\\d{2}):(?P<minute>\\d{2}):(?P<second>\\d{2})(?::(?P<microsecond>\\d+))?'
    var_4 = module_0.compile(var_3)
    var_5 = 'typesystem.formats'
    var_6 = '25:00:00'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.0.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = var_2.is_native_type(var_4)
    assert var_5 is True

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::1'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.is_native_type(var_4)
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_raises_value_error_for_invalid_date_components. Retrieved 3/13 statements.


def test_case_0():
    var_0 = '2023-13-01'
    var_1 = 'ValueError was not raised by datetime.date'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #28
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Must be a real datetime.'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.DateTimeFormat(*var_3, **var_4)
    var_6 = '2023-02-30'
    var_7 = var_5.validate(var_6)
    var_8 = 'Should have raised ValidationError'
    var_9 = AssertionError(var_8)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '192.168.1.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not-an-ip'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid IP format.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real IP.'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_valid_utc_z. Retrieved 7/11 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 10
    var_7 = 27
    var_8 = 30

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '202lag-10-27T10:30:00+05:00'
    var_4 = var_2.validate(var_3)
    var_5 = '2023-10-27T10:30:00+05:30'
    var_6 = var_2.validate(var_5)
    var_7 = 5
    var_8 = 30
    var_9 = []
    var_10 = 'hours'
    var_11 = 'minutes'
    var_12 = {var_10: var_7, var_11: var_8}
    var_13 = module_1.timedelta(*var_9, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = var_6.tzinfo
    var_18 = bool(var_6.tzinfo == var_16)
    assert var_18 is True
    var_19 = var_6.year
    assert var_19 == 2023

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00.123456'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'not-a-date'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid datetime format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T10:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real datetime'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00-05:00'
    var_4 = var_2.validate(var_3)
    var_5 = -5
    var_6 = 0
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.timedelta(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.timezone(*var_12, **var_13)
    var_15 = var_4.tzinfo
    var_16 = bool(var_4.tzinfo == var_14)
    assert var_16 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    assert var_5 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_raises_error_on_invalid_uuid_string. Retrieved 5/14 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = 'Validator should have raised validation error for invalid input'
    var_6 = AssertionError(var_5)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 14
    var_4 = 30
    var_5 = 5
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '14:30:05'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 12
    var_4 = 0
    var_5 = 123
    var_6 = [var_3, var_4, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '12:00:00.123000'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'not a time object'
    var_4 = var_2.serialize(var_3)
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com'
    var_4 = var_2.serialize(var_3)
    assert var_4 == 'https://example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://localhost:8080/path'
    var_4 = var_2.serialize(var_3)
    assert var_4 == 'http://localhost:8080/path'



# Parsed testcases at query #3
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.serialize(var_3)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.serialize(var_3)
    assert var_4 == ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serialize_replaces_utc_offset_with_z. Retrieved 7/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = [var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = var_2.serialize(var_11)
    assert var_12 == '2023-10-27T15:30:45'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 5
    var_4 = 30
    var_5 = []
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.timedelta(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 10
    var_15 = 27
    var_16 = 15
    var_17 = 45
    var_18 = [var_13, var_14, var_15, var_16, var_4, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_12}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    assert var_22 == '2023-10-27T15:30:45+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -8
    var_4 = 0
    var_5 = []
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.timedelta(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 10
    var_15 = 27
    var_16 = 15
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = var_2.serialize(var_22)
    assert var_23 == '2023-10-27T15:30:45-08:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = var_2.serialize(var_12)
    assert var_13 == '2023-10-27T15:30:45.123456'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_replaces_utc_offset_with_z. Retrieved 7/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 0
    var_9 = [var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = var_2.serialize(var_11)
    assert var_12 == '2023-10-05T14:30:00'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 2023
    var_12 = 10
    var_13 = 5
    var_14 = 14
    var_15 = 30
    var_16 = 0
    var_17 = [var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_10}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-10-05T14:30:00+02:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -5
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 2023
    var_12 = 10
    var_13 = 5
    var_14 = 14
    var_15 = 30
    var_16 = 0
    var_17 = [var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_10}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-10-05T14:30:00-05:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 0
    var_9 = 123456
    var_10 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = var_2.serialize(var_12)
    assert var_13 == '2023-10-05T14:30:00.123456'



# Parsed testcases at query #6
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'testexample.com'
    var_4 = var_2.validate(var_3)
    var_5 = 'format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@'
    var_4 = var_2.validate(var_3)
    var_5 = 'format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user@mail.sub.domain.org'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user@mail.sub.domain.org'



# Parsed testcases at query #7
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://www.google.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://www.google.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'www.google.com'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https:///path/to/resource'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_date_format_validate_success. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-10-25'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 10
    var_7 = var_4.day
    assert var_7 == 25

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 'not-a-date'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid date format.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-13-01'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real date.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real date.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2024-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2024
    var_6 = var_4.month
    assert var_6 == 2
    var_7 = var_4.day
    assert var_7 == 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real date.'



# Parsed testcases at query #9
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.0.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.168.0.1'

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::1'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '2001:db8::1'

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'fe80::1%eth0'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == 'fe80::1%eth0'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.0.1'
    var_4 = var_2.serialize(var_3)



# Parsed testcases at query #10
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 5
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '2023-10-05'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-10-05'
    var_4 = var_2.serialize(var_3)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 9999
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '9999-12-31'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_3, var_3]
    var_5 = {}
    var_6 = module_1.date(*var_4, **var_5)
    var_7 = var_2.serialize(var_6)
    assert var_7 == '0001-01-01'



# Parsed testcases at query #11
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 30
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 30
    var_7 = 45
    var_8 = 123456
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.time(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 30
    var_7 = 45
    var_8 = 123000
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.time(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'not-a-time'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid time format.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:61'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:00:61'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid time format.'



# Parsed testcases at query #12
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid-object'
    var_4 = var_2.serialize(var_3)
    var_5 = bool(True)
    assert var_5 is True
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 'invalid'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:00:70'
    var_4 = var_2.validate(var_3)
    var_5 = 'invalid'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_raises_invalid_on_out_of_range_values. Retrieved 1/18 statements.


def test_case_0():
    var_0 = '25:00:00'



# Parsed testcases at query #15
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_raises_invalid_on_out_of_range_time. Retrieved 1/18 statements.


def test_case_0():
    var_0 = '25:00:00'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_DateTimeFormat_validate_success_utc. Retrieved 4/16 statements.
# Partially parsed test_DateTimeFormat_validate_success_offset. Retrieved 4/10 statements.
# Partially parsed test_DateTimeFormat_validate_success_microsecond. Retrieved 1/6 statements.
# Partially parsed test_DateTimeFormat_validate_format_error. Retrieved 1/7 statements.
# Partially parsed test_DateTimeFormat_validate_invalid_date_error. Retrieved 1/7 statements.
# Partially parsed test_DateTimeFormat_validate_negative_offset. Retrieved 4/10 statements.


def test_case_0():
    var_0 = '202rypt-01-01T00:00:00Z'
    var_1 = 'typesystem.formats'
    var_2 = '(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})T(?P<hour>\\d{2}):(?P<minute>\\d{2}):(?P<second>\\d{2})(?P<microsecond>\\d*)?(?P<tzinfo>Z|[+-]\\d{2}:\\d{2})?'
    var_3 = '2023-10-27T15:30:45Z'

import datetime as module_0

def test_case_0():
    var_0 = '2023-10-27T15:30:45+02:00'
    var_1 = None
    var_2 = 2
    var_3 = []
    var_4 = 'hours'
    var_5 = {var_4: var_2}
    var_6 = module_0.timedelta(*var_3, **var_5)

def test_case_0():
    var_0 = '2023-10-27T15:30:45.123Z'

def test_case_0():
    var_0 = 'not-a-date'

def test_case_0():
    var_0 = '2023-02-30T15:30:45Z'

import datetime as module_0

def test_case_0():
    var_0 = '2023-10-27T15:30:45-05:00'
    var_1 = None
    var_2 = -5
    var_3 = []
    var_4 = 'hours'
    var_5 = {var_4: var_2}
    var_6 = module_0.timedelta(*var_3, **var_5)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_success. Retrieved 3/7 statements.
# Partially parsed test_validate_format_error. Retrieved 4/8 statements.
# Partially parsed test_validate_invalid_date_error. Retrieved 4/9 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-10-25'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 10
    var_7 = var_4.day
    assert var_7 == 25

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 'invalid-date'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'Must be a valid date format'
    var_7 = bool('Must be a valid date format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'Must be a real date'
    var_7 = bool('Must be a real date' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #19
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_raises_value_error_on_invalid_date. Retrieved 1/11 statements.


def test_case_0():
    var_0 = '2023-02-30'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_raises_value_error_on_invalid_date. Retrieved 5/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = 'Should have raised a validation error'
    var_6 = AssertionError(var_5)
    var_7 = 'invalid'
    var_8 = bool('invalid' in var_4)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_uuid_format_validate_success. Retrieved 7/12 statements.
# Partially parsed test_uuid_format_validate_hex_only. Retrieved 3/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-123ument-1234-1234-123456789abc'
    var_4 = 'ument'
    var_5 = '5678'
    var_6 = '12345678-1234-5678-1234-567812345678'
    var_7 = var_2.validate(var_6)
    var_8 = str(var_7)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(True)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(True)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = bool(True)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678123456781234567812345678'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_uuid_format_validate_valid_hex. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_valid_no_hyphens. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_3)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678123456781234567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hex
    var_6 = bool(var_4.hex == var_3)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_uuid_format_validate_success. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_success_with_braces. Retrieved 4/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 123456789
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_uuid_format_validate_invalid_string_raises_error. Retrieved 3/12 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '192.168.1.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not-an-ip'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_raises_format_error_on_invalid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not-an-ip-address'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = 'Must be a valid IP format.'
    var_7 = bool('Must be a valid IP format.' in var_5)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_with_valid_string. Retrieved 3/35 statements.


def test_case_0():
    var_0 = 'format'
    var_1 = 'invalid'
    var_2 = 'error'
    var_3 = 'error'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda self, key: MockValidationError(key)
    var_6 = '2023-10-27T10:00:00Z'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_success. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 10
    var_7 = var_4.day
    assert var_7 == 27
    var_8 = var_4.tzinfo



# Parsed testcases at query #30
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00.123Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 123000
    var_6 = var_4.year
    assert var_6 == 2023



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_tzinfo_not_z. Retrieved 7/37 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = 'format'
    var_1 = 'invalid'
    var_2 = 'error'
    var_3 = 'error'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.DateTimeFormat(*var_5, **var_6)
    var_8 = '2023-01-01T12:00:00+05:00'
    var_9 = var_7.validate(var_8)
    var_10 = 5
    var_11 = []
    var_12 = 'hours'
    var_13 = {var_12: var_10}
    var_14 = module_1.timedelta(*var_11, **var_13)
    var_15 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_raises_format_error_when_neither_ipv4_nor_ipv6_regex_matches. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not-an-ip-address'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = 'Must be a valid IP format.'
    var_7 = bool('Must be a valid IP format.' in var_5)
    assert var_7 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_datetime_format_validate_valid_iso. Retrieved 7/9 statements.
# Partially parsed test_datetime_format_validate_invalid_format_raises_error. Retrieved 6/10 statements.
# Partially parsed test_datetime_format_validate_invalid_values_raises_error. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 10
    var_7 = 27
    var_8 = 30

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 5
    var_6 = 30
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.timedelta(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.timezone(*var_12, **var_13)
    var_15 = 2023
    var_16 = 10
    var_17 = 27
    var_18 = [var_15, var_16, var_17, var_16, var_6]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_14}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = bool(var_4 == var_21)
    assert var_22 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00-04:00'
    var_4 = var_2.validate(var_3)
    var_5 = -4
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 10
    var_15 = 27
    var_16 = 30
    var_17 = [var_13, var_14, var_15, var_14, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = bool(var_4 == var_20)
    assert var_21 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00.12Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 120000

import typesystem.formats as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'format_error'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.ValueError(*var_4, **var_5)
    var_7 = 'not-a-date'
    var_8 = var_2.validate(var_7)
    var_9 = 'format'

import typesystem.formats as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'invalid_value'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.ValueError(*var_4, **var_5)
    var_7 = '2023-13-27T10:30:00Z'
    var_8 = var_2.validate(var_7)
    var_9 = 'invalid'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    assert var_5 is None
    var_6 = var_4.hour
    assert var_6 == 10



