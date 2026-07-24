####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_uuid_format_serialize_returns_string_for_valid_uuid. Retrieved 4/7 statements.


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
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.serialize(var_3)
    var_5 = 'Should have raised AssertionError for non-UUID type'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #2
#--------------------------




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
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.time(*var_8, **var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

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
    var_3 = '08:05'
    var_4 = var_2.validate(var_3)
    var_5 = 8
    var_6 = 5
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

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
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:61:00'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real time.'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '10:10:10.1'
    var_4 = var_2.validate(var_3)
    var_5 = 10
    var_6 = 100000
    var_7 = [var_5, var_5, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_UUIDFormat_validate_valid_hex_string. Retrieved 4/8 statements.
# Partially parsed test_UUIDFormat_validate_valid_no_hyphens. Retrieved 4/8 statements.


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
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = 'Should have raised validation error'
    var_6 = AssertionError(var_5)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    var_5 = 'Should have raised validation error'
    var_6 = AssertionError(var_5)
    var_7 = bool(True)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 123456789
    var_4 = var_2.validate(var_3)
    var_5 = 'Should have raised validation error'
    var_6 = AssertionError(var_5)
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 25
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '2023-10-25'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-10-25'
    var_4 = var_2.serialize(var_3)
    var_5 = 'serialize should assert isinstance(obj, datetime.date)'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_invalid_format_raises_error. Retrieved 4/7 statements.
# Partially parsed test_validate_invalid_ip_range_raises_error. Retrieved 4/7 statements.


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
    var_3 = 'not.an.ip.format'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'invalid'
    var_7 = bool('invalid' in var_5)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = 'invalid'
    var_2 = 'format_err'
    var_3 = 'invalid_err'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.DateFormat(*var_5, **var_6)
    var_8 = '2023-05-20'
    var_9 = var_7.validate(var_8)
    var_10 = var_9.year
    assert var_10 == 2023
    var_11 = var_9.month
    assert var_11 == 5
    var_12 = var_9.day
    assert var_12 == 20

import typesystem.formats as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = 'invalid'
    var_2 = 'format_err'
    var_3 = 'invalid_err'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.DateFormat(*var_5, **var_6)
    var_8 = '20-05-2023'
    var_9 = var_7.validate(var_8)

import typesystem.formats as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = 'invalid'
    var_2 = 'format_err'
    var_3 = 'invalid_err'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.DateFormat(*var_5, **var_6)
    var_8 = '2023-02-30'
    var_9 = var_7.validate(var_8)



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
    var_5 = 'Must be a real URL.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https:///path'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real URL.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'ftp://files.server.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'ftp://files.server.com'



# Parsed testcases at query #8
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
    var_3 = 'invalidemail.com'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_uuid_format_validate_invalid_string_raises_error. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'not-a-uuid'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_uuid_format_validate_raises_error_on_invalid_string. Retrieved 7/20 statements.


import re as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    var_1 = module_0.compile(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.UUIDFormat(*var_2, **var_3)
    var_5 = 'not-a-uuid'
    var_6 = var_4.validate(var_5)
    var_7 = 'validate() should have raised ValueError for invalid input'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_success_with_offset_plus. Retrieved 8/11 statements.
# Partially parsed test_validate_success_with_offset_minus. Retrieved 5/6 statements.


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
    var_8 = var_4.hour
    assert var_8 == 10
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.tzinfo

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00.123Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 123000

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '202string-10-27T10:30:00+02:00'
    var_4 = '202string'
    var_5 = '2023'
    var_6 = '2023-10-27T10:30:00+02:00'
    var_7 = var_2.validate(var_6)
    var_8 = 2
    var_9 = []
    var_10 = 'hours'
    var_11 = {var_10: var_8}
    var_12 = module_1.timedelta(*var_9, **var_11)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00-05:00'
    var_4 = var_2.validate(var_3)
    var_5 = -5
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'not-a-date'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-13-27T10:30:00Z'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    assert var_5 is None



# Parsed testcases at query #12
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #13
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
    var_3 = '192.168.1.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.168.1.1'

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
    var_3 = '192.168.1.1'
    var_4 = var_2.serialize(var_3)
    var_5 = 'Should have raised AssertionError for string input'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #14
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
    var_5 = 'format'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)
    var_5 = 'invalid'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_raises_value_error_on_invalid_date_date_format_invalid_path. Retrieved 4/36 statements.


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Must be a real date.'
    var_2 = {var_0: var_1}
    var_3 = 'typesystem.formats'
    var_4 = '2023-13-01'
    var_5 = '2023-13-01'



# Parsed testcases at query #16
#--------------------------






####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_valid_with_offset. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_with_negative_offset. Retrieved 5/6 statements.
# Partially parsed test_validate_invalid_format_raises_error. Retrieved 4/7 statements.
# Partially parsed test_validate_invalid_date_values_raises_error. Retrieved 4/7 statements.


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
    var_8 = var_4.hour
    assert var_8 == 10
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.tzinfo

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.hour
    assert var_6 == 10
    var_7 = var_4.minute
    assert var_7 == 30
    var_8 = 5
    var_9 = 30
    var_10 = []
    var_11 = 'hours'
    var_12 = 'minutes'
    var_13 = {var_11: var_8, var_12: var_9}
    var_14 = module_1.timedelta(*var_10, **var_13)

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

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00.123456Z'
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
    var_5 = str(var_3)
    var_6 = 'Must be a valid datetime format'
    var_7 = bool('Must be a valid datetime format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-13-45T10:30:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'Must be a real datetime'
    var_7 = bool('Must be a real datetime' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    assert var_5 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_uuid_format_serialize_returns_string_for_valid_uuid. Retrieved 4/7 statements.


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
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.serialize(var_3)
    var_5 = 'Should have raised AssertionError for non-UUID type'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dateformat_validate_success. Retrieved 3/5 statements.


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
    var_3 = '25-10-2023'
    var_4 = var_2.validate(var_3)
    var_5 = 'Did not raise ValidationError for invalid format'
    var_6 = AssertionError(var_5)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = 'Did not raise ValidationError for invalid date values'
    var_6 = AssertionError(var_5)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 'not-a-date'
    var_4 = var_2.validate(var_3)
    var_5 = 'Did not raise ValidationError for malformed string'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #4
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
    var_3 = 'invalidemail.com'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'test@'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_uuid_format_validate_valid_hex. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_valid_no_hyphens. Retrieved 3/6 statements.
# Partially parsed test_uuid_format_validate_invalid_string_raises_error. Retrieved 4/8 statements.


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
    var_5 = str(var_4)
    var_6 = 'Must be a valid UUID format.'
    var_7 = bool('Must be a valid UUID format.' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '1234'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #6
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
    var_3 = '192.168.1.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.168.1.1'

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
    var_3 = 'not an address object'
    var_4 = var_2.serialize(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_tzinfo_not_none_and_not_Z. Retrieved 4/18 statements.


import datetime as module_0

def test_case_0():
    var_0 = '(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})T(?P<hour>\\d{2}):(?P<minute>\\d{2}):(?P<second>\\d{2})(?P<microsecond>\\d{6})?(?P<tzinfo>[Z+-]\\d{2}:?\\d{2})?'
    var_1 = '2023-01-01T12:00:00+01:00'
    var_2 = 1
    var_3 = []
    var_4 = 'hours'
    var_5 = {var_4: var_2}
    var_6 = module_0.timedelta(*var_3, **var_5)
    var_7 = bool(var_0 == var_6)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a valid UUID format'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_timeformat_validate_success. Retrieved 4/27 statements.
# Partially parsed test_timeformat_validate_format_error. Retrieved 2/16 statements.
# Partially parsed test_timeformat_validate_invalid_value_error. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'format'
    var_1 = 'invalid'
    var_2 = 'error'
    var_3 = 'error'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '12:30'
    var_6 = '12:30:45'
    var_7 = '12:30:45.123'

def test_case_0():
    var_0 = 'format'
    var_1 = 'Must be a valid time format.'
    var_2 = {var_0: var_1}
    var_3 = 'invalid-string'

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Must be a real time.'
    var_2 = {var_0: var_1}
    var_3 = '25:00'



# Parsed testcases at query #10
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
    var_3 = 'http://localhost:8080/api/v1'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'http://localhost:8080/api/v1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'www.google.com'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real URL.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https:///path/only'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real URL.'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a real URL.'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_uuid_format_validate_invalid_string_raises_error. Retrieved 6/28 statements.


import re as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = 'Must be a valid UUID format.'
    var_2 = {var_0: var_1}
    var_3 = '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    var_4 = module_0.compile(var_3)
    var_5 = 'typesystem.formats'
    var_6 = 'format'
    var_7 = 'Must be a valid UUID format.'
    var_8 = {var_6: var_7}
    var_9 = 'not-a-uuid'



# Parsed testcases at query #12
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
    var_6 = 123
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '12:30:45.000123'

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

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'not a time object'
    var_4 = var_2.serialize(var_3)

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
    var_4 = 5
    var_5 = 9
    var_6 = 0
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '14:05:09'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 14
    var_4 = 5
    var_5 = 9
    var_6 = 123456
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '14:05:09.123456'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_datetime_format_validate_success_offset. Retrieved 7/8 statements.
# Partially parsed test_datetime_format_validate_success_negative_offset. Retrieved 6/7 statements.
# Partially parsed test_datetime_format_serialize_utc_z. Retrieved 6/9 statements.


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
    var_8 = var_4.hour
    assert var_8 == 10
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.tzinfo

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = None
    var_6 = 5
    var_7 = 30
    var_8 = []
    var_9 = 'hours'
    var_10 = 'minutes'
    var_11 = {var_9: var_6, var_10: var_7}
    var_12 = module_1.timedelta(*var_8, **var_11)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-10-27T10:30:00-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = None
    var_6 = -8
    var_7 = []
    var_8 = 'hours'
    var_9 = {var_8: var_6}
    var_10 = module_1.timedelta(*var_7, **var_9)

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
    var_3 = '2023-10-27T10:30:00.123'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '27-10-2023 10:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-13-27T10:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 30
    var_7 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 30
    var_7 = 0
    var_8 = 5
    var_9 = []
    var_10 = 'hours'
    var_11 = 'minutes'
    var_12 = {var_10: var_8, var_11: var_6}
    var_13 = module_1.timedelta(*var_9, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = [var_3, var_4, var_5, var_4, var_6, var_7]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-10-27T10:30:00+05:30'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serialize_utc_z. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_microseconds. Retrieved 8/9 statements.


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
    var_8 = 30
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_invalid_format_raises_error. Retrieved 4/7 statements.
# Partially parsed test_validate_invalid_ip_value_raises_error. Retrieved 4/7 statements.


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
    var_5 = str(var_3)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    var_6 = 'invalid'
    var_7 = bool('invalid' in var_5)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_date_format_validate_invalid_date_value. Retrieved 1/10 statements.


def test_case_0():
    var_0 = '2023-02-30'



