####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUID(var_0)
    var_2 = module_1.UUIDFormat()
    var_3 = var_2.serialize(var_1)
    assert var_3 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None



# Parsed testcases at query #2
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.0.2.1'
    var_2 = module_1.IPv4Address(var_1)
    var_3 = var_0.serialize(var_2)
    assert var_3 == '192.0.2.1'

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '2001:db8::'
    var_2 = module_1.IPv6Address(var_1)
    var_3 = var_0.serialize(var_2)
    assert var_3 == '2001:db8::'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '::ffff:192.0.2.1'
    var_2 = module_1.IPv6Address(var_1)
    var_3 = var_0.serialize(var_2)
    assert var_3 == '::ffff:192.0.2.1'

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'fe80::1%eth0'
    var_2 = module_1.IPv6Address(var_1)
    var_3 = var_0.serialize(var_2)
    assert var_3 == 'fe80::1%eth0'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_partial_microseconds. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #4
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'invalid-email'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_datetime_without_timezone. Retrieved 5/7 statements.
# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 5/8 statements.
# Partially parsed test_serialize_datetime_with_positive_timezone. Retrieved 8/11 statements.
# Partially parsed test_serialize_datetime_with_negative_timezone. Retrieved 8/11 statements.
# Partially parsed test_serialize_datetime_with_microseconds. Retrieved 6/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 5
    var_2 = 30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = -3
    var_2 = -45
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = 123456



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_without_hyphens. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '{12345678-1234-5678-1234-567812345678}'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678123456781234567812345678'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'invalid-uuid-string'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-56781234567'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678-extra'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_valid_date_string. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-15'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '15-01-2023'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 12345
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_valid_date_string. Retrieved 5/6 statements.
# Partially parsed test_validate_date_object. Retrieved 3/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '01-01-2023'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = module_0.DateFormat()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_valid_time_string. Retrieved 6/7 statements.
# Partially parsed test_validate_with_valid_time_string_with_microseconds. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'not-a-time'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_with_valid_ipv6. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '192.168.1.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'invalid_ip'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.168.1.1'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 7/9 statements.
# Partially parsed test_validate_with_valid_datetime_string_with_offset. Retrieved 10/12 statements.
# Partially parsed test_validate_with_valid_datetime_string_with_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 5
    var_8 = 30
    var_9 = module_1.timedelta()

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00.123456Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T12:00:00Z'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_returns_ipaddress_on_success. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc_timezone. Retrieved 7/9 statements.
# Partially parsed test_validate_valid_datetime_with_positive_offset. Retrieved 10/12 statements.
# Partially parsed test_validate_valid_datetime_with_negative_offset. Retrieved 10/12 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 8/10 statements.
# Partially parsed test_validate_valid_datetime_without_timezone. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 12
    var_9 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00-03:45'
    var_2 = var_0.validate(var_1)
    var_3 = -3
    var_4 = -45
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 12
    var_9 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00.123456Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T12:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #14
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUIDFormat()
    var_2 = var_1.validate(var_0)
    var_3 = module_1.UUID(var_0)



# Parsed testcases at query #15
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-31T12:00:00Z'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_time_without_microseconds. Retrieved 4/6 statements.
# Partially parsed test_serialize_time_with_microseconds. Retrieved 5/7 statements.
# Partially parsed test_serialize_time_with_tzinfo. Retrieved 6/9 statements.
# Partially parsed test_serialize_time_with_fold. Retrieved 5/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = module_0.TimeFormat()

import typesystem.formats as module_0

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = 123456
    var_4 = module_0.TimeFormat()

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2
    var_1 = module_0.timedelta()
    var_2 = 12
    var_3 = 30
    var_4 = 45
    var_5 = module_1.TimeFormat()

import typesystem.formats as module_0

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = 1
    var_4 = module_0.TimeFormat()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serialize_utc_datetime_ends_with_Z. Retrieved 5/9 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 0
    var_4 = 'Z'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_urn_prefix. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUIDFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)

import typesystem.formats as module_0

def test_case_0():
    var_0 = '{12345678-1234-5678-1234-567812345678}'
    var_1 = module_0.UUIDFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUIDFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'invalid-uuid-format'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #19
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://www.example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://www.example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'www.example.com'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https:'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #20
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #21
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serialize_with_valid_time_object. Retrieved 4/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = module_0.TimeFormat()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_partial_microseconds. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12-34-56'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serialize_datetime_without_timezone. Retrieved 5/7 statements.
# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 5/8 statements.
# Partially parsed test_serialize_datetime_with_positive_offset. Retrieved 8/11 statements.
# Partially parsed test_serialize_datetime_with_negative_offset. Retrieved 8/11 statements.
# Partially parsed test_serialize_datetime_with_microseconds. Retrieved 6/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 5
    var_2 = 30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = -3
    var_2 = -45
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = 123456



# Parsed testcases at query #25
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.0.0.1'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serialize_utc_datetime. Retrieved 5/8 statements.
# Partially parsed test_serialize_datetime_with_timezone. Retrieved 8/11 statements.
# Partially parsed test_serialize_naive_datetime. Retrieved 5/7 statements.
# Partially parsed test_serialize_datetime_with_microseconds. Retrieved 6/9 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 5
    var_2 = 30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = 123456



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc. Retrieved 7/9 statements.
# Partially parsed test_validate_valid_datetime_with_offset. Retrieved 10/12 statements.
# Partially parsed test_validate_valid_datetime_with_negative_offset. Retrieved 9/11 statements.
# Partially parsed test_validate_valid_datetime_without_tzinfo. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 5
    var_8 = 30
    var_9 = module_1.timedelta()

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00-03:00'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = -3
    var_8 = module_1.timedelta()

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00.123456Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T12:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #28
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serialize_utc_timezone_replaces_plus_0000_with_Z. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = 'Z'



# Parsed testcases at query #30
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_raises_invalid_error_for_invalid_ip. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.256.256.256'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_1)



# Parsed testcases at query #32
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_partial_microseconds. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '192.168.1.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'invalid_ip'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.168.1.1'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '192.168.1.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'invalid_ip'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.168.1.1'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUIDFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_partial_microseconds. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #38
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T12:00:00Z'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #39
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T12:00:00Z'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #40
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None



# Parsed testcases at query #41
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-31T12:00:00Z'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #42
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_validate_raises_error_for_invalid_ip. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'invalid.ip.address'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_1)
    assert var_3 == 'Must be a real IP.'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUIDFormat()
    var_2 = var_1.validate(var_0)
    var_3 = str(var_2)



# Parsed testcases at query #45
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_serialize_utc_datetime. Retrieved 5/8 statements.
# Partially parsed test_serialize_datetime_with_timezone. Retrieved 8/11 statements.
# Partially parsed test_serialize_datetime_without_timezone. Retrieved 5/7 statements.
# Partially parsed test_serialize_datetime_with_microseconds. Retrieved 6/9 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = module_0.DateTimeFormat()

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = 5
    var_5 = 30
    var_6 = module_0.timedelta()
    var_7 = module_1.DateTimeFormat()

import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = module_0.DateTimeFormat()

import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = 123456
    var_5 = module_0.DateTimeFormat()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_partial_microseconds. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_serialize_datetime_without_timezone. Retrieved 5/7 statements.
# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 5/8 statements.
# Partially parsed test_serialize_datetime_with_positive_offset. Retrieved 8/11 statements.
# Partially parsed test_serialize_datetime_with_negative_offset. Retrieved 8/11 statements.
# Partially parsed test_serialize_datetime_with_microseconds. Retrieved 6/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = 5
    var_6 = 30
    var_7 = module_1.timedelta()

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = -3
    var_6 = -45
    var_7 = module_1.timedelta()

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = 123456



# Parsed testcases at query #49
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUIDFormat()
    var_2 = var_1.validate(var_0)
    var_3 = module_1.UUID(var_0)



