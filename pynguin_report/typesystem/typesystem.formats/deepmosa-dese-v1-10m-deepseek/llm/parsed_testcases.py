####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_valid_date. Retrieved 6/7 statements.
# Partially parsed test_validate_leap_year. Retrieved 6/7 statements.
# Partially parsed test_validate_year_min. Retrieved 4/5 statements.
# Partially parsed test_validate_year_max. Retrieved 6/7 statements.
# Partially parsed test_validate_single_digit_month. Retrieved 5/6 statements.
# Partially parsed test_validate_single_digit_day. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 25

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023/12/25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2020-02-29'
    var_2 = var_0.validate(var_1)
    var_3 = 2020
    var_4 = 2
    var_5 = 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-29'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-13-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-32'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '0001-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '9999-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 9999
    var_4 = 12
    var_5 = 31

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-1-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-1'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_parses_valid_datetime_without_timezone. Retrieved 9/10 statements.
# Partially parsed test_validate_parses_valid_datetime_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_parses_valid_datetime_with_utc_zulu. Retrieved 9/11 statements.
# Partially parsed test_validate_parses_valid_datetime_with_positive_timezone. Retrieved 11/13 statements.
# Partially parsed test_validate_parses_valid_datetime_with_negative_timezone. Retrieved 11/13 statements.
# Partially parsed test_validate_parses_valid_datetime_with_timezone_no_minutes. Retrieved 11/13 statements.
# Partially parsed test_validate_parses_valid_datetime_with_microseconds_and_timezone. Retrieved 12/14 statements.
# Partially parsed test_validate_pads_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_handles_midnight. Retrieved 7/8 statements.
# Partially parsed test_validate_handles_leap_second_support. Retrieved 9/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 14
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.987654-03:00'
    var_2 = var_0.validate(var_1)
    var_3 = -3
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 987654

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'not-a-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T14:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T00:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-12-31T23:59:60'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 31
    var_6 = 23
    var_7 = 59
    var_8 = 60



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 9/10 statements.
# Partially parsed test_validate_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_with_utc_zulu. Retrieved 9/11 statements.
# Partially parsed test_validate_with_positive_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_negative_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_timezone_offset_no_minutes. Retrieved 11/13 statements.
# Partially parsed test_validate_with_partial_microseconds_padded. Retrieved 10/11 statements.
# Partially parsed test_validate_with_timezone_offset_and_microseconds. Retrieved 12/14 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 10
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 10
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 10
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 10
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 10
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 10
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T10:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 10
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45.654321-05:00'
    var_2 = var_0.validate(var_1)
    var_3 = -5
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 10
    var_9 = 30
    var_10 = 45
    var_11 = 654321



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serialize_with_valid_date. Retrieved 4/6 statements.
# Partially parsed test_serialize_with_leap_year_date. Retrieved 4/6 statements.
# Partially parsed test_serialize_with_min_date. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_max_date. Retrieved 4/6 statements.
# Partially parsed test_serialize_with_single_digit_month_and_day. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2020
    var_2 = 2
    var_3 = 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 9999
    var_2 = 12
    var_3 = 31

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 1



# Parsed testcases at query #5
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

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'not-a-uuid'
    var_2 = var_0.serialize(var_1)

import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '00000000-0000-0000-0000-000000000000'
    var_1 = module_0.UUID(var_0)
    var_2 = module_1.UUIDFormat()
    var_3 = var_2.serialize(var_1)
    assert var_3 == '00000000-0000-0000-0000-000000000000'

import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'ABCDEFAB-1234-5678-9ABC-DEF123456789'
    var_1 = module_0.UUID(var_0)
    var_2 = module_1.UUIDFormat()
    var_3 = var_2.serialize(var_1)
    assert var_3 == 'abcdefab-1234-5678-9abc-def123456789'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_returns_isoformat_for_naive_datetime. Retrieved 9/11 statements.
# Partially parsed test_serialize_returns_isoformat_with_z_for_utc_timezone. Retrieved 9/12 statements.
# Partially parsed test_serialize_returns_isoformat_with_offset_for_non_utc_timezone. Retrieved 11/14 statements.
# Partially parsed test_serialize_converts_plus_00_00_to_z. Retrieved 11/14 statements.
# Partially parsed test_serialize_handles_datetime_with_no_microseconds. Retrieved 8/10 statements.
# Partially parsed test_serialize_handles_datetime_with_microseconds_zero. Retrieved 9/11 statements.


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
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-01-15T14:30:45.123456'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-01-15T14:30:45.123456Z'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 5
    var_2 = 30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 1
    var_6 = 15
    var_7 = 14
    var_8 = 45
    var_9 = 123456
    var_10 = '2023-01-15T14:30:45.123456+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 0
    var_2 = module_1.timedelta()
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = '2023-01-15T14:30:45.123456Z'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = '2023-01-15T14:30:45'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 0
    var_8 = '2023-01-15T14:30:45'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'not a datetime'
    var_2 = var_0.serialize(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_valid_time_with_hour_minute. Retrieved 5/6 statements.
# Partially parsed test_validate_valid_time_with_hour_minute_second. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_microseconds_padded. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_leading_zeros. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_midnight. Retrieved 4/5 statements.
# Partially parsed test_validate_valid_time_max_hour. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34

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
    var_1 = '12'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'ab:cd'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '24:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '01:02:03.004005'
    var_2 = var_0.validate(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4005

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '00:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_short_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_single_digit_hour. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_two_digit_hour. Retrieved 5/6 statements.
# Partially parsed test_validate_valid_time_with_microseconds_padded. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_microseconds_all_zero. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_microseconds_max. Retrieved 6/7 statements.


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
    var_1 = '01:23:45'
    var_2 = var_0.validate(var_1)
    var_3 = 1
    var_4 = 23
    var_5 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '24:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.001'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 1000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.000000'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_uuidformat_validate_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_urn. Retrieved 4/6 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-56781234567'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-5678123456789'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-56781234567g'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-56781234567-'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 12345678
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_mapped_ipv6. Retrieved 4/6 statements.


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
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '999.999.999.999'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '010.010.010.010'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '::1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '::ffff:192.0.2.1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '::ffff:192.0.2.1'



# Parsed testcases at query #12
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

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user@sub.example.co.uk'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user@sub.example.co.uk'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user+tag@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user+tag@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'first.last@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'first.last@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user123@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user123@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user_name@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user_name@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user-name@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user-name@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'userexample.com'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user@'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = '@example.com'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user name@example.com'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user@name@example.com'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user@example.c'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user@example.info'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user@example.info'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user@example.museum'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user@example.museum'



# Parsed testcases at query #13
#--------------------------




import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '192.168.1.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = '192.168.1.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2001:db8::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = '2001:db8::1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'fe80::1%eth0'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = 'fe80::1%eth0'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '::ffff:192.168.1.1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = '::ffff:192.168.1.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.


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
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '999.999.999.999'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '010.010.010.010'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '2001:db8::1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '2001:db8::1'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_assert_isinstance_with_date_object. Retrieved 4/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_format. Retrieved 4/6 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = 'urn:uuid:{12345678-1234-5678-1234-567812345678}'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_version_1. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_version_4. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_all_zero. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_all_f. Retrieved 4/6 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'c232ab00-9414-11ec-b3c8-9f6b6a116ef5'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == 'c232ab00-9414-11ec-b3c8-9f6b6a116ef5'

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
    var_1 = '00000000-0000-0000-0000-000000000000'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '00000000-0000-0000-0000-000000000000'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'ffffffff-ffff-ffff-ffff-ffffffffffff'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == 'ffffffff-ffff-ffff-ffff-ffffffffffff'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serialize_assert_isinstance_with_date_object. Retrieved 4/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_time_with_invalid_microsecond. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1234567'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_valid_date. Retrieved 6/7 statements.
# Partially parsed test_validate_leap_year. Retrieved 6/7 statements.
# Partially parsed test_validate_single_digit_month. Retrieved 6/7 statements.
# Partially parsed test_validate_single_digit_day. Retrieved 5/6 statements.
# Partially parsed test_validate_min_year. Retrieved 4/5 statements.
# Partially parsed test_validate_max_year. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-15'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023/05/15'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-13-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2020-02-29'
    var_2 = var_0.validate(var_1)
    var_3 = 2020
    var_4 = 2
    var_5 = 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-29'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-5-15'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-5'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 5

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '0001-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '9999-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 9999
    var_4 = 12
    var_5 = 31

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-00-15'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-15 extra'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = ' 2023-05-15 '
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 9/10 statements.
# Partially parsed test_validate_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_with_short_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_with_utc_z. Retrieved 9/11 statements.
# Partially parsed test_validate_with_positive_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_negative_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_timezone_offset_no_minutes. Retrieved 11/13 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 14
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T14:30:45'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_short_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_single_digit_hour. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_two_digit_hour. Retrieved 5/6 statements.
# Partially parsed test_validate_valid_time_with_microseconds_padded. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_microseconds_max. Retrieved 6/7 statements.


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
    var_1 = '01:23:45'
    var_2 = var_0.validate(var_1)
    var_3 = 1
    var_4 = 23
    var_5 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12-34-56'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '24:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.001'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 1000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_serialize_assert_isinstance_with_date_object. Retrieved 4/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_with_valid_datetime_should_not_raise_invalid_error. Retrieved 7/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 9/10 statements.
# Partially parsed test_validate_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_with_short_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_with_utc_zulu. Retrieved 9/11 statements.
# Partially parsed test_validate_with_positive_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_negative_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_offset_no_minutes. Retrieved 11/13 statements.
# Partially parsed test_validate_with_offset_minutes_only. Retrieved 10/12 statements.
# Partially parsed test_validate_with_microseconds_and_timezone. Retrieved 12/14 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 14
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+00:30'
    var_2 = var_0.validate(var_1)
    var_3 = 30
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-13-45T25:61:61'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.987654-05:00'
    var_2 = var_0.validate(var_1)
    var_3 = -5
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 987654



# Parsed testcases at query #26
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'example.com'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'http://'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_short_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_single_digit_hour. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_zero_hour. Retrieved 4/5 statements.
# Partially parsed test_validate_valid_time_with_max_values. Retrieved 6/7 statements.


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
    var_1 = '01:02:03'
    var_2 = var_0.validate(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '00:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'ab:cd:ef'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56+01:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '24:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '-01:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56 extra'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_returns_ipv4_address_when_value_is_valid_ipv4_string. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address_when_value_is_valid_ipv6_string. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv4_address_when_value_is_valid_ipv4_string_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address_when_value_is_valid_compressed_ipv6_string. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv4_address_when_value_is_valid_ipv4_string_with_max_values. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address_when_value_is_valid_ipv6_string_with_max_values. Retrieved 4/6 statements.


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
    var_1 = '010.010.010.010'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '2001:db8::1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '255.255.255.255'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '255.255.255.255'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'



# Parsed testcases at query #29
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_valid_date. Retrieved 6/7 statements.
# Partially parsed test_validate_leap_year. Retrieved 6/7 statements.
# Partially parsed test_validate_single_digit_month_and_day. Retrieved 5/6 statements.
# Partially parsed test_validate_leading_zeros. Retrieved 5/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 25

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023/12/25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2020-02-29'
    var_2 = var_0.validate(var_1)
    var_3 = 2020
    var_4 = 2
    var_5 = 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-29'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-13-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-32'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '0000-01-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-1-1'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #31
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.256.256.256'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_date. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 25



# Parsed testcases at query #33
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #34
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_validate_valid_date_string. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_date_string_single_digit. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_date_string_leading_zeros. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_date_string_min_date. Retrieved 4/5 statements.
# Partially parsed test_validate_valid_date_string_max_date. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_date_string_leap_year. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 25

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023/12/25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-13-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-29'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-1-5'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 5

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-05'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 5

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023.12.25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-Dec-25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '0001-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '9999-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 9999
    var_4 = 12
    var_5 = 31

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '10000-01-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '-2023-12-25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-00-25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2024-02-29'
    var_2 = var_0.validate(var_1)
    var_3 = 2024
    var_4 = 2
    var_5 = 29



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_date. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_serialize_with_date_object. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 9/10 statements.
# Partially parsed test_validate_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_with_short_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_with_utc_z. Retrieved 9/11 statements.
# Partially parsed test_validate_with_positive_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_negative_timezone_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_with_timezone_offset_no_minutes. Retrieved 11/13 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 14
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T14:30:45'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_validate_time_with_invalid_microsecond. Retrieved 7/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1234567'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456



# Parsed testcases at query #40
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.256.256.256'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_serialize_with_valid_date. Retrieved 4/6 statements.
# Partially parsed test_serialize_with_leap_year_date. Retrieved 4/6 statements.
# Partially parsed test_serialize_with_single_digit_month_and_day. Retrieved 3/5 statements.
# Partially parsed test_serialize_with_max_date. Retrieved 4/6 statements.
# Partially parsed test_serialize_with_min_date. Retrieved 2/4 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 12
    var_3 = 25

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2020
    var_2 = 2
    var_3 = 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 9999
    var_2 = 12
    var_3 = 31

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 1



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_string_with_urn. Retrieved 4/6 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-56781234567'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-56781234567g'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'not-a-uuid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = module_1.UUID(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 24197857161011715162171839636988778104
    var_2 = module_1.UUID(int=var_1)
    var_3 = var_0.validate(var_2)



# Parsed testcases at query #2
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.BaseFormat()
    var_1 = 'test_value'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #3
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123456'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.123'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'ab:cd:ef'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '24:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '00:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.


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
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '999.999.999.999'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '010.010.010.010'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '2001:db8::1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 3232235777
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1:ffff'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #6
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'example.com'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'http://'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://example.com/path'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://example.com/path'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://example.com?query=value'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://example.com?query=value'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://example.com#fragment'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://example.com#fragment'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'ftp://example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'ftp://example.com'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_valid_date_string. Retrieved 6/7 statements.
# Partially parsed test_validate_leap_year. Retrieved 6/7 statements.
# Partially parsed test_validate_year_min. Retrieved 4/5 statements.
# Partially parsed test_validate_year_max. Retrieved 6/7 statements.
# Partially parsed test_validate_single_digit_month. Retrieved 6/7 statements.
# Partially parsed test_validate_single_digit_day. Retrieved 5/6 statements.
# Partially parsed test_validate_leading_zeros. Retrieved 5/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-15'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023/05/15'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-13-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2020-02-29'
    var_2 = var_0.validate(var_1)
    var_3 = 2020
    var_4 = 2
    var_5 = 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-29'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-00-15'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-32'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '0001-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '9999-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 9999
    var_4 = 12
    var_5 = 31

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-1-15'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-5'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 5

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '0023-005-005'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 5



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_time_with_invalid_microsecond. Retrieved 7/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1234567'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 123456



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_uuidformat_validate_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_urn. Retrieved 4/6 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-56781234567'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-56781234567g'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = 'not-a-uuid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #10
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-31'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #11
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1234567'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #12
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

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user@sub.example.co.uk'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user@sub.example.co.uk'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'user+tag@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'user+tag@example.com'



# Parsed testcases at query #13
#--------------------------




import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '192.168.1.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = '192.168.1.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2001:db8::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = '2001:db8::1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'fe80::1%eth0'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = 'fe80::1%eth0'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '::ffff:192.168.1.1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = module_1.IPAddressFormat()
    var_3 = var_2.serialize(var_1)
    var_4 = '::ffff:192.168.1.1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    var_3 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_mapped_ipv6. Retrieved 4/6 statements.


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
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '999.999.999.999'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '010.010.010.010'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '::1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '::ffff:192.0.2.1'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '::ffff:192.0.2.1'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_valid_date_should_not_raise_invalid_error. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_valid_datetime_without_timezone. Retrieved 9/10 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_valid_datetime_with_short_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_valid_datetime_with_utc_zulu. Retrieved 9/11 statements.
# Partially parsed test_validate_valid_datetime_with_positive_timezone. Retrieved 10/12 statements.
# Partially parsed test_validate_valid_datetime_with_negative_timezone. Retrieved 11/13 statements.
# Partially parsed test_validate_valid_datetime_with_timezone_no_minutes. Retrieved 11/13 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-05T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-05T14:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-05T14:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-05T14:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-05T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 4
    var_8 = 14
    var_9 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-05T14:30:45-03:00'
    var_2 = var_0.validate(var_1)
    var_3 = -3
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 4
    var_7 = 5
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-05T14:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 4
    var_7 = 5
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T14:30:45'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_valid_datetime_without_timezone. Retrieved 9/10 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_valid_datetime_with_short_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_valid_datetime_with_utc_z. Retrieved 9/11 statements.
# Partially parsed test_validate_valid_datetime_with_positive_timezone. Retrieved 11/13 statements.
# Partially parsed test_validate_valid_datetime_with_negative_timezone. Retrieved 11/13 statements.
# Partially parsed test_validate_valid_datetime_with_timezone_no_minutes. Retrieved 11/13 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-15T14:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-15T14:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 4
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 4
    var_8 = 15
    var_9 = 14
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 4
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-04-15T14:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 4
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 'not-a-datetime'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-30T14:30:45'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serialize_returns_isoformat_for_naive_datetime. Retrieved 9/11 statements.
# Partially parsed test_serialize_returns_isoformat_with_z_for_utc_timezone. Retrieved 9/12 statements.
# Partially parsed test_serialize_returns_isoformat_with_offset_for_non_utc_timezone. Retrieved 10/13 statements.
# Partially parsed test_serialize_returns_isoformat_with_negative_offset. Retrieved 12/15 statements.
# Partially parsed test_serialize_returns_isoformat_without_microseconds. Retrieved 8/10 statements.
# Partially parsed test_serialize_converts_utc_offset_to_z. Retrieved 11/14 statements.


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
    var_2 = 5
    var_3 = 17
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-05-17T14:30:45.123456'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 17
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-05-17T14:30:45.123456Z'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 5
    var_2 = 30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 17
    var_6 = 14
    var_7 = 45
    var_8 = 123456
    var_9 = '2023-05-17T14:30:45.123456+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = -5
    var_2 = -30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 5
    var_6 = 17
    var_7 = 14
    var_8 = 30
    var_9 = 45
    var_10 = 123456
    var_11 = '2023-05-17T14:30:45.123456-05:30'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 17
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = '2023-05-17T14:30:45'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 0
    var_2 = module_1.timedelta()
    var_3 = 2023
    var_4 = 5
    var_5 = 17
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = '2023-05-17T14:30:45.123456Z'



# Parsed testcases at query #19
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00+25:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serialize_returns_isoformat_for_naive_datetime. Retrieved 9/11 statements.
# Partially parsed test_serialize_returns_isoformat_with_z_for_utc_timezone. Retrieved 9/12 statements.
# Partially parsed test_serialize_returns_isoformat_with_offset_for_non_utc_timezone. Retrieved 10/13 statements.
# Partially parsed test_serialize_converts_plus_00_00_to_z. Retrieved 11/14 statements.
# Partially parsed test_serialize_handles_datetime_with_no_microseconds. Retrieved 8/10 statements.
# Partially parsed test_serialize_handles_datetime_with_microseconds_zero. Retrieved 9/11 statements.
# Partially parsed test_serialize_handles_datetime_with_negative_timezone_offset. Retrieved 11/14 statements.


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
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-05-15T14:30:45.123456'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-05-15T14:30:45.123456Z'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 5
    var_2 = 30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 15
    var_6 = 14
    var_7 = 45
    var_8 = 123456
    var_9 = '2023-05-15T14:30:45.123456+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 0
    var_2 = module_1.timedelta()
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = '2023-05-15T14:30:45.123456Z'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = '2023-05-15T14:30:45'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 0
    var_8 = '2023-05-15T14:30:45'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = -5
    var_2 = module_1.timedelta()
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = '2023-05-15T14:30:45.123456-05:00'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_braces. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/9 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_lowercase. Retrieved 2/9 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = module_0.DateTimeFormat()
    var_5 = 'Z'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime. Retrieved 9/12 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_date. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_valid_time_with_hour_minute. Retrieved 5/6 statements.
# Partially parsed test_validate_valid_time_with_hour_minute_second. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_microseconds_padded. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_midnight. Retrieved 4/5 statements.
# Partially parsed test_validate_valid_time_max. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = 45
    var_6 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = 45
    var_6 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'ab:cd:ef'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '25:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30:45.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '00:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = module_0.DateTimeFormat()
    var_5 = 'Z'



# Parsed testcases at query #27
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '256.256.256.256'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_valid_datetime_without_timezone. Retrieved 9/10 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_valid_datetime_with_short_microseconds. Retrieved 10/11 statements.
# Partially parsed test_validate_valid_datetime_with_utc_timezone. Retrieved 9/11 statements.
# Partially parsed test_validate_valid_datetime_with_positive_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_valid_datetime_with_negative_offset. Retrieved 11/13 statements.
# Partially parsed test_validate_valid_datetime_with_offset_no_minutes. Retrieved 11/13 statements.
# Partially parsed test_validate_invalid_datetime_leap_year_february. Retrieved 9/10 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds_and_timezone. Retrieved 12/14 statements.
# Partially parsed test_validate_valid_datetime_with_short_microseconds_and_timezone. Retrieved 10/12 statements.
# Partially parsed test_validate_valid_datetime_with_negative_offset_with_minutes. Retrieved 12/14 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 14
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+02'
    var_2 = var_0.validate(var_1)
    var_3 = 2
    var_4 = module_1.timedelta()
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023/01/15T14:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-32T14:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-13-15T14:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T25:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:60:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2024-02-29T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = 2024
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = 30
    var_8 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-02-29T14:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 14
    var_10 = 45
    var_11 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123000

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:30'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = -30
    var_5 = module_1.timedelta()
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 14
    var_10 = 30
    var_11 = 45



# Parsed testcases at query #29
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_for_valid_date. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_short_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_single_digit_hour. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_two_digit_hour. Retrieved 5/6 statements.
# Partially parsed test_validate_valid_time_with_microseconds_padded. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_microseconds_max. Retrieved 6/7 statements.


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
    var_1 = '01:23:45'
    var_2 = var_0.validate(var_1)
    var_3 = 1
    var_4 = 23
    var_5 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '24:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.001'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 34
    var_5 = 56
    var_6 = 1000

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_does_not_raise_value_error_for_valid_datetime. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-01T12:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_raises_format_error_for_invalid_ip_string. Retrieved 4/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = None
    var_2 = 'not_an_ip'
    var_3 = var_0.validate(var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_serialize_returns_isoformat_for_naive_datetime. Retrieved 9/11 statements.
# Partially parsed test_serialize_returns_isoformat_with_timezone. Retrieved 10/13 statements.
# Partially parsed test_serialize_converts_utc_to_z_suffix. Retrieved 9/12 statements.
# Partially parsed test_serialize_handles_microsecond_padding. Retrieved 9/11 statements.
# Partially parsed test_serialize_handles_no_microseconds. Retrieved 8/10 statements.
# Partially parsed test_serialize_handles_negative_timezone_offset. Retrieved 12/15 statements.


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
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-05-15T14:30:45.123456'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 5
    var_2 = 30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 15
    var_6 = 14
    var_7 = 45
    var_8 = 123456
    var_9 = '2023-05-15T14:30:45.123456+05:30'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123456
    var_8 = '2023-05-15T14:30:45.123456Z'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = 123
    var_8 = '2023-05-15T14:30:45.000123'

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = 2023
    var_2 = 5
    var_3 = 15
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = '2023-05-15T14:30:45'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = -5
    var_2 = -30
    var_3 = module_1.timedelta()
    var_4 = 2023
    var_5 = 5
    var_6 = 15
    var_7 = 14
    var_8 = 30
    var_9 = 45
    var_10 = 123456
    var_11 = '2023-05-15T14:30:45.123456-05:30'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime. Retrieved 3/5 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_microseconds. Retrieved 3/5 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_timezone. Retrieved 3/5 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_utc_z. Retrieved 3/5 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_short_microseconds. Retrieved 3/5 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_negative_timezone. Retrieved 3/5 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_timezone_no_minutes. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45Z'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = module_0.DateTimeFormat()
    var_5 = 'Z'



# Parsed testcases at query #37
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '999.999.999.999'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_uuid_format_validate_returns_uuid_instance. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_validate_valid_time_without_microseconds. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_short_microseconds. Retrieved 7/8 statements.
# Partially parsed test_validate_valid_time_with_single_digit_hour. Retrieved 6/7 statements.
# Partially parsed test_validate_valid_time_with_zero_hour. Retrieved 4/5 statements.
# Partially parsed test_validate_valid_time_with_max_values. Retrieved 6/7 statements.


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
    var_1 = '01:23:45'
    var_2 = var_0.validate(var_1)
    var_3 = 1
    var_4 = 23
    var_5 = 45

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '00:00:00'
    var_2 = var_0.validate(var_1)
    var_3 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '23:59:59.999999'
    var_2 = var_0.validate(var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12-34-56'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56 extra'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '24:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:60:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:60'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:34:56.1000000'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '-1:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_date. Retrieved 6/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-31'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.


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
    var_1 = '12345678123456781234567812345678'
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
    var_1 = '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime. Retrieved 3/7 statements.
# Partially parsed test_validate_handles_microseconds_correctly. Retrieved 3/7 statements.
# Partially parsed test_validate_handles_utc_timezone. Retrieved 3/7 statements.
# Partially parsed test_validate_handles_positive_timezone_offset. Retrieved 6/11 statements.
# Partially parsed test_validate_handles_negative_timezone_offset. Retrieved 5/10 statements.
# Partially parsed test_validate_handles_short_timezone_offset. Retrieved 5/10 statements.
# Partially parsed test_validate_handles_edge_case_datetime. Retrieved 3/7 statements.
# Partially parsed test_validate_handles_leap_year. Retrieved 3/7 statements.
# Partially parsed test_validate_handles_midnight. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45.123456'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45Z'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05:30'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = 30
    var_5 = module_1.timedelta()

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45-08:00'
    var_2 = var_0.validate(var_1)
    var_3 = -8
    var_4 = module_1.timedelta()

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T14:30:45+05'
    var_2 = var_0.validate(var_1)
    var_3 = 5
    var_4 = module_1.timedelta()

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '0001-01-01T00:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2024-02-29T12:00:00'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-12-31T23:59:59.999999'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_validate_valid_date. Retrieved 6/7 statements.
# Partially parsed test_validate_leap_year. Retrieved 6/7 statements.
# Partially parsed test_validate_single_digit_month_and_day. Retrieved 5/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 25

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023/12/25'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-30'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2020-02-29'
    var_2 = var_0.validate(var_1)
    var_3 = 2020
    var_4 = 2
    var_5 = 29

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-02-29'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-13-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-32'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '0000-01-01'
    var_2 = var_0.validate(var_1)

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-1-1'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25T00:00:00'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = module_0.DateTimeFormat()
    var_5 = 'Z'



# Parsed testcases at query #45
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = 'not_an_ip'
    var_2 = var_0.validate(var_1)



