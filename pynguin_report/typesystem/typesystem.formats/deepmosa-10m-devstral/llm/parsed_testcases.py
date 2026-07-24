####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 5/8 statements.
# Partially parsed test_serialize_datetime_with_microseconds_and_utc. Retrieved 6/9 statements.


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
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.datetime(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '2023-01-01T12:00:00'

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
    var_14 = 1
    var_15 = 12
    var_16 = 0
    var_17 = [var_13, var_14, var_14, var_15, var_16, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -3
    var_4 = -45
    var_5 = []
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.timedelta(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 12
    var_16 = 0
    var_17 = [var_13, var_14, var_14, var_15, var_16, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00-03:45'

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
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 123456



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_utc_timezone_replaces_plus_0000_with_Z. Retrieved 4/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serialize_date_min. Retrieved 2/4 statements.
# Partially parsed test_serialize_date_max. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.DateFormat(*var_6, **var_7)
    var_9 = var_8.serialize(var_5)
    assert var_9 == '2023-05-15'

import typesystem.formats as module_0

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = {}
    var_3 = module_0.DateFormat(*var_1, **var_2)

import typesystem.formats as module_0

def test_case_0():
    var_0 = 12
    var_1 = 31
    var_2 = []
    var_3 = {}
    var_4 = module_0.DateFormat(*var_2, **var_3)



# Parsed testcases at query #4
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12:34:56.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12:34:56.123'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12-34-56'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #5
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
    var_3 = 'invalid-email'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.0.2.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.0.2.1'

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '2001:db8::'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #7
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.time(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.TimeFormat(*var_6, **var_7)
    var_9 = var_8.serialize(var_5)
    assert var_9 == '12:30:45'

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = 123456
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.time(*var_4, **var_5)
    var_7 = []
    var_8 = {}
    var_9 = module_1.TimeFormat(*var_7, **var_8)
    var_10 = var_9.serialize(var_6)
    assert var_10 == '12:30:45.123456'

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2
    var_1 = []
    var_2 = 'hours'
    var_3 = {var_2: var_0}
    var_4 = module_0.timedelta(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.timezone(*var_5, **var_6)
    var_8 = 12
    var_9 = 30
    var_10 = 45
    var_11 = [var_8, var_9, var_10]
    var_12 = 'tzinfo'
    var_13 = {var_12: var_7}
    var_14 = module_0.time(*var_11, **var_13)
    var_15 = []
    var_16 = {}
    var_17 = module_1.TimeFormat(*var_15, **var_16)
    var_18 = var_17.serialize(var_14)
    assert var_18 == '12:30:45+02:00'

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = 1
    var_4 = [var_0, var_1, var_2]
    var_5 = 'fold'
    var_6 = {var_5: var_3}
    var_7 = module_0.time(*var_4, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_1.TimeFormat(*var_8, **var_9)
    var_11 = var_10.serialize(var_7)
    assert var_11 == '12:30:45'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc_timezone. Retrieved 7/9 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 5
    var_10 = 30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = -3
    var_10 = []
    var_11 = 'hours'
    var_12 = {var_11: var_9}
    var_13 = module_1.timedelta(*var_10, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = bool(var_4 == var_20)
    assert var_21 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'invalid-datetime'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-15'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
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
    var_3 = '15-01-2023'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc. Retrieved 7/9 statements.
# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 5
    var_10 = 30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = -3
    var_10 = []
    var_11 = 'hours'
    var_12 = {var_11: var_9}
    var_13 = module_1.timedelta(*var_10, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = bool(var_4 == var_20)
    assert var_21 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #11
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #12
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.BaseFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.is_native_type(var_3)



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = '+00'
    var_1 = len(var_0)
    var_2 = 3
    var_3 = var_1 > var_2
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://www.example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://www.example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'www.example.com'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https:'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #15
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc_timezone. Retrieved 7/9 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456
    var_10 = [var_5, var_6, var_6, var_7, var_8, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = bool(var_4 == var_12)
    assert var_13 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 5
    var_10 = 30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00-05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = -5
    var_10 = -30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023/01/01 12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.UUID(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.UUID(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.UUID(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'invalid-uuid-string'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUID(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.UUIDFormat(*var_2, **var_3)
    var_5 = var_4.validate(var_1)
    var_6 = bool(var_5 == var_1)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_valid_datetime_with_microseconds. Retrieved 9/11 statements.
# Partially parsed test_validate_valid_datetime_without_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:34:56.123456Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 34
    var_9 = 56
    var_10 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:34:56Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 34
    var_9 = 56

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:34:56+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 34
    var_9 = 56
    var_10 = 5
    var_11 = 30
    var_12 = []
    var_13 = 'hours'
    var_14 = 'minutes'
    var_15 = {var_13: var_10, var_14: var_11}
    var_16 = module_1.timedelta(*var_12, **var_15)
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.timezone(*var_17, **var_18)
    var_20 = [var_5, var_6, var_6, var_7, var_8, var_9]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_19}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:34:56-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 34
    var_9 = 56
    var_10 = -3
    var_11 = []
    var_12 = 'hours'
    var_13 = {var_12: var_10}
    var_14 = module_1.timedelta(*var_11, **var_13)
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.timezone(*var_15, **var_16)
    var_18 = [var_5, var_6, var_6, var_7, var_8, var_9]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_17}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = bool(var_4 == var_21)
    assert var_22 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01 12:34:56'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:34:56Z'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)



# Parsed testcases at query #21
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12:34:56'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12:34:56.123'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12-34-56'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 34
    var_2 = 56
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.time(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.TimeFormat(*var_6, **var_7)
    var_9 = var_8.validate(var_5)
    var_10 = bool(var_9 == var_5)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #23
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #24
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-31T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #25
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_with_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_with_valid_ipv6. Retrieved 4/6 statements.


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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #27
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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 31
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
    var_3 = '31-12-2023'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.validate(var_8)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_with_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_with_valid_ipv6. Retrieved 4/6 statements.


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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_returns_ipaddress_on_success. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #31
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #32
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv4Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv6Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = module_1.UUID(var_0)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'invalid_ip'



# Parsed testcases at query #36
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True



# Parsed testcases at query #38
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv4Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv6Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #39
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #40
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-15'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
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
    var_3 = '15-01-2023'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #41
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_validate_valid_datetime_with_zulu_timezone. Retrieved 7/9 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456
    var_10 = [var_5, var_6, var_6, var_7, var_8, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = bool(var_4 == var_12)
    assert var_13 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 5
    var_10 = 30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = -3
    var_10 = []
    var_11 = 'hours'
    var_12 = {var_11: var_9}
    var_13 = module_1.timedelta(*var_10, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = bool(var_4 == var_20)
    assert var_21 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #43
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = module_1.UUID(var_0)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_validate_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_braces. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_urn. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = '{12345678-1234-5678-1234-567812345678}'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'invalid-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #45
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #46
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-15'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
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
    var_3 = '15-01-2023'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #47
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12:34:56.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12:34:56.123'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12-34-56'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #48
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #49
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_validate_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_without_hyphens. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = '{12345678-1234-5678-1234-567812345678}'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678123456781234567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'invalid-uuid-string'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #51
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-31T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #52
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #53
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True



# Parsed testcases at query #55
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #56
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #57
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #58
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1
    var_8 = var_4.hour
    assert var_8 == 12
    var_9 = var_4.minute
    assert var_9 == 0
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1
    var_8 = var_4.hour
    assert var_8 == 12
    var_9 = var_4.minute
    assert var_9 == 0
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.microsecond
    assert var_11 == 123456
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1
    var_8 = var_4.hour
    assert var_8 == 12
    var_9 = var_4.minute
    assert var_9 == 0
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1
    var_8 = var_4.hour
    assert var_8 == 12
    var_9 = var_4.minute
    assert var_9 == 0
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = 5
    var_13 = 30
    var_14 = []
    var_15 = 'hours'
    var_16 = 'minutes'
    var_17 = {var_15: var_12, var_16: var_13}
    var_18 = module_1.timedelta(*var_14, **var_17)
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.timezone(*var_19, **var_20)
    var_22 = var_4.tzinfo
    var_23 = bool(var_4.tzinfo == var_21)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1
    var_8 = var_4.hour
    assert var_8 == 12
    var_9 = var_4.minute
    assert var_9 == 0
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = -3
    var_13 = []
    var_14 = 'hours'
    var_15 = {var_14: var_12}
    var_16 = module_1.timedelta(*var_13, **var_15)
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.timezone(*var_17, **var_18)
    var_20 = var_4.tzinfo
    var_21 = bool(var_4.tzinfo == var_19)
    assert var_21 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'invalid-datetime'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'invalid-email'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUID(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.UUIDFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 6/9 statements.


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
    var_4 = 1
    var_5 = 12
    var_6 = 30
    var_7 = 45
    var_8 = [var_3, var_4, var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.datetime(*var_8, **var_9)
    var_11 = var_2.serialize(var_10)
    assert var_11 == '2023-01-01T12:30:45'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 30
    var_7 = 45

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
    var_14 = 1
    var_15 = 12
    var_16 = 45
    var_17 = [var_13, var_14, var_14, var_15, var_4, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:30:45+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -3
    var_4 = -45
    var_5 = []
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.timedelta(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 12
    var_16 = 30
    var_17 = 45
    var_18 = [var_13, var_14, var_14, var_15, var_16, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_12}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    assert var_22 == '2023-01-01T12:30:45-03:45'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 30
    var_7 = 45
    var_8 = 123456
    var_9 = [var_3, var_4, var_4, var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = var_2.serialize(var_11)
    assert var_12 == '2023-01-01T12:30:45.123456'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_with_valid_ipv6. Retrieved 4/6 statements.


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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://www.example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://www.example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'www.example.com'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https:'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #6
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-15'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
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
    var_3 = '15-01-2023'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-15 12:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-Jan-15'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #7
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.0.2.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '192.0.2.1'

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '2001:db8::'

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
    var_3 = '::ffff:192.0.2.1'
    var_4 = module_1.IPv6Address(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '::ffff:192.0.2.1'

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



# Parsed testcases at query #8
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.DateFormat(*var_6, **var_7)
    var_9 = var_8.serialize(var_5)
    assert var_9 == '2023-05-15'

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.DateFormat(*var_6, **var_7)
    var_9 = var_8.serialize(var_5)
    assert var_9 == '2023-01-05'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 7/9 statements.
# Partially parsed test_validate_with_valid_datetime_string_with_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 5
    var_10 = 30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T25:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #10
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 31
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
    var_3 = '31-12-2023'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #12
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
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
    var_3 = '12:34:56.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123456
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
    var_3 = 'invalid_time'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 34
    var_2 = 56
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.time(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.TimeFormat(*var_6, **var_7)
    var_9 = var_8.validate(var_5)
    var_10 = bool(var_9 == var_5)
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_returns_ipaddress_on_valid_input. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 7/9 statements.
# Partially parsed test_validate_with_valid_datetime_string_with_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 5
    var_10 = 30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00-05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = -5
    var_10 = -30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'invalid-datetime'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.time(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.TimeFormat(*var_6, **var_7)
    var_9 = var_8.serialize(var_5)
    assert var_9 == '12:30:45'

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = 123456
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.time(*var_4, **var_5)
    var_7 = []
    var_8 = {}
    var_9 = module_1.TimeFormat(*var_7, **var_8)
    var_10 = var_9.serialize(var_6)
    assert var_10 == '12:30:45.123456'

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2
    var_1 = []
    var_2 = 'hours'
    var_3 = {var_2: var_0}
    var_4 = module_0.timedelta(*var_1, **var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.timezone(*var_5, **var_6)
    var_8 = 12
    var_9 = 30
    var_10 = 45
    var_11 = [var_8, var_9, var_10]
    var_12 = 'tzinfo'
    var_13 = {var_12: var_7}
    var_14 = module_0.time(*var_11, **var_13)
    var_15 = []
    var_16 = {}
    var_17 = module_1.TimeFormat(*var_15, **var_16)
    var_18 = var_17.serialize(var_14)
    assert var_18 == '12:30:45+02:00'

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 12
    var_1 = 30
    var_2 = 45
    var_3 = 1
    var_4 = [var_0, var_1, var_2]
    var_5 = 'fold'
    var_6 = {var_5: var_3}
    var_7 = module_0.time(*var_4, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_1.TimeFormat(*var_8, **var_9)
    var_11 = var_10.serialize(var_7)
    assert var_11 == '12:30:45'



# Parsed testcases at query #16
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.date(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '01-01-2023'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_with_valid_datetime_string. Retrieved 7/9 statements.
# Partially parsed test_validate_with_valid_datetime_string_with_microseconds. Retrieved 8/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 5
    var_10 = 30
    var_11 = []
    var_12 = 'hours'
    var_13 = 'minutes'
    var_14 = {var_12: var_9, var_13: var_10}
    var_15 = module_1.timedelta(*var_11, **var_14)
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'invalid-datetime'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = -3
    var_10 = []
    var_11 = 'hours'
    var_12 = {var_11: var_9}
    var_13 = module_1.timedelta(*var_10, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = bool(var_4 == var_20)
    assert var_21 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_uuid_with_urn_prefix. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = '{12345678-1234-5678-1234-567812345678}'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'invalid-uuid-string'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serialize_utc_timezone_ends_with_Z. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 'Z'



# Parsed testcases at query #20
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #21
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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = module_1.UUID(var_0)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.0.0.1'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #24
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.UUID(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = '12345678-1234-5678-1234-567812345678'
    var_6 = module_1.UUID(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = '12345678-1234-5678-1234-567812345678'
    var_6 = module_1.UUID(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'invalid-uuid-string'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = module_0.UUID(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.UUIDFormat(*var_2, **var_3)
    var_5 = var_4.validate(var_1)
    var_6 = bool(var_5 == var_1)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc. Retrieved 7/9 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00+02:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 2
    var_10 = []
    var_11 = 'hours'
    var_12 = {var_11: var_9}
    var_13 = module_1.timedelta(*var_10, **var_12)
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.timezone(*var_14, **var_15)
    var_17 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = bool(var_4 == var_20)
    assert var_21 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = [var_5, var_6, var_6, var_7, var_8, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01T12:00:00.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 0
    var_9 = 123456
    var_10 = [var_5, var_6, var_6, var_7, var_8, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = bool(var_4 == var_12)
    assert var_13 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-01 12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-32T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #27
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_returns_ipaddress_on_success. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #29
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.date(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '01-01-2023'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_with_invalid_date_format. Retrieved 4/6 statements.
# Partially parsed test_validate_with_invalid_date_value. Retrieved 4/6 statements.
# Partially parsed test_validate_with_empty_string. Retrieved 4/6 statements.
# Partially parsed test_validate_with_partial_date. Retrieved 4/6 statements.
# Partially parsed test_validate_with_extra_characters. Retrieved 4/6 statements.
# Partially parsed test_validate_with_non_numeric_characters. Retrieved 4/6 statements.
# Partially parsed test_validate_with_non_leap_year_february_29. Retrieved 4/6 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.date(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '01-01-2023'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'invalid'
    var_7 = bool('invalid' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-01 12:00'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-AB'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2020-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = 2020
    var_6 = 2
    var_7 = 29
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
    var_3 = '2021-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'invalid'
    var_7 = bool('invalid' in var_5)
    assert var_7 is True

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = []
    var_6 = {}
    var_7 = module_1.DateFormat(*var_5, **var_6)
    var_8 = var_7.validate(var_4)
    var_9 = bool(var_8 == var_4)
    assert var_9 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_serialize_with_utc_timezone. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = []
    var_5 = {}
    var_6 = module_0.DateTimeFormat(*var_4, **var_5)
    var_7 = 'Z'



# Parsed testcases at query #32
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #33
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 5/8 statements.


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
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.datetime(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '2023-01-01T12:00:00'

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
    var_17 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_16}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00+05:30'

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
    var_7 = -3
    var_8 = -45
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
    assert var_21 == '2023-01-01T12:00:00-03:45'

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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serialize_with_utc_timezone. Retrieved 5/9 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = []
    var_4 = {}
    var_5 = module_0.DateTimeFormat(*var_3, **var_4)
    var_6 = 'Z'



# Parsed testcases at query #37
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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_validate_with_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_with_valid_ipv6. Retrieved 4/6 statements.


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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_validate_returns_ipaddress_on_success. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #41
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-31T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True



# Parsed testcases at query #43
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #44
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 5/8 statements.
# Partially parsed test_serialize_datetime_with_microseconds_and_utc. Retrieved 6/9 statements.


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
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.datetime(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '2023-01-01T12:00:00'

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
    var_14 = 1
    var_15 = 12
    var_16 = 0
    var_17 = [var_13, var_14, var_14, var_15, var_16, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -3
    var_4 = -45
    var_5 = []
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.timedelta(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 12
    var_16 = 0
    var_17 = [var_13, var_14, var_14, var_15, var_16, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00-03:45'

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
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 123456



# Parsed testcases at query #46
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_validate_raises_invalid_error_for_invalid_ip. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.300.999.1000'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_3)
    assert var_5 == 'Must be a real IP.'



# Parsed testcases at query #48
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-31T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #49
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_validate_valid_date_string. Retrieved 7/9 statements.
# Partially parsed test_validate_invalid_date_format. Retrieved 4/6 statements.
# Partially parsed test_validate_invalid_date_value. Retrieved 4/6 statements.
# Partially parsed test_validate_none_value. Retrieved 4/6 statements.
# Partially parsed test_validate_non_string_value. Retrieved 4/6 statements.
# Partially parsed test_validate_date_object. Retrieved 5/7 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-15'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
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
    var_3 = '15-01-2023'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'invalid'
    var_7 = bool('invalid' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import datetime as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = []
    var_7 = {}
    var_8 = module_1.DateFormat(*var_6, **var_7)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_serialize_datetime_with_utc_timezone. Retrieved 5/8 statements.


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
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = [var_3, var_4, var_4, var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.datetime(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '2023-01-01T12:00:00'

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
    var_14 = 1
    var_15 = 12
    var_16 = 0
    var_17 = [var_13, var_14, var_14, var_15, var_16, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00+05:30'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -3
    var_4 = -45
    var_5 = []
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.timedelta(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 12
    var_16 = 0
    var_17 = [var_13, var_14, var_14, var_15, var_16, var_16]
    var_18 = 'tzinfo'
    var_19 = {var_18: var_12}
    var_20 = module_1.datetime(*var_17, **var_19)
    var_21 = var_2.serialize(var_20)
    assert var_21 == '2023-01-01T12:00:00-03:45'

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



# Parsed testcases at query #52
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = module_1.UUID(var_0)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True



# Parsed testcases at query #53
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv4Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv6Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #54
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv4Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv6Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #55
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-15'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
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
    var_3 = '15-01-2023'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #56
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #57
#--------------------------




import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = module_1.UUID(var_0)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True



# Parsed testcases at query #58
#--------------------------




import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv4Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = module_1.IPv6Address(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #59
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_serialize_with_utc_timezone. Retrieved 5/9 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = []
    var_4 = {}
    var_5 = module_0.DateTimeFormat(*var_3, **var_4)
    var_6 = 'Z'



# Parsed testcases at query #61
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #62
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T12:00:00Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_validate_with_invalid_date_format. Retrieved 4/6 statements.
# Partially parsed test_validate_with_invalid_date. Retrieved 4/6 statements.
# Partially parsed test_validate_with_non_string_input. Retrieved 4/6 statements.
# Partially parsed test_validate_with_none_input. Retrieved 4/6 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 31
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
    var_3 = '31-12-2023'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'invalid'
    var_7 = bool('invalid' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = str(var_2)
    var_6 = 'format'
    var_7 = bool('format' in var_5)
    assert var_7 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_validate_with_valid_uuid_string. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = '12345678-1234-5678-1234-567812345678'
    var_1 = []
    var_2 = {}
    var_3 = module_0.UUIDFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)
    var_5 = str(var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_serialize_utc_timezone_replaces_plus_00_00_with_z. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 0
    var_7 = 'Z'



# Parsed testcases at query #66
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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'invalid_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.168.1.1'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #67
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '25:00:00'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_validate_returns_ipaddress_on_success. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '192.168.1.1'
    var_4 = var_2.validate(var_3)



