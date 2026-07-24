####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_utc_zulu. Retrieved 9/11 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_1.datetime(*var_11, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123456
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123000
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 5
    var_6 = 30
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.timedelta(*var_7, **var_10)
    var_12 = 2023
    var_13 = 1
    var_14 = 15
    var_15 = 14
    var_16 = 45
    var_17 = [var_11]
    var_18 = {}
    var_19 = module_1.timezone(*var_17, **var_18)
    var_20 = [var_12, var_13, var_14, var_15, var_6, var_16]
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
    var_3 = '2023-01-15T14:30:45-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = -3
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = 2023
    var_11 = 1
    var_12 = 15
    var_13 = 14
    var_14 = 30
    var_15 = 45
    var_16 = [var_9]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_10, var_11, var_12, var_13, var_14, var_15]
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
    var_3 = '2023-01-15T14:30:45+02'
    var_4 = var_2.validate(var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = 2023
    var_11 = 1
    var_12 = 15
    var_13 = 14
    var_14 = 30
    var_15 = 45
    var_16 = [var_9]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_10, var_11, var_12, var_13, var_14, var_15]
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
    var_3 = '2023-02-30T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.987654-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = -8
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = 2023
    var_11 = 1
    var_12 = 15
    var_13 = 14
    var_14 = 30
    var_15 = 45
    var_16 = 987654
    var_17 = [var_9]
    var_18 = {}
    var_19 = module_1.timezone(*var_17, **var_18)
    var_20 = [var_10, var_11, var_12, var_13, var_14, var_15, var_16]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_19}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'http://example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com/path'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com/path'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'example.com'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http:'
    var_4 = var_2.validate(var_3)



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
    var_3 = 'hello'
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
    var_5 = bool(var_4 == var_3)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_uuidformat_validate_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_urn. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '12345678-1234-5678-1234-56781234567'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-56781234567g'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'not-a-uuid-at-all'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



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

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user@sub.example.co.uk'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user@sub.example.co.uk'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user+tag@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user+tag@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'first.last@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'first.last@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user123@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user123@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user_name@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user_name@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user-name@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user-name@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'userexample.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user@'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = '@example.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user name@example.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user@name@example.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #6
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
    var_5 = 45
    var_6 = 123456
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    var_11 = '14:30:45.123456'
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 14
    var_4 = 30
    var_5 = 45
    var_6 = 0
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    var_11 = '14:30:45'
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 14
    var_4 = 30
    var_5 = 45
    var_6 = 123
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    var_11 = '14:30:45.000123'
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
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
    var_13 = 14
    var_14 = 45
    var_15 = [var_13, var_4, var_14]
    var_16 = 'tzinfo'
    var_17 = {var_16: var_12}
    var_18 = module_1.time(*var_15, **var_17)
    var_19 = var_2.serialize(var_18)
    var_20 = '14:30:45+05:30'
    var_21 = bool(var_19 == var_20)
    assert var_21 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 14
    var_4 = 30
    var_5 = 45
    var_6 = 1
    var_7 = [var_3, var_4, var_5]
    var_8 = 'fold'
    var_9 = {var_8: var_6}
    var_10 = module_1.time(*var_7, **var_9)
    var_11 = var_2.serialize(var_10)
    var_12 = '14:30:45'
    var_13 = bool(var_11 == var_12)
    assert var_13 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 0
    var_4 = [var_3, var_3, var_3]
    var_5 = {}
    var_6 = module_1.time(*var_4, **var_5)
    var_7 = var_2.serialize(var_6)
    var_8 = '00:00:00'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999
    var_6 = [var_3, var_4, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    var_10 = '23:59:59.999999'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    var_10 = '2023-05-15'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

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
    var_3 = 2020
    var_4 = 2
    var_5 = 29
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    var_10 = '2020-02-29'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

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
    var_8 = '0001-01-01'
    var_9 = bool(var_7 == var_8)
    assert var_9 is True

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
    var_10 = '9999-12-31'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_1.date(*var_5, **var_6)
    var_8 = var_2.serialize(var_7)
    var_9 = '2023-01-01'
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

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
    var_9 = var_2.serialize(var_8)
    var_10 = '2023-12-31'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
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
    var_3 = '2023/12/25'
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
    var_3 = '2023-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-13-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-32'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '0000-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-1-1'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.date(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '0001-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 1
    var_6 = [var_5, var_5, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '9999-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = 9999
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
    var_3 = '2023-12-25T00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023.12.25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023--25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 20231225
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '0200-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 200
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
    var_3 = '2023-04-31'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-06-31'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-09-31'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-11-31'
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '01:23:45'
    var_4 = var_2.validate(var_3)
    var_5 = 1
    var_6 = 23
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
    var_3 = '00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 0
    var_6 = [var_5, var_5, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '23:59:59'
    var_4 = var_2.validate(var_3)
    var_5 = 23
    var_6 = 59
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56abc'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '24:00:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:60:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:60'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1000000'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '-1:34:56'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 100000
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
    var_3 = '12:34:56.12'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 120000
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123400
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
    var_3 = '12:34:56.12345'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123450
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_returns_isoformat_with_z_for_utc_timezone. Retrieved 9/12 statements.


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
    var_4 = 5
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = var_2.serialize(var_12)
    var_14 = '2023-05-15T14:30:45.123456'
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = '2023-05-15T14:30:45.123456Z'

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
    var_14 = 15
    var_15 = 14
    var_16 = 45
    var_17 = 123456
    var_18 = [var_13, var_3, var_14, var_15, var_4, var_16, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_12}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    var_23 = '2023-05-15T14:30:45.123456+05:30'
    var_24 = bool(var_22 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = -5
    var_4 = -30
    var_5 = []
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.timedelta(*var_5, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 5
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = 123456
    var_20 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_12}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = var_2.serialize(var_23)
    var_25 = '2023-05-15T14:30:45.123456-05:30'
    var_26 = bool(var_24 == var_25)
    assert var_26 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 0
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 2023
    var_12 = 5
    var_13 = 15
    var_14 = 14
    var_15 = 30
    var_16 = 45
    var_17 = 123456
    var_18 = [var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_10}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    var_23 = '2023-05-15T14:30:45.123456Z'
    var_24 = bool(var_22 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = [var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = var_2.serialize(var_11)
    var_13 = '2023-05-15T14:30:45'
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 0
    var_10 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = var_2.serialize(var_12)
    var_14 = '2023-05-15T14:30:45'
    var_15 = bool(var_13 == var_14)
    assert var_15 is True



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime. Retrieved 3/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    assert var_12 is None



# Parsed testcases at query #13
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
    var_6 = '12345678-1234-5678-1234-567812345678'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '00000000-0000-0000-0000-000000000000'
    var_1 = module_0.UUID(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.UUIDFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    var_6 = '00000000-0000-0000-0000-000000000000'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'ABCDEFAB-1234-5678-9ABC-DEF123456789'
    var_1 = module_0.UUID(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.UUIDFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    var_6 = 'abcdefab-1234-5678-9abc-def123456789'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import uuid as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '6ba7b810-9dad-11d1-80b4-00c04fd430c8'
    var_1 = module_0.UUID(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.UUIDFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    var_6 = '6ba7b810-9dad-11d1-80b4-00c04fd430c8'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
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
    var_3 = '2023/12/25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-13-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_3 = '2023-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-00-15'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-04-31'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-1-5'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 5
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '0001-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 1
    var_6 = [var_5, var_5, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '9999-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = 9999
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
    var_3 = 'not-a-date'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25 extra'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-2-3'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 2
    var_7 = 3
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
    var_3 = '10000-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #16
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.BaseFormat(*var_0, **var_1)
    var_3 = 'test_value'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_urn. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_uppercase. Retrieved 2/7 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.
# Partially parsed test_uuid_format_validate_with_uuid_object. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_uuid_object_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_uuid_object_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_uuid_object_urn. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_uuid_object_uppercase. Retrieved 2/7 statements.
# Partially parsed test_uuid_format_validate_with_uuid_object_mixed_case. Retrieved 2/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678123456781234567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'urn:uuid:12345678-1234-5678-1234-567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #19
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234567'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_curly_braces. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/8 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix_and_curly_braces. Retrieved 4/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = 'urn:uuid:{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_returns_time_object_without_raising_value_error. Retrieved 3/7 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 30
    var_7 = var_4.second
    assert var_7 == 45
    var_8 = var_4.microsecond
    assert var_8 == 0
    var_9 = var_4.tzinfo
    assert var_9 is None



# Parsed testcases at query #22
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 30
    var_7 = var_4.second
    assert var_7 == 0
    var_8 = var_4.microsecond
    assert var_8 == 0
    var_9 = var_4.tzinfo
    assert var_9 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 30
    var_7 = var_4.second
    assert var_7 == 45
    var_8 = var_4.microsecond
    assert var_8 == 0
    var_9 = var_4.tzinfo
    assert var_9 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 30
    var_7 = var_4.second
    assert var_7 == 45
    var_8 = var_4.microsecond
    assert var_8 == 123456
    var_9 = var_4.tzinfo
    assert var_9 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 30
    var_7 = var_4.second
    assert var_7 == 45
    var_8 = var_4.microsecond
    assert var_8 == 123000
    var_9 = var_4.tzinfo
    assert var_9 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'ab:cd:ef'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '24:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:60:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:60'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.1000000'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '05:07:09.000123'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 5
    var_6 = var_4.minute
    assert var_6 == 7
    var_7 = var_4.second
    assert var_7 == 9
    var_8 = var_4.microsecond
    assert var_8 == 123
    var_9 = var_4.tzinfo
    assert var_9 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 0
    var_6 = var_4.minute
    assert var_6 == 0
    var_7 = var_4.second
    assert var_7 == 0
    var_8 = var_4.microsecond
    assert var_8 == 0
    var_9 = var_4.tzinfo
    assert var_9 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '23:59:59.999999'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 23
    var_6 = var_4.minute
    assert var_6 == 59
    var_7 = var_4.second
    assert var_7 == 59
    var_8 = var_4.microsecond
    assert var_8 == 999999
    var_9 = var_4.tzinfo
    assert var_9 is None



# Parsed testcases at query #23
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 25
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_with_utc_zulu. Retrieved 9/11 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_1.datetime(*var_11, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123456
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123000
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05:30'
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
    var_16 = 1
    var_17 = 15
    var_18 = 14
    var_19 = 45
    var_20 = [var_15, var_16, var_17, var_18, var_6, var_19]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_14}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45-03:00'
    var_4 = var_2.validate(var_3)
    var_5 = -3
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+02'
    var_4 = var_2.validate(var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
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
    var_3 = '2023-02-30T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+00:45'
    var_4 = var_2.validate(var_3)
    var_5 = 45
    var_6 = []
    var_7 = 'minutes'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = [var_13, var_14, var_15, var_16, var_17, var_5]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_12}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = bool(var_4 == var_21)
    assert var_22 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '2001:db8::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8::1'



# Parsed testcases at query #26
#--------------------------




import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '192.168.1.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '192.168.1.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2001:db8::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc_zulu. Retrieved 9/11 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_1.datetime(*var_11, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True
    var_15 = var_4.tzinfo
    assert var_15 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123456
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True
    var_16 = var_4.tzinfo
    assert var_16 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123000
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True
    var_16 = var_4.tzinfo
    assert var_16 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05:30'
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
    var_16 = 1
    var_17 = 15
    var_18 = 14
    var_19 = 45
    var_20 = [var_15, var_16, var_17, var_18, var_6, var_19]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_14}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = -8
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+02'
    var_4 = var_2.validate(var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 'not-a-datetime'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_mapped_ipv6. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::ffff:192.0.2.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::ffff:192.0.2.1'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_version_1. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_version_4. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_version_5. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'c232ab00-9414-11ec-b3c8-9f6b6a716856'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'c232ab00-9414-11ec-b3c8-9f6b6a716856'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'f47ac10b-58cc-4372-a567-0e02b2c3d479'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '74738ff5-5367-5958-9aee-98fffdcd1876'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '74738ff5-5367-5958-9aee-98fffdcd1876'



# Parsed testcases at query #30
#--------------------------




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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_returns_ipv4_address. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address. Retrieved 4/6 statements.


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
    var_3 = '::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::1'



# Parsed testcases at query #32
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '01:23:45'
    var_4 = var_2.validate(var_3)
    var_5 = 1
    var_6 = 23
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
    var_3 = '23:59:59'
    var_4 = var_2.validate(var_3)
    var_5 = 23
    var_6 = 59
    var_7 = [var_5, var_6, var_6]
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
    var_3 = '00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 0
    var_6 = [var_5, var_5, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'ab:cd:ef'
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

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '24:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:60:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:60'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1000000'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 100000
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
    var_3 = '12:34:56.12'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 120000
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123400
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
    var_3 = '12:34:56.12345'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123450
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
    var_3 = '12:34:56.123456Z'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_mapped_ipv6. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::ffff:192.0.2.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::ffff:192.0.2.1'



# Parsed testcases at query #37
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 3232235777
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #39
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_validate_time_with_invalid_microsecond. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '12:34:56.1234567'
    var_1 = None
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_date. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 12
    var_7 = var_4.day
    assert var_7 == 25



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_validate_with_valid_datetime_with_positive_offset_should_not_raise_value_error. Retrieved 7/9 statements.
# Partially parsed test_validate_with_valid_datetime_with_negative_offset_should_not_raise_value_error. Retrieved 6/8 statements.
# Partially parsed test_validate_with_valid_datetime_with_short_offset_should_not_raise_value_error. Retrieved 6/8 statements.
# Partially parsed test_validate_with_valid_datetime_with_microseconds_and_timezone_should_not_raise_value_error. Retrieved 6/8 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 123456
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = None
    var_13 = 5
    var_14 = 30
    var_15 = []
    var_16 = 'hours'
    var_17 = 'minutes'
    var_18 = {var_16: var_13, var_17: var_14}
    var_19 = module_1.timedelta(*var_15, **var_18)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = None
    var_13 = -8
    var_14 = []
    var_15 = 'hours'
    var_16 = {var_15: var_13}
    var_17 = module_1.timedelta(*var_14, **var_16)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = None
    var_13 = 5
    var_14 = []
    var_15 = 'hours'
    var_16 = {var_15: var_13}
    var_17 = module_1.timedelta(*var_14, **var_16)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.987654+02:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 987654
    var_12 = None
    var_13 = 2
    var_14 = []
    var_15 = 'hours'
    var_16 = {var_15: var_13}
    var_17 = module_1.timedelta(*var_14, **var_16)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 123000
    var_12 = var_4.tzinfo
    assert var_12 is None



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
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
    var_3 = '2023/12/25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-13-01'
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
    var_3 = '2023-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2024-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = 2024
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
    var_3 = '2023-00-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '0000-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023 12 25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25T00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-2-5'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '0001-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 1
    var_6 = [var_5, var_5, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '9999-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = 9999
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
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '2023-05-15'

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
    var_3 = 2020
    var_4 = 2
    var_5 = 29
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '2020-02-29'

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
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_1.date(*var_5, **var_6)
    var_8 = var_2.serialize(var_7)
    assert var_8 == '2023-01-01'



# Parsed testcases at query #3
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
    var_5 = 45
    var_6 = 123456
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '14:30:45.123456'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 9
    var_4 = 15
    var_5 = 0
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '09:15:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 0
    var_4 = [var_3, var_3, var_3]
    var_5 = {}
    var_6 = module_1.time(*var_4, **var_5)
    var_7 = var_2.serialize(var_6)
    assert var_7 == '00:00:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999
    var_6 = [var_3, var_4, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '23:59:59.999999'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 5
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 12
    var_12 = 0
    var_13 = [var_11, var_12, var_12]
    var_14 = 'tzinfo'
    var_15 = {var_14: var_10}
    var_16 = module_1.time(*var_13, **var_15)
    var_17 = var_2.serialize(var_16)
    assert var_17 == '12:00:00+05:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = -8
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 8
    var_12 = 30
    var_13 = 15
    var_14 = [var_11, var_12, var_13]
    var_15 = 'tzinfo'
    var_16 = {var_15: var_10}
    var_17 = module_1.time(*var_14, **var_16)
    var_18 = var_2.serialize(var_17)
    assert var_18 == '08:30:15-08:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 1
    var_4 = 30
    var_5 = 0
    var_6 = [var_3, var_4, var_5]
    var_7 = 'fold'
    var_8 = {var_7: var_3}
    var_9 = module_1.time(*var_6, **var_8)
    var_10 = var_2.serialize(var_9)
    assert var_10 == '01:30:00'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 3232235777
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '02:34:56'
    var_4 = var_2.validate(var_3)
    var_5 = 2
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
    var_3 = '00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 0
    var_6 = [var_5, var_5, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '23:59:59'
    var_4 = var_2.validate(var_3)
    var_5 = 23
    var_6 = 59
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '24:00:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:60:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:60'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1000000'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 100000
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
    var_3 = '12:34:56.12'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 120000
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123400
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
    var_3 = '12:34:56.12345'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
    var_7 = 56
    var_8 = 123450
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



# Parsed testcases at query #6
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

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
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '00000000-0000-0000-0000-000000000000'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == '00000000-0000-0000-0000-000000000000'

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'ABCDEFAB-1234-5678-9ABC-DEF123456789'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.serialize(var_4)
    assert var_5 == 'abcdefab-1234-5678-9abc-def123456789'



# Parsed testcases at query #7
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234567'
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



# Parsed testcases at query #8
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 12
    var_7 = var_4.day
    assert var_7 == 25



# Parsed testcases at query #9
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'http://example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com/path'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com/path'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://example.com?query=value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'http://example.com?query=value'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'example.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http:'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_utc_zulu. Retrieved 9/11 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 0
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_1.datetime(*var_11, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 0
    var_11 = 123456
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00.123'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 0
    var_11 = 123000
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 0

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00+05:30'
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
    var_16 = 1
    var_17 = 15
    var_18 = 14
    var_19 = 0
    var_20 = [var_15, var_16, var_17, var_18, var_6, var_19]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_14}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = -8
    var_6 = 0
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.timedelta(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.timezone(*var_12, **var_13)
    var_15 = 2023
    var_16 = 1
    var_17 = 15
    var_18 = 14
    var_19 = 30
    var_20 = [var_15, var_16, var_17, var_18, var_19, var_6]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_14}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00+02'
    var_4 = var_2.validate(var_3)
    var_5 = 2
    var_6 = 0
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.timedelta(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.timezone(*var_12, **var_13)
    var_15 = 2023
    var_16 = 1
    var_17 = 15
    var_18 = 14
    var_19 = 30
    var_20 = [var_15, var_16, var_17, var_18, var_19, var_6]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_14}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

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
    var_3 = '2023-13-45T25:61:61'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-12-31T23:59:59.999999-11:30'
    var_4 = var_2.validate(var_3)
    var_5 = -11
    var_6 = -30
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.timedelta(*var_7, **var_10)
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.timezone(*var_12, **var_13)
    var_15 = 2023
    var_16 = 12
    var_17 = 31
    var_18 = 23
    var_19 = 59
    var_20 = 999999
    var_21 = [var_15, var_16, var_17, var_18, var_19, var_19, var_20]
    var_22 = 'tzinfo'
    var_23 = {var_22: var_14}
    var_24 = module_1.datetime(*var_21, **var_23)
    var_25 = bool(var_4 == var_24)
    assert var_25 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc_z. Retrieved 9/11 statements.
# Partially parsed test_validate_valid_datetime_with_short_microseconds_padded. Retrieved 10/12 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_1.datetime(*var_11, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True
    var_15 = var_4.tzinfo
    assert var_15 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = var_4.tzinfo

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = 5
    var_6 = 30
    var_7 = []
    var_8 = 'hours'
    var_9 = 'minutes'
    var_10 = {var_8: var_5, var_9: var_6}
    var_11 = module_1.timedelta(*var_7, **var_10)
    var_12 = 2023
    var_13 = 1
    var_14 = 15
    var_15 = 14
    var_16 = 45
    var_17 = [var_11]
    var_18 = {}
    var_19 = module_1.timezone(*var_17, **var_18)
    var_20 = [var_12, var_13, var_14, var_15, var_6, var_16]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_19}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True
    var_25 = var_4.tzinfo._offset
    var_26 = bool(var_4.tzinfo._offset == var_11)
    assert var_26 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = -8
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = 2023
    var_11 = 1
    var_12 = 15
    var_13 = 14
    var_14 = 30
    var_15 = 45
    var_16 = [var_9]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_10, var_11, var_12, var_13, var_14, var_15]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True
    var_24 = var_4.tzinfo._offset
    var_25 = bool(var_4.tzinfo._offset == var_9)
    assert var_25 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123456
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True
    var_16 = var_4.microsecond
    assert var_16 == 123456

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456+02:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = 2023
    var_11 = 1
    var_12 = 15
    var_13 = 14
    var_14 = 30
    var_15 = 45
    var_16 = 123456
    var_17 = [var_9]
    var_18 = {}
    var_19 = module_1.timezone(*var_17, **var_18)
    var_20 = [var_10, var_11, var_12, var_13, var_14, var_15, var_16]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_19}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True
    var_25 = var_4.microsecond
    assert var_25 == 123456
    var_26 = var_4.tzinfo._offset
    var_27 = bool(var_4.tzinfo._offset == var_9)
    assert var_27 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123000
    var_12 = var_4.microsecond
    assert var_12 == 123000

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
    var_3 = '2023-02-30T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05'
    var_4 = var_2.validate(var_3)
    var_5 = 5
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = 2023
    var_11 = 1
    var_12 = 15
    var_13 = 14
    var_14 = 30
    var_15 = 45
    var_16 = [var_9]
    var_17 = {}
    var_18 = module_1.timezone(*var_16, **var_17)
    var_19 = [var_10, var_11, var_12, var_13, var_14, var_15]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_18}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True
    var_24 = var_4.tzinfo._offset
    var_25 = bool(var_4.tzinfo._offset == var_9)
    assert var_25 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime. Retrieved 3/6 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_microseconds. Retrieved 3/6 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_utc_timezone. Retrieved 3/6 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_positive_offset. Retrieved 6/10 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_negative_offset. Retrieved 5/9 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_short_microseconds. Retrieved 3/6 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_timezone_no_minutes. Retrieved 5/9 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_leap_day. Retrieved 3/6 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_all_fields. Retrieved 3/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 123456
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    var_13 = bool(var_4.tzinfo is not None)
    assert var_13 is True
    var_14 = 5
    var_15 = 30
    var_16 = []
    var_17 = 'hours'
    var_18 = 'minutes'
    var_19 = {var_17: var_14, var_18: var_15}
    var_20 = module_1.timedelta(*var_16, **var_19)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    var_13 = bool(var_4.tzinfo is not None)
    assert var_13 is True
    var_14 = -8
    var_15 = []
    var_16 = 'hours'
    var_17 = {var_16: var_14}
    var_18 = module_1.timedelta(*var_15, **var_17)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 123000
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    var_13 = bool(var_4.tzinfo is not None)
    assert var_13 is True
    var_14 = 5
    var_15 = []
    var_16 = 'hours'
    var_17 = {var_16: var_14}
    var_18 = module_1.timedelta(*var_15, **var_17)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2024-02-29T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2024
    var_6 = var_4.month
    assert var_6 == 2
    var_7 = var_4.day
    assert var_7 == 29
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-12-31T23:59:59.999999+00:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 12
    var_7 = var_4.day
    assert var_7 == 31
    var_8 = var_4.hour
    assert var_8 == 23
    var_9 = var_4.minute
    assert var_9 == 59
    var_10 = var_4.second
    assert var_10 == 59
    var_11 = var_4.microsecond
    assert var_11 == 999999
    var_12 = var_4.tzinfo



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serialize_returns_isoformat_string_with_z_for_utc_timezone. Retrieved 9/12 statements.


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
    var_4 = 5
    var_5 = 17
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = var_2.serialize(var_12)
    var_14 = '2023-05-17T14:30:45.123456'
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 17
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = '2023-05-17T14:30:45.123456Z'

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
    var_14 = 17
    var_15 = 14
    var_16 = 45
    var_17 = 123456
    var_18 = [var_13, var_3, var_14, var_15, var_4, var_16, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_12}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    var_23 = '2023-05-17T14:30:45.123456+05:30'
    var_24 = bool(var_22 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 0
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
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
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    var_23 = '2023-05-17T14:30:45.123456Z'
    var_24 = bool(var_22 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 17
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = [var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.datetime(*var_9, **var_10)
    var_12 = var_2.serialize(var_11)
    var_13 = '2023-05-17T14:30:45'
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 17
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 0
    var_10 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = var_2.serialize(var_12)
    var_14 = '2023-05-17T14:30:45'
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

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
    var_12 = 5
    var_13 = 17
    var_14 = 14
    var_15 = 30
    var_16 = 45
    var_17 = 123456
    var_18 = [var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_10}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    var_23 = '2023-05-17T14:30:45.123456-05:00'
    var_24 = bool(var_22 == var_23)
    assert var_24 is True



# Parsed testcases at query #14
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

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user@sub.example.co.uk'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user@sub.example.co.uk'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'user+tag@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'user+tag@example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.EmailFormat(*var_0, **var_1)
    var_3 = 'first.last@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'first.last@example.com'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc_timezone. Retrieved 9/11 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = [var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_1.datetime(*var_11, **var_12)
    var_14 = bool(var_4 == var_13)
    assert var_14 is True
    var_15 = var_4.tzinfo
    assert var_15 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123456
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True
    var_16 = var_4.tzinfo
    assert var_16 is None

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123000
    var_12 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_1.datetime(*var_12, **var_13)
    var_15 = bool(var_4 == var_14)
    assert var_15 is True
    var_16 = var_4.tzinfo
    assert var_16 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+05:30'
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
    var_16 = 1
    var_17 = 15
    var_18 = 14
    var_19 = 45
    var_20 = [var_15, var_16, var_17, var_18, var_6, var_19]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_14}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = -8
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+02'
    var_4 = var_2.validate(var_3)
    var_5 = 2
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_12}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.987654-05:00'
    var_4 = var_2.validate(var_3)
    var_5 = -5
    var_6 = []
    var_7 = 'hours'
    var_8 = {var_7: var_5}
    var_9 = module_1.timedelta(*var_6, **var_8)
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = 15
    var_16 = 14
    var_17 = 30
    var_18 = 45
    var_19 = 987654
    var_20 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19]
    var_21 = 'tzinfo'
    var_22 = {var_21: var_12}
    var_23 = module_1.datetime(*var_20, **var_22)
    var_24 = bool(var_4 == var_23)
    assert var_24 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023/01/15T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-02-30T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T25:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:60:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:60'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.9999999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45+25:00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------




import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '192.168.1.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    var_6 = '192.168.1.1'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2001:db8::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    var_6 = '2001:db8::1'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'fe80::1%eth0'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    var_6 = 'fe80::1%eth0'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '::ffff:192.168.1.1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    var_6 = '::ffff:192.168.1.1'
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_uuidformat_validate_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_urn. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = 'invalid-uuid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-56781234567'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-56781234567g'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == var_4)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 12
    var_7 = var_4.day
    assert var_7 == 31



# Parsed testcases at query #19
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
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
    var_3 = 'invalid'
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
    var_3 = '2023-13-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-32'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-00-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2024-02-29'
    var_4 = var_2.validate(var_3)
    var_5 = 2024
    var_6 = 2
    var_7 = 29
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_1.date(*var_8, **var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string. Retrieved 4/9 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_without_hyphens. Retrieved 4/9 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_braces. Retrieved 4/9 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_urn_prefix. Retrieved 4/9 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_urn_prefix_and_braces. Retrieved 4/9 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = 'urn:uuid:{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_format. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = 'urn:uuid:{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_returns_time_object_without_raising_value_error. Retrieved 3/5 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 30
    var_7 = var_4.second
    assert var_7 == 45
    var_8 = var_4.microsecond
    assert var_8 == 0
    var_9 = var_4.tzinfo
    assert var_9 is None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_urn_and_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_version_1. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_version_4. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_version_5. Retrieved 4/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = '{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

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
    var_3 = 'urn:uuid:{12345678-1234-5678-1234-567812345678}'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'c232ab00-9414-11ec-b3c8-9a6bdfc4b925'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'c232ab00-9414-11ec-b3c8-9a6bdfc4b925'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '12345678-1234-4234-8234-567812345678'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '12345678-1234-4234-8234-567812345678'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = '74738ff5-5367-5958-9aee-98fffdcd1876'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '74738ff5-5367-5958-9aee-98fffdcd1876'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_datetime_with_valid_input. Retrieved 3/6 statements.
# Partially parsed test_validate_datetime_with_microseconds. Retrieved 3/6 statements.
# Partially parsed test_validate_datetime_with_utc_zulu. Retrieved 3/6 statements.
# Partially parsed test_validate_datetime_with_timezone_offset. Retrieved 6/10 statements.
# Partially parsed test_validate_datetime_with_negative_timezone. Retrieved 5/9 statements.
# Partially parsed test_validate_datetime_with_short_timezone. Retrieved 5/9 statements.
# Partially parsed test_validate_datetime_with_partial_microseconds. Retrieved 3/6 statements.
# Partially parsed test_validate_datetime_with_all_fields. Retrieved 6/10 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 4
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 12
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = var_4.tzinfo
    assert var_12 is None

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    var_6 = bool(var_4.tzinfo is not None)
    assert var_6 is True
    var_7 = 5
    var_8 = 30
    var_9 = []
    var_10 = 'hours'
    var_11 = 'minutes'
    var_12 = {var_10: var_7, var_11: var_8}
    var_13 = module_1.timedelta(*var_9, **var_12)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    var_6 = bool(var_4.tzinfo is not None)
    assert var_6 is True
    var_7 = -8
    var_8 = []
    var_9 = 'hours'
    var_10 = {var_9: var_7}
    var_11 = module_1.timedelta(*var_8, **var_10)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45+05'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo
    var_6 = bool(var_4.tzinfo is not None)
    assert var_6 is True
    var_7 = 5
    var_8 = []
    var_9 = 'hours'
    var_10 = {var_9: var_7}
    var_11 = module_1.timedelta(*var_8, **var_10)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.microsecond
    assert var_5 == 123000

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-15T12:30:45.123456+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 4
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 12
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 123456
    var_12 = var_4.tzinfo
    var_13 = bool(var_4.tzinfo is not None)
    assert var_13 is True
    var_14 = 5
    var_15 = 30
    var_16 = []
    var_17 = 'hours'
    var_18 = 'minutes'
    var_19 = {var_17: var_14, var_18: var_15}
    var_20 = module_1.timedelta(*var_16, **var_19)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_raises_invalid_error_for_invalid_ip. Retrieved 3/6 statements.


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = module_1.ip_address(var_3)



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234567'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_mapped_ipv6. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

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
    var_3 = '::ffff:192.0.2.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::ffff:192.0.2.1'



# Parsed testcases at query #29
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_time_with_invalid_microsecond. Retrieved 6/13 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234567'
    var_4 = 'microsecond'
    var_5 = 6
    var_6 = '0'
    var_7 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_mapped_ipv6. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::ffff:192.0.2.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::ffff:192.0.2.1'



# Parsed testcases at query #33
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_valid_ipv4. Retrieved 4/6 statements.
# Partially parsed test_validate_valid_ipv6. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv6_compressed. Retrieved 4/6 statements.
# Partially parsed test_validate_ipv4_mapped_ipv6. Retrieved 4/6 statements.


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
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::ffff:192.0.2.1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::ffff:192.0.2.1'



# Parsed testcases at query #35
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_validate_does_not_raise_value_error_for_valid_datetime_with_timezone. Retrieved 6/9 statements.


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
    var_11 = var_4.tzinfo
    var_12 = bool(var_4.tzinfo is not None)
    assert var_12 is True
    var_13 = 5
    var_14 = 30
    var_15 = []
    var_16 = 'hours'
    var_17 = 'minutes'
    var_18 = {var_16: var_13, var_17: var_14}
    var_19 = module_1.timedelta(*var_15, **var_18)



# Parsed testcases at query #37
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



# Parsed testcases at query #38
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'not_an_ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #39
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '01:02:03'
    var_4 = var_2.validate(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
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
    var_3 = '23:59:59'
    var_4 = var_2.validate(var_3)
    var_5 = 23
    var_6 = 59
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.time(*var_7, **var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'ab:cd:ef'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56 extra'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '24:00:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:60:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:60'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1000000'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 0
    var_6 = [var_5, var_5, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '23:59:59.999999'
    var_4 = var_2.validate(var_3)
    var_5 = 23
    var_6 = 59
    var_7 = 999999
    var_8 = [var_5, var_6, var_6, var_7]
    var_9 = {}
    var_10 = module_1.time(*var_8, **var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True



# Parsed testcases at query #40
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '256.256.256.256'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_validate_does_not_raise_value_error_for_valid_datetime. Retrieved 3/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:00Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 14
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 0
    var_11 = var_4.tzinfo



# Parsed testcases at query #43
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = '12:34:56.1234567'
    var_1 = []
    var_2 = {}
    var_3 = module_0.TimeFormat(*var_1, **var_2)
    var_4 = var_3.validate(var_0)



# Parsed testcases at query #44
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_validate_returns_ipv4_address_when_value_is_valid_ipv4_string. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address_when_value_is_valid_ipv6_string. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv4_address_when_value_is_valid_ipv4_string_with_leading_zeros. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address_when_value_is_valid_compressed_ipv6_string. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv4_address_when_value_is_valid_ipv4_string_with_max_values. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address_when_value_is_valid_ipv6_string_with_max_values. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv4_address_when_value_is_valid_ipv4_string_with_min_values. Retrieved 4/6 statements.
# Partially parsed test_validate_returns_ipv6_address_when_value_is_valid_ipv6_string_with_min_values. Retrieved 4/6 statements.


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
    var_3 = '010.010.010.010'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '10.10.10.10'

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
    var_3 = '255.255.255.255'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '255.255.255.255'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '0.0.0.0'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '0.0.0.0'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '::'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '::'



