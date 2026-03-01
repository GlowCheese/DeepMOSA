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
    var_3 = 1999
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '1999-12-31'

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



# Parsed testcases at query #2
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'http://example.com'
    var_4 = var_2.is_native_type(var_3)
    assert var_4 is False
    var_5 = None
    var_6 = var_2.is_native_type(var_5)
    assert var_6 is False
    var_7 = 123
    var_8 = var_2.is_native_type(var_7)
    assert var_8 is False
    var_9 = []
    var_10 = var_2.is_native_type(var_9)
    assert var_10 is False



# Parsed testcases at query #3
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 14
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
    var_3 = '14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 14
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
    var_3 = '14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 14
    var_6 = 30
    var_7 = 45
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
    var_3 = '04:05:06'
    var_4 = var_2.validate(var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
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
    var_3 = '14:30'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 'not a time'
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
    var_3 = '14:60:00'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '14:30:60'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '14:30:45.1000000'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '14:30:45.001'
    var_4 = var_2.validate(var_3)
    var_5 = 14
    var_6 = 30
    var_7 = 45
    var_8 = 1000
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
    var_3 = '14:30:45.999999'
    var_4 = var_2.validate(var_3)
    var_5 = 14
    var_6 = 30
    var_7 = 45
    var_8 = 999999
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
    var_3 = '14:30:45.000000'
    var_4 = var_2.validate(var_3)
    var_5 = 14
    var_6 = 30
    var_7 = 45
    var_8 = 0
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
    var_3 = '04:05:06.007'
    var_4 = var_2.validate(var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = 7000
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_1.time(*var_9, **var_10)
    var_12 = bool(var_4 == var_11)
    assert var_12 is True



# Parsed testcases at query #4
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
    var_3 = ' 2023-12-25 '
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
    var_3 = '2023-12'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25-10'
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
    var_3 = '23-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '10000-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-31'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 1
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
    var_3 = '2023-00-25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_returns_isoformat_with_z_for_utc_timezone. Retrieved 9/12 statements.
# Partially parsed test_serialize_converts_utc_offset_to_z. Retrieved 10/17 statements.


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
    var_10 = '+00:00'
    var_11 = 'Z'

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
    var_12 = 5
    var_13 = 15
    var_14 = 14
    var_15 = 30
    var_16 = 45
    var_17 = 0
    var_18 = [var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = 'tzinfo'
    var_20 = {var_19: var_10}
    var_21 = module_1.datetime(*var_18, **var_20)
    var_22 = var_2.serialize(var_21)
    var_23 = '2023-05-15T14:30:45+02:00'
    var_24 = bool(var_22 == var_23)
    assert var_24 is True



# Parsed testcases at query #6
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

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '::ffff:192.168.1.1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '::ffff:192.168.1.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '127.0.0.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '127.0.0.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '::1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '255.255.255.255'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '255.255.255.255'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '10.0.0.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '10.0.0.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'fd00::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == 'fd00::1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '224.0.0.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '224.0.0.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'ff00::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == 'ff00::1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '0.0.0.0'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '0.0.0.0'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '::'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '::'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '169.254.0.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '169.254.0.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'fe80::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == 'fe80::1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = 'fec0::1'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == 'fec0::1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2001:0:4137:9e76:0:0:0:0'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '2001:0:4137:9e76::'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2002:c000:0204::'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '2002:c000:204::'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '8.8.8.8'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '8.8.8.8'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2001:4860:4860::8888'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '2001:4860:4860::8888'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '240.0.0.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '240.0.0.1'

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '100::'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.serialize(var_1)
    assert var_5 == '100::'



# Parsed testcases at query #7
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
    var_3 = '2023-01-15T14:30:45-03:45'
    var_4 = var_2.validate(var_3)
    var_5 = -3
    var_6 = -45
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
    var_20 = 45
    var_21 = [var_15, var_16, var_17, var_18, var_19, var_20]
    var_22 = 'tzinfo'
    var_23 = {var_22: var_14}
    var_24 = module_1.datetime(*var_21, **var_23)
    var_25 = bool(var_4 == var_24)
    assert var_25 is True

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

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T25:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




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
    var_3 = 'https://example.com/path'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com/path'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com?query=value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com?query=value'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com#fragment'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com#fragment'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'ftp://example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'ftp://example.com'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'file:///path/to/file'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'file:///path/to/file'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_uuid_format_validate_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_valid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_valid_string_with_urn. Retrieved 4/6 statements.


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
    var_3 = 'not-a-uuid'
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



# Parsed testcases at query #11
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

# Partially parsed test_validate_with_utc_zulu. Retrieved 9/11 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-05T14:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 5
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
    var_3 = '2023-04-05T14:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 5
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
    var_3 = '2023-04-05T14:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 5
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
    var_3 = '2023-04-05T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 5
    var_8 = 14
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-05T14:30:45+05:30'
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
    var_16 = 4
    var_17 = 14
    var_18 = 45
    var_19 = [var_15, var_16, var_5, var_17, var_6, var_18]
    var_20 = 'tzinfo'
    var_21 = {var_20: var_14}
    var_22 = module_1.datetime(*var_19, **var_21)
    var_23 = bool(var_4 == var_22)
    assert var_23 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-05T14:30:45-08:00'
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
    var_14 = 4
    var_15 = 5
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
    var_3 = '2023-04-05T14:30:45+02'
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
    var_14 = 4
    var_15 = 5
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
    var_3 = '2024-02-29T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = 2024
    var_6 = 2
    var_7 = 29
    var_8 = 12
    var_9 = 0
    var_10 = [var_5, var_6, var_7, var_8, var_9, var_9]
    var_11 = {}
    var_12 = module_1.datetime(*var_10, **var_11)
    var_13 = bool(var_4 == var_12)
    assert var_13 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-05T14:30:45.987654-05:00'
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
    var_14 = 4
    var_15 = 5
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_uuidformat_validate_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_urn. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_uuid_object. Retrieved 5/7 statements.


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
    var_3 = 'not-a-uuid'
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
    var_5 = str(var_4)
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime. Retrieved 3/6 statements.
# Partially parsed test_validate_handles_microseconds_correctly. Retrieved 3/6 statements.
# Partially parsed test_validate_handles_utc_timezone. Retrieved 3/6 statements.
# Partially parsed test_validate_handles_positive_timezone_offset. Retrieved 7/10 statements.
# Partially parsed test_validate_handles_negative_timezone_offset. Retrieved 6/9 statements.
# Partially parsed test_validate_handles_short_timezone_offset. Retrieved 6/9 statements.
# Partially parsed test_validate_handles_edge_case_datetime. Retrieved 3/6 statements.
# Partially parsed test_validate_handles_leap_year. Retrieved 3/6 statements.


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
    var_5 = var_4.microsecond
    assert var_5 == 123456

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.tzinfo

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
    var_15 = var_4.tzinfo
    var_16 = bool(var_4.tzinfo == var_14)
    assert var_16 is True

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
    var_13 = var_4.tzinfo
    var_14 = bool(var_4.tzinfo == var_12)
    assert var_14 is True

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
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.timezone(*var_10, **var_11)
    var_13 = var_4.tzinfo
    var_14 = bool(var_4.tzinfo == var_12)
    assert var_14 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '0001-01-01T00:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 1
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1
    var_8 = var_4.hour
    assert var_8 == 0
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
    var_3 = '2024-02-29T12:00:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2024
    var_6 = var_4.month
    assert var_6 == 2
    var_7 = var_4.day
    assert var_7 == 29
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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
    var_3 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8:85a3::8a2e:370:7334'



# Parsed testcases at query #19
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

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '01:02:03.004005'
    var_4 = var_2.validate(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4005
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
    var_3 = ''
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.123456Z'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #20
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
    var_3 = '0001-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 1
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '9999-12-31'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 9999
    var_6 = var_4.month
    assert var_6 == 12
    var_7 = var_4.day
    assert var_7 == 31

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 1

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-04-30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 4
    var_7 = var_4.day
    assert var_7 == 30

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-07-31'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 7
    var_7 = var_4.day
    assert var_7 == 31



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


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



# Parsed testcases at query #22
#--------------------------




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
    var_5 = 0
    var_6 = [var_3, var_4, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '23:59:59'

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
    var_11 = 10
    var_12 = 15
    var_13 = 30
    var_14 = [var_11, var_12, var_13]
    var_15 = 'tzinfo'
    var_16 = {var_15: var_10}
    var_17 = module_1.time(*var_14, **var_16)
    var_18 = var_2.serialize(var_17)
    assert var_18 == '10:15:30+05:00'

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 1
    var_4 = 30
    var_5 = [var_3, var_4]
    var_6 = 'fold'
    var_7 = {var_6: var_3}
    var_8 = module_1.time(*var_5, **var_7)
    var_9 = var_2.serialize(var_8)
    assert var_9 == '01:30:00'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_for_valid_date. Retrieved 7/11 statements.


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
    var_12 = var_4.year
    assert var_12 == 2023
    var_13 = var_4.month
    assert var_13 == 12
    var_14 = var_4.day
    assert var_14 == 25



# Parsed testcases at query #24
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



# Parsed testcases at query #25
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



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_without_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_curly_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_returns_uuid_for_valid_string_with_urn_prefix_and_curly_braces. Retrieved 4/6 statements.


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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


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



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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
    var_3 = ''
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



# Parsed testcases at query #31
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
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_random. Retrieved 4/6 statements.


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
    var_3 = 'c232ab00-9414-11ec-b3c8-9f6b385d64be'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'c232ab00-9414-11ec-b3c8-9f6b385d64be'

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
    var_3 = '00000000-0000-0000-0000-000000000000'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '00000000-0000-0000-0000-000000000000'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'ffffffff-ffff-ffff-ffff-ffffffffffff'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'ffffffff-ffff-ffff-ffff-ffffffffffff'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.UUIDFormat(*var_0, **var_1)
    var_3 = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'



# Parsed testcases at query #32
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



# Parsed testcases at query #33
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
    var_3 = '12-34-56'
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
    var_3 = 12345
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
    var_3 = 'not.an.ip'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serialize_ends_with_plus_00_00. Retrieved 6/10 statements.


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



# Parsed testcases at query #36
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
    var_3 = '2023-01-15T14:30:45.987654+09:00'
    var_4 = var_2.validate(var_3)
    var_5 = 9
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



# Parsed testcases at query #37
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
    var_3 = '2001:db8::1'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '2001:db8::1'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = 3232235777
    var_4 = var_2.validate(var_3)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '192.168.1.1'
    var_1 = module_0.IPv4Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.is_native_type(var_1)
    assert var_5 is True

import ipaddress as module_0
import typesystem.formats as module_1

def test_case_0():
    var_0 = '2001:db8::'
    var_1 = module_0.IPv6Address(var_0)
    var_2 = []
    var_3 = {}
    var_4 = module_1.IPAddressFormat(*var_2, **var_3)
    var_5 = var_4.is_native_type(var_1)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = '192.168.1.1'
    var_1 = []
    var_2 = {}
    var_3 = module_0.IPAddressFormat(*var_1, **var_2)
    var_4 = var_3.is_native_type(var_0)
    assert var_4 is False

import typesystem.formats as module_0

def test_case_0():
    var_0 = 123
    var_1 = []
    var_2 = {}
    var_3 = module_0.IPAddressFormat(*var_1, **var_2)
    var_4 = var_3.is_native_type(var_0)
    assert var_4 is False

import typesystem.formats as module_0

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = {}
    var_3 = module_0.IPAddressFormat(*var_1, **var_2)
    var_4 = var_3.is_native_type(var_0)
    assert var_4 is False



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------




import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-05-15'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 5
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
    var_3 = '2023/05/15'
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-5-9'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 5
    var_7 = 9
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
    var_3 = '2023-00-15'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-05-00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-05'
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
    var_3 = 20230515
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '0023-05-15'
    var_4 = var_2.validate(var_3)
    var_5 = 23
    var_6 = 5
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
    var_3 = '10000-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #4
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
    var_3 = 2024
    var_4 = 2
    var_5 = 29
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    var_10 = '2024-02-29'
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



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------




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
    var_3 = 'https://example.com/path'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com/path'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com?query=value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com?query=value'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'https://example.com#fragment'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'https://example.com#fragment'

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.URLFormat(*var_0, **var_1)
    var_3 = 'ftp://example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'ftp://example.com'



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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
    var_3 = '12:34'
    var_4 = var_2.validate(var_3)
    var_5 = 12
    var_6 = 34
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
    var_3 = '25:00:00'
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
    var_3 = '12:34:56 extra'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_valid_datetime_with_utc_z. Retrieved 9/11 statements.


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-01T12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 1
    var_8 = 12
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
    var_3 = '2023-04-01T12:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 1
    var_8 = 12
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
    var_3 = '2023-04-01T12:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 1
    var_8 = 12
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
    var_3 = '2023-04-01T12:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 4
    var_7 = 1
    var_8 = 12
    var_9 = 30
    var_10 = 45

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-04-01T12:30:45+05:30'
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
    var_16 = 4
    var_17 = 1
    var_18 = 12
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
    var_3 = '2023-04-01T12:30:45-08:00'
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
    var_14 = 4
    var_15 = 1
    var_16 = 12
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
    var_3 = '2023-04-01T12:30:45+02'
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
    var_14 = 4
    var_15 = 1
    var_16 = 12
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
    var_3 = '2023-02-30T12:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_uuidformat_validate_valid_string. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_valid_string_with_urn. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_lowercase_hex. Retrieved 4/6 statements.
# Partially parsed test_uuidformat_validate_uppercase_hex. Retrieved 2/7 statements.


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
    var_3 = '12345678-1234-5678-1234-567812345678'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_braces. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_urn_prefix. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_mixed_case. Retrieved 2/7 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_lowercase. Retrieved 2/7 statements.


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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime. Retrieved 3/7 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_microseconds. Retrieved 3/7 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_utc_timezone. Retrieved 3/7 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_positive_offset. Retrieved 6/11 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_negative_offset. Retrieved 5/10 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_short_offset. Retrieved 5/10 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_all_fields. Retrieved 6/11 statements.
# Partially parsed test_validate_does_not_raise_invalid_error_for_valid_datetime_with_padded_microseconds. Retrieved 3/7 statements.


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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T14:30:45.123456+05:30'
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



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_uuid_format_validate_with_valid_uuid_string. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_no_hyphens. Retrieved 4/6 statements.
# Partially parsed test_uuid_format_validate_with_valid_uuid_string_with_braces. Retrieved 4/6 statements.
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
    var_3 = 'c232ab00-9414-11ec-b3c8-9f6b6d1167f4'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'c232ab00-9414-11ec-b3c8-9f6b6d1167f4'

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
    var_3 = '74738ff5-5367-5958-9aee-98fffdcd1876'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == '74738ff5-5367-5958-9aee-98fffdcd1876'



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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
    var_3 = '10000-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-1-01'
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
    var_3 = '2023-12-1'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 1
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
    var_3 = '02023-12-01'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 1
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
    var_3 = ' 2023-12-01 '
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #19
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
    var_3 = 9
    var_4 = 15
    var_5 = 30
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_1.time(*var_6, **var_7)
    var_9 = var_2.serialize(var_8)
    var_10 = '09:15:30'
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

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
    var_3 = 5
    var_4 = []
    var_5 = 'hours'
    var_6 = {var_5: var_3}
    var_7 = module_1.timedelta(*var_4, **var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.timezone(*var_8, **var_9)
    var_11 = 18
    var_12 = 45
    var_13 = 20
    var_14 = 500000
    var_15 = [var_11, var_12, var_13, var_14]
    var_16 = 'tzinfo'
    var_17 = {var_16: var_10}
    var_18 = module_1.time(*var_15, **var_17)
    var_19 = var_2.serialize(var_18)
    var_20 = '18:45:20.500000+05:00'
    var_21 = bool(var_19 == var_20)
    assert var_21 is True

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = 23
    var_4 = 59
    var_5 = 999999
    var_6 = 1
    var_7 = [var_3, var_4, var_4, var_5]
    var_8 = 'fold'
    var_9 = {var_8: var_6}
    var_10 = module_1.time(*var_7, **var_9)
    var_11 = var_2.serialize(var_10)
    var_12 = '23:59:59.999999'
    var_13 = bool(var_11 == var_12)
    assert var_13 is True



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234567'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 34
    var_7 = var_4.second
    assert var_7 == 56
    var_8 = var_4.microsecond
    assert var_8 == 123456



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:34:56.1234567'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.hour
    assert var_5 == 12
    var_6 = var_4.minute
    assert var_6 == 34
    var_7 = var_4.second
    assert var_7 == 56
    var_8 = var_4.microsecond
    assert var_8 == 123456



# Parsed testcases at query #24
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



# Parsed testcases at query #25
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
    var_3 = '0023-01-01'
    var_4 = var_2.validate(var_3)
    var_5 = 23
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
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-1-01'
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
    var_3 = '2023-12-1'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 1
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
    var_3 = '2023-13-01'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-32'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '10000-01-01'
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
    var_3 = '-2023-12-25'
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
    var_3 = '2023-12-01'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 12
    var_7 = 1
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

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-02-28'
    var_4 = var_2.validate(var_3)
    var_5 = 2023
    var_6 = 2
    var_7 = 28
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
    var_3 = '2024-02-30'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = ' 2023-12-25 '
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25\n'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '\n2023-12-25'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25\t'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25\r'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25\x00'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25©'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25😀'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '<span>2023-12-25</span>'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = "2023-12-25'; DROP TABLE users; --"
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = "2023-12-25<script>alert('xss')</script>"
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)
    var_3 = '2023-12-25\\'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateFormat(*var_0, **var_1)



# Parsed testcases at query #26
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
    var_3 = '12'
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
    var_3 = '25:30:45'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:60:45'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:60'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '12:30:45.1000000'
    var_4 = var_2.validate(var_3)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.TimeFormat(*var_0, **var_1)
    var_3 = '05:07:09.000123'
    var_4 = var_2.validate(var_3)
    var_5 = 5
    var_6 = 7
    var_7 = 9
    var_8 = 123
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



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_with_valid_datetime_does_not_raise_invalid_error. Retrieved 3/6 statements.
# Partially parsed test_validate_with_valid_datetime_with_microseconds_does_not_raise_invalid_error. Retrieved 3/6 statements.
# Partially parsed test_validate_with_valid_datetime_with_utc_timezone_does_not_raise_invalid_error. Retrieved 3/6 statements.
# Partially parsed test_validate_with_valid_datetime_with_positive_offset_does_not_raise_invalid_error. Retrieved 6/10 statements.
# Partially parsed test_validate_with_valid_datetime_with_negative_offset_does_not_raise_invalid_error. Retrieved 5/9 statements.
# Partially parsed test_validate_with_valid_datetime_with_short_offset_does_not_raise_invalid_error. Retrieved 5/9 statements.
# Partially parsed test_validate_with_valid_datetime_with_all_fields_does_not_raise_invalid_error. Retrieved 5/9 statements.
# Partially parsed test_validate_with_valid_datetime_with_partial_microseconds_does_not_raise_invalid_error. Retrieved 3/6 statements.


import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T10:30:45'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 10
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
    var_3 = '2023-01-15T10:30:45.123456'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 10
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
    var_3 = '2023-01-15T10:30:45Z'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 10
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
    var_3 = '2023-01-15T10:30:45+05:30'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 10
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = 5
    var_13 = 30
    var_14 = []
    var_15 = 'hours'
    var_16 = 'minutes'
    var_17 = {var_15: var_12, var_16: var_13}
    var_18 = module_1.timedelta(*var_14, **var_17)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T10:30:45-08:00'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 10
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = -8
    var_13 = []
    var_14 = 'hours'
    var_15 = {var_14: var_12}
    var_16 = module_1.timedelta(*var_13, **var_15)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T10:30:45+05'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 10
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 0
    var_12 = 5
    var_13 = []
    var_14 = 'hours'
    var_15 = {var_14: var_12}
    var_16 = module_1.timedelta(*var_13, **var_15)

import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-12-31T23:59:59.999999+12:00'
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
    var_12 = 12
    var_13 = []
    var_14 = 'hours'
    var_15 = {var_14: var_12}
    var_16 = module_1.timedelta(*var_13, **var_15)

import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.DateTimeFormat(*var_0, **var_1)
    var_3 = '2023-01-15T10:30:45.123'
    var_4 = var_2.validate(var_3)
    var_5 = var_4.year
    assert var_5 == 2023
    var_6 = var_4.month
    assert var_6 == 1
    var_7 = var_4.day
    assert var_7 == 15
    var_8 = var_4.hour
    assert var_8 == 10
    var_9 = var_4.minute
    assert var_9 == 30
    var_10 = var_4.second
    assert var_10 == 45
    var_11 = var_4.microsecond
    assert var_11 == 123000
    var_12 = var_4.tzinfo
    assert var_12 is None



# Parsed testcases at query #29
#--------------------------




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




import typesystem.formats as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.IPAddressFormat(*var_0, **var_1)
    var_3 = '999.999.999.999'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_time_with_invalid_microsecond. Retrieved 7/15 statements.


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
    var_8 = None
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------




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



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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



# Parsed testcases at query #37
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



# Parsed testcases at query #38
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



