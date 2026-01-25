####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 8/17 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_no_args. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_negative_timedelta. Retrieved 5/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = 3
    var_10 = [var_2, var_3, var_9]
    var_11 = 4
    var_12 = [var_2, var_3, var_11]
    var_13 = [var_2, var_3, var_5]
    var_14 = 6
    var_15 = [var_2, var_3, var_14]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = -1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_datetime_with_default_params. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2025
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2025)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True
    var_6 = var_3.tzinfo
    var_7 = str(var_6)
    assert var_7 == 'UTC'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2025
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2025)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True
    var_6 = var_3.tzinfo
    var_7 = str(var_6)
    assert var_7 == 'UTC'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 3/6 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 5/8 statements.
# Partially parsed test_duration_custom_unit. Retrieved 2/7 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()
    var_3 = 60
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)
    var_5 = 60
    var_6 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 3600
    var_3 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 1
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_duration_with_non_integer_values. Retrieved 5/17 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '5'
    var_3 = 10
    var_4 = 5
    var_5 = '10'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_datetime_with_timezone_raises_import_error_when_pytz_not_available. Retrieved 4/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = str(var_2)
    assert var_4 == 'Timezones are supported only with pytz'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2025
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2025)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_duration_predicate_false. Retrieved 4/8 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_valid_input. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_with_empty_kwargs. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_invalid_range. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_none_input. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 6/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = 6
    var_10 = [var_2, var_3, var_9]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4]
    var_9 = 6
    var_10 = [var_2, var_3, var_3, var_9, var_4]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_duration_predicate_false. Retrieved 4/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 10
    var_4 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_timedelta_must_be_positive. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 6/13 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 6/13 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 6/13 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = [var_2, var_3, var_5]
    var_10 = 4
    var_11 = [var_2, var_3, var_10]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_3, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4]
    var_10 = 3
    var_11 = [var_2, var_3, var_3, var_10, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_3, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_3]
    var_9 = [var_2, var_3, var_3, var_4, var_6]
    var_10 = 3
    var_11 = [var_2, var_3, var_3, var_4, var_10]



# Parsed testcases at query #14
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_duration_default. Retrieved 2/5 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 1/6 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 1
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(var_4 is not False)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_empty_kwargs. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = []
    var_6 = []



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_bulk_create_datetimes_basic. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_error. Retrieved 5/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = [var_2, var_3, var_5]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_3, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_duration_with_default_parameters. Retrieved 4/7 statements.
# Partially parsed test_duration_with_custom_min_and_max. Retrieved 4/7 statements.
# Partially parsed test_duration_with_custom_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_none_unit. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()
    var_3 = 1
    var_4 = []
    var_5 = 10
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)
    var_5 = []
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 15
    var_3 = 5
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2025
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2025)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True
    var_6 = var_3.tzinfo
    var_7 = str(var_6)
    assert var_7 == 'UTC'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_duration_with_non_integer_parameters. Retrieved 3/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '5'
    var_3 = 10



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_datetime_with_default_params. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.
# Partially parsed test_datetime_with_pytz_timezone. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2010
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2010
    var_6 = bool(2010 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2020)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'America/New_York'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True
    var_6 = var_3.tzinfo
    var_7 = str(var_6)
    var_8 = 'America/New_York'
    var_9 = bool('America/New_York' in var_7)
    assert var_9 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_none_unit. Retrieved 5/8 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()
    var_3 = 1
    var_4 = []
    var_5 = 10
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)
    var_5 = []
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)
    var_4 = 1
    var_5 = []
    var_6 = 10
    var_7 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 1
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_duration_with_non_integer_parameters. Retrieved 3/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '5'
    var_3 = 10



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_duration_predicate_false. Retrieved 3/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = 1
    var_6 = '10'
    var_7 = var_1.duration(var_5, var_6)
    var_8 = '1'
    var_9 = '10'
    var_10 = var_1.duration(var_8, var_9)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2022
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2022)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #7
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_valid_inputs. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_empty_kwargs. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_invalid_range. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_missing_args. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = [var_2, var_3, var_5]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 2/5 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 1/6 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 1
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_duration_with_invalid_types. Retrieved 5/13 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = 1
    var_5 = '10'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_timedelta_predicate_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = []



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()
    var_3 = 1
    var_4 = []
    var_5 = 10
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)
    var_5 = []
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 15
    var_3 = 5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_none_unit. Retrieved 4/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()
    var_3 = 1
    var_4 = []
    var_5 = 10
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)
    var_5 = []
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)
    var_4 = 1
    var_5 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 1
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = 1
    var_6 = '10'
    var_7 = var_1.duration(var_5, var_6)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_timedelta_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = []



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2022
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2022)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_value_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range_raises_value_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_value_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_missing_dates_raises_value_error. Retrieved 3/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = 4
    var_10 = [var_2, var_3, var_9]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 3
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_duration_with_non_integer_parameters. Retrieved 3/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '5'
    var_3 = 10



# Parsed testcases at query #20
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = 1
    var_6 = '10'
    var_7 = var_1.duration(var_5, var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_duration_with_non_integer_values. Retrieved 5/13 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = 1
    var_5 = '10'



# Parsed testcases at query #22
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #23
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = '10'
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_valueerror_for_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_missing_args_raises_error. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = [var_2, var_3, var_5]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_duration_with_integer_parameters. Retrieved 4/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)



# Parsed testcases at query #27
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #28
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = 1
    var_6 = '10'
    var_7 = var_1.duration(var_5, var_6)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 0



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2025
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2025)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



