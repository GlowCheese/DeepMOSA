####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_datetime_without_timezone. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year
    var_4 = bool(var_2.year >= var_1._CURRENT_YEAR)
    assert var_4 is True
    var_5 = var_2.year
    var_6 = bool(var_2.year <= var_1._CURRENT_YEAR)
    assert var_6 is True
    var_7 = var_2.hour
    var_8 = bool(var_2.hour >= 0)
    assert var_8 is True
    var_9 = var_2.hour
    var_10 = bool(var_2.hour <= 23)
    assert var_10 is True
    var_11 = var_2.minute
    var_12 = bool(var_2.minute >= 0)
    assert var_12 is True
    var_13 = var_2.minute
    var_14 = bool(var_2.minute <= 59)
    assert var_14 is True
    var_15 = var_2.second
    var_16 = bool(var_2.second >= 0)
    assert var_16 is True
    var_17 = var_2.second
    var_18 = bool(var_2.second <= 59)
    assert var_18 is True

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
    var_2 = 2020
    var_3 = 2022
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = var_4.year
    var_6 = bool(var_4.year >= 2020)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2022)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_empty_dates_raises_value_error. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_start_after_end_raises_value_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_step_raises_value_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes_step. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_1, var_4]
    var_6 = [var_0, var_1, var_1, var_1]
    var_7 = [var_0, var_1, var_1, var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_1, var_2, var_4]
    var_6 = [var_0, var_1, var_1, var_2, var_1]
    var_7 = [var_0, var_1, var_1, var_2, var_4]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_timestamp_posix_format. Retrieved 1/4 statements.
# Partially parsed test_timestamp_rfc_3339_format. Retrieved 1/5 statements.
# Partially parsed test_timestamp_iso_8601_format. Retrieved 1/5 statements.
# Partially parsed test_timestamp_default_format. Retrieved 2/3 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = {}
    var_3 = var_1.timestamp(**var_2)



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_duration_with_default_values. Retrieved 4/7 statements.
# Partially parsed test_duration_with_custom_values. Retrieved 3/8 statements.
# Partially parsed test_duration_with_random_unit. Retrieved 3/4 statements.


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
    var_4 = []
    var_5 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 5
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1.5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10.5
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2021
    var_4 = 'UTC'
    var_5 = var_1.datetime(var_2, var_3, var_4)



# Parsed testcases at query #7
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_when_date_start_larger_than_date_end. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2022
    var_4 = [var_3, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_invalid_start_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_missing_start_end. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_same_start_end. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = 6
    var_8 = [var_0, var_1, var_7]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1

def test_case_0():
    var_0 = None
    var_1 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = 2
    var_5 = [var_0, var_1, var_4]



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_input. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_hours_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_minutes_step. Retrieved 6/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = {}
    var_6 = module_0.Datetime(**var_5)
    var_7 = 2
    var_8 = [var_0, var_1, var_7]
    var_9 = [var_0, var_1, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 10
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = {}
    var_6 = module_0.Datetime(**var_5)
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = {}
    var_6 = module_0.Datetime(**var_5)
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2]
    var_4 = 10
    var_5 = [var_0, var_1, var_1, var_4, var_2]
    var_6 = {}
    var_7 = module_0.Datetime(**var_6)
    var_8 = [var_0, var_1, var_1, var_1, var_2]
    var_9 = [var_0, var_1, var_1, var_4, var_2]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2]
    var_4 = 50
    var_5 = [var_0, var_1, var_1, var_2, var_4]
    var_6 = {}
    var_7 = module_0.Datetime(**var_6)
    var_8 = 10
    var_9 = [var_0, var_1, var_1, var_2, var_8]
    var_10 = [var_0, var_1, var_1, var_2, var_4]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_value_error_when_date_start_larger_than_date_end. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2022
    var_4 = [var_3, var_1, var_1]
    var_5 = 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_pytz_not_available. Retrieved 4/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'pytz'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'UTC'
    var_4 = var_2.datetime(timezone=var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_when_date_start_larger_than_date_end. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2022
    var_4 = [var_3, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_same_start_end. Retrieved 3/8 statements.
# Partially parsed test_bulk_create_datetimes_invalid_start_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_missing_start_end. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = 2
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1

def test_case_0():
    var_0 = None
    var_1 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_when_date_start_larger_than_date_end. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2022
    var_4 = [var_3, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_empty_range. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = 11
    var_8 = [var_0, var_1, var_7]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 10
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_pytz_not_installed. Retrieved 4/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'pytz'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'UTC'
    var_4 = var_2.datetime(timezone=var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = 'days'
    var_6 = {var_5: var_1}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_datetime_timezone_requires_pytz. Retrieved 5/7 statements.


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
    var_2 = var_1.datetime()
    var_3 = var_2.tzinfo
    assert var_3 is None

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'pytz'
    var_3 = None
    var_4 = 'UTC'
    var_5 = var_1.datetime(timezone=var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = 'days'
    var_6 = 0
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_duration_valid_input. Retrieved 3/8 statements.
# Partially parsed test_duration_default_unit. Retrieved 4/7 statements.
# Partially parsed test_duration_random_unit. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = []
    var_5 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = []
    var_6 = []

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = None
    var_5 = var_1.duration(var_2, var_3, var_4)

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
    var_2 = 1.5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_missing_dates. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_3]
    var_8 = 4
    var_9 = [var_0, var_1, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.
# Partially parsed test_datetime_raises_import_error_for_missing_pytz. Retrieved 2/5 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 2/6 statements.


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
    var_2 = 2000
    var_3 = 2010
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2000
    var_6 = bool(2000 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2010)
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
    var_2 = 'pytz is installed'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = '10'
    var_4 = var_1.duration(var_2, var_3)



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = 'days'
    var_6 = 'hours'
    var_7 = 'minutes'
    var_8 = 'seconds'
    var_9 = 'microseconds'
    var_10 = 0
    var_11 = {var_5: var_10, var_6: var_10, var_7: var_10, var_8: var_10, var_9: var_10}



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

# Partially parsed test_datetime_with_default_values. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
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

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_input. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_step. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_hours_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_minutes_step. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = {}
    var_6 = module_0.Datetime(**var_5)
    var_7 = 2
    var_8 = [var_0, var_1, var_7]
    var_9 = [var_0, var_1, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 10
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = {}
    var_6 = module_0.Datetime(**var_5)
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = {}
    var_6 = module_0.Datetime(**var_5)
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = 10
    var_5 = [var_0, var_1, var_1, var_4]
    var_6 = {}
    var_7 = module_0.Datetime(**var_6)
    var_8 = [var_0, var_1, var_1, var_1]
    var_9 = [var_0, var_1, var_1, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2]
    var_4 = 10
    var_5 = [var_0, var_1, var_1, var_2, var_4]
    var_6 = {}
    var_7 = module_0.Datetime(**var_6)
    var_8 = [var_0, var_1, var_1, var_2, var_1]
    var_9 = [var_0, var_1, var_1, var_2, var_4]



# Parsed testcases at query #10
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

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = '10'
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_duration_with_valid_integer_parameters. Retrieved 4/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_empty_date_start_and_end. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_date_start_larger_than_end. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 5/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]
    var_7 = 2
    var_8 = [var_2, var_3, var_7]
    var_9 = 11
    var_10 = [var_2, var_3, var_9]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 10
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
    var_5 = 10
    var_6 = [var_2, var_3, var_5]
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_pytz_not_available_and_timezone_provided. Retrieved 4/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'pytz'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'UTC'
    var_4 = var_2.datetime(timezone=var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_valid_input. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_empty_input. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_with_start_larger_than_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_custom_step. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2]
    var_4 = 2
    var_5 = [var_0, var_1, var_1, var_4, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_2]
    var_7 = [var_0, var_1, var_1, var_4, var_2]



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '1'
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = '10'
    var_4 = var_1.duration(var_2, var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_duration_with_minutes_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_hours_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_days_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_seconds_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_random_unit. Retrieved 5/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2
    var_3 = 7

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 30
    var_3 = 120

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = None
    var_5 = var_1.duration(var_2, var_3, var_4)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1.5
    var_3 = 5.5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
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

# Partially parsed test_bulk_create_datetimes_with_valid_inputs. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_empty_dates. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_with_start_after_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes_step. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2]
    var_4 = 2
    var_5 = [var_0, var_1, var_1, var_4, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_2]
    var_7 = [var_0, var_1, var_1, var_4, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2]
    var_4 = 2
    var_5 = [var_0, var_1, var_1, var_2, var_4]
    var_6 = [var_0, var_1, var_1, var_2, var_1]
    var_7 = [var_0, var_1, var_1, var_2, var_4]



# Parsed testcases at query #19
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #20
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1.5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10.5
    var_4 = var_1.duration(var_2, var_3)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 'days'
    var_6 = 0
    var_7 = {var_5: var_6}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_datetime_with_default_values. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.
# Partially parsed test_datetime_with_timezone_and_custom_year. Retrieved 5/6 statements.


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
    var_2 = 2000
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2000
    var_6 = bool(2000 <= var_4.year)
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
    var_2 = 2010
    var_3 = 2015
    var_4 = 'UTC'
    var_5 = var_1.datetime(var_2, var_3, var_4)
    var_6 = 2010
    var_7 = bool(2010 <= var_5.year)
    assert var_7 is True
    var_8 = var_5.year
    var_9 = bool(var_5.year <= 2015)
    assert var_9 is True
    var_10 = var_5.tzinfo
    var_11 = bool(var_5.tzinfo is not None)
    assert var_11 is True



