####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_formatted_datetime_default_format. Retrieved 3/5 statements.
# Partially parsed test_formatted_datetime_custom_format. Retrieved 4/6 statements.
# Partially parsed test_formatted_datetime_iso_format. Retrieved 4/6 statements.
# Partially parsed test_formatted_datetime_with_start_year. Retrieved 4/6 statements.
# Partially parsed test_formatted_datetime_with_end_year. Retrieved 4/6 statements.
# Partially parsed test_formatted_datetime_with_timezone. Retrieved 5/7 statements.
# Partially parsed test_formatted_datetime_format_percent_d. Retrieved 4/6 statements.
# Partially parsed test_formatted_datetime_format_percent_m. Retrieved 4/6 statements.
# Partially parsed test_formatted_datetime_format_percent_Y. Retrieved 4/6 statements.
# Partially parsed test_formatted_datetime_multiple_calls_different_results. Retrieved 6/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = {}
    var_3 = var_1.formatted_datetime(**var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%Y-%m-%d %H:%M:%S'
    var_3 = {}
    var_4 = var_1.formatted_datetime(var_2, **var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%Y-%m-%dT%H:%M:%S'
    var_3 = {}
    var_4 = var_1.formatted_datetime(var_2, **var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%Y-%m-%d'
    var_3 = 2020
    var_4 = 'start'
    var_5 = 'end'
    var_6 = {var_4: var_3, var_5: var_3}
    var_7 = var_1.formatted_datetime(var_2, **var_6)
    var_8 = '2020'
    var_9 = bool('2020' in var_7)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%Y-%m-%d'
    var_3 = 2015
    var_4 = 'start'
    var_5 = 'end'
    var_6 = {var_4: var_3, var_5: var_3}
    var_7 = var_1.formatted_datetime(var_2, **var_6)
    var_8 = '2015'
    var_9 = bool('2015' in var_7)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%Y-%m-%d %H:%M:%S'
    var_3 = 'UTC'
    var_4 = 'timezone'
    var_5 = {var_4: var_3}
    var_6 = var_1.formatted_datetime(var_2, **var_5)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%d'
    var_3 = {}
    var_4 = var_1.formatted_datetime(var_2, **var_3)
    var_5 = int(var_4)
    var_6 = 1
    var_7 = bool(1 <= var_5)
    assert var_7 is True
    var_8 = bool(var_5 <= 31)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%m'
    var_3 = {}
    var_4 = var_1.formatted_datetime(var_2, **var_3)
    var_5 = int(var_4)
    var_6 = 1
    var_7 = bool(1 <= var_5)
    assert var_7 is True
    var_8 = bool(var_5 <= 12)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%Y'
    var_3 = 2010
    var_4 = 'start'
    var_5 = 'end'
    var_6 = {var_4: var_3, var_5: var_3}
    var_7 = var_1.formatted_datetime(var_2, **var_6)
    assert var_7 == '2010'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = '%Y-%m-%d %H:%M:%S'
    var_3 = 2000
    var_4 = 2025
    var_5 = 'start'
    var_6 = 'end'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = var_1.formatted_datetime(var_2, **var_7)
    var_9 = 'start'
    var_10 = 'end'
    var_11 = {var_9: var_3, var_10: var_4}
    var_12 = var_1.formatted_datetime(var_2, **var_11)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_duration_default_parameters. Retrieved 2/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/8 statements.
# Partially parsed test_duration_seconds_unit. Retrieved 3/10 statements.
# Partially parsed test_duration_hours_unit. Retrieved 3/10 statements.
# Partially parsed test_duration_days_unit. Retrieved 3/10 statements.
# Partially parsed test_duration_microseconds_unit. Retrieved 3/10 statements.
# Partially parsed test_duration_none_unit. Retrieved 5/9 statements.
# Partially parsed test_duration_equal_min_max. Retrieved 2/9 statements.


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
    var_5 = 5 * 60

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 20
    var_4 = 10

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 5
    var_4 = 3600

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 3
    var_4 = 86400

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1000
    var_3 = 5000
    var_4 = 0.001

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
    var_2 = 20
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_duration must be less or equal to max_duration'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5.5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_duration and max_duration must be integers'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 10.5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_duration and max_duration must be integers'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_timestamp_posix_format. Retrieved 1/6 statements.
# Partially parsed test_timestamp_rfc_3339_format. Retrieved 1/6 statements.
# Partially parsed test_timestamp_iso_8601_format. Retrieved 1/6 statements.
# Partially parsed test_timestamp_default_format. Retrieved 2/4 statements.
# Partially parsed test_timestamp_with_custom_year_range. Retrieved 3/8 statements.
# Partially parsed test_timestamp_posix_returns_integer. Retrieved 1/6 statements.
# Partially parsed test_timestamp_rfc_3339_returns_string. Retrieved 1/6 statements.
# Partially parsed test_timestamp_iso_8601_returns_string. Retrieved 1/6 statements.
# Partially parsed test_timestamp_rfc_3339_format_structure. Retrieved 4/13 statements.
# Partially parsed test_timestamp_iso_8601_format_structure. Retrieved 1/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'T'
    var_3 = 'Z'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'T'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = {}
    var_3 = var_1.timestamp(**var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2021

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
    var_2 = 'T'
    var_3 = 0
    var_4 = '-'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'T'
    var_3 = '-'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_datetime_default_parameters. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_custom_years. Retrieved 5/8 statements.
# Partially parsed test_datetime_with_same_start_and_end. Retrieved 4/7 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 4/7 statements.
# Partially parsed test_datetime_without_timezone. Retrieved 4/7 statements.
# Partially parsed test_datetime_generates_valid_date. Retrieved 3/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test datetime generation with default parameters.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = var_2.datetime()
    var_4 = var_3.year

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test datetime generation with custom year range.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 2010
    var_4 = 2020
    var_5 = var_2.datetime(var_3, var_4)
    var_6 = 2010
    var_7 = bool(2010 <= var_5.year)
    assert var_7 is True
    var_8 = var_5.year
    var_9 = bool(var_5.year <= 2020)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test datetime generation with same start and end year.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 2015
    var_4 = var_2.datetime(var_3, var_3)
    var_5 = var_4.year
    assert var_5 == 2015

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test that generated datetime has valid time component.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = var_2.datetime()
    var_4 = 0
    var_5 = bool(0 <= var_3.hour)
    assert var_5 is True
    var_6 = var_3.hour
    var_7 = bool(var_3.hour <= 23)
    assert var_7 is True
    var_8 = 0
    var_9 = bool(0 <= var_3.minute)
    assert var_9 is True
    var_10 = var_3.minute
    var_11 = bool(var_3.minute <= 59)
    assert var_11 is True
    var_12 = 0
    var_13 = bool(0 <= var_3.second)
    assert var_13 is True
    var_14 = var_3.second
    var_15 = bool(var_3.second <= 59)
    assert var_15 is True
    var_16 = 0
    var_17 = bool(0 <= var_3.microsecond)
    assert var_17 is True
    var_18 = var_3.microsecond
    var_19 = bool(var_3.microsecond <= 999999)
    assert var_19 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test datetime generation with timezone.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'US/Eastern'
    var_4 = var_2.datetime(timezone=var_3)
    var_5 = var_4.tzinfo
    var_6 = bool(var_4.tzinfo is not None)
    assert var_6 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test datetime generation without timezone.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = None
    var_4 = var_2.datetime(timezone=var_3)
    var_5 = var_4.tzinfo
    assert var_5 is None

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test that invalid timezone raises appropriate error.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'Invalid/Timezone'
    var_4 = var_2.datetime(timezone=var_3)
    var_5 = bool(False)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test that multiple calls produce different datetime objects.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 2000
    var_4 = 2023
    var_5 = var_2.datetime(var_3, var_4)
    var_6 = var_2.datetime(var_3, var_4)
    var_7 = type(var_6)
    var_8 = isinstance(var_5, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test that datetime respects year boundaries.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 2005
    var_4 = 2010
    var_5 = var_2.datetime(var_3, var_4)
    var_6 = bool(var_3 <= var_5.year)
    assert var_6 is True
    var_7 = var_5.year
    var_8 = bool(var_5.year <= var_4)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test that generated datetime has valid date component.'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = var_2.datetime()
    var_4 = 1
    var_5 = bool(1 <= var_3.month)
    assert var_5 is True
    var_6 = var_3.month
    var_7 = bool(var_3.month <= 12)
    assert var_7 is True
    var_8 = 1
    var_9 = bool(1 <= var_3.day)
    assert var_9 is True
    var_10 = var_3.day
    var_11 = bool(var_3.day <= 31)
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_datetime_default_parameters. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/7 statements.
# Partially parsed test_datetime_with_same_start_and_end_year. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone_utc. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone_us_eastern. Retrieved 3/6 statements.


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
    var_2 = 2015
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    assert var_4 == 2015

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
    var_2 = 'US/Eastern'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2000
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 0
    var_6 = bool(0 <= var_4.hour)
    assert var_6 is True
    var_7 = var_4.hour
    var_8 = bool(var_4.hour <= 23)
    assert var_8 is True
    var_9 = 0
    var_10 = bool(0 <= var_4.minute)
    assert var_10 is True
    var_11 = var_4.minute
    var_12 = bool(var_4.minute <= 59)
    assert var_12 is True
    var_13 = 0
    var_14 = bool(0 <= var_4.second)
    assert var_14 is True
    var_15 = var_4.second
    var_16 = bool(var_4.second <= 59)
    assert var_16 is True
    var_17 = 0
    var_18 = bool(0 <= var_4.microsecond)
    assert var_18 is True
    var_19 = var_4.microsecond
    var_20 = bool(var_4.microsecond <= 999999)
    assert var_20 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1990
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = var_1.datetime(var_2, var_3)
    var_6 = bool(var_4 != var_5)
    assert var_6 is True

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
    var_2 = 1980
    var_3 = 2010
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 1980
    var_6 = bool(1980 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2010)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_pytz_not_available. Retrieved 20/45 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = []
    var_4 = 10
    var_5 = 30
    var_6 = 45
    var_7 = []
    var_8 = 'pytz'
    var_9 = None
    var_10 = 'UTC'
    var_11 = 2023
    var_12 = 5
    var_13 = 15
    var_14 = 10
    var_15 = 30
    var_16 = 45
    var_17 = False
    var_18 = 'Timezones are supported only with pytz'
    var_19 = ImportError(var_18)
    var_20 = True
    var_21 = bool(var_20)
    assert var_21 is True
    var_22 = 'pytz'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_multiple_units. Retrieved 6/13 statements.
# Partially parsed test_bulk_create_datetimes_no_dates_raises_error. Retrieved 3/6 statements.
# Partially parsed test_bulk_create_datetimes_start_greater_than_end_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_zero_timedelta_raises_error. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_negative_timedelta_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_same_start_and_end. Retrieved 3/9 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 6/14 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
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
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 2
    var_6 = [var_2, var_3, var_5]
    var_7 = 12
    var_8 = 0
    var_9 = [var_2, var_3, var_5, var_7, var_8, var_8]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'You must pass date_start and date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'date_start can not be larger than date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_4, var_3]
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_4, var_4, var_3]
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_6, var_4]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_8, var_4]
    var_10 = [var_2, var_3, var_3, var_4, var_6, var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_datetime_default_parameters. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_custom_years. Retrieved 4/7 statements.
# Partially parsed test_datetime_with_same_start_and_end_year. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone_and_custom_years. Retrieved 5/8 statements.
# Partially parsed test_datetime_returns_valid_time_components. Retrieved 4/7 statements.


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
    var_2 = 2015
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    assert var_4 == 2015

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
    var_2 = 2005
    var_3 = 2010
    var_4 = 'US/Eastern'
    var_5 = var_1.datetime(var_2, var_3, var_4)
    var_6 = 2005
    var_7 = bool(2005 <= var_5.year)
    assert var_7 is True
    var_8 = var_5.year
    var_9 = bool(var_5.year <= 2010)
    assert var_9 is True
    var_10 = var_5.tzinfo
    var_11 = bool(var_5.tzinfo is not None)
    assert var_11 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2000
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 0
    var_6 = bool(0 <= var_4.hour)
    assert var_6 is True
    var_7 = var_4.hour
    var_8 = bool(var_4.hour <= 23)
    assert var_8 is True
    var_9 = 0
    var_10 = bool(0 <= var_4.minute)
    assert var_10 is True
    var_11 = var_4.minute
    var_12 = bool(var_4.minute <= 59)
    assert var_12 is True
    var_13 = 0
    var_14 = bool(0 <= var_4.second)
    assert var_14 is True
    var_15 = var_4.second
    var_16 = bool(var_4.second <= 59)
    assert var_16 is True
    var_17 = 0
    var_18 = bool(0 <= var_4.microsecond)
    assert var_18 is True
    var_19 = var_4.microsecond
    var_20 = bool(var_4.microsecond <= 999999)
    assert var_20 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_timestamp_rfc_3339_format. Retrieved 2/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'T'
    var_3 = 'Z'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_bulk_create_datetimes_basic. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_raises_when_both_dates_missing. Retrieved 3/6 statements.
# Partially parsed test_bulk_create_datetimes_raises_when_start_greater_than_end. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_raises_with_non_positive_timedelta. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_raises_with_negative_timedelta. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_same_start_and_end. Retrieved 3/9 statements.
# Partially parsed test_bulk_create_datetimes_multiple_kwargs. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
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
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_6, var_4]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_8, var_4]
    var_10 = [var_2, var_3, var_3, var_4, var_6, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'You must pass date_start and date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'date_start can not be larger than date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_6]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_8]
    var_10 = [var_2, var_3, var_3, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_4, var_8]
    var_10 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_6, var_4, var_4, var_4]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_bulk_create_datetimes_positive_timedelta. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_start_equals_end. Retrieved 3/9 statements.
# Partially parsed test_bulk_create_datetimes_no_dates_raises_error. Retrieved 3/6 statements.
# Partially parsed test_bulk_create_datetimes_start_after_end_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_zero_timedelta_raises_error. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_negative_timedelta_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds. Retrieved 6/14 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
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
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_6, var_4]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_8, var_4]
    var_10 = [var_2, var_3, var_3, var_4, var_6, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'You must pass date_start and date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'date_start can not be larger than date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_6]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_8]
    var_10 = [var_2, var_3, var_3, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_4, var_8]
    var_10 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_pytz_not_available. Retrieved 6/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'pytz'
    var_1 = None
    var_2 = {}
    var_3 = module_0.Datetime(**var_2)
    var_4 = 'UTC'
    var_5 = var_3.datetime(timezone=var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'pytz'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_bulk_create_datetimes_positive_timedelta. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_datetime_default_parameters. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/7 statements.
# Partially parsed test_datetime_with_different_start_end_years. Retrieved 4/7 statements.
# Partially parsed test_datetime_has_time_component. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone_europe_london. Retrieved 3/6 statements.
# Partially parsed test_datetime_without_timezone. Retrieved 3/6 statements.


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
    var_2 = 2015
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2015
    var_6 = bool(2015 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2020)
    assert var_8 is True

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
    var_2 = var_1.datetime()
    var_3 = 0
    var_4 = bool(0 <= var_2.hour)
    assert var_4 is True
    var_5 = var_2.hour
    var_6 = bool(var_2.hour <= 23)
    assert var_6 is True
    var_7 = 0
    var_8 = bool(0 <= var_2.minute)
    assert var_8 is True
    var_9 = var_2.minute
    var_10 = bool(var_2.minute <= 59)
    assert var_10 is True
    var_11 = 0
    var_12 = bool(0 <= var_2.second)
    assert var_12 is True
    var_13 = var_2.second
    var_14 = bool(var_2.second <= 59)
    assert var_14 is True

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
    var_2 = 'Europe/London'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    assert var_4 is None

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2000
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = var_1.datetime(var_2, var_3)
    var_6 = bool(var_4 != var_5)
    assert var_6 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2015
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    assert var_4 == 2015

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = type(var_2)
    var_4 = var_3.__name__
    assert var_4 == 'datetime'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_bulk_create_datetimes_basic. Retrieved 5/14 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 6/15 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_same_start_and_end. Retrieved 3/10 statements.
# Partially parsed test_bulk_create_datetimes_error_both_none. Retrieved 3/6 statements.
# Partially parsed test_bulk_create_datetimes_error_date_start_larger. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_error_non_positive_timedelta. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_error_negative_timedelta. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_combined_kwargs. Retrieved 5/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
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
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = 2
    var_10 = [var_2, var_3, var_3, var_9, var_4, var_4]
    var_11 = [var_2, var_3, var_3, var_6, var_4, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_4, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_4, var_3, var_4]
    var_9 = [var_2, var_3, var_3, var_4, var_6, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 10
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_6]
    var_8 = 2
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_8]
    var_10 = [var_2, var_3, var_3, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4, var_4]
    var_6 = 100
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]
    var_8 = 25
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_4, var_8]
    var_10 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'You must pass date_start and date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = [var_2, var_3, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_6]
    var_8 = 1
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'date_start can not be larger than date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_6, var_4, var_4, var_4]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_bulk_create_datetimes_predicate_line_46_false. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 46 evaluates to False with positive timedelta.'
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 0
    var_5 = [var_1, var_2, var_2, var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_1, var_2, var_6, var_3, var_4, var_4]



# Parsed testcases at query #20
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_duration_default_parameters. Retrieved 2/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/9 statements.
# Partially parsed test_duration_with_hours_unit. Retrieved 3/11 statements.
# Partially parsed test_duration_with_seconds_unit. Retrieved 3/11 statements.
# Partially parsed test_duration_with_days_unit. Retrieved 3/11 statements.
# Partially parsed test_duration_min_equals_max. Retrieved 3/7 statements.
# Partially parsed test_duration_with_none_unit. Retrieved 5/9 statements.
# Partially parsed test_duration_with_microseconds_unit. Retrieved 3/11 statements.


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
    var_2 = 1
    var_3 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 20

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 3

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = var_1.duration(var_2, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_duration must be less or equal to max_duration'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5.5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_duration and max_duration must be integers'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 10.5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_duration and max_duration must be integers'

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
    var_2 = 1000
    var_3 = 5000



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_no_arguments_raises_error. Retrieved 3/7 statements.
# Partially parsed test_bulk_create_datetimes_start_after_end_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_single_point_range. Retrieved 3/9 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_negative_timedelta_raises_error. Retrieved 5/11 statements.


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
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'You must pass date_start and date_end'

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
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'date_start can not be larger than date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_4, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_4, var_3, var_4]
    var_9 = [var_2, var_3, var_3, var_4, var_6, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_4, var_3]
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_4, var_4, var_3]
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]

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
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_datetime_default_parameters. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_custom_years. Retrieved 4/7 statements.
# Partially parsed test_datetime_with_same_start_end_year. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/8 statements.
# Partially parsed test_datetime_includes_time_component. Retrieved 2/7 statements.
# Partially parsed test_datetime_includes_date_component. Retrieved 2/7 statements.


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
    var_3 = 2015
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2010
    var_6 = bool(2010 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2015)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    assert var_4 == 2020

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2015
    var_3 = 2020
    var_4 = 'US/Eastern'
    var_5 = var_1.datetime(var_2, var_3, var_4)
    var_6 = var_5.tzinfo
    var_7 = bool(var_5.tzinfo is not None)
    assert var_7 is True
    var_8 = 2015
    var_9 = bool(2015 <= var_5.year)
    assert var_9 is True
    var_10 = var_5.year
    var_11 = bool(var_5.year <= 2020)
    assert var_11 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = 0
    var_4 = bool(0 <= var_2.hour)
    assert var_4 is True
    var_5 = var_2.hour
    var_6 = bool(var_2.hour <= 23)
    assert var_6 is True
    var_7 = 0
    var_8 = bool(0 <= var_2.minute)
    assert var_8 is True
    var_9 = var_2.minute
    var_10 = bool(var_2.minute <= 59)
    assert var_10 is True
    var_11 = 0
    var_12 = bool(0 <= var_2.second)
    assert var_12 is True
    var_13 = var_2.second
    var_14 = bool(var_2.second <= 59)
    assert var_14 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = 1
    var_4 = bool(1 <= var_2.month)
    assert var_4 is True
    var_5 = var_2.month
    var_6 = bool(var_2.month <= 12)
    assert var_6 is True
    var_7 = 1
    var_8 = bool(1 <= var_2.day)
    assert var_8 is True
    var_9 = var_2.day
    var_10 = bool(var_2.day <= 31)
    assert var_10 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_pytz_not_available. Retrieved 9/19 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = []
    var_4 = 14
    var_5 = 30
    var_6 = 45
    var_7 = []
    var_8 = {}
    var_9 = module_0.Datetime(**var_8)
    var_10 = 'America/New_York'
    var_11 = var_9.datetime(timezone=var_10)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_no_dates_raises_error. Retrieved 3/6 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_error. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_same_start_and_end. Retrieved 3/9 statements.
# Partially parsed test_bulk_create_datetimes_with_multiple_kwargs. Retrieved 5/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
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
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_4, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_4, var_3, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'You must pass date_start and date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'date_start can not be larger than date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_4, var_3]
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_6]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_6, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_bulk_create_datetimes_predicate_line_46_evaluates_to_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_0, var_1, var_3]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_datetime_default_parameters. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/7 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone_different_zones. Retrieved 5/9 statements.
# Partially parsed test_datetime_start_equals_end. Retrieved 3/6 statements.
# Partially parsed test_datetime_without_timezone. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_old_year_range. Retrieved 4/7 statements.


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
    var_3 = 2015
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2010
    var_6 = bool(2010 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2015)
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
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = 'US/Eastern'
    var_5 = var_1.datetime(timezone=var_4)
    var_6 = var_3.tzinfo
    var_7 = bool(var_3.tzinfo is not None)
    assert var_7 is True
    var_8 = var_5.tzinfo
    var_9 = bool(var_5.tzinfo is not None)
    assert var_9 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    assert var_4 == 2020

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2000
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 0
    var_6 = bool(0 <= var_4.hour)
    assert var_6 is True
    var_7 = var_4.hour
    var_8 = bool(var_4.hour <= 23)
    assert var_8 is True
    var_9 = 0
    var_10 = bool(0 <= var_4.minute)
    assert var_10 is True
    var_11 = var_4.minute
    var_12 = bool(var_4.minute <= 59)
    assert var_12 is True
    var_13 = 0
    var_14 = bool(0 <= var_4.second)
    assert var_14 is True
    var_15 = var_4.second
    var_16 = bool(var_4.second <= 59)
    assert var_16 is True
    var_17 = 0
    var_18 = bool(0 <= var_4.microsecond)
    assert var_18 is True
    var_19 = var_4.microsecond
    var_20 = bool(var_4.microsecond <= 999999)
    assert var_20 is True

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
    var_2 = 1950
    var_3 = 1960
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 1950
    var_6 = bool(1950 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 1960)
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]



# Parsed testcases at query #9
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_bulk_create_datetimes_predicate_line_46_false. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 46 evaluates to False (timedelta is positive).'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 5
    var_5 = [var_1, var_2, var_4]
    var_6 = 2
    var_7 = [var_1, var_2, var_6]



# Parsed testcases at query #11
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = 3
    var_8 = [var_0, var_1, var_7]
    var_9 = 4
    var_10 = [var_0, var_1, var_9]
    var_11 = [var_0, var_1, var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_datetime_default_parameters. Retrieved 2/5 statements.
# Partially parsed test_datetime_with_custom_years. Retrieved 4/7 statements.
# Partially parsed test_datetime_with_same_start_and_end_year. Retrieved 3/6 statements.
# Partially parsed test_datetime_with_timezone_pytz. Retrieved 3/7 statements.
# Partially parsed test_datetime_with_timezone_us_eastern. Retrieved 3/6 statements.


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
    var_3 = 2015
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2010
    var_6 = bool(2010 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2015)
    assert var_8 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    assert var_4 == 2020

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
    var_2 = 'US/Eastern'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    var_5 = bool(var_3.tzinfo is not None)
    assert var_5 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'pytz'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = 0
    var_4 = bool(0 <= var_2.hour)
    assert var_4 is True
    var_5 = var_2.hour
    var_6 = bool(var_2.hour <= 23)
    assert var_6 is True
    var_7 = 0
    var_8 = bool(0 <= var_2.minute)
    assert var_8 is True
    var_9 = var_2.minute
    var_10 = bool(var_2.minute <= 59)
    assert var_10 is True
    var_11 = 0
    var_12 = bool(0 <= var_2.second)
    assert var_12 is True
    var_13 = var_2.second
    var_14 = bool(var_2.second <= 59)
    assert var_14 is True
    var_15 = 0
    var_16 = bool(0 <= var_2.microsecond)
    assert var_16 is True
    var_17 = var_2.microsecond
    var_18 = bool(var_2.microsecond <= 999999)
    assert var_18 is True

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2000
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2000
    var_6 = bool(2000 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2023)
    assert var_8 is True
    var_9 = 1
    var_10 = bool(1 <= var_4.month)
    assert var_10 is True
    var_11 = var_4.month
    var_12 = bool(var_4.month <= 12)
    assert var_12 is True
    var_13 = 1
    var_14 = bool(1 <= var_4.day)
    assert var_14 is True
    var_15 = var_4.day
    var_16 = bool(var_4.day <= 31)
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_bulk_create_datetimes_predicate_line_46_false. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 0
    var_4 = [var_0, var_1, var_1, var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_0, var_1, var_5, var_2, var_3, var_3]
    var_7 = 2
    var_8 = [var_0, var_1, var_7, var_2, var_3, var_3]



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_pytz_not_available. Retrieved 6/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'pytz'
    var_1 = None
    var_2 = {}
    var_3 = module_0.Datetime(**var_2)
    var_4 = 'UTC'
    var_5 = var_3.datetime(timezone=var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'pytz'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_bulk_create_datetimes_basic. Retrieved 5/16 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_raises_error_when_both_none. Retrieved 3/6 statements.
# Partially parsed test_bulk_create_datetimes_raises_error_when_start_greater_than_end. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_raises_error_when_timedelta_not_positive. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_raises_error_when_timedelta_negative. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_equal_start_and_end. Retrieved 3/9 statements.
# Partially parsed test_bulk_create_datetimes_combined_kwargs. Retrieved 5/14 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
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
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 5
    var_7 = [var_2, var_3, var_3, var_4, var_6, var_4]
    var_8 = [var_2, var_3, var_3, var_4, var_3, var_4]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_4, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4, var_4]
    var_6 = 3
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]
    var_8 = [var_2, var_3, var_3, var_4, var_4, var_4, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'You must pass date_start and date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_2, var_3, var_3]
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'date_start can not be larger than date_end'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 5
    var_6 = [var_2, var_3, var_5]
    var_7 = -1
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'timedelta must be positive'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_6, var_6, var_4, var_4]



