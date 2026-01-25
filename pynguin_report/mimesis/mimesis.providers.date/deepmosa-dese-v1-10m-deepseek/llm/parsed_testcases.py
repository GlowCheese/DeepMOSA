####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_duration_with_default_parameters. Retrieved 6/7 statements.
# Partially parsed test_duration_with_custom_duration_unit. Retrieved 5/8 statements.
# Partially parsed test_duration_with_custom_min_and_max_duration. Retrieved 6/7 statements.
# Partially parsed test_duration_with_random_duration_unit. Retrieved 3/4 statements.


import mimesis.providers.date as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 1
    var_3 = module_1.timedelta()
    var_4 = 10
    var_5 = module_1.timedelta()

import mimesis.providers.date as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = module_1.timedelta()
    var_3 = 10
    var_4 = module_1.timedelta()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 5
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10.5
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2
    var_2 = 8
    var_3 = var_0.duration(var_1, var_2)
    var_4 = module_1.timedelta()
    var_5 = module_1.timedelta()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_datetime_without_timezone. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = 'UTC'
    var_4 = var_0.datetime(var_1, var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = 'Invalid/Timezone'
    var_4 = var_0.datetime(var_1, var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = 'UTC'
    var_4 = var_0.datetime(var_1, var_2, var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs. Retrieved 3/7 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_no_start_and_end. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 1

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_error_for_non_positive_timedelta. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = 'days'
    var_4 = 'hours'
    var_5 = 'minutes'
    var_6 = 'seconds'
    var_7 = 'microseconds'
    var_8 = 0
    var_9 = {var_3: var_8, var_4: var_8, var_5: var_8, var_6: var_8, var_7: var_8}



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 10
    var_3 = 'days'
    var_4 = 0
    var_5 = {var_3: var_4}



# Parsed testcases at query #7
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_empty_range. Retrieved 2/6 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_invalid_step. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_missing_arguments. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 0

def test_case_0():
    var_0 = None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_datetime_with_default_values. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2000
    var_2 = 2010
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Invalid/Timezone'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_timezone_provided_but_pytz_not_installed. Retrieved 4/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'pytz'
    var_1 = module_0.Datetime()
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 10
    var_3 = 'days'
    var_4 = 0
    var_5 = {var_3: var_4}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_error. Retrieved 3/7 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end_raises_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes_step. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_datetime_with_default_values. Retrieved 2/3 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2022
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Invalid/Timezone'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #14
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 10
    var_3 = 'days'
    var_4 = 0
    var_5 = {var_3: var_4}



# Parsed testcases at query #16
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #17
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
    var_2 = 3
    var_3 = 2

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_duration_with_minutes. Retrieved 5/12 statements.
# Partially parsed test_duration_with_hours. Retrieved 5/12 statements.
# Partially parsed test_duration_with_days. Retrieved 5/12 statements.
# Partially parsed test_duration_with_random_unit. Retrieved 5/6 statements.
# Partially parsed test_duration_with_invalid_min_max. Retrieved 3/6 statements.
# Partially parsed test_duration_with_non_integer_min_max. Retrieved 3/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = 60
    var_4 = 600

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = 3600
    var_4 = 36000

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = 86400
    var_4 = 864000

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = None
    var_4 = var_0.duration(var_1, var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 10.5



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_datetime. Retrieved 2/3 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2000
    var_2 = 2010
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'INVALID'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #3
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_timezone_provided_but_pytz_not_installed. Retrieved 4/9 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'pytz'
    var_1 = module_0.Datetime()
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_duration_with_minutes_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_hours_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_days_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_seconds_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_random_unit. Retrieved 5/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 15

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2
    var_2 = 7

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 30
    var_2 = 120

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = None
    var_4 = var_0.duration(var_1, var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 5
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 5.5
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_value_error_when_no_dates_provided. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_raises_value_error_when_start_after_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_raises_value_error_when_non_positive_timedelta. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_returns_correct_list_of_dates. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds_step. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_with_multiple_timedelta_args. Retrieved 8/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2022
    var_3 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = module_0.Datetime()
    var_4 = 0

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Datetime()
    var_4 = 2

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2
    var_4 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2
    var_4 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2
    var_4 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2000
    var_4 = module_0.Datetime()
    var_5 = 1000

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 30
    var_5 = module_0.Datetime()
    var_6 = 12
    var_7 = 15



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_duration_with_valid_integer_parameters. Retrieved 4/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #9
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_datetime_timezone_requires_pytz. Retrieved 5/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'pytz'
    var_2 = None
    var_3 = 'UTC'
    var_4 = var_0.datetime(timezone=var_3)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 2/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2000
    var_2 = 2010
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'INVALID_TZ'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = 'days'
    var_4 = {var_3: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_valid_inputs. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_empty_dates. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_with_start_after_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes_step. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2

def test_case_0():
    var_0 = None
    var_1 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = 2



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_valid_inputs. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_missing_dates. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_with_start_after_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_datetime_with_default_values. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Invalid/Timezone'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #16
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = 'days'
    var_4 = 0
    var_5 = {var_3: var_4}



# Parsed testcases at query #18
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_duration_with_minutes_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_hours_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_seconds_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_days_unit. Retrieved 3/6 statements.
# Partially parsed test_duration_with_random_unit. Retrieved 5/7 statements.
# Partially parsed test_duration_with_invalid_min_max. Retrieved 3/6 statements.
# Partially parsed test_duration_with_non_integer_min_max. Retrieved 3/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 15

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 30
    var_2 = 120

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 7

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = None
    var_4 = var_0.duration(var_1, var_2, var_3)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 5.5



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 10
    var_3 = 'days'
    var_4 = {var_3: var_1}



# Parsed testcases at query #21
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #22
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = '10'
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = '10'
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 10
    var_3 = 'days'
    var_4 = 0
    var_5 = {var_3: var_4}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Invalid/Timezone'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #25
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_value_error_for_non_positive_timedelta. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = 'days'
    var_4 = 0
    var_5 = {var_3: var_4}



