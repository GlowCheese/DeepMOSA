####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_timestamp_with_posix_format. Retrieved 1/4 statements.
# Partially parsed test_timestamp_with_rfc_3339_format. Retrieved 2/7 statements.
# Partially parsed test_timestamp_with_iso_8601_format. Retrieved 1/4 statements.
# Partially parsed test_timestamp_with_custom_datetime_kwargs. Retrieved 3/6 statements.
# Partially parsed test_timestamp_with_timezone. Retrieved 2/5 statements.
# Partially parsed test_timestamp_default_format_is_posix. Retrieved 2/3 statements.


import mimesis.providers.date as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Z'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'T'
    var_3 = '.'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2021


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = {}
    var_3 = var_1.timestamp(**var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'Invalid/Timezone'
    var_3 = 'timezone'
    var_4 = {var_3: var_2}
    var_5 = var_1.timestamp(**var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_duration_with_default_parameters. Retrieved 4/7 statements.
# Partially parsed test_duration_with_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_with_hours_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_days_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_seconds_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_microseconds_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_milliseconds_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_weeks_unit. Retrieved 3/8 statements.
# Partially parsed test_duration_with_none_unit. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()
    var_3 = 1
    var_4 = []
    var_5 = 10
    var_6 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)
    var_5 = []
    var_6 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = []
    var_4 = 10
    var_5 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1.5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10.5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 8/13 statements.
# Partially parsed test_datetime_returns_datetime_instance. Retrieved 4/5 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = bool(var_2 <= var_4.year)
    assert var_5 is True
    var_6 = var_4.year
    var_7 = bool(var_4.year <= var_3)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2022
    var_3 = 5
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = [var_6, var_7, var_8]
    var_10 = var_1.datetime(var_2, var_2)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2021
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    var_5 = bool(var_3.year == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2019
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_timezone_given_and_pytz_not_installed. Retrieved 4/7 statements.



def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'UTC'
    var_4 = var_2.datetime(timezone=var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_bulk_create_datetimes_raises_value_error_when_no_dates_provided. Retrieved 2/5 statements.
# Partially parsed test_bulk_create_datetimes_raises_value_error_when_date_start_greater_than_date_end. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_raises_value_error_when_timedelta_not_positive. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_creates_list_with_correct_step. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_creates_list_with_hours_step. Retrieved 5/13 statements.
# Partially parsed test_bulk_create_datetimes_returns_empty_list_when_start_equals_end_and_step_positive. Retrieved 3/8 statements.
# Partially parsed test_bulk_create_datetimes_handles_microseconds_step. Retrieved 6/14 statements.
# Partially parsed test_bulk_create_datetimes_works_with_combined_timedelta_arguments. Retrieved 5/12 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 2022
    var_6 = [var_5, var_3, var_3]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 2
    var_6 = [var_2, var_3, var_5]
    var_7 = 0


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
    var_9 = [var_2, var_3, var_5]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_3, var_6, var_4, var_4]
    var_8 = [var_2, var_3, var_3, var_3, var_4, var_4]
    var_9 = [var_2, var_3, var_3, var_6, var_4, var_4]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = [var_2, var_3, var_3]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4, var_4]
    var_6 = 2000
    var_7 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]
    var_8 = 1000
    var_9 = [var_2, var_3, var_3, var_4, var_4, var_4, var_8]
    var_10 = [var_2, var_3, var_3, var_4, var_4, var_4, var_6]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = [var_2, var_3, var_3, var_4, var_4, var_4]
    var_6 = 2
    var_7 = [var_2, var_3, var_6, var_3, var_3, var_3]
    var_8 = [var_2, var_3, var_6, var_3, var_3, var_3]



# Parsed testcases at query #6
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 7/16 statements.
# Partially parsed test_bulk_create_datetimes_empty_start_and_end. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_zero_timedelta. Retrieved 3/7 statements.
# Partially parsed test_bulk_create_datetimes_hours_step. Retrieved 7/16 statements.
# Partially parsed test_bulk_create_datetimes_minutes_step. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_seconds_step. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_microseconds_step. Retrieved 6/13 statements.
# Partially parsed test_bulk_create_datetimes_combined_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_exact_match. Retrieved 3/8 statements.


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
    var_12 = 6
    var_13 = [var_0, var_1, var_12]

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_2, var_2]
    var_7 = 2
    var_8 = [var_0, var_1, var_1, var_7, var_2, var_2]
    var_9 = 3
    var_10 = [var_0, var_1, var_1, var_9, var_2, var_2]
    var_11 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_12 = 5
    var_13 = [var_0, var_1, var_1, var_12, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 2
    var_5 = [var_0, var_1, var_1, var_2, var_4, var_2]
    var_6 = [var_0, var_1, var_1, var_2, var_1, var_2]
    var_7 = [var_0, var_1, var_1, var_2, var_4, var_2]
    var_8 = 3
    var_9 = [var_0, var_1, var_1, var_2, var_8, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 2
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_4]
    var_6 = [var_0, var_1, var_1, var_2, var_2, var_1]
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_4]
    var_8 = 3
    var_9 = [var_0, var_1, var_1, var_2, var_2, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2, var_2]
    var_4 = 2000
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_6 = 1000
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_2, var_6]
    var_8 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_9 = 3000
    var_10 = [var_0, var_1, var_1, var_2, var_2, var_2, var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 30
    var_5 = [var_0, var_1, var_1, var_1, var_4, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_4, var_2]
    var_7 = 3
    var_8 = [var_0, var_1, var_1, var_7, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = 2
    var_5 = [var_0, var_1, var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 7/16 statements.
# Partially parsed test_bulk_create_datetimes_empty_date_start_and_date_end. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_date_start_larger_than_date_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 7/15 statements.
# Partially parsed test_bulk_create_datetimes_single_step_exact_end. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_microseconds_step. Retrieved 8/17 statements.


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
    var_12 = 6
    var_13 = [var_0, var_1, var_12]

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 6
    var_5 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_6 = 2
    var_7 = [var_0, var_1, var_1, var_6, var_2, var_2]
    var_8 = 4
    var_9 = [var_0, var_1, var_1, var_8, var_2, var_2]
    var_10 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_11 = 8
    var_12 = [var_0, var_1, var_1, var_11, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2, var_2]
    var_4 = 2000
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_6 = 500
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_2, var_6]
    var_8 = 1000
    var_9 = [var_0, var_1, var_1, var_2, var_2, var_2, var_8]
    var_10 = 1500
    var_11 = [var_0, var_1, var_1, var_2, var_2, var_2, var_10]
    var_12 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_13 = 2500
    var_14 = [var_0, var_1, var_1, var_2, var_2, var_2, var_13]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone_with_pytz. Retrieved 4/6 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 10/16 statements.
# Partially parsed test_datetime_start_greater_than_end. Retrieved 4/5 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2023)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Timezones are supported only with pytz'


def test_case_0():
    var_0 = 'pytz.timezone'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'America/New_York'
    var_4 = var_2.datetime(timezone=var_3)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2022
    var_3 = 5
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = [var_6, var_7, var_8]
    var_10 = 'date'
    var_11 = 'time'
    var_12 = var_1.datetime()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2023)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 6/15 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]
    var_7 = 'days'
    var_8 = {var_7: var_3}



# Parsed testcases at query #11
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_positive_timedelta. Retrieved 4/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 7/16 statements.
# Partially parsed test_bulk_create_datetimes_empty_start_and_end. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_negative_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_hours_step. Retrieved 7/16 statements.
# Partially parsed test_bulk_create_datetimes_minutes_step. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_seconds_step. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_microseconds_step. Retrieved 6/13 statements.
# Partially parsed test_bulk_create_datetimes_combined_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_single_step_exact_end. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_step_larger_than_range. Retrieved 5/10 statements.


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
    var_12 = 6
    var_13 = [var_0, var_1, var_12]

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = -1
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_2, var_2]
    var_7 = 2
    var_8 = [var_0, var_1, var_1, var_7, var_2, var_2]
    var_9 = 3
    var_10 = [var_0, var_1, var_1, var_9, var_2, var_2]
    var_11 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_12 = 5
    var_13 = [var_0, var_1, var_1, var_12, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 2
    var_5 = [var_0, var_1, var_1, var_2, var_4, var_2]
    var_6 = [var_0, var_1, var_1, var_2, var_1, var_2]
    var_7 = [var_0, var_1, var_1, var_2, var_4, var_2]
    var_8 = 3
    var_9 = [var_0, var_1, var_1, var_2, var_8, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 2
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_4]
    var_6 = [var_0, var_1, var_1, var_2, var_2, var_1]
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_4]
    var_8 = 3
    var_9 = [var_0, var_1, var_1, var_2, var_2, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2, var_2]
    var_4 = 2000
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_6 = 1000
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_2, var_6]
    var_8 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_9 = 3000
    var_10 = [var_0, var_1, var_1, var_2, var_2, var_2, var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 30
    var_5 = [var_0, var_1, var_1, var_1, var_4, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_4, var_2]
    var_7 = 3
    var_8 = [var_0, var_1, var_1, var_7, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 5
    var_6 = 6
    var_7 = [var_0, var_1, var_6]



# Parsed testcases at query #14
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_datetime_timezone_without_pytz. Retrieved 4/7 statements.



def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'UTC'
    var_4 = var_2.datetime(timezone=var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_bulk_create_datetimes_positive_timedelta. Retrieved 4/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone_with_pytz. Retrieved 4/6 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 10/16 statements.
# Partially parsed test_datetime_start_greater_than_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_same_start_and_end. Retrieved 3/4 statements.
# Partially parsed test_datetime_negative_years. Retrieved 4/5 statements.
# Partially parsed test_datetime_large_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_timezone_none. Retrieved 3/4 statements.
# Partially parsed test_datetime_timezone_empty_string. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2010
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = bool(var_2 <= var_4.year)
    assert var_5 is True
    var_6 = var_4.year
    var_7 = bool(var_4.year <= var_3)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)


def test_case_0():
    var_0 = 'pytz.timezone'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'America/New_York'
    var_4 = var_2.datetime(timezone=var_3)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 5
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = [var_6, var_7, var_8]
    var_10 = 'date'
    var_11 = 'time'
    var_12 = var_1.datetime()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2020
    var_3 = 2010
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2010
    var_6 = bool(2010 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2020)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2015
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    var_5 = bool(var_3.year == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = -100
    var_3 = -50
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = -100
    var_6 = bool(-100 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= -50)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 9999
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 1
    var_6 = bool(1 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 9999)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    assert var_4 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = ''
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    assert var_4 is None



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone_with_pytz. Retrieved 9/16 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 10/16 statements.
# Partially parsed test_datetime_with_start_greater_than_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_timezone_parameter_none. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2010
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = bool(var_2 <= var_4.year)
    assert var_5 is True
    var_6 = var_4.year
    var_7 = bool(var_4.year <= var_3)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = 'Timezones are supported only with pytz'


def test_case_0():
    var_0 = 'mimesis.providers.datetime.pytz'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'mimesis.providers.datetime.pytz.timezone'
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0
    var_8 = [var_4, var_5, var_5, var_6, var_7, var_7]
    var_9 = 'America/New_York'
    var_10 = var_2.datetime(timezone=var_9)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 5
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = [var_6, var_7, var_8]
    var_10 = 'date'
    var_11 = 'time'
    var_12 = var_1.datetime()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 2020
    var_6 = bool(2020 <= var_4.year)
    assert var_6 is True
    var_7 = var_4.year
    var_8 = bool(var_4.year <= 2023)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = type(var_2)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = range(var_2)
    var_4 = 2020
    var_5 = [var_1.datetime(var_4, var_4) for _ in var_3]
    var_6 = set(var_5)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 1)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2015
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    var_5 = bool(var_3.year == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.datetime(timezone=var_2)
    var_4 = var_3.tzinfo
    assert var_4 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_duration_with_default_parameters. Retrieved 2/5 statements.
# Partially parsed test_duration_with_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_with_different_units. Retrieved 1/13 statements.
# Partially parsed test_duration_with_none_unit. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.duration()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 5
    var_3 = 15
    var_4 = var_1.duration(var_2, var_3)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = var_1.duration(duration_unit=var_2)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 10
    var_3 = 5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1.5
    var_3 = 10
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 1
    var_3 = 10.5
    var_4 = var_1.duration(var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_timezone_given_and_pytz_not_installed. Retrieved 6/14 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 'pytz'
    var_4 = 'UTC'
    var_5 = var_1.datetime(timezone=var_4)
    var_6 = 'pytz'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 5/12 statements.
# Partially parsed test_bulk_create_datetimes_empty_arguments. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 7/16 statements.
# Partially parsed test_bulk_create_datetimes_single_step_exact_end. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds. Retrieved 8/17 statements.
# Partially parsed test_bulk_create_datetimes_negative_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_zero_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_combined_timedelta. Retrieved 6/12 statements.


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
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
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
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_2, var_2]
    var_7 = 2
    var_8 = [var_0, var_1, var_1, var_7, var_2, var_2]
    var_9 = 3
    var_10 = [var_0, var_1, var_1, var_9, var_2, var_2]
    var_11 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_12 = 5
    var_13 = [var_0, var_1, var_1, var_12, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2, var_2]
    var_4 = 2000
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_6 = 500
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_2, var_6]
    var_8 = 1000
    var_9 = [var_0, var_1, var_1, var_2, var_2, var_2, var_8]
    var_10 = 1500
    var_11 = [var_0, var_1, var_1, var_2, var_2, var_2, var_10]
    var_12 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_13 = 2500
    var_14 = [var_0, var_1, var_1, var_2, var_2, var_2, var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = -1
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
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 2
    var_5 = 30
    var_6 = [var_0, var_1, var_1, var_4, var_5, var_2]
    var_7 = [var_0, var_1, var_1, var_1, var_5, var_2]
    var_8 = 3
    var_9 = [var_0, var_1, var_1, var_8, var_2, var_2]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone_with_pytz. Retrieved 9/16 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 10/15 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2010
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = bool(var_2 <= var_4.year)
    assert var_5 is True
    var_6 = var_4.year
    var_7 = bool(var_4.year <= var_3)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)


def test_case_0():
    var_0 = 'mimesis.providers.datetime.pytz'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'mimesis.providers.datetime.pytz.timezone'
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0
    var_8 = [var_4, var_5, var_5, var_6, var_7, var_7]
    var_9 = 'America/New_York'
    var_10 = var_2.datetime(timezone=var_9)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 5
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = [var_6, var_7, var_8]
    var_10 = 'date'
    var_11 = 'time'
    var_12 = var_1.datetime()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 2022
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = var_4.year
    assert var_5 == 2023


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = type(var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #7
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #10
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/12 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_datetime_raises_import_error_when_timezone_given_and_pytz_not_installed. Retrieved 6/14 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = None
    var_3 = 'pytz'
    var_4 = 'UTC'
    var_5 = var_1.datetime(timezone=var_4)
    var_6 = 'pytz'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_bulk_create_datetimes_positive_timedelta. Retrieved 4/13 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 10
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_datetime_with_default_parameters. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_custom_year_range. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone_with_pytz. Retrieved 9/15 statements.
# Partially parsed test_datetime_combines_date_and_time. Retrieved 10/18 statements.
# Partially parsed test_datetime_returns_valid_datetime_object. Retrieved 4/5 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = var_1.datetime()
    var_3 = var_2.year


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2010
    var_3 = 2020
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = bool(var_2 <= var_4.year)
    assert var_5 is True
    var_6 = var_4.year
    var_7 = bool(var_4.year <= var_3)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)


def test_case_0():
    var_0 = 'mimesis.providers.datetime.pytz'
    var_1 = {}
    var_2 = module_0.Datetime(**var_1)
    var_3 = 'mimesis.providers.datetime.pytz.timezone'
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 0
    var_8 = [var_4, var_5, var_5, var_6, var_7, var_7]
    var_9 = 'America/New_York'
    var_10 = var_2.datetime(timezone=var_9)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'date'
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = 'time'
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = [var_8, var_9, var_10]
    var_12 = var_1.datetime()
    var_13 = [var_3, var_4, var_5]
    var_14 = [var_8, var_9, var_10]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 2015
    var_3 = var_1.datetime(var_2, var_2)
    var_4 = var_3.year
    var_5 = bool(var_3.year == var_2)
    assert var_5 is True


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
    var_17 = 0
    var_18 = bool(0 <= var_4.hour)
    assert var_18 is True
    var_19 = var_4.hour
    var_20 = bool(var_4.hour <= 23)
    assert var_20 is True
    var_21 = 0
    var_22 = bool(0 <= var_4.minute)
    assert var_22 is True
    var_23 = var_4.minute
    var_24 = bool(var_4.minute <= 59)
    assert var_24 is True
    var_25 = 0
    var_26 = bool(0 <= var_4.second)
    assert var_26 is True
    var_27 = var_4.second
    var_28 = bool(var_4.second <= 59)
    assert var_28 is True
    var_29 = 0
    var_30 = bool(0 <= var_4.microsecond)
    assert var_30 is True
    var_31 = var_4.microsecond
    var_32 = bool(var_4.microsecond <= 999999)
    assert var_32 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_dates. Retrieved 1/3 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_hours_step. Retrieved 4/10 statements.
# Partially parsed test_bulk_create_datetimes_minutes_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_seconds_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_microseconds_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_combined_step. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_exact_end. Retrieved 4/9 statements.
# Partially parsed test_bulk_create_datetimes_single_step. Retrieved 3/8 statements.


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
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True

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
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 5
    var_5 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_6 = [var_0, var_1, var_1, var_1, var_2, var_2]
    var_7 = [var_0, var_1, var_1, var_4, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 10
    var_5 = [var_0, var_1, var_1, var_2, var_4, var_2]
    var_6 = 2
    var_7 = [var_0, var_1, var_1, var_2, var_6, var_2]
    var_8 = [var_0, var_1, var_1, var_2, var_4, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 10
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_4]
    var_6 = 2
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_6]
    var_8 = [var_0, var_1, var_1, var_2, var_2, var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2, var_2]
    var_4 = 1000
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_6 = 200
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_2, var_6]
    var_8 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 2
    var_5 = 12
    var_6 = [var_0, var_1, var_4, var_5, var_2, var_2]
    var_7 = [var_0, var_1, var_1, var_5, var_2, var_2]
    var_8 = [var_0, var_1, var_4, var_5, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 6
    var_6 = [var_0, var_1, var_5]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = 2
    var_5 = [var_0, var_1, var_4]



# Parsed testcases at query #16
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Datetime(**var_0)
    var_2 = 'UTC'
    var_3 = var_1.datetime(timezone=var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range_and_step. Retrieved 7/16 statements.
# Partially parsed test_bulk_create_datetimes_empty_start_and_end_raises_value_error. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_start_greater_than_end_raises_value_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_value_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_hours_step. Retrieved 7/15 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes_step. Retrieved 8/16 statements.
# Partially parsed test_bulk_create_datetimes_with_seconds_step. Retrieved 7/15 statements.
# Partially parsed test_bulk_create_datetimes_with_microseconds_step. Retrieved 7/15 statements.
# Partially parsed test_bulk_create_datetimes_with_combined_step. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_single_step_exact_end. Retrieved 4/10 statements.


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
    var_12 = 6
    var_13 = [var_0, var_1, var_12]

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 6
    var_5 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_6 = 2
    var_7 = [var_0, var_1, var_1, var_6, var_2, var_2]
    var_8 = 4
    var_9 = [var_0, var_1, var_1, var_8, var_2, var_2]
    var_10 = [var_0, var_1, var_1, var_4, var_2, var_2]
    var_11 = 8
    var_12 = [var_0, var_1, var_1, var_11, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 10
    var_5 = [var_0, var_1, var_1, var_2, var_4, var_2]
    var_6 = 3
    var_7 = [var_0, var_1, var_1, var_2, var_6, var_2]
    var_8 = 6
    var_9 = [var_0, var_1, var_1, var_2, var_8, var_2]
    var_10 = 9
    var_11 = [var_0, var_1, var_1, var_2, var_10, var_2]
    var_12 = 12
    var_13 = [var_0, var_1, var_1, var_2, var_12, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 12
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_4]
    var_6 = 4
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_6]
    var_8 = 8
    var_9 = [var_0, var_1, var_1, var_2, var_2, var_8]
    var_10 = [var_0, var_1, var_1, var_2, var_2, var_4]
    var_11 = 16
    var_12 = [var_0, var_1, var_1, var_2, var_2, var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2, var_2]
    var_4 = 300000
    var_5 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_6 = 100000
    var_7 = [var_0, var_1, var_1, var_2, var_2, var_2, var_6]
    var_8 = 200000
    var_9 = [var_0, var_1, var_1, var_2, var_2, var_2, var_8]
    var_10 = [var_0, var_1, var_1, var_2, var_2, var_2, var_4]
    var_11 = 400000
    var_12 = [var_0, var_1, var_1, var_2, var_2, var_2, var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = [var_0, var_1, var_1, var_2, var_2, var_2]
    var_4 = 2
    var_5 = 12
    var_6 = [var_0, var_1, var_4, var_5, var_2, var_2]
    var_7 = [var_0, var_1, var_4, var_5, var_2, var_2]
    var_8 = 4
    var_9 = [var_0, var_1, var_8, var_2, var_2, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = 3
    var_7 = [var_0, var_1, var_6]



