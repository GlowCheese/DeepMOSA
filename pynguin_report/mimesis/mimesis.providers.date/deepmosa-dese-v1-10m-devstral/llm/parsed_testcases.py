####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2025
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)
    var_3 = var_2.tzinfo
    var_4 = str(var_3)
    assert var_4 == 'UTC'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Invalid/Timezone'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 6/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 6/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 5/8 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/4 statements.


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
    var_1 = 5
    var_2 = 15
    var_3 = var_0.duration(var_1, var_2)
    var_4 = module_1.timedelta()
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
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 15
    var_2 = 5
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_bulk_create_datetimes_basic. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs. Retrieved 3/7 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = -1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 0
    var_4 = 2



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)
    var_4 = 1
    var_5 = '10'
    var_6 = var_0.duration(var_4, var_5)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_duration_with_invalid_types. Retrieved 5/17 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = 1
    var_4 = '10'



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_valid_input. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_with_empty_kwargs. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_start_larger_than_end. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_missing_start_and_end. Retrieved 3/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 6

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = -1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_bulk_create_datetimes_empty_args. Retrieved 2/4 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 2022

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 2
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 3



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/6 statements.


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
    var_3 = var_2.tzinfo
    var_4 = str(var_3)
    assert var_4 == 'UTC'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Invalid/Timezone'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 5/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2025
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)
    var_3 = var_2.tzinfo
    var_4 = str(var_3)
    assert var_4 == 'UTC'

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Invalid/Timezone'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_duration_default. Retrieved 2/5 statements.
# Partially parsed test_duration_custom_range. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 1/6 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 15
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 1
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_valid_inputs. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_empty_kwargs. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_with_invalid_range. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_missing_dates. Retrieved 3/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 2

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = -1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2025
    var_3 = var_0.datetime(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)
    var_4 = 1
    var_5 = 10.5
    var_6 = var_0.duration(var_4, var_5)
    var_7 = 1.5
    var_8 = 10.5
    var_9 = var_0.duration(var_7, var_8)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3



# Parsed testcases at query #17
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_non_positive_timedelta. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = 0



# Parsed testcases at query #19
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_timedelta_must_be_positive. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = 0
    var_4 = str(var_0)
    assert var_4 == 'timedelta must be positive'



# Parsed testcases at query #21
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 6/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 6/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 5/8 statements.
# Partially parsed test_duration_none_unit. Retrieved 7/8 statements.


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
    var_1 = 5
    var_2 = 15
    var_3 = var_0.duration(var_1, var_2)
    var_4 = module_1.timedelta()
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
import datetime as module_1

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)
    var_3 = 1
    var_4 = module_1.timedelta()
    var_5 = 10
    var_6 = module_1.timedelta()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 1
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_duration_predicate_false. Retrieved 3/8 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 2/5 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 1/6 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 15
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 1
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)
    var_4 = 1
    var_5 = '10'
    var_6 = var_0.duration(var_4, var_5)



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)
    var_4 = 1
    var_5 = '10'
    var_6 = var_0.duration(var_4, var_5)
    var_7 = '1'
    var_8 = '10'
    var_9 = var_0.duration(var_7, var_8)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_duration_with_non_integer_values. Retrieved 9/13 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 1.5
    var_3 = '1'
    var_4 = 10
    var_5 = var_0.duration(var_3, var_4)
    var_6 = 1.5
    var_7 = 10
    var_8 = var_0.duration(var_6, var_7)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_datetime_default. Retrieved 2/3 statements.
# Partially parsed test_datetime_with_start_and_end. Retrieved 4/5 statements.
# Partially parsed test_datetime_with_timezone. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2025
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_duration_with_invalid_types. Retrieved 5/13 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 10
    var_3 = 1
    var_4 = 10.5



# Parsed testcases at query #9
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_value_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end_raises_value_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_value_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_with_minutes. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = -1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 0
    var_4 = 5

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 0
    var_4 = 5



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_empty_kwargs. Retrieved 5/8 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2
    var_3 = module_0.timedelta()
    var_4 = module_0.timedelta()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_duration_with_integer_arguments. Retrieved 4/5 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_duration_with_invalid_types. Retrieved 3/7 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 10



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_bulk_create_datetimes_with_positive_timedelta. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2



# Parsed testcases at query #17
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_start_larger_than_end_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 2

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = -1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 0
    var_4 = 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_duration_default_params. Retrieved 2/5 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 1/6 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/6 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 15
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 1
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1.5
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #20
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_range. Retrieved 6/12 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_missing_args. Retrieved 3/5 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 6/12 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 6

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 0

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 0
    var_4 = 5
    var_5 = 6



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 2/5 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 4/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 1/6 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/4 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = 15
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 10
    var_2 = 5
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #23
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)
    var_4 = 1
    var_5 = '10'
    var_6 = var_0.duration(var_4, var_5)
    var_7 = '1'
    var_8 = '10'
    var_9 = var_0.duration(var_7, var_8)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_timedelta_must_be_positive. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = 'days'
    var_4 = 0
    var_5 = {var_3: var_4}



# Parsed testcases at query #25
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_duration_default_values. Retrieved 6/7 statements.
# Partially parsed test_duration_custom_min_max. Retrieved 6/7 statements.
# Partially parsed test_duration_custom_unit. Retrieved 5/8 statements.
# Partially parsed test_duration_none_unit. Retrieved 3/4 statements.


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
    var_1 = 5
    var_2 = 15
    var_3 = var_0.duration(var_1, var_2)
    var_4 = module_1.timedelta()
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
    var_1 = None
    var_2 = var_0.duration(duration_unit=var_1)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 15
    var_2 = 5
    var_3 = var_0.duration(var_1, var_2)

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = '1'
    var_2 = 10
    var_3 = var_0.duration(var_1, var_2)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_bulk_create_datetimes_valid_input. Retrieved 5/11 statements.
# Partially parsed test_bulk_create_datetimes_empty_kwargs_raises_error. Retrieved 4/8 statements.
# Partially parsed test_bulk_create_datetimes_invalid_range_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_non_positive_timedelta_raises_error. Retrieved 5/9 statements.
# Partially parsed test_bulk_create_datetimes_with_hours. Retrieved 5/11 statements.


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 2

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = -1

import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 0
    var_4 = 2



# Parsed testcases at query #28
#--------------------------




import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'UTC'
    var_2 = var_0.datetime(timezone=var_1)



