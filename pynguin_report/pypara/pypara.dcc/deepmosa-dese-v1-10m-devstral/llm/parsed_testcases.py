####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_register_successfully_adds_dcc_to_main_buffer. Retrieved 6/10 statements.
# Partially parsed test_register_successfully_adds_dcc_to_alternative_buffer. Retrieved 6/10 statements.
# Partially parsed test_register_raises_typeerror_for_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_raises_typeerror_for_duplicate_alternative_name. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = {}
    var_3 = {}
    var_4 = 0.5
    var_5 = {}
    var_6 = {}

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = 'AnotherDCC'
    var_7 = {var_2}
    var_8 = {}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_DCCRegistryMachinery_constructor_initializes_buffers. Retrieved 11/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '_buffer_main'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._buffer_main
    var_4 = var_0._buffer_main
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = '_buffer_altn'
    var_7 = hasattr(var_0, var_6)
    var_8 = var_0._buffer_altn
    var_9 = var_0._buffer_altn
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #3
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 7/19 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 14/34 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_same_start_and_asof_month. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 9
    var_6 = 4

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 2
    var_6 = 7
    var_7 = 8
    var_8 = 4
    var_9 = 30
    var_10 = 2012
    var_11 = 15
    var_12 = 2016
    var_13 = 6

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = 2015
    var_3 = 10
    var_4 = 6
    var_5 = 4

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = 2015
    var_4 = 4
    var_5 = 30



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_date_range_empty. Retrieved 2/6 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 3/9 statements.
# Partially parsed test_get_date_range_multiple_days. Retrieved 5/13 statements.
# Partially parsed test_get_date_range_year_boundary. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 4
    var_3 = 2
    var_4 = 3

def test_case_0():
    var_0 = 2019
    var_1 = 12
    var_2 = 30
    var_3 = 2020
    var_4 = 1
    var_5 = 3
    var_6 = 31
    var_7 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_register_raises_typeerror_for_duplicate_main_name. Retrieved 7/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 0.5
    var_5 = set()
    var_6 = set()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_calculate_daily_fraction_with_valid_dates. Retrieved 11/17 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_equal_to_start. Retrieved 10/16 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_before_start. Retrieved 11/17 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_after_end. Retrieved 11/17 statements.
# Partially parsed test_calculate_daily_fraction_with_custom_frequency. Retrieved 11/18 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 365
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = 12
    var_9 = 31
    var_10 = var_6 / var_3

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 365
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 31
    var_9 = var_6 / var_3

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 365
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 2022
    var_8 = 12
    var_9 = 31
    var_10 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 365
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 2024
    var_8 = 12
    var_9 = 31
    var_10 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 365
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3) * f if f else Decimal((a - s).days / var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = 12
    var_9 = 31
    var_10 = var_7 / var_3



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_calculate_daily_fraction_asof_minus_1_not_less_than_start. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = 1
    var_4 = 2023
    var_5 = 2
    var_6 = 3
    var_7 = 0



# Parsed testcases at query #9
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #10
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 30
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 1
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0._construct_date(var_0, var_1, var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = -1
    var_1 = -1
    var_2 = -1
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_30_360_german_basic_cases. Retrieved 18/42 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33055555555556'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcfc_30_360_german_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_german_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33055555555556'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_interest_calculates_accrued_interest_correctly. Retrieved 20/30 statements.
# Partially parsed test_interest_uses_asof_as_end_when_end_is_none. Retrieved 18/27 statements.
# Partially parsed test_interest_returns_zero_when_asof_before_start. Retrieved 14/23 statements.
# Partially parsed test_interest_returns_zero_when_asof_after_end. Retrieved 14/23 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = 360
    var_6 = 1000
    var_7 = module_0.Currency(var_2)
    var_8 = '0.05'
    var_9 = 2023
    var_10 = 1
    var_11 = 6
    var_12 = 12
    var_13 = 31
    var_14 = 0.05
    var_15 = var_6 * var_14
    var_16 = 151
    var_17 = var_15 * var_16
    var_18 = var_17 / var_5
    var_19 = module_0.Currency(var_2)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = 360
    var_6 = 1000
    var_7 = module_0.Currency(var_2)
    var_8 = '0.05'
    var_9 = 2023
    var_10 = 1
    var_11 = 6
    var_12 = 0.05
    var_13 = var_6 * var_12
    var_14 = 151
    var_15 = var_13 * var_14
    var_16 = var_15 / var_5
    var_17 = module_0.Currency(var_2)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = 360
    var_6 = 1000
    var_7 = module_0.Currency(var_2)
    var_8 = '0.05'
    var_9 = 2023
    var_10 = 1
    var_11 = 2022
    var_12 = 12
    var_13 = 31

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = 360
    var_6 = 1000
    var_7 = module_0.Currency(var_2)
    var_8 = '0.05'
    var_9 = 2023
    var_10 = 1
    var_11 = 2024
    var_12 = 12
    var_13 = 31



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 1
    var_6 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33333333333333'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_existing_dcc. Retrieved 9/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = '0.5'
    var_5 = var_0.find(var_1)
    var_6 = var_0.find(var_2)
    var_7 = 'act/act'
    var_8 = var_0.find(var_7)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None
    var_3 = '  '
    var_4 = var_0.find(var_3)
    assert var_4 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33333333333333'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dcfc_nl_365_with_no_leap_day. Retrieved 7/12 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day. Retrieved 8/13 statements.
# Partially parsed test_dcfc_nl_365_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_nl_365_cross_year. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16986301369863'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16986301369863'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08219178082192'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32602739726027'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_register_raises_typeerror_for_duplicate_main_name. Retrieved 6/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = set()
    var_5 = set()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33333333333333'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_returns_correct_dcc_with_stripped_uppercase_name. Retrieved 7/9 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = 'Actual/Actual'
    var_4 = [var_2, var_3]
    var_5 = ' act/act '
    var_6 = var_0.find(var_5)



# Parsed testcases at query #22
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dcfc_30_360_german_asof_day_31_or_last_day_of_feb. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1
    var_5 = '57'
    var_6 = '360'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_calculate_daily_fraction_when_asof_minus_1_less_than_start. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = 3



# Parsed testcases at query #25
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_calculate_fraction_with_valid_dates. Retrieved 12/20 statements.
# Partially parsed test_calculate_fraction_with_invalid_dates. Retrieved 13/20 statements.
# Partially parsed test_calculate_fraction_with_frequency. Retrieved 13/21 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = '0.5'
    var_7 = 2023
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = '0.5'
    var_7 = 2023
    var_8 = 1
    var_9 = 2022
    var_10 = 6
    var_11 = 12
    var_12 = 31

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = '0.5'
    var_7 = 2023
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = '0.75'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dcfc_act_act_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_act_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16942884946478'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.17216108990194'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08243131970956'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32625945055768'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_d1_equals_31. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = '1.33333333333333'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dcfc_act_365_a_without_leap_day. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_365_a_with_leap_day. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_a_full_year_without_leap. Retrieved 5/10 statements.
# Partially parsed test_dcfc_act_365_a_full_year_with_leap. Retrieved 6/12 statements.
# Partially parsed test_dcfc_act_365_a_partial_period. Retrieved 6/12 statements.
# Partially parsed test_dcfc_act_365_a_same_start_and_asof. Retrieved 5/10 statements.
# Partially parsed test_dcfc_act_365_a_invalid_date_range. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 6
    var_3 = 30
    var_4 = 12
    var_5 = 31
    var_6 = '0.5'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 6
    var_3 = 30
    var_4 = 12
    var_5 = 31
    var_6 = '0.5013698630137'
    var_7 = 14

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1.0'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1.0027397260274'
    var_5 = 14

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 15
    var_3 = 31
    var_4 = '0.0410958904109589'
    var_5 = 14

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '0.0'

def test_case_0():
    var_0 = 2021
    var_1 = 12
    var_2 = 31
    var_3 = 1
    var_4 = 6
    var_5 = 30
    var_6 = '0.0'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_34_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = 2008
    var_4 = 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_construct_date_with_valid_date. Retrieved 4/6 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_register_altname_conflict. Retrieved 8/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 1
    var_5 = 'Test2'
    var_6 = {var_1}
    var_7 = set()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_next_payment_date_annual_no_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_annual_with_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_semiannual_no_eom. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_semiannual_with_eom. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_quarterly_no_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_quarterly_with_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_monthly_no_eom. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_monthly_with_eom. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_invalid_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_february_eom. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = None
    var_3 = 2015

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 15
    var_3 = 2015

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2
    var_3 = None
    var_4 = 7

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2
    var_3 = 15
    var_4 = 7

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 4
    var_3 = None

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 4
    var_3 = 15

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 12
    var_3 = None
    var_4 = 2

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 12
    var_3 = 15
    var_4 = 2

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = 2015

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = 29
    var_4 = 2015



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_last_payment_date_predicate. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dcfc_30_360_german_asof_day_31_or_last_day_of_feb. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = 3
    var_6 = '30'
    var_7 = '360'
    var_8 = '28'



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    pass



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_register_successful. Retrieved 8/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 14/23 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 14/23 statements.


import pypara.dcc as module_0
import pypara.currencies as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = module_1.Currency(var_4)
    var_6 = {var_5}
    var_7 = '0.5'

import pypara.dcc as module_0
import pypara.currencies as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = module_1.Currency(var_4)
    var_6 = {var_5}
    var_7 = '0.5'
    var_8 = 'TestAlt2'
    var_9 = {var_8}
    var_10 = 'EUR'
    var_11 = module_1.Currency(var_10)
    var_12 = {var_11}
    var_13 = '0.6'

import pypara.dcc as module_0
import pypara.currencies as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = module_1.Currency(var_4)
    var_6 = {var_5}
    var_7 = '0.5'
    var_8 = 'Test2'
    var_9 = {var_2}
    var_10 = 'EUR'
    var_11 = module_1.Currency(var_10)
    var_12 = {var_11}
    var_13 = '0.6'



# Parsed testcases at query #2
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_register_successful. Retrieved 8/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 14/23 statements.
# Partially parsed test_register_duplicate_alternative_name. Retrieved 14/23 statements.


import pypara.dcc as module_0
import pypara.currencies as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = module_1.Currency(var_4)
    var_6 = {var_5}
    var_7 = 0.5

import pypara.dcc as module_0
import pypara.currencies as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = module_1.Currency(var_4)
    var_6 = {var_5}
    var_7 = 0.5
    var_8 = 'TestAlt2'
    var_9 = {var_8}
    var_10 = 'EUR'
    var_11 = module_1.Currency(var_10)
    var_12 = {var_11}
    var_13 = 0.6

import pypara.dcc as module_0
import pypara.currencies as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = module_1.Currency(var_4)
    var_6 = {var_5}
    var_7 = 0.5
    var_8 = 'Test2'
    var_9 = {var_2}
    var_10 = 'EUR'
    var_11 = module_1.Currency(var_10)
    var_12 = {var_11}
    var_13 = 0.6



# Parsed testcases at query #4
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33333333333333'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_empty_range. Retrieved 2/6 statements.
# Partially parsed test_single_day_range. Retrieved 3/9 statements.
# Partially parsed test_multi_day_range. Retrieved 6/15 statements.
# Partially parsed test_year_boundary. Retrieved 8/17 statements.
# Partially parsed test_leap_year. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = 2
    var_4 = 3
    var_5 = 4

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 30
    var_3 = 2023
    var_4 = 1
    var_5 = 3
    var_6 = 31
    var_7 = 2

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 27
    var_3 = 3
    var_4 = 28
    var_5 = 29
    var_6 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dcfc_act_act_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_act_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16942884946478'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.17216108990194'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08243131970956'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32625945055768'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_existing_dcc_by_main_name. Retrieved 7/9 statements.
# Partially parsed test_find_existing_dcc_by_alternative_name. Retrieved 7/9 statements.
# Partially parsed test_find_existing_dcc_case_insensitive. Retrieved 8/10 statements.
# Partially parsed test_find_existing_dcc_with_whitespace. Retrieved 8/10 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = 1.0
    var_5 = lambda s, e, r: var_4
    var_6 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = 1.0
    var_5 = lambda s, e, r: var_4
    var_6 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = 1.0
    var_5 = lambda s, e, r: var_4
    var_6 = 'act/act'
    var_7 = var_0.find(var_6)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = 1.0
    var_5 = lambda s, e, r: var_4
    var_6 = '  Act/Act  '
    var_7 = var_0.find(var_6)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16942884946478'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.17216108990194'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08243131970956'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32625945055768'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_last_day_of_february_in_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_february_in_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_april. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_december. Retrieved 3/5 statements.
# Partially parsed test_not_last_day_of_month. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2021
    var_1 = 4
    var_2 = 30

def test_case_0():
    var_0 = 2021
    var_1 = 12
    var_2 = 31

def test_case_0():
    var_0 = 2021
    var_1 = 5
    var_2 = 15



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_30_e_360_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_360_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33055555555556'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_date_range_basic. Retrieved 6/15 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 3/9 statements.
# Partially parsed test_get_date_range_empty. Retrieved 3/7 statements.
# Partially parsed test_get_date_range_year_boundary. Retrieved 8/17 statements.
# Partially parsed test_get_date_range_month_boundary. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = 2
    var_4 = 3
    var_5 = 4

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = []

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 30
    var_3 = 2023
    var_4 = 1
    var_5 = 3
    var_6 = 31
    var_7 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = 2
    var_4 = 3
    var_5 = 31



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dcfc_30_360_us_basic_cases. Retrieved 18/42 statements.
# Partially parsed test_dcfc_30_360_us_edge_cases. Retrieved 10/29 statements.
# Partially parsed test_dcfc_30_360_us_invalid_date_order. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'
    var_7 = 29
    var_8 = '0.16944444444444'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08333333333333'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '0.0'
    var_6 = '30'
    var_7 = '360'
    var_8 = 29
    var_9 = 3

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = 3
    var_4 = '0'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_26. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_last_payment_date_annual_same_year. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_annual_previous_year. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_semi_annual_july. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_august. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_april. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_june_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_december_start_january_end. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_december_start_december_end. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 2
    var_6 = 7

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 8
    var_4 = 31
    var_5 = 2
    var_6 = 7

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 4
    var_4 = 30
    var_5 = 2

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = 2015
    var_4 = 4
    var_5 = 30

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = 2015
    var_3 = 10
    var_4 = 6
    var_5 = 4

def test_case_0():
    var_0 = 2014
    var_1 = 12
    var_2 = 9
    var_3 = 2015
    var_4 = 4
    var_5 = 1

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = 2016
    var_4 = 1
    var_5 = 6
    var_6 = 2
    var_7 = 2015

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = 2015
    var_4 = 31
    var_5 = 2



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_coupon_basic_case. Retrieved 15/24 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/25 statements.
# Partially parsed test_coupon_asof_before_start. Retrieved 15/24 statements.
# Partially parsed test_coupon_asof_after_end. Retrieved 16/25 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 14/23 statements.
# Partially parsed test_coupon_zero_principal. Retrieved 15/24 statements.
# Partially parsed test_coupon_high_frequency. Retrieved 15/24 statements.
# Partially parsed test_coupon_low_frequency. Retrieved 15/24 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.05'
    var_7 = 2020
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 2
    var_13 = None
    var_14 = '25.00'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.05'
    var_7 = 2020
    var_8 = 1
    var_9 = 15
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = 2
    var_14 = 15
    var_15 = '25.00'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.05'
    var_7 = 2020
    var_8 = 6
    var_9 = 1
    var_10 = 12
    var_11 = 31
    var_12 = 2
    var_13 = None
    var_14 = '0.00'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.05'
    var_7 = 2020
    var_8 = 1
    var_9 = 2021
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = 2
    var_14 = None
    var_15 = '0.00'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.00'
    var_7 = 2020
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 2
    var_13 = None

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 0
    var_5 = 'USD'
    var_6 = '0.05'
    var_7 = 2020
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 2
    var_13 = None
    var_14 = '0.00'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.05'
    var_7 = 2020
    var_8 = 1
    var_9 = 2
    var_10 = 12
    var_11 = 31
    var_12 = 12
    var_13 = None
    var_14 = '4.17'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.05'
    var_7 = 2020
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 1
    var_13 = None
    var_14 = '25.00'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_calculate_daily_fraction_with_valid_dates. Retrieved 10/15 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_before_start. Retrieved 12/17 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_after_end. Retrieved 10/15 statements.
# Partially parsed test_calculate_daily_fraction_with_equal_dates. Retrieved 8/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = module_0.DCC()
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = var_6 / var_7

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = module_0.DCC()
    var_5 = 2023
    var_6 = 1
    var_7 = 2022
    var_8 = 12
    var_9 = 31
    var_10 = 3
    var_11 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = module_0.DCC()
    var_5 = 2023
    var_6 = 1
    var_7 = 4
    var_8 = 3
    var_9 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = module_0.DCC()
    var_5 = 2023
    var_6 = 1
    var_7 = 0



# Parsed testcases at query #18
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_register_raises_typeerror_for_duplicate_altname. Retrieved 8/15 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'ALT1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 'Test2'
    var_6 = {var_2}
    var_7 = set()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_existing_main_name. Retrieved 5/7 statements.
# Partially parsed test_find_existing_alternative_name. Retrieved 5/7 statements.
# Partially parsed test_find_case_insensitive. Retrieved 8/10 statements.
# Partially parsed test_find_whitespace_insensitive. Retrieved 8/10 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = [var_2]
    var_4 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistingDCC'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = [var_2]
    var_4 = 'testdcc'
    var_5 = var_0.find(var_4)
    var_6 = 'ALTTESTDCC'
    var_7 = var_0.find(var_6)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = [var_2]
    var_4 = '  TestDCC  '
    var_5 = var_0.find(var_4)
    var_6 = '  AltTestDCC  '
    var_7 = var_0.find(var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dcfc_act_act_icma. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example_4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33333333333333'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dcfc_30_e_360_start_day_31. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = '1'
    var_5 = 360



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 1
    var_6 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_d1_not_31. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16666666666667'



# Parsed testcases at query #26
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_has_leap_day_with_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_without_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_start. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_end. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_multiple_leap_years_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_no_leap_years_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_single_day_range_on_leap_day. Retrieved 3/6 statements.
# Partially parsed test_has_leap_day_with_single_day_range_not_on_leap_day. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = 29

def test_case_0():
    var_0 = 2016
    var_1 = 1
    var_2 = 2024
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 2023
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 28



# Parsed testcases at query #28
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dcfc_30_360_us_d2_not_31. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_coupon_standard_case. Retrieved 15/23 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/24 statements.
# Partially parsed test_coupon_zero_principal. Retrieved 14/22 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 15/23 statements.
# Partially parsed test_coupon_asof_before_start. Retrieved 15/23 statements.
# Partially parsed test_coupon_asof_after_end. Retrieved 15/23 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2020
    var_9 = 1
    var_10 = 6
    var_11 = 2021
    var_12 = 2
    var_13 = None
    var_14 = 25

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2020
    var_9 = 1
    var_10 = 15
    var_11 = 6
    var_12 = 2021
    var_13 = 2
    var_14 = 15
    var_15 = 25

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 0
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2020
    var_9 = 1
    var_10 = 6
    var_11 = 2021
    var_12 = 2
    var_13 = None

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0'
    var_8 = 2020
    var_9 = 1
    var_10 = 6
    var_11 = 2021
    var_12 = 2
    var_13 = None
    var_14 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2020
    var_9 = 6
    var_10 = 1
    var_11 = 2021
    var_12 = 2
    var_13 = None
    var_14 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2020
    var_9 = 1
    var_10 = 2021
    var_11 = 6
    var_12 = 2
    var_13 = None
    var_14 = 0



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_last_day_of_month. Retrieved 9/19 statements.
# Partially parsed test_not_last_day_of_month. Retrieved 9/19 statements.
# Partially parsed test_leap_year_february. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = 3
    var_6 = 4
    var_7 = 30
    var_8 = 5

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = 2
    var_4 = 27
    var_5 = 3
    var_6 = 4
    var_7 = 29
    var_8 = 5

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = 28



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_interest_with_valid_inputs. Retrieved 14/24 statements.
# Partially parsed test_interest_with_end_none. Retrieved 12/21 statements.
# Partially parsed test_interest_with_invalid_dates. Retrieved 14/24 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = '0.5'
    var_6 = 1000
    var_7 = '0.1'
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = 50

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = '0.5'
    var_6 = 1000
    var_7 = '0.1'
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 50

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = '0.5'
    var_6 = 1000
    var_7 = '0.1'
    var_8 = 2023
    var_9 = 12
    var_10 = 31
    var_11 = 6
    var_12 = 1
    var_13 = 0



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 7/16 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 0.5
    var_5 = set()
    var_6 = set()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_eom_not_false_or_none. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 1
    var_5 = 15



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dcfc_act_365_a_without_leap_day. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_365_a_with_leap_day. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_365_a_cross_year. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16986301369863'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.17213114754098'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08196721311475'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32513661202186'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dcfc_act_365_a_without_leap_day. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_365_a_with_leap_day. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_365_a_crossing_year_boundary. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16986301369863'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.17213114754098'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08196721311475'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32513661202186'



# Parsed testcases at query #37
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 30
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = -2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 0
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #38
#--------------------------






# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_predicate. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 7/19 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 14/34 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_start_after_asof. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 9
    var_6 = 4

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 2
    var_6 = 7
    var_7 = 8
    var_8 = 4
    var_9 = 30
    var_10 = 2012
    var_11 = 15
    var_12 = 2016
    var_13 = 6

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = 2015
    var_3 = 10
    var_4 = 6
    var_5 = 4

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = 2015
    var_4 = 4
    var_5 = 30



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_30_360_isda_ex1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_ex2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_ex3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_ex4. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.16944444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.33333333333333'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dcfc_act_act_icma. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_august. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_april. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_june_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_december_start_january_end. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_december_start_december_end. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 2
    var_6 = 7

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 8
    var_4 = 31
    var_5 = 2
    var_6 = 7

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 4
    var_4 = 30
    var_5 = 2

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = 2015
    var_4 = 4
    var_5 = 30

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = 2015
    var_3 = 10
    var_4 = 6
    var_5 = 4

def test_case_0():
    var_0 = 2014
    var_1 = 12
    var_2 = 9
    var_3 = 2015
    var_4 = 4
    var_5 = 1

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = 2016
    var_4 = 1
    var_5 = 6
    var_6 = 2
    var_7 = 2015

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = 2015
    var_4 = 31
    var_5 = 2



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_register_successful. Retrieved 6/12 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = {}
    var_3 = {}
    var_4 = 0.5
    var_5 = {}
    var_6 = {}

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = 'AnotherTest'
    var_7 = {var_2}
    var_8 = {}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_calculate_daily_fraction_with_valid_dates. Retrieved 9/15 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_equal_to_start. Retrieved 9/15 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_equal_to_end. Retrieved 8/14 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_before_start. Retrieved 9/15 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_after_end. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = var_5 / var_6

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 2023
    var_5 = 1
    var_6 = 3
    var_7 = 2
    var_8 = var_5 / var_7

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 2023
    var_5 = 1
    var_6 = 3
    var_7 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = lambda s, a, e, f: Decimal((a - s).days / (e - s).days)
    var_4 = 2023
    var_5 = 1
    var_6 = 4
    var_7 = 3
    var_8 = 0



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_register_successful. Retrieved 7/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 10/19 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TEST1'
    var_3 = 'TEST2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0.5

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 0.5
    var_5 = set()
    var_6 = set()
    var_7 = 0.6

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'TEST'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.5
    var_6 = 'Test2'
    var_7 = {var_2}
    var_8 = set()
    var_9 = 0.6



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_calculate_daily_fraction_predicate_false. Retrieved 8/15 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = 1
    var_4 = 2023
    var_5 = 2
    var_6 = 3
    var_7 = module_0.timedelta()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dcfc_act_act_basic_calculation. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_cross_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_invalid_date_range. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16942884946478'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.17216108990194'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08243131970956'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32625945055768'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2022
    var_3 = 12
    var_4 = 31
    var_5 = '0'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_last_day_of_february_in_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_february_in_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_april. Retrieved 3/5 statements.
# Partially parsed test_non_last_day_of_month. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_december. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2019
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_calculate_fraction_with_valid_dates. Retrieved 9/17 statements.
# Partially parsed test_calculate_fraction_with_invalid_dates. Retrieved 9/16 statements.
# Partially parsed test_calculate_fraction_with_freq. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 2023
    var_5 = 1
    var_6 = 6
    var_7 = 12
    var_8 = 31

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 2023
    var_5 = 12
    var_6 = 31
    var_7 = 6
    var_8 = 1

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '4'
    var_4 = '0.25'
    var_5 = '0.5'
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 12
    var_10 = 31



# Parsed testcases at query #11
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = '1.08333333333333'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_register_raises_typeerror_for_duplicate_main_name. Retrieved 7/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = set()
    var_6 = set()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_interest_basic_calculation. Retrieved 13/23 statements.
# Partially parsed test_interest_without_end_date. Retrieved 12/21 statements.
# Partially parsed test_interest_zero_fraction. Retrieved 12/22 statements.
# Partially parsed test_interest_with_frequency. Retrieved 15/27 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.10'
    var_7 = 2023
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 50

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = 2000
    var_5 = 'EUR'
    var_6 = '0.05'
    var_7 = 2023
    var_8 = 1
    var_9 = 3
    var_10 = 31
    var_11 = 25

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0'
    var_4 = 5000
    var_5 = 'GBP'
    var_6 = '0.08'
    var_7 = 2023
    var_8 = 1
    var_9 = 12
    var_10 = 31
    var_11 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.3'
    var_4 = '0.5'
    var_5 = 10000
    var_6 = 'JPY'
    var_7 = '0.02'
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = '2'
    var_14 = 60



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_DCCRegistryMachinery_constructor_initializes_buffers. Retrieved 11/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '_buffer_main'
    var_2 = hasattr(var_0, var_1)
    var_3 = '_buffer_altn'
    var_4 = hasattr(var_0, var_3)
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn
    var_7 = var_0._buffer_main
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = var_0._buffer_altn
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #16
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_last_day_of_january. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_february_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_february_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_april. Retrieved 3/5 statements.
# Partially parsed test_non_last_day. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 3
    var_2 = 15



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_existing_dcc. Retrieved 3/5 statements.
# Partially parsed test_find_existing_dcc_with_altname. Retrieved 5/7 statements.
# Partially parsed test_find_with_stripped_uppercase. Retrieved 4/6 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = ' act/act '
    var_3 = var_0.find(var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dates_order_check. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2022
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_calculate_daily_fraction_when_asof_minus_1_is_not_less_than_start. Retrieved 9/16 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '1.0'
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = module_0.timedelta()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_coupon_basic_case. Retrieved 14/24 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/26 statements.
# Partially parsed test_coupon_zero_principal. Retrieved 13/23 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 14/24 statements.
# Partially parsed test_coupon_full_period. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.1'
    var_7 = 2020
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 2
    var_13 = 50

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.1'
    var_7 = 2020
    var_8 = 1
    var_9 = 15
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = 2
    var_14 = 15
    var_15 = 50

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 0
    var_5 = 'USD'
    var_6 = '0.1'
    var_7 = 2020
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 2

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0'
    var_7 = 2020
    var_8 = 1
    var_9 = 6
    var_10 = 12
    var_11 = 31
    var_12 = 2
    var_13 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '1.0'
    var_4 = 1000
    var_5 = 'USD'
    var_6 = '0.1'
    var_7 = 2020
    var_8 = 1
    var_9 = 12
    var_10 = 31
    var_11 = 1
    var_12 = 100



# Parsed testcases at query #22
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_calculate_fraction_returns_zero_when_dates_are_invalid. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 1
    var_4 = 2023
    var_5 = 2
    var_6 = 3



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dcfc_nl_365_basic_cases. Retrieved 17/41 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16986301369863'
    var_15 = '1.08219178082192'
    var_16 = '1.32602739726027'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor_initializes_buffers. Retrieved 7/9 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn
    var_3 = var_0._buffer_main
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_0._buffer_altn
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dcc_registry_machinery_initialization. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_last_payment_date_predicate. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 8/24 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 11/27 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_with_eom. Retrieved 12/24 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 8
    var_6 = 4
    var_7 = 30

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 2
    var_6 = 7
    var_7 = 8
    var_8 = 4
    var_9 = 30
    var_10 = 6

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = 2015
    var_3 = 10
    var_4 = 6
    var_5 = 4

def test_case_0():
    var_0 = 2014
    var_1 = 12
    var_2 = 9
    var_3 = 2015
    var_4 = 4
    var_5 = 1
    var_6 = 2012
    var_7 = 15
    var_8 = 2016
    var_9 = 6
    var_10 = 2
    var_11 = 31



# Parsed testcases at query #29
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_last_day_of_february_in_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_february_in_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_april. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_december. Retrieved 3/5 statements.
# Partially parsed test_not_last_day_of_month. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2021
    var_1 = 4
    var_2 = 30

def test_case_0():
    var_0 = 2021
    var_1 = 12
    var_2 = 31

def test_case_0():
    var_0 = 2021
    var_1 = 5
    var_2 = 15



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dcfc_30_360_isda_predicate_false. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 2
    var_4 = 3
    var_5 = '0.1666666666666666666666666667'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dcfc_act_act_icma_returns_correct_fraction. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_init_initializes_buffers. Retrieved 7/9 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '_buffer_main'
    var_2 = hasattr(var_0, var_1)
    var_3 = '_buffer_altn'
    var_4 = hasattr(var_0, var_3)
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_init_initializes_buffers. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dcfc_act_act_basic. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_cross_year. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16942884946478'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 14
    var_7 = '0.17216108990194'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = 14
    var_7 = '1.08243131970956'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = 14
    var_7 = '1.32625945055768'



# Parsed testcases at query #36
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_interest_uses_calculate_fraction. Retrieved 14/28 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = '100'
    var_5 = 'USD'
    var_6 = module_0.Currency(var_5)
    var_7 = '0.1'
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = '1'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 1
    var_6 = 1



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_register_successful. Retrieved 6/10 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 10/19 statements.
# Partially parsed test_register_duplicate_alternative_name. Retrieved 10/19 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = 'AnotherAlt'
    var_7 = {var_6}
    var_8 = {}
    var_9 = 0.6

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = 'Another'
    var_7 = {var_2}
    var_8 = {}
    var_9 = 0.6



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_calculate_daily_fraction_asof_minus_1_less_than_start. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 1
    var_4 = 2023
    var_5 = 2
    var_6 = 3



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_last_day_of_month. Retrieved 16/40 statements.
# Partially parsed test_not_last_day_of_month. Retrieved 16/40 statements.
# Partially parsed test_leap_year_february. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = 3
    var_6 = 4
    var_7 = 30
    var_8 = 5
    var_9 = 6
    var_10 = 7
    var_11 = 8
    var_12 = 9
    var_13 = 10
    var_14 = 11
    var_15 = 12

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = 2
    var_4 = 27
    var_5 = 3
    var_6 = 4
    var_7 = 29
    var_8 = 5
    var_9 = 6
    var_10 = 7
    var_11 = 8
    var_12 = 9
    var_13 = 10
    var_14 = 11
    var_15 = 12

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = 28



