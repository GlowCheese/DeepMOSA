####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_next_payment_date_standard_annual. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_with_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_quarterly. Retrieved 6/9 statements.
# Partially parsed test_next_payment_date_monthly_with_invalid_eom. Retrieved 7/10 statements.
# Partially parsed test_next_payment_date_decimal_frequency. Retrieved 5/9 statements.
# Partially parsed test_next_payment_date_leap_year_transition. Retrieved 7/10 statements.


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
    var_0 = 2023
    var_1 = 3
    var_2 = 10
    var_3 = 4
    var_4 = None
    var_5 = 6

def test_case_0():
    var_0 = 2023
    var_1 = 3
    var_2 = 15
    var_3 = 12
    var_4 = 31
    var_5 = 4
    var_6 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '2.0'
    var_3 = None
    var_4 = 7

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = 1
    var_4 = None
    var_5 = 2025
    var_6 = 28



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_constructor_registry_property_is_empty_list.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #3
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #4
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = lambda x, y, z, f: var_5
    var_7 = module_0.DCC()
    var_8 = var_0.register(var_7)
    var_9 = var_0.find(var_1)
    var_10 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = lambda x, y, z, f: var_5
    var_7 = module_0.DCC()
    var_8 = 'Other'
    var_9 = {var_8}
    var_10 = set()
    var_11 = lambda x, y, z, f: var_5
    var_12 = module_0.DCC()
    var_13 = var_0.register(var_7)
    var_14 = var_0.register(var_12)
    var_15 = 'TypeError not raised'
    var_16 = AssertionError(var_15)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = lambda x, y, z, f: var_5
    var_7 = module_0.DCC()
    var_8 = 'NEW'
    var_9 = {var_2}
    var_10 = set()
    var_11 = lambda x, y, z, f: var_5
    var_12 = module_0.DCC()
    var_13 = var_0.register(var_7)
    var_14 = var_0.register(var_12)
    var_15 = 'TypeError not raised'
    var_16 = AssertionError(var_15)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = lambda x, y, z, f: var_5
    var_7 = module_0.DCC()
    var_8 = 'Other'
    var_9 = {var_8}
    var_10 = set()
    var_11 = lambda x, y, z, f: var_5
    var_12 = module_0.DCC()
    var_13 = var_0.register(var_7)
    var_14 = var_0.register(var_12)
    var_15 = 'TypeError not raised'
    var_16 = AssertionError(var_15)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_has_leap_day_contains_leap_day. Retrieved 3/9 statements.
# Partially parsed test_has_leap_day_boundary_leap_day. Retrieved 3/9 statements.
# Partially parsed test_has_leap_day_no_leap_day_in_range. Retrieved 4/10 statements.
# Partially parsed test_has_leap_day_range_around_leap_year_without_feb_29. Retrieved 4/10 statements.
# Partially parsed test_has_leap_day_long_range_with_leap_year. Retrieved 3/9 statements.
# Partially parsed test_has_leap_day_single_day_non_leap. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 3

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = 2021

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 2023

def test_case_0():
    var_0 = 2021
    var_1 = 6
    var_2 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dcfc_30_360_isda_calculation_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_isda_calculation_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_isda_calculation_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_isda_calculation_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31_adjustment. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_360_isda_both_days_31_adjustment. Retrieved 5/12 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = '1'
    var_5 = '360'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = 31
    var_4 = '0'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_calculate_fraction_valid_range. Retrieved 14/37 statements.
# Partially parsed test_calculate_fraction_invalid_range_asof_too_early. Retrieved 14/37 statements.
# Partially parsed test_calculate_fraction_invalid_range_asof_too_late. Retrieved 13/36 statements.
# Partially parsed test_calculate_fraction_with_frequency. Retrieved 13/37 statements.


def test_case_0():
    var_0 = 'DCC'
    var_1 = 'name'
    var_2 = 'altnames'
    var_3 = 'currencies'
    var_4 = 'calculate_fraction_method'
    var_5 = 'Actual/360'
    var_6 = 'A/360'
    var_7 = {var_6}
    var_8 = set()
    var_9 = 2023
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = '0.5'

def test_case_0():
    var_0 = 'DCC'
    var_1 = 'name'
    var_2 = 'altnames'
    var_3 = 'currencies'
    var_4 = 'calculate_fraction_method'
    var_5 = '0'
    var_6 = 'Actual/360'
    var_7 = set()
    var_8 = set()
    var_9 = 2023
    var_10 = 1
    var_11 = 5
    var_12 = 2
    var_13 = 10

def test_case_0():
    var_0 = 'DCC'
    var_1 = 'name'
    var_2 = 'altnames'
    var_3 = 'currencies'
    var_4 = 'calculate_fraction_method'
    var_5 = '0'
    var_6 = 'Actual/360'
    var_7 = set()
    var_8 = set()
    var_9 = 2023
    var_10 = 1
    var_11 = 15
    var_12 = 10

def test_case_0():
    var_0 = 'DCC'
    var_1 = 'name'
    var_2 = 'altnames'
    var_3 = 'currencies'
    var_4 = 'calculate_fraction_method'
    var_5 = 'Actual/360'
    var_6 = set()
    var_7 = set()
    var_8 = 2020
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = '2.0'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_register_raises_type_error_on_duplicate_altname. Retrieved 12/20 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.1'
    var_2 = 'ACT/360'
    var_3 = 'ACT360'
    var_4 = 'ACT/360_ALT'
    var_5 = {var_3, var_4}
    var_6 = set()
    var_7 = '30/360'
    var_8 = {var_3}
    var_9 = set()
    var_10 = 'TypeError was not raised for duplicate altname'
    var_11 = AssertionError(var_10)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_returns_dcc_by_main_name. Retrieved 5/11 statements.
# Partially parsed test_find_returns_dcc_by_altname. Retrieved 5/11 statements.
# Partially parsed test_find_handles_case_insensitivity_and_stripping. Retrieved 7/13 statements.
# Partially parsed test_find_prefers_exact_match_over_normalized. Retrieved 6/14 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = 'Act/360'
    var_2 = [var_1]
    var_3 = module_0.DCCRegistryMachinery()
    var_4 = var_3.find(var_0)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = 'Actual/360'
    var_2 = [var_0, var_1]
    var_3 = module_0.DCCRegistryMachinery()
    var_4 = var_3.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = [var_0]
    var_2 = module_0.DCCRegistryMachinery()
    var_3 = '  act/360  '
    var_4 = var_2.find(var_3)
    var_5 = 'act/360'
    var_6 = var_2.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'FIRST'
    var_1 = []
    var_2 = 'SECOND'
    var_3 = []
    var_4 = module_0.DCCRegistryMachinery()
    var_5 = var_4.find(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_last_day_of_month_true_january. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false_january. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_true_leap_year_february. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false_leap_year_february. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_true_non_leap_year_february. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_true_april_30. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false_april_29. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 28

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
    var_1 = 4
    var_2 = 29



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_act_act_icma_calculation. Retrieved 9/21 statements.
# Partially parsed test_dcfc_act_act_icma_default_freq. Retrieved 9/20 statements.
# Partially parsed test_dcfc_act_act_icma_invalid_date_range. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_icma_same_dates. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '2'
    var_7 = '191'
    var_8 = '366'

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '191'
    var_7 = '366'
    var_8 = '1'

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 2021
    var_4 = 2020
    var_5 = '731'
    var_6 = '366'

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 2
    var_3 = '0'
    var_4 = '1'



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_constructor_registry_property_is_empty.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_interest_valid_dates. Retrieved 12/53 statements.
# Partially parsed test_interest_invalid_dates_returns_zero. Retrieved 12/53 statements.
# Partially parsed test_interest_end_date_is_asof_when_none. Retrieved 10/49 statements.


def test_case_0():
    var_0 = '0.5'
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = '1000'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 12
    var_10 = 31
    var_11 = '25.0'

def test_case_0():
    var_0 = '0.5'
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = '1000'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 6
    var_8 = 1
    var_9 = 12
    var_10 = 31
    var_11 = '0'

def test_case_0():
    var_0 = '1.0'
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = '100'
    var_5 = '0.1'
    var_6 = 2023
    var_7 = 1
    var_8 = None
    var_9 = '10.0'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_30_e_360_standard_case. Retrieved 6/11 statements.
# Partially parsed test_dcfc_30_e_360_leap_year_case. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_e_360_end_of_month_case. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_e_360_long_period_case. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_e_360_day_31_adjustment_start. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_360_day_31_adjustment_asof. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.1666666666666666666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = '0.1694444444444444444444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = '1.0833333333333333333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = '1.3305555555555555555555555556'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '28'
    var_6 = '360'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 31
    var_4 = '89'
    var_5 = '360'



# Parsed testcases at query #2
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'ACT/360'
    var_3 = 'Actual/360'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0.1
    var_7 = lambda s, a, e, f: var_6
    var_8 = module_0.DCC()
    var_9 = var_0.register(var_8)
    var_10 = var_0.find(var_1)
    var_11 = var_0.find(var_2)
    var_12 = var_0.find(var_3)
    var_13 = var_0.registry
    var_14 = len(var_13)
    assert var_14 == 1

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = set()
    var_3 = set()
    var_4 = 0.1
    var_5 = lambda s, a, e, f: var_4
    var_6 = module_0.DCC()
    var_7 = set()
    var_8 = set()
    var_9 = 0.2
    var_10 = lambda s, a, e, f: var_9
    var_11 = module_0.DCC()
    var_12 = var_0.register(var_6)
    var_13 = var_0.register(var_11)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'ACT/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.1
    var_6 = lambda s, a, e, f: var_5
    var_7 = module_0.DCC()
    var_8 = 'Other'
    var_9 = {var_2}
    var_10 = set()
    var_11 = 0.2
    var_12 = lambda s, a, e, f: var_11
    var_13 = module_0.DCC()
    var_14 = var_0.register(var_7)
    var_15 = var_0.register(var_13)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'Duplicate'
    var_3 = {var_1, var_2}
    var_4 = set()
    var_5 = 0.1
    var_6 = lambda s, a, e, f: var_5
    var_7 = module_0.DCC()
    var_8 = var_0.register(var_7)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_construct_date_valid. Retrieved 4/7 statements.
# Partially parsed test_construct_date_auto_decrement_day. Retrieved 5/8 statements.
# Partially parsed test_construct_date_auto_decrement_leap_year. Retrieved 5/8 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0._construct_date(var_0, var_1, var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = -1
    var_2 = 1
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = -5
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 30
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = 28

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 30
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = 29

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 1
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_case1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_case2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_case3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_case4. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_boundary_start_31. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_boundary_asof_31. Retrieved 5/12 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '28'
    var_6 = '360'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = '30'
    var_4 = '360'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_constructor_registry_property_is_empty_list.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dcfc_30_360_isda_standard_case. Retrieved 6/11 statements.
# Partially parsed test_dcfc_30_360_isda_leap_year_case. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_isda_month_end_case. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_isda_long_period_case. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_isda_day_31_adjustment. Retrieved 6/11 statements.
# Partially parsed test_dcfc_30_360_isda_both_days_31_adjustment. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.1666666666666666666666666667'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = '0.1694444444444444444444444444'

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = '1.0833333333333333333333333333'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 2009
    var_4 = 5
    var_5 = 31
    var_6 = '1.3333333333333333333333333333'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '27/360'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 3
    var_4 = '60/360'



# Parsed testcases at query #7
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = 'ACT360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = lambda x, y, z, f: var_5
    var_7 = module_0.DCC()
    var_8 = 'OTHER'
    var_9 = {var_8}
    var_10 = set()
    var_11 = lambda x, y, z, f: var_5
    var_12 = module_0.DCC()
    var_13 = var_0.register(var_7)
    var_14 = var_0.register(var_12)
    var_15 = 'TypeError was not raised'
    var_16 = AssertionError(var_15)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_interest_valid_dates. Retrieved 11/58 statements.
# Partially parsed test_interest_invalid_dates_returns_zero. Retrieved 11/54 statements.
# Partially parsed test_interest_default_end_date. Retrieved 9/54 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.05'
    var_5 = 2023
    var_6 = 1
    var_7 = 6
    var_8 = 12
    var_9 = 31
    var_10 = '0.5'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.05'
    var_5 = 2023
    var_6 = 6
    var_7 = 1
    var_8 = 12
    var_9 = 31
    var_10 = '0'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.10'
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = '1.0'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dcfc_30_360_us_standard_case. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_leap_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_month_end_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_long_period_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_zero_days. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '0'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_has_leap_day_true_within_range. Retrieved 3/6 statements.
# Partially parsed test_has_leap_day_true_boundary_start. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_true_boundary_end. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_false_no_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_false_leap_year_exists_but_february_not_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_false_leap_year_exists_but_after_range. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 3

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = 3
    var_4 = 1

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = 29

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 2024



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_30_360_german_standard_case. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_german_leap_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_month_end_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_long_period_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_day_31_adjustment. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_360_german_february_leap_adjustment. Retrieved 7/14 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '28'
    var_6 = '360'

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = 3
    var_4 = 1
    var_5 = '1'
    var_6 = '360'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcfc_30_360_german_predicate_true_via_day_31. Retrieved 5/11 statements.
# Partially parsed test_dcfc_30_360_german_predicate_true_via_february_last_day. Retrieved 5/10 statements.
# Partially parsed test_dcfc_30_360_german_predicate_true_via_february_last_day_non_leap. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = '0'

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = 3
    var_4 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_constructor_property_registry_is_empty.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_last_payment_date_basic_annual. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_annual_same_year. Retrieved 4/10 statements.


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
    pass



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcfc_nl_365_standard_calculation. Retrieved 7/15 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_extended_leap_year_range. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_same_day. Retrieved 3/10 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '0'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_returns_correct_dcc_when_exact_name_exists. Retrieved 3/9 statements.
# Partially parsed test_find_returns_correct_dcc_when_altname_exists. Retrieved 5/11 statements.
# Partially parsed test_find_returns_correct_dcc_with_case_insensitivity_and_stripping. Retrieved 4/10 statements.
# Partially parsed test_find_returns_none_when_not_found. Retrieved 4/10 statements.
# Partially parsed test_find_returns_correct_dcc_for_alternative_name_with_padding. Retrieved 6/12 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'ACTUAL/360'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = '  act/360  '
    var_3 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'NonExistent'
    var_3 = var_0.find(var_2)
    assert var_3 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360 US'
    var_2 = '30/360 US ISDA'
    var_3 = [var_2]
    var_4 = '  30/360 US ISDA  '
    var_5 = var_0.find(var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_date_range_basic. Retrieved 2/19 statements.
# Partially parsed test_get_date_range_empty. Retrieved 1/14 statements.
# Partially parsed test_get_date_range_negative_delta. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 4

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 10
    var_1 = 5



