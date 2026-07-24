####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_register_successful. Retrieved 6/10 statements.
# Partially parsed test_register_with_altnames. Retrieved 8/12 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 10/19 statements.
# Partially parsed test_register_altname_conflicts_with_main_name. Retrieved 9/18 statements.
# Partially parsed test_register_main_name_conflicts_with_existing_altname. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test successful registration of a DCC.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC'
    var_3 = set()
    var_4 = set()
    var_5 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test registration of a DCC with alternative names.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC'
    var_3 = 'Alt1'
    var_4 = 'Alt2'
    var_5 = {var_3, var_4}
    var_6 = set()
    var_7 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test that registering duplicate main name raises TypeError.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC'
    var_3 = set()
    var_4 = set()
    var_5 = 0
    var_6 = set()
    var_7 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test that registering duplicate alternative name raises TypeError.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC1'
    var_3 = 'Alt1'
    var_4 = {var_3}
    var_5 = set()
    var_6 = 0
    var_7 = 'Test/DCC2'
    var_8 = {var_3}
    var_9 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test that registering altname that conflicts with existing main name raises TypeError.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC1'
    var_3 = set()
    var_4 = set()
    var_5 = 0
    var_6 = 'Test/DCC2'
    var_7 = {var_2}
    var_8 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test that registering main name that conflicts with existing altname raises TypeError.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC1'
    var_3 = 'Alt1'
    var_4 = {var_3}
    var_5 = set()
    var_6 = 0
    var_7 = set()
    var_8 = set()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_day. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_day. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_full_year. Retrieved 4/12 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 16
    var_4 = '1'
    var_5 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = '2'
    var_5 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 31
    var_4 = '16'
    var_5 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 2009
    var_3 = '360'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_calculate_fraction_valid_dates. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_asof_equals_start. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_equals_end. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_before_start. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_asof_after_end. Retrieved 9/19 statements.
# Partially parsed test_calculate_fraction_with_frequency. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 6
    var_6 = 15
    var_7 = 12
    var_8 = 31
    var_9 = '0.5'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 31
    var_7 = '0.25'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 31
    var_7 = '1.0'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 6
    var_5 = 15
    var_6 = 1
    var_7 = 12
    var_8 = 31
    var_9 = '0'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 2024
    var_6 = 12
    var_7 = 31
    var_8 = '0'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 6
    var_6 = 15
    var_7 = 12
    var_8 = 31
    var_9 = '4'
    var_10 = '0.4'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_a_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_over_year. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_same_day. Retrieved 4/11 statements.
# Partially parsed test_dcfc_act_365_a_one_day. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_365_a_with_freq_parameter. Retrieved 8/17 statements.


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

def test_case_0():
    var_0 = 2008
    var_1 = 3
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 3
    var_2 = 15
    var_3 = 16
    var_4 = '1'
    var_5 = '366'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '2'
    var_6 = 14
    var_7 = '0.16986301369863'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coupon_basic_calculation. Retrieved 11/25 statements.
# Partially parsed test_coupon_with_eom. Retrieved 14/28 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 10/22 statements.
# Partially parsed test_coupon_different_frequencies. Retrieved 11/27 statements.
# Partially parsed test_coupon_decimal_frequency. Retrieved 11/26 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.05'
    var_5 = 2014
    var_6 = 1
    var_7 = 6
    var_8 = 2015
    var_9 = 1
    var_10 = '0.5'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '5000'
    var_4 = '0.03'
    var_5 = 2014
    var_6 = 1
    var_7 = 31
    var_8 = 3
    var_9 = 15
    var_10 = 2015
    var_11 = 2
    var_12 = 31
    var_13 = '0.25'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0'
    var_5 = 2014
    var_6 = 1
    var_7 = 6
    var_8 = 2015
    var_9 = 1

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = '0.04'
    var_5 = 2014
    var_6 = 1
    var_7 = 4
    var_8 = 2015
    var_9 = 4
    var_10 = '0.1'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '10000'
    var_4 = '0.06'
    var_5 = 2014
    var_6 = 1
    var_7 = 8
    var_8 = 2015
    var_9 = '2'
    var_10 = '0.33'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_next_payment_date_annual_frequency_no_eom. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_annual_frequency_with_eom. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_semi_annual_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_quarterly_frequency. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_monthly_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_eom_invalid_day. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_eom_february_leap_year. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_decimal_frequency. Retrieved 5/12 statements.


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
    var_2 = 4
    var_3 = None

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 12
    var_3 = None
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

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2
    var_3 = None
    var_4 = 7



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_with_exact_name. Retrieved 4/12 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/11 statements.
# Partially parsed test_find_case_insensitive. Retrieved 4/10 statements.
# Partially parsed test_find_with_whitespace. Retrieved 5/11 statements.
# Partially parsed test_find_multiple_registrations. Retrieved 7/15 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = '  act/act  '
    var_4 = var_0.find(var_3)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent/DCC'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360'
    var_2 = []
    var_3 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'BOND BASIS'
    var_2 = []
    var_3 = '  bond basis  '
    var_4 = var_0.find(var_3)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'UnknownDCC'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = []
    var_3 = 'ACT/365'
    var_4 = []
    var_5 = var_0.find(var_1)
    var_6 = var_0.find(var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_multiple_leap_years_with_one_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_multiple_leap_years_none_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_start_equals_leap_day. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_end_equals_leap_day. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_range_is_leap_day. Retrieved 3/8 statements.
# Partially parsed test_has_leap_day_before_leap_day. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_after_leap_day. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_multi_year_range_with_multiple_leap_days. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = 3

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2019
    var_1 = 2
    var_2 = 1
    var_3 = 2020
    var_4 = 3

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = 2024
    var_4 = 2

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = 3
    var_4 = 1

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = 29

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = 28

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2019
    var_1 = 2
    var_2 = 1
    var_3 = 2024
    var_4 = 3



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_day_difference. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 14/22 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 15/23 statements.


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
    var_0 = 2008
    var_1 = 3
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 3
    var_2 = 15
    var_3 = 16
    var_4 = '1'
    var_5 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 29
    var_5 = 30
    var_6 = var_4 - var_5
    var_7 = var_3 - var_1
    var_8 = var_5 * var_7
    var_9 = var_6 + var_8
    var_10 = 360
    var_11 = var_0 - var_0
    var_12 = var_10 * var_11
    var_13 = var_9 + var_12

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 31
    var_4 = 1
    var_5 = 2
    var_6 = var_4 - var_2
    var_7 = 30
    var_8 = var_5 - var_1
    var_9 = var_7 * var_8
    var_10 = var_6 + var_9
    var_11 = 360
    var_12 = var_0 - var_0
    var_13 = var_11 * var_12
    var_14 = var_10 + var_13



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dcfc_30_e_360_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_e_360_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_e_360_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_e_360_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_e_360_start_day_31. Retrieved 8/19 statements.
# Partially parsed test_dcfc_30_e_360_asof_day_31. Retrieved 5/12 statements.
# Partially parsed test_dcfc_30_e_360_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_360_one_year_apart. Retrieved 5/12 statements.
# Partially parsed test_dcfc_30_e_360_month_difference. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_e_360_both_end_of_month_31. Retrieved 7/16 statements.


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
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 11
    var_4 = 30
    var_5 = '1'
    var_6 = '12'
    var_7 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 30
    var_3 = 31
    var_4 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 6
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2007
    var_1 = 6
    var_2 = 15
    var_3 = 2008
    var_4 = '1'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 3
    var_4 = '60'
    var_5 = '360'

def test_case_0():
    var_0 = 2007
    var_1 = 8
    var_2 = 31
    var_3 = 9
    var_4 = 30
    var_5 = '30'
    var_6 = '360'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 6/14 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Act/Act'
    var_1 = set()
    var_2 = set()
    var_3 = set()
    var_4 = set()
    var_5 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = '  act/act  '
    var_4 = var_0.find(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_annual_same_year. Retrieved 4/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_august. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_april. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_june. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_december_payment. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_semi_annual_december. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_semi_annual_december_year_end. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_before_first_payment. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_with_eom_parameter. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_february_eom_handling. Retrieved 6/12 statements.


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

def test_case_0():
    var_0 = 2015
    var_1 = 6
    var_2 = 1
    var_3 = 3
    var_4 = 31

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = 2015
    var_4 = 12

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = 2015
    var_4 = 2
    var_5 = 28



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_date_range. Retrieved 6/18 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 2/9 statements.
# Partially parsed test_get_date_range_two_days. Retrieved 3/12 statements.
# Partially parsed test_get_date_range_end_exclusive. Retrieved 3/10 statements.
# Partially parsed test_get_date_range_returns_generator. Retrieved 5/13 statements.
# Partially parsed test_get_date_range_long_period. Retrieved 4/13 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = '__iter__'
    var_4 = '__next__'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 30



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_30_days. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_31. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30

def test_case_0():
    var_0 = 2020
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
    var_1 = 12
    var_2 = 31



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_day. Retrieved 4/10 statements.
# Partially parsed test_dcfc_act_act_non_leap_year. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 6/14 statements.


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
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = '0'

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 28
    var_3 = 29
    var_4 = '1'
    var_5 = '366'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_360_isda_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 15/23 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_30_asof_day_31. Retrieved 14/22 statements.
# Partially parsed test_dcfc_30_360_isda_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_isda_one_day_difference. Retrieved 5/13 statements.
# Partially parsed test_dcfc_30_360_isda_month_difference. Retrieved 13/21 statements.
# Partially parsed test_dcfc_30_360_isda_year_difference. Retrieved 13/21 statements.


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
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = 2008
    var_4 = 1
    var_5 = 15
    var_6 = 30
    var_7 = var_5 - var_6
    var_8 = var_4 - var_1
    var_9 = var_6 * var_8
    var_10 = var_7 + var_9
    var_11 = 360
    var_12 = var_3 - var_0
    var_13 = var_11 * var_12
    var_14 = var_10 + var_13

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 30
    var_3 = 2008
    var_4 = 1
    var_5 = 31
    var_6 = var_2 - var_2
    var_7 = var_4 - var_1
    var_8 = var_2 * var_7
    var_9 = var_6 + var_8
    var_10 = 360
    var_11 = var_3 - var_0
    var_12 = var_10 * var_11
    var_13 = var_9 + var_12

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 0

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 16
    var_4 = 360

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 2
    var_4 = var_2 - var_2
    var_5 = 30
    var_6 = var_3 - var_1
    var_7 = var_5 * var_6
    var_8 = var_4 + var_7
    var_9 = 360
    var_10 = var_0 - var_0
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 15
    var_3 = 2008
    var_4 = var_2 - var_2
    var_5 = 30
    var_6 = var_1 - var_1
    var_7 = var_5 * var_6
    var_8 = var_4 + var_7
    var_9 = 360
    var_10 = var_3 - var_0
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dcfc_nl_365_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_nl_365_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_extended_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_nl_365_one_day. Retrieved 5/14 statements.
# Partially parsed test_dcfc_nl_365_with_freq_parameter. Retrieved 6/15 statements.


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
    var_0 = 2020
    var_1 = 1
    var_2 = '0'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 6
    var_3 = 30
    var_4 = '4'
    var_5 = '0'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 7/9 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_coupon_basic. Retrieved 12/24 statements.
# Partially parsed test_coupon_with_eom. Retrieved 13/25 statements.
# Partially parsed test_coupon_semi_annual. Retrieved 14/26 statements.
# Partially parsed test_coupon_quarterly. Retrieved 14/26 statements.
# Partially parsed test_coupon_with_decimal_frequency. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.05'
    var_5 = 2014
    var_6 = 1
    var_7 = 6
    var_8 = 15
    var_9 = 2015
    var_10 = 1
    var_11 = '25'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = '0.04'
    var_5 = 2014
    var_6 = 1
    var_7 = 15
    var_8 = 6
    var_9 = 2015
    var_10 = 2
    var_11 = 15
    var_12 = '20'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '5000'
    var_4 = '0.06'
    var_5 = 2012
    var_6 = 12
    var_7 = 15
    var_8 = 2015
    var_9 = 31
    var_10 = 2016
    var_11 = 6
    var_12 = 2
    var_13 = '750'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '10000'
    var_4 = '0.08'
    var_5 = 2008
    var_6 = 7
    var_7 = 2015
    var_8 = 10
    var_9 = 6
    var_10 = 2016
    var_11 = 1
    var_12 = 4
    var_13 = '200'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.1'
    var_5 = 2014
    var_6 = 1
    var_7 = 7
    var_8 = 2015
    var_9 = '1'
    var_10 = '50'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_leap_day_on_start_date. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_leap_day_on_end_date. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_multiple_leap_years_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_before_leap_day. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_after_leap_day. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_single_day_leap_day. Retrieved 3/8 statements.
# Partially parsed test_has_leap_day_single_day_non_leap_day. Retrieved 3/8 statements.
# Partially parsed test_has_leap_day_century_leap_year. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_century_non_leap_year. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1

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
    var_0 = 2020
    var_1 = 1
    var_2 = 2024
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = 28

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2000
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1

def test_case_0():
    var_0 = 1900
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1



# Parsed testcases at query #22
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_57. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dcfc_act_act_basic. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_act_leap_day. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_across_years. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_same_date. Retrieved 3/8 statements.
# Partially parsed test_dcfc_act_act_one_day. Retrieved 5/12 statements.
# Partially parsed test_dcfc_act_act_non_leap_year. Retrieved 5/12 statements.
# Partially parsed test_dcfc_act_act_with_freq_parameter. Retrieved 8/15 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '366'

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '2'
    var_6 = 14
    var_7 = '0.16942884946478'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_54_evaluates_to_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 1
    var_3 = None
    var_4 = 2013



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_eom_parameter_false_condition. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 15
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = 1
    var_7 = 10



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_next_payment_date_with_eom. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 15
    var_3 = 2015



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_1_false. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_find_evaluates_predicate_to_true. Retrieved 5/10 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = var_0.find(var_1)
    var_3 = '  act/act  '
    var_4 = var_0.find(var_3)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_day. Retrieved 4/11 statements.
# Partially parsed test_dcfc_act_act_with_freq_parameter. Retrieved 8/17 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_act_non_leap_year. Retrieved 7/16 statements.


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
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = '0'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '2'
    var_6 = 14
    var_7 = '0.16942884946478'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 28
    var_3 = 29
    var_4 = '1'
    var_5 = '366'

def test_case_0():
    var_0 = 2007
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1
    var_5 = '1'
    var_6 = '365'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_has_leap_day_predicate_evaluates_to_true. Retrieved 8/21 statements.
# Partially parsed test_has_leap_day_with_leap_year_range. Retrieved 8/21 statements.
# Partially parsed test_has_leap_day_with_multiple_leap_years. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = False
    var_5 = 2
    var_6 = 29
    var_7 = True
    assert var_7 is True

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 1
    var_3 = 3
    var_4 = False
    var_5 = 2
    var_6 = 29
    var_7 = True
    assert var_7 is True

def test_case_0():
    var_0 = 2000
    var_1 = 1
    var_2 = 2008
    var_3 = 12
    var_4 = 31
    var_5 = False
    var_6 = 2
    var_7 = 29
    var_8 = True
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 7/9 statements.


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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_get_date_range. Retrieved 6/17 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 2/8 statements.
# Partially parsed test_get_date_range_two_days. Retrieved 3/11 statements.
# Partially parsed test_get_date_range_month_boundary. Retrieved 5/15 statements.
# Partially parsed test_get_date_range_year_boundary. Retrieved 6/15 statements.
# Partially parsed test_get_date_range_is_generator. Retrieved 5/12 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = 2
    var_4 = 31

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 31
    var_3 = 2023
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = '__iter__'
    var_4 = '__next__'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dcfc_act_act_example_1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example_2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example_3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example_4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_date. Retrieved 3/10 statements.
# Partially parsed test_dcfc_act_act_one_day. Retrieved 5/14 statements.
# Partially parsed test_dcfc_act_act_non_leap_year. Retrieved 5/14 statements.
# Partially parsed test_dcfc_act_act_with_freq_parameter. Retrieved 6/15 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '366'

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = '2'
    var_5 = '0'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dcfc_act_act_icma_basic. Retrieved 7/16 statements.
# Partially parsed test_dcfc_act_act_icma_same_dates. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_end_date. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_with_freq. Retrieved 8/18 statements.
# Partially parsed test_dcfc_act_act_icma_one_day. Retrieved 6/16 statements.
# Partially parsed test_dcfc_act_act_icma_half_year. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 2020
    var_4 = '0'

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 2020
    var_4 = '1'

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '2'
    var_7 = '0.2622950820'

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 2
    var_3 = 365
    var_4 = '1'
    var_5 = '364'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 7
    var_3 = 12
    var_4 = 31
    var_5 = '182'
    var_6 = '365'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 5/11 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 5/11 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_day_difference. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_month_difference. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_year_difference. Retrieved 5/13 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 15

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 2
    var_4 = 31

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 16
    var_4 = '1'
    var_5 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 2
    var_4 = '30'
    var_5 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 2009
    var_4 = '360'



# Parsed testcases at query #38
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = -1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = -1
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dcfc_nl_365_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_nl_365_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_nl_365_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_nl_365_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_nl_365_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_nl_365_one_day. Retrieved 5/13 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day. Retrieved 6/14 statements.
# Partially parsed test_dcfc_nl_365_across_leap_day. Retrieved 7/15 statements.


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
    var_0 = 2020
    var_1 = 1
    var_2 = '0'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = 29
    var_4 = '0'
    var_5 = '365'

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = 3
    var_4 = 1
    var_5 = '1'
    var_6 = '365'



# Parsed testcases at query #40
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dcfc_30_360_isda_line_29_predicate. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = 31
    var_4 = 30
    var_5 = 3
    var_6 = 360



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 7/9 statements.


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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_dcfc_30_e_360_asof_day_31_predicate. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 390
    var_6 = 360



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_dcfc_30_360_german_example_1. Retrieved 7/16 statements.
# Partially parsed test_dcfc_30_360_german_example_2. Retrieved 8/17 statements.
# Partially parsed test_dcfc_30_360_german_example_3. Retrieved 8/17 statements.
# Partially parsed test_dcfc_30_360_german_example_4. Retrieved 8/17 statements.
# Partially parsed test_dcfc_30_360_german_same_dates. Retrieved 4/12 statements.
# Partially parsed test_dcfc_30_360_german_month_31st_adjustment. Retrieved 7/17 statements.
# Partially parsed test_dcfc_30_360_german_february_last_day_not_end. Retrieved 8/18 statements.
# Partially parsed test_dcfc_30_360_german_year_difference. Retrieved 4/14 statements.
# Partially parsed test_dcfc_30_360_german_with_freq_parameter. Retrieved 6/17 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 15
    var_5 = '15'
    var_6 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 29
    var_3 = 3
    var_4 = 15
    var_5 = 20
    var_6 = '15'
    var_7 = '360'

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 2008
    var_3 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 2
    var_3 = '4'
    var_4 = '30'
    var_5 = '360'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_dcfc_30_360_german_line_25_predicate_true. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '0.08333333333333'
    var_6 = 29
    var_7 = 3
    var_8 = 2007



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_find_with_stripped_uppercase_fallback. Retrieved 4/9 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test that find method falls back to stripped uppercase name lookup.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = '  act/act  '
    var_3 = var_1.find(var_2)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_dcfc_30_360_german_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_german_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_360_german_one_day. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_360_german_with_freq_parameter. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_start_day_31. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_german_asof_day_31. Retrieved 7/16 statements.
# Partially parsed test_dcfc_30_360_german_february_last_day_not_end. Retrieved 6/13 statements.
# Partially parsed test_dcfc_30_360_german_february_last_day_is_end. Retrieved 7/16 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 16
    var_4 = '1'
    var_5 = '360'

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '2'
    var_6 = 14
    var_7 = '0.16666666666667'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '28'
    var_6 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 31
    var_4 = 2
    var_5 = '16'
    var_6 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 29
    var_3 = 3
    var_4 = 31
    var_5 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 29
    var_5 = '29'
    var_6 = '360'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_buffer_main_initialization. Retrieved 4/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_main
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_57. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_last_payment_date_examples. Retrieved 17/59 statements.


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
    var_11 = 2008
    var_12 = 10
    var_13 = 9
    var_14 = 2012
    var_15 = 15
    var_16 = 2016



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_calculate_fraction_predicate_evaluates_to_false. Retrieved 11/22 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 6
    var_6 = 15
    var_7 = 12
    var_8 = 31
    var_9 = None
    var_10 = '0.5'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_us_with_freq_parameter. Retrieved 7/16 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 3
    var_3 = 31
    var_4 = '2'
    var_5 = 14
    var_6 = '0.25'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_57_true. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_has_leap_day_predicate_evaluates_true. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_august. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_april. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_june_start. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_december. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_semi_annual_december. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_semi_annual_december_year_end. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_with_decimal_frequency. Retrieved 5/13 statements.
# Partially parsed test_last_payment_date_before_start_date. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_with_eom_parameter. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_eom_day_adjustment. Retrieved 6/12 statements.


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

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2015
    var_1 = 6
    var_2 = 1
    var_3 = 5
    var_4 = 31

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = 2015
    var_4 = 12

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = 2015
    var_4 = 2
    var_5 = 28



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_dcfc_act_act_icma_predicate_line_22. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = 1
    var_7 = 0



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_57. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 1
    var_1 = 15
    var_2 = 10
    var_3 = 2020
    var_4 = 12
    var_5 = 31
    var_6 = 0



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 4/17 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 4/17 statements.
# Partially parsed test_has_leap_day_leap_day_at_start. Retrieved 5/18 statements.
# Partially parsed test_has_leap_day_leap_day_at_end. Retrieved 4/17 statements.
# Partially parsed test_has_leap_day_multiple_leap_years. Retrieved 5/18 statements.


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
    var_0 = 2019
    var_1 = 1
    var_2 = 2021
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_1_false. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = '1'
    var_6 = 15



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_360_isda_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_30_asof_day_31. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_360_isda_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_isda_one_month_difference. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_360_isda_one_year_difference. Retrieved 13/20 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 15
    var_5 = 30
    var_6 = var_4 - var_5
    var_7 = var_3 - var_1
    var_8 = var_5 * var_7
    var_9 = var_6 + var_8
    var_10 = 360
    var_11 = var_0 - var_0
    var_12 = var_10 * var_11
    var_13 = var_9 + var_12

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 30
    var_3 = 2
    var_4 = 31
    var_5 = var_2 - var_2
    var_6 = var_3 - var_1
    var_7 = var_2 * var_6
    var_8 = var_5 + var_7
    var_9 = 360
    var_10 = var_0 - var_0
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 2
    var_4 = var_2 - var_2
    var_5 = 30
    var_6 = var_3 - var_1
    var_7 = var_5 * var_6
    var_8 = var_4 + var_7
    var_9 = 360
    var_10 = var_0 - var_0
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 15
    var_3 = 2008
    var_4 = var_2 - var_2
    var_5 = 30
    var_6 = var_1 - var_1
    var_7 = var_5 * var_6
    var_8 = var_4 + var_7
    var_9 = 360
    var_10 = var_3 - var_0
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_register_valid_dcc. Retrieved 5/9 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_conflicting_altname. Retrieved 10/19 statements.
# Partially parsed test_register_altname_conflicts_with_main_name. Retrieved 9/18 statements.
# Partially parsed test_register_multiple_valid_dccs. Retrieved 11/19 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = set()
    var_6 = set()
    var_7 = '0.6'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC1'
    var_2 = 'Common'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = 'TestDCC2'
    var_7 = {var_2}
    var_8 = set()
    var_9 = '0.6'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC1'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = 'TestDCC2'
    var_6 = {var_1}
    var_7 = set()
    var_8 = '0.6'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC1'
    var_2 = 'Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = 'TestDCC2'
    var_7 = 'Alt2'
    var_8 = {var_7}
    var_9 = set()
    var_10 = '0.6'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 5/9 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name_raises_error. Retrieved 9/17 statements.
# Partially parsed test_register_duplicate_altname_raises_error. Retrieved 11/19 statements.
# Partially parsed test_register_altname_conflicts_with_main_name_raises_error. Retrieved 10/18 statements.
# Partially parsed test_register_multiple_dccs_without_conflicts. Retrieved 10/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = set()
    var_6 = set()
    var_7 = False
    var_8 = True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'SharedAlt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = 'Test/DCC2'
    var_7 = {var_2}
    var_8 = set()
    var_9 = False
    var_10 = True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = 'Test/DCC2'
    var_6 = {var_1}
    var_7 = set()
    var_8 = False
    var_9 = True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = 'Test/DCC2'
    var_7 = 'Alt2'
    var_8 = {var_7}
    var_9 = set()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_date_range. Retrieved 6/17 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 2/9 statements.
# Partially parsed test_get_date_range_two_days. Retrieved 3/11 statements.
# Partially parsed test_get_date_range_across_months. Retrieved 6/17 statements.
# Partially parsed test_get_date_range_across_years. Retrieved 7/17 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = 2
    var_4 = 3
    var_5 = 31

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 31
    var_3 = 2023
    var_4 = 1
    var_5 = 3
    var_6 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 14/22 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 8/30 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_day. Retrieved 5/13 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 15
    var_5 = 30
    var_6 = var_4 - var_5
    var_7 = var_3 - var_1
    var_8 = var_5 * var_7
    var_9 = var_6 + var_8
    var_10 = 360
    var_11 = var_0 - var_0
    var_12 = var_10 * var_11
    var_13 = var_9 + var_12

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 2
    var_4 = 31
    var_5 = 3
    var_6 = 30
    var_7 = 360

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 0

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 16
    var_4 = 360



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dcfc_act_act_icma. Retrieved 21/62 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'
    var_7 = 1
    var_8 = 12
    var_9 = 31
    var_10 = '0'
    var_11 = '1'
    var_12 = 6
    var_13 = 30
    var_14 = '2'
    var_15 = '181'
    var_16 = '365'
    var_17 = 7
    var_18 = 2021
    var_19 = '182'
    var_20 = '366'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dcfc_30_360_isda_start_day_not_31. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16666666666666666666666666667'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_next_payment_date_annual_frequency_no_eom. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_annual_frequency_with_eom. Retrieved 4/8 statements.
# Partially parsed test_next_payment_date_semi_annual_frequency. Retrieved 5/9 statements.
# Partially parsed test_next_payment_date_quarterly_frequency. Retrieved 4/8 statements.
# Partially parsed test_next_payment_date_monthly_frequency. Retrieved 5/9 statements.
# Partially parsed test_next_payment_date_with_eom_february_leap_year. Retrieved 4/8 statements.
# Partially parsed test_next_payment_date_with_eom_february_non_leap_year. Retrieved 4/8 statements.
# Partially parsed test_next_payment_date_decimal_frequency. Retrieved 5/11 statements.
# Partially parsed test_next_payment_date_multiple_years. Retrieved 6/10 statements.
# Partially parsed test_next_payment_date_eom_valid_day. Retrieved 6/10 statements.


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
    var_2 = 4
    var_3 = None

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 12
    var_3 = None
    var_4 = 2

def test_case_0():
    var_0 = 2016
    var_1 = 1
    var_2 = 31
    var_3 = 2017

def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = 31
    var_3 = 2016

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = '2'
    var_3 = None
    var_4 = 7

def test_case_0():
    var_0 = 2010
    var_1 = 6
    var_2 = 15
    var_3 = 1
    var_4 = None
    var_5 = 2011

def test_case_0():
    var_0 = 2014
    var_1 = 3
    var_2 = 15
    var_3 = 2
    var_4 = 20
    var_5 = 9



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_isda_example2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_isda_example3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_isda_example4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_isda_same_dates. Retrieved 3/8 statements.
# Partially parsed test_dcfc_30_360_isda_one_day_difference. Retrieved 5/13 statements.
# Partially parsed test_dcfc_30_360_isda_with_start_day_31. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_30_asof_day_31. Retrieved 7/15 statements.


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
    var_0 = 2010
    var_1 = 1
    var_2 = '0'

def test_case_0():
    var_0 = 2010
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '360'

def test_case_0():
    var_0 = 2010
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 28
    var_5 = '28'
    var_6 = '360'

def test_case_0():
    var_0 = 2010
    var_1 = 1
    var_2 = 30
    var_3 = False
    var_4 = 2
    var_5 = 31
    var_6 = 28



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 7/9 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 5/9 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_altname_conflicts_with_main_name. Retrieved 9/18 statements.
# Partially parsed test_register_altname_conflicts_with_altname. Retrieved 10/19 statements.
# Partially parsed test_register_multiple_dcc_no_conflicts. Retrieved 11/19 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/360'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/360'
    var_2 = 'Test360'
    var_3 = 'T360'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/360'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = set()
    var_6 = set()
    var_7 = '0.6'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/360'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = 'Other/360'
    var_6 = {var_1}
    var_7 = set()
    var_8 = '0.6'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/360'
    var_2 = 'T360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = 'Other/360'
    var_7 = {var_2}
    var_8 = set()
    var_9 = '0.6'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/360'
    var_2 = 'T360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = 'Other/365'
    var_7 = 'O365'
    var_8 = {var_7}
    var_9 = set()
    var_10 = '0.6'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_register_successful. Retrieved 6/10 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_duplicate_alternative_name. Retrieved 9/18 statements.
# Partially parsed test_register_multiple_alternative_names. Retrieved 8/12 statements.
# Partially parsed test_register_main_name_conflicts_with_existing_alternative. Retrieved 8/17 statements.
# Partially parsed test_register_empty_altnames. Retrieved 7/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Test/Alternative'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = set()
    var_6 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'Test/Alt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = 'Test/DCC2'
    var_7 = {var_2}
    var_8 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = 'Alt3'
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'Test/Conflict'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = set()
    var_7 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = var_0._buffer_altn
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dcfc_30_360_german_example1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_german_example2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_same_date. Retrieved 4/11 statements.
# Partially parsed test_dcfc_30_360_german_one_day. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_360_german_month_adjustment. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_german_february_last_day_not_end_date. Retrieved 9/18 statements.
# Partially parsed test_dcfc_30_360_german_february_last_day_is_end_date. Retrieved 6/15 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 16
    var_4 = '1'
    var_5 = '360'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 30
    var_5 = var_1 - var_4
    var_6 = 360

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 28
    var_3 = 29
    var_4 = 3
    var_5 = 31
    var_6 = 30
    var_7 = var_6 - var_2
    var_8 = 360

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 28
    var_3 = 29
    var_4 = var_3 - var_2
    var_5 = 360



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = 2008
    var_4 = 11
    var_5 = 30
    var_6 = '1.08333333333333'



# Parsed testcases at query #16
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
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
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = -2023
    var_1 = 1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = -1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = -15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dcfc_30_360_german_line_31_predicate. Retrieved 8/22 statements.


def test_case_0():
    var_0 = '\n    Test that the predicate at line 31 evaluates to True.\n    The predicate is: asof.day == 31 or (asof.month == 2 and _is_last_day_of_month(asof) and end != asof)\n    '
    var_1 = 2008
    var_2 = 1
    var_3 = 31
    var_4 = 2
    var_5 = 28
    var_6 = 29
    var_7 = 3



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_calculate_fraction_valid_dates. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_start_equals_asof. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_equals_end. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_before_start. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_asof_after_end. Retrieved 9/19 statements.
# Partially parsed test_calculate_fraction_with_freq_parameter. Retrieved 11/24 statements.
# Partially parsed test_calculate_fraction_returns_zero_for_invalid_order. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 6
    var_6 = 15
    var_7 = 12
    var_8 = 31
    var_9 = '0.5'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 31
    var_7 = '0.25'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 31
    var_7 = '0.75'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 6
    var_5 = 15
    var_6 = 1
    var_7 = 12
    var_8 = 31
    var_9 = '0'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 2024
    var_6 = 12
    var_7 = 31
    var_8 = '0'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 6
    var_6 = 15
    var_7 = 12
    var_8 = 31
    var_9 = '2'
    var_10 = '1'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 12
    var_5 = 31
    var_6 = 6
    var_7 = 15
    var_8 = 1
    var_9 = '0'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dcfc_30_e_360_example_1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_360_example_2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_example_3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_example_4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_start_day_31. Retrieved 6/13 statements.
# Partially parsed test_dcfc_30_e_360_asof_day_31. Retrieved 6/13 statements.
# Partially parsed test_dcfc_30_e_360_both_days_31. Retrieved 5/12 statements.
# Partially parsed test_dcfc_30_e_360_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_360_one_year_apart. Retrieved 5/11 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 2
    var_4 = 29
    var_5 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 3
    var_4 = 31
    var_5 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = 3
    var_4 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = 2009
    var_4 = '1'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_with_exact_name. Retrieved 4/12 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 5/11 statements.
# Partially parsed test_find_case_insensitive. Retrieved 5/11 statements.
# Partially parsed test_find_with_whitespace_and_case. Retrieved 5/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = '  act/act  '
    var_4 = var_0.find(var_3)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent/DCC'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360 US'
    var_2 = []
    var_3 = '30/360 us'
    var_4 = var_0.find(var_3)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'BOND BASIS'
    var_2 = []
    var_3 = '  bond basis  '
    var_4 = var_0.find(var_3)



# Parsed testcases at query #21
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_register_successfully_registers_dcc. Retrieved 7/11 statements.
# Partially parsed test_register_raises_error_when_main_name_already_registered. Retrieved 7/16 statements.
# Partially parsed test_register_raises_error_when_altname_conflicts_with_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_raises_error_when_altname_conflicts_with_existing_altname. Retrieved 9/18 statements.
# Partially parsed test_register_with_empty_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_with_multiple_altnames. Retrieved 8/12 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Test'
    var_3 = 'TESTDCC'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = set()
    var_6 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = 'Test/DCC2'
    var_6 = {var_1}
    var_7 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = 'Test/DCC2'
    var_7 = {var_2}
    var_8 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = var_0._buffer_altn
    var_6 = len(var_5)
    assert var_6 == 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = 'Alt3'
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_august. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_april. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_frequency_june. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_frequency_december. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_end. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_with_eom_parameter. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_end_of_month_handling. Retrieved 6/11 statements.


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

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 15
    var_3 = 2015
    var_4 = 12
    var_5 = 31

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = 2015
    var_4 = 2
    var_5 = 28



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_1_false. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #25
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_last_payment_date_predicate. Retrieved 17/59 statements.


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
    var_11 = 2008
    var_12 = 10
    var_13 = 9
    var_14 = 2012
    var_15 = 15
    var_16 = 2016



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_interest_basic_calculation. Retrieved 12/41 statements.
# Partially parsed test_interest_without_end_date. Retrieved 10/37 statements.
# Partially parsed test_interest_zero_rate. Retrieved 11/39 statements.


def test_case_0():
    var_0 = 'Test Convention'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.05'
    var_5 = 2024
    var_6 = 1
    var_7 = 6
    var_8 = 30
    var_9 = 12
    var_10 = 31
    var_11 = '25'

def test_case_0():
    var_0 = 'Test Convention'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = '0.10'
    var_5 = 2024
    var_6 = 1
    var_7 = 3
    var_8 = 31
    var_9 = '50'

def test_case_0():
    var_0 = 'Test Convention'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0'
    var_5 = 2024
    var_6 = 1
    var_7 = 6
    var_8 = 30
    var_9 = 12
    var_10 = 31



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_last_payment_date_predicate_false. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_last_payment_date_predicate_false. Retrieved 9/24 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 2015
    var_3 = 12
    var_4 = 31
    var_5 = 1
    var_6 = None
    var_7 = '2'
    var_8 = 15



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_find_with_exact_name. Retrieved 4/12 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_nonexistent_name. Retrieved 5/11 statements.
# Partially parsed test_find_case_insensitive. Retrieved 5/11 statements.
# Partially parsed test_find_with_whitespace. Retrieved 5/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = '  act/act  '
    var_4 = var_0.find(var_3)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = 'NonExistent'
    var_4 = var_0.find(var_3)
    assert var_4 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360 US'
    var_2 = []
    var_3 = '30/360 us'
    var_4 = var_0.find(var_3)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'BOND BASIS'
    var_2 = []
    var_3 = '   bond basis   '
    var_4 = var_0.find(var_3)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_date. Retrieved 3/8 statements.
# Partially parsed test_dcfc_act_act_one_day_non_leap. Retrieved 5/13 statements.


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
    var_0 = 2020
    var_1 = 1
    var_2 = '0'

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_calculate_daily_fraction_basic. Retrieved 9/20 statements.
# Partially parsed test_calculate_daily_fraction_with_increasing_fractions. Retrieved 11/25 statements.
# Partially parsed test_calculate_daily_fraction_asof_equals_start. Retrieved 8/19 statements.
# Partially parsed test_calculate_daily_fraction_with_freq_parameter. Retrieved 10/24 statements.
# Partially parsed test_calculate_daily_fraction_without_freq_parameter. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 3
    var_6 = 12
    var_7 = 31
    var_8 = '0'

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'Test DCC'
    var_3 = set()
    var_4 = set()
    var_5 = 2023
    var_6 = 1
    var_7 = 3
    var_8 = 12
    var_9 = 31
    var_10 = '0.1'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 31
    var_7 = '0.3'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 5
    var_6 = 12
    var_7 = 31
    var_8 = '4'
    var_9 = '0.1'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 10
    var_6 = 12
    var_7 = 31
    var_8 = '0'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dcfc_30_360_german_line_31_predicate_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = '\n    Test that the predicate at line 31 evaluates to False.\n    The predicate is: asof.day == 31 or (asof.month == 2 and _is_last_day_of_month(asof) and end != asof)\n    \n    For it to be False:\n    - asof.day must not be 31\n    - AND (asof.month != 2 OR _is_last_day_of_month(asof) is False OR end == asof)\n    '
    var_1 = 2008
    var_2 = 1
    var_3 = 15
    var_4 = 3



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_calculate_daily_fraction_basic. Retrieved 8/19 statements.
# Partially parsed test_calculate_daily_fraction_with_asof_minus_1_before_start. Retrieved 8/19 statements.
# Partially parsed test_calculate_daily_fraction_with_different_values. Retrieved 11/28 statements.
# Partially parsed test_calculate_daily_fraction_asof_equals_start. Retrieved 7/20 statements.
# Partially parsed test_calculate_daily_fraction_with_freq_parameter. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = 2
    var_6 = 31
    var_7 = '0'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = 2
    var_6 = 31
    var_7 = '0.05'

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = []
    var_3 = 'Test'
    var_4 = set()
    var_5 = set()
    var_6 = 2024
    var_7 = 1
    var_8 = 2
    var_9 = 31
    var_10 = '0.05'

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = 31
    var_6 = '0.02'

def test_case_0():
    var_0 = []
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 2024
    var_5 = 1
    var_6 = 3
    var_7 = 31
    var_8 = '4'
    var_9 = len(var_0)
    assert var_9 == 2



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_last_payment_date_predicate. Retrieved 17/59 statements.


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
    var_11 = 2008
    var_12 = 10
    var_13 = 9
    var_14 = 2012
    var_15 = 15
    var_16 = 2016



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_coupon_basic. Retrieved 11/23 statements.
# Partially parsed test_coupon_with_eom. Retrieved 13/25 statements.
# Partially parsed test_coupon_annual_frequency. Retrieved 12/24 statements.
# Partially parsed test_coupon_quarterly_frequency. Retrieved 11/23 statements.
# Partially parsed test_coupon_with_decimal_frequency. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = '0.05'
    var_5 = 2014
    var_6 = 1
    var_7 = 6
    var_8 = 2015
    var_9 = 2
    var_10 = '25'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = '0.1'
    var_5 = 2014
    var_6 = 1
    var_7 = 15
    var_8 = 7
    var_9 = 2015
    var_10 = 2
    var_11 = 15
    var_12 = '50'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '5000'
    var_4 = '0.02'
    var_5 = 2014
    var_6 = 1
    var_7 = 12
    var_8 = 31
    var_9 = 2015
    var_10 = 1
    var_11 = '100'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '10000'
    var_4 = '0.08'
    var_5 = 2014
    var_6 = 1
    var_7 = 4
    var_8 = 7
    var_9 = 4
    var_10 = '80'

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '3000'
    var_4 = '0.06'
    var_5 = 2014
    var_6 = 1
    var_7 = 7
    var_8 = 2015
    var_9 = '2'
    var_10 = '90'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_30. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_31. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 30

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2024
    var_1 = 4
    var_2 = 30

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_non_leap_year. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_multiple_leap_years. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_exact_leap_day_start. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_exact_leap_day_end. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_single_day_leap_day. Retrieved 3/8 statements.
# Partially parsed test_has_leap_day_single_day_not_leap_day. Retrieved 3/8 statements.
# Partially parsed test_has_leap_day_range_before_leap_day. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_range_after_leap_day. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = 3

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 1
    var_3 = 3

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2024
    var_3 = 12
    var_4 = 31

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = 3
    var_4 = 1

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = 29

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = 28

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_a_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_multi_year. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_same_day. Retrieved 3/10 statements.
# Partially parsed test_dcfc_act_365_a_one_day. Retrieved 5/14 statements.
# Partially parsed test_dcfc_act_365_a_with_freq_parameter. Retrieved 5/14 statements.


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

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = '0'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 6
    var_3 = '2'
    var_4 = '0'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dcfc_act_365_l_basic. Retrieved 7/16 statements.
# Partially parsed test_dcfc_act_365_l_leap_year_feb_29. Retrieved 8/17 statements.
# Partially parsed test_dcfc_act_365_l_across_years. Retrieved 8/17 statements.
# Partially parsed test_dcfc_act_365_l_long_period. Retrieved 8/17 statements.
# Partially parsed test_dcfc_act_365_l_same_date. Retrieved 4/12 statements.
# Partially parsed test_dcfc_act_365_l_one_day. Retrieved 5/15 statements.
# Partially parsed test_dcfc_act_365_l_non_leap_year. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16939890710383'

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
    var_7 = '1.32876712328767'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = '0'

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = '1'
    var_4 = '366'

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 2
    var_3 = '1'
    var_4 = '365'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_next_payment_date_with_eom. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 15
    var_3 = 2015



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_last_payment_date_line_1_predicate. Retrieved 17/68 statements.


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
    var_11 = 2008
    var_12 = 10
    var_13 = 9
    var_14 = 2012
    var_15 = 15
    var_16 = 2016



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_register_raises_error_when_altname_already_registered. Retrieved 10/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Convention1'
    var_1 = 'Alt1'
    var_2 = 'Alt2'
    var_3 = {var_1, var_2}
    var_4 = set()
    var_5 = 'Convention2'
    var_6 = 'Alt3'
    var_7 = {var_1, var_6}
    var_8 = set()
    var_9 = module_0.DCCRegistryMachinery()



