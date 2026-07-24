####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_last_day_of_month_true. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_feb_non_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_feb_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_april. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dcfc_30_360_isda_basic_case. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_leap_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_31_day_month_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_multi_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_same_date_case. Retrieved 3/6 statements.
# Partially parsed test_dcfc_30_360_isda_end_of_month_adjustment. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = 14
    var_9 = '0.16666666666667'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = 14
    var_10 = '0.16944444444444'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = 14
    var_10 = '1.08333333333333'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = 14
    var_10 = '1.33333333333333'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 0
    var_4 = [var_3]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 14
    var_9 = '0.08333333333333'
    var_10 = [var_9]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 6/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 10/19 statements.
# Partially parsed test_register_duplicate_name_in_alt_names. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = '0.1'
    var_5 = [var_4]
    var_6 = module_0.DCCRegistryMachinery()

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = '0.1'
    var_4 = [var_3]
    var_5 = set()
    var_6 = set()
    var_7 = '0.2'
    var_8 = [var_7]
    var_9 = module_0.DCCRegistryMachinery()
    var_10 = bool(False)
    assert var_10 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC1'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = set()
    var_4 = '0.1'
    var_5 = [var_4]
    var_6 = 'TestDCC2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = '0.2'
    var_10 = [var_9]
    var_11 = module_0.DCCRegistryMachinery()
    var_12 = bool(False)
    assert var_12 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC1'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = set()
    var_4 = '0.1'
    var_5 = [var_4]
    var_6 = set()
    var_7 = set()
    var_8 = '0.2'
    var_9 = [var_8]
    var_10 = module_0.DCCRegistryMachinery()
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_next_payment_date_basic. Retrieved 3/6 statements.
# Partially parsed test_next_payment_date_with_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_monthly_frequency. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_eom_february. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_quarterly_frequency. Retrieved 3/6 statements.
# Partially parsed test_next_payment_date_semi_annual_frequency. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = 2015
    var_5 = [var_4, var_1, var_3]

def test_case_0():
    var_0 = 2023
    var_1 = 3
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 12
    var_5 = 4
    var_6 = [var_0, var_5, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 4
    var_4 = [var_0, var_3, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = 7
    var_5 = [var_0, var_4, var_1]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 6/12 statements.
# Partially parsed test_register_existing_dcc. Retrieved 9/18 statements.
# Partially parsed test_register_conflicting_altname. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'TestAlt2'
    var_8 = {var_7}
    var_9 = {}
    var_10 = [var_5]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'TestDCC2'
    var_8 = {var_2}
    var_9 = {}
    var_10 = [var_5]
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dcfc_30_360_us_with_valid_dates. Retrieved 6/10 statements.
# Partially parsed test_dcfc_30_360_us_with_leap_year. Retrieved 7/11 statements.
# Partially parsed test_dcfc_30_360_us_with_month_end_dates. Retrieved 7/11 statements.
# Partially parsed test_dcfc_30_360_us_with_long_period. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16666666666667'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.16944444444444'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08333333333333'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.33333333333333'
    var_9 = [var_8]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 6/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 9/18 statements.
# Partially parsed test_register_duplicate_alt_name_in_main. Retrieved 8/17 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = module_0.DCCRegistryMachinery()

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = set()
    var_6 = set()
    var_7 = [var_3]
    var_8 = module_0.DCCRegistryMachinery()
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC1'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'TestDCC2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = [var_4]
    var_10 = module_0.DCCRegistryMachinery()
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC1'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = 'TestAlt'
    var_6 = set()
    var_7 = set()
    var_8 = [var_3]
    var_9 = module_0.DCCRegistryMachinery()
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dcfc_30_e_360_example1. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_e_360_example2. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_360_example3. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_360_example4. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.16666666666667'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = 14
    var_9 = '0.16944444444444'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = 14
    var_9 = '1.08333333333333'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = 14
    var_9 = '1.33055555555556'
    var_10 = [var_9]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dcfc_nl_365_with_same_day. Retrieved 3/8 statements.
# Partially parsed test_dcfc_nl_365_with_one_day. Retrieved 4/9 statements.
# Partially parsed test_dcfc_nl_365_with_leap_year_in_range. Retrieved 6/12 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day_in_range. Retrieved 7/13 statements.
# Partially parsed test_dcfc_nl_365_with_multiple_years. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = [var_0, var_1, var_1]
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = '0.002739726027397260273972602739726027397260274'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2016
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = 14
    var_8 = '0.99726027397260'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2016
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 14
    var_9 = '0.00821917808219'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2017
    var_4 = [var_3, var_1, var_1]
    var_5 = [var_3, var_1, var_1]
    var_6 = 14
    var_7 = '2.00273972602740'
    var_8 = [var_7]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_last_day_of_month_true. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_december. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_april. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_date_range_with_single_day. Retrieved 3/9 statements.
# Partially parsed test_get_date_range_with_multiple_days. Retrieved 5/13 statements.
# Partially parsed test_get_date_range_with_zero_days. Retrieved 3/7 statements.
# Partially parsed test_get_date_range_with_negative_days. Retrieved 4/8 statements.
# Partially parsed test_get_date_range_with_leap_year. Retrieved 6/14 statements.
# Partially parsed test_get_date_range_with_month_crossing. Retrieved 6/15 statements.
# Partially parsed test_get_date_range_with_year_crossing. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 4
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]
    var_8 = 3
    var_9 = [var_0, var_1, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = []

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_1]
    var_6 = [var_0, var_1, var_2]
    var_7 = 29
    var_8 = [var_0, var_1, var_7]
    var_9 = 1
    var_10 = [var_0, var_4, var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 3
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_1, var_2]
    var_8 = 31
    var_9 = [var_0, var_1, var_8]
    var_10 = [var_0, var_4, var_1]
    var_11 = [var_0, var_4, var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = 2
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_0, var_1, var_2]
    var_9 = 31
    var_10 = [var_0, var_1, var_9]
    var_11 = [var_4, var_5, var_5]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_date_range_with_single_day. Retrieved 3/9 statements.
# Partially parsed test_get_date_range_with_multiple_days. Retrieved 5/13 statements.
# Partially parsed test_get_date_range_with_zero_days. Retrieved 3/7 statements.
# Partially parsed test_get_date_range_across_month. Retrieved 5/13 statements.
# Partially parsed test_get_date_range_with_leap_year. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 4
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]
    var_8 = 3
    var_9 = [var_0, var_1, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_4]
    var_6 = [var_0, var_1, var_2]
    var_7 = 31
    var_8 = [var_0, var_1, var_7]
    var_9 = [var_0, var_4, var_1]

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_1]
    var_6 = [var_0, var_1, var_2]
    var_7 = 29
    var_8 = [var_0, var_1, var_7]
    var_9 = 1
    var_10 = [var_0, var_4, var_9]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_register_successfully_adds_new_dcc. Retrieved 6/12 statements.
# Partially parsed test_register_raises_error_on_duplicate_main_name. Retrieved 9/18 statements.
# Partially parsed test_register_raises_error_on_duplicate_altname_in_main_buffer. Retrieved 9/18 statements.
# Partially parsed test_register_raises_error_on_duplicate_altname_in_alt_buffer. Retrieved 10/19 statements.
# Partially parsed test_register_raises_error_when_main_name_exists_as_altname. Retrieved 9/18 statements.
# Partially parsed test_register_adds_all_altnames_to_buffer. Retrieved 8/15 statements.
# Partially parsed test_register_main_buffer_contains_dcc_by_main_name. Retrieved 6/10 statements.
# Partially parsed test_register_alt_buffer_contains_dcc_by_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_empty_altnames_does_not_add_to_alt_buffer. Retrieved 7/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'Other'
    var_7 = {var_6}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'SecondDCC'
    var_7 = {var_1}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = 'AltName'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'SecondDCC'
    var_8 = {var_2}
    var_9 = set()
    var_10 = '0.3'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = 'SecondDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = set()
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = 'Alt3'
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = '0.5'
    var_8 = [var_7]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = var_0._buffer_main['TestDCC']
    var_8 = 'TestDCC'
    var_9 = bool('TestDCC' not in var_0._buffer_altn)
    assert var_9 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TDCC'
    var_3 = 'Test'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = var_0._buffer_altn['TDCC']
    var_9 = var_0._buffer_altn['Test']
    var_10 = 'TDCC'
    var_11 = bool('TDCC' not in var_0._buffer_main)
    assert var_11 is True
    var_12 = 'Test'
    var_13 = bool('Test' not in var_0._buffer_main)
    assert var_13 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = var_0._buffer_altn
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0._buffer_main['TestDCC']



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_register_successfully_adds_new_dcc. Retrieved 6/12 statements.
# Partially parsed test_register_raises_error_if_main_name_already_registered. Retrieved 9/18 statements.
# Partially parsed test_register_raises_error_if_altname_conflicts_with_main_name. Retrieved 9/18 statements.
# Partially parsed test_register_raises_error_if_altname_conflicts_with_existing_altname. Retrieved 10/19 statements.
# Partially parsed test_register_adds_all_altnames_to_buffer. Retrieved 7/13 statements.
# Partially parsed test_register_does_not_modify_existing_registries. Retrieved 11/23 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TD'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'Other'
    var_7 = {var_6}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'AnotherDCC'
    var_7 = {var_1}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'Alt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'AnotherDCC'
    var_8 = {var_2}
    var_9 = set()
    var_10 = '0.3'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'First'
    var_2 = 'F1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.1'
    var_6 = [var_5]
    var_7 = 'Second'
    var_8 = 'S1'
    var_9 = {var_8}
    var_10 = set()
    var_11 = '0.2'
    var_12 = [var_11]



# Parsed testcases at query #6
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dcfc_30_360_us_basic. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_leap_year. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_last_day_of_month_start. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_last_day_of_month_both. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_360_us_d1_31_d2_31. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_d1_30_d2_31. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_d1_31_d2_30. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_cross_year. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16666666666667'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.16944444444444'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08333333333333'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.33333333333333'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = '0.07777777777778'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = '0.07777777777778'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = '0.07777777777778'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.08333333333333'
    var_8 = [var_7]
    var_9 = 14



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_register_raises_type_error_when_altname_conflict. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'DCC1'
    var_1 = 'ALT1'
    var_2 = {var_1}
    var_3 = set()
    var_4 = '0'
    var_5 = [var_4]
    var_6 = 'DCC2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = [var_4]
    var_10 = module_0.DCCRegistryMachinery()
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_last_day_of_month_true. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_december. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_april. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_example2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example4. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31_adjustment. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_30_asof_day_31_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_360_isda_no_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_360_isda_cross_year. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_360_isda_leap_year_feb29. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_360_isda_same_date. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = '0.16666666666667'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '0.16944444444444'
    var_10 = [var_9]
    var_11 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '1.08333333333333'
    var_10 = [var_9]
    var_11 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '1.33333333333333'
    var_10 = [var_9]
    var_11 = 14

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 30
    var_9 = var_5 - var_8
    var_10 = var_4 - var_1
    var_11 = var_8 * var_10
    var_12 = var_9 + var_11
    var_13 = 360
    var_14 = var_0 - var_0
    var_15 = var_13 * var_14
    var_16 = var_12 + var_15
    var_17 = [var_16]
    var_18 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = var_2 - var_2
    var_9 = var_4 - var_1
    var_10 = var_2 * var_9
    var_11 = var_8 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 3
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 6
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_0, var_4, var_2]
    var_7 = var_2 - var_2
    var_8 = 30
    var_9 = var_4 - var_1
    var_10 = var_8 * var_9
    var_11 = var_7 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 3
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = var_2 - var_2
    var_9 = 30
    var_10 = var_5 - var_1
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11
    var_13 = 360
    var_14 = var_4 - var_0
    var_15 = var_13 * var_14
    var_16 = var_12 + var_15
    var_17 = [var_16]
    var_18 = [var_13]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 30
    var_9 = var_8 - var_2
    var_10 = var_4 - var_1
    var_11 = var_8 * var_10
    var_12 = var_9 + var_11
    var_13 = 360
    var_14 = var_0 - var_0
    var_15 = var_13 * var_14
    var_16 = var_12 + var_15
    var_17 = [var_16]
    var_18 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 10
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = 0
    var_7 = [var_6]
    var_8 = 360
    var_9 = [var_8]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_30_e_360_example1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_360_example2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_example3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_example4. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_start_day_31_adjustment. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_e_360_asof_day_31_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_360_both_days_31_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_360_no_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_360_cross_year. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_e_360_same_date. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = '0.16666666666667'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '0.16944444444444'
    var_10 = [var_9]
    var_11 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '1.08333333333333'
    var_10 = [var_9]
    var_11 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '1.33055555555556'
    var_10 = [var_9]
    var_11 = 14

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 30
    var_9 = var_5 - var_8
    var_10 = var_4 - var_1
    var_11 = var_8 * var_10
    var_12 = var_9 + var_11
    var_13 = 360
    var_14 = var_0 - var_0
    var_15 = var_13 * var_14
    var_16 = var_12 + var_15
    var_17 = [var_16]
    var_18 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = var_2 - var_2
    var_9 = var_4 - var_1
    var_10 = var_2 * var_9
    var_11 = var_8 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_0, var_4, var_2]
    var_7 = 30
    var_8 = var_7 - var_7
    var_9 = var_4 - var_1
    var_10 = var_7 * var_9
    var_11 = var_8 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = var_5 - var_2
    var_9 = var_4 - var_1
    var_10 = var_2 * var_9
    var_11 = var_8 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = var_2 - var_2
    var_9 = 30
    var_10 = var_5 - var_1
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11
    var_13 = 360
    var_14 = var_4 - var_0
    var_15 = var_13 * var_14
    var_16 = var_12 + var_15
    var_17 = [var_16]
    var_18 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = 0
    var_7 = [var_6]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcfc_nl_365_basic_calculation. Retrieved 7/13 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day_in_range. Retrieved 8/14 statements.
# Partially parsed test_dcfc_nl_365_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_nl_365_another_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_nl_365_same_start_and_asof. Retrieved 3/8 statements.
# Partially parsed test_dcfc_nl_365_no_leap_day_in_range. Retrieved 6/13 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day_excluded. Retrieved 7/14 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day_included. Retrieved 6/13 statements.
# Partially parsed test_dcfc_nl_365_crossing_multiple_leap_years. Retrieved 7/17 statements.
# Partially parsed test_dcfc_nl_365_freq_parameter_ignored. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16986301369863'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.16986301369863'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08219178082192'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.32602739726027'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = '364'
    var_7 = [var_6]
    var_8 = '365'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]
    var_7 = '2'
    var_8 = [var_7]
    var_9 = '365'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '365'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2016
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2020
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = '365'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '2'
    var_8 = [var_7]
    var_9 = '0.16986301369863'
    var_10 = [var_9]
    var_11 = 14



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_same_year_annual. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_semiannual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semiannual_before_midyear. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semiannual_early_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_start_midyear. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_near_year_end. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semiannual_december_start. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semiannual_end_of_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_eom_handling. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_month_end_adjustment. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_frequency_decimal. Retrieved 5/10 statements.
# Partially parsed test_last_payment_date_start_date_returned. Retrieved 2/6 statements.
# Partially parsed test_last_payment_date_before_first_payment. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = 7
    var_9 = [var_3, var_8, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 8
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = 7
    var_9 = [var_3, var_8, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 4
    var_5 = 30
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 4
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 10
    var_5 = 6
    var_6 = [var_3, var_4, var_5]
    var_7 = 4
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 12
    var_2 = 9
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 4
    var_6 = [var_4, var_1, var_5]
    var_7 = 1
    var_8 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2016
    var_5 = 1
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = 2
    var_9 = 2015
    var_10 = [var_9, var_1, var_2]

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 31
    var_6 = [var_4, var_1, var_5]
    var_7 = 2
    var_8 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 3
    var_6 = 15
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 3
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = 1
    var_9 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_1]
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_1, var_2]



# Parsed testcases at query #14
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcfc_act_365_l_basic_calculation. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_365_l_leap_year. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_l_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_l_another_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_l_same_date. Retrieved 3/8 statements.
# Partially parsed test_dcfc_act_365_l_non_leap_year_denominator. Retrieved 6/13 statements.
# Partially parsed test_dcfc_act_365_l_leap_year_denominator. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16939890710383'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.17213114754098'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08196721311475'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.32876712328767'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = '364'
    var_7 = [var_6]
    var_8 = '365'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = '365'
    var_7 = [var_6]
    var_8 = '366'
    var_9 = [var_8]



# Parsed testcases at query #16
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_next_payment_date_no_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_with_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_frequency_two. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_frequency_two_with_eom. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_frequency_four. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_frequency_four_with_eom. Retrieved 4/7 statements.
# Partially parsed test_next_payment_date_invalid_eom. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_decimal_frequency. Retrieved 5/9 statements.
# Partially parsed test_next_payment_date_decimal_frequency_with_eom. Retrieved 5/9 statements.
# Partially parsed test_next_payment_date_frequency_six. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_frequency_six_with_eom. Retrieved 5/8 statements.
# Partially parsed test_next_payment_date_leap_year. Retrieved 7/10 statements.
# Partially parsed test_next_payment_date_leap_year_with_eom. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = None
    var_4 = 2015
    var_5 = [var_4, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = 2015
    var_5 = [var_4, var_1, var_3]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = None
    var_5 = 7
    var_6 = [var_0, var_5, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = 31
    var_5 = 7
    var_6 = [var_0, var_5, var_4]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 4
    var_4 = None
    var_5 = [var_0, var_3, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 4
    var_4 = 30
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2014
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 31
    var_5 = 2015
    var_6 = [var_5, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = None
    var_6 = 2026
    var_7 = [var_6, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = 10
    var_6 = 2026
    var_7 = [var_6, var_1, var_5]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = None
    var_5 = 3
    var_6 = [var_0, var_5, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = 28
    var_5 = 3
    var_6 = [var_0, var_5, var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = None
    var_6 = 2021
    var_7 = 28
    var_8 = [var_6, var_1, var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2021
    var_6 = 28
    var_7 = [var_5, var_1, var_6]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_same_year_annual. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_before_mid_year. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_early_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_start_mid_year. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_december_start. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semi_annual_december_end_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_with_eom_override. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_frequency_decimal. Retrieved 6/11 statements.
# Partially parsed test_last_payment_date_edge_case_negative_year. Retrieved 3/7 statements.
# Partially parsed test_last_payment_date_monthly_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_bi_monthly_frequency. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 1
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = 1
    var_7 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = 7
    var_9 = [var_3, var_8, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 8
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = 7
    var_9 = [var_3, var_8, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 4
    var_5 = 30
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 4
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = 1
    var_9 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 10
    var_5 = 6
    var_6 = [var_3, var_4, var_5]
    var_7 = 4
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 12
    var_2 = 9
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 4
    var_6 = [var_4, var_1, var_5]
    var_7 = 1
    var_8 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2016
    var_5 = 1
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = 2
    var_9 = 2015
    var_10 = [var_9, var_1, var_2]

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 31
    var_6 = [var_4, var_1, var_5]
    var_7 = 2
    var_8 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = [var_4, var_5, var_2]
    var_7 = 1
    var_8 = 15
    var_9 = 15
    var_10 = [var_4, var_1, var_9]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = '1'
    var_8 = [var_7]
    var_9 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_1, var_1, var_1]
    var_4 = 1

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = 20
    var_7 = [var_4, var_5, var_6]
    var_8 = 12
    var_9 = [var_4, var_5, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = 20
    var_7 = [var_4, var_5, var_6]
    var_8 = 6
    var_9 = 11
    var_10 = [var_4, var_9, var_2]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_coupon_basic_annual. Retrieved 13/25 statements.
# Partially parsed test_coupon_semi_annual. Retrieved 14/26 statements.
# Partially parsed test_coupon_quarterly. Retrieved 14/26 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/28 statements.
# Partially parsed test_coupon_asof_on_start. Retrieved 12/24 statements.
# Partially parsed test_coupon_asof_on_end. Retrieved 12/24 statements.
# Partially parsed test_coupon_fraction_zero. Retrieved 12/24 statements.
# Partially parsed test_coupon_fraction_full_period. Retrieved 12/24 statements.
# Partially parsed test_coupon_with_high_frequency. Retrieved 13/25 statements.
# Partially parsed test_coupon_negative_rate. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 7
    var_14 = [var_10, var_13, var_11]
    var_15 = 2021
    var_16 = [var_15, var_11, var_11]
    var_17 = 1
    var_18 = '25'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = '2000'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '0.03'
    var_9 = [var_8]
    var_10 = 2019
    var_11 = 6
    var_12 = 15
    var_13 = [var_10, var_11, var_12]
    var_14 = 9
    var_15 = [var_10, var_14, var_12]
    var_16 = 2020
    var_17 = [var_16, var_11, var_12]
    var_18 = 2
    var_19 = '15'
    var_20 = [var_19]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.1'
    var_4 = [var_3]
    var_5 = '5000'
    var_6 = [var_5]
    var_7 = 'GBP'
    var_8 = '0.04'
    var_9 = [var_8]
    var_10 = 2021
    var_11 = 3
    var_12 = 10
    var_13 = [var_10, var_11, var_12]
    var_14 = 4
    var_15 = [var_10, var_14, var_12]
    var_16 = 2022
    var_17 = [var_16, var_11, var_12]
    var_18 = 4
    var_19 = '20'
    var_20 = [var_19]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.3'
    var_4 = [var_3]
    var_5 = '1500'
    var_6 = [var_5]
    var_7 = 'JPY'
    var_8 = '0.02'
    var_9 = [var_8]
    var_10 = 2018
    var_11 = 2
    var_12 = 28
    var_13 = [var_10, var_11, var_12]
    var_14 = 5
    var_15 = 31
    var_16 = [var_10, var_14, var_15]
    var_17 = 2019
    var_18 = [var_17, var_11, var_12]
    var_19 = 1
    var_20 = 31
    var_21 = '9'
    var_22 = [var_21]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.0'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = [var_10, var_11, var_11]
    var_14 = 2021
    var_15 = [var_14, var_11, var_11]
    var_16 = 1
    var_17 = '0'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '1.0'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 2021
    var_14 = [var_13, var_11, var_11]
    var_15 = [var_13, var_11, var_11]
    var_16 = 1
    var_17 = '50'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.0'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = [var_10, var_11, var_11]
    var_14 = 2021
    var_15 = [var_14, var_11, var_11]
    var_16 = 2
    var_17 = '0'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '1.0'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 2021
    var_14 = [var_13, var_11, var_11]
    var_15 = [var_13, var_11, var_11]
    var_16 = 1
    var_17 = '50'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.08333'
    var_4 = [var_3]
    var_5 = '12000'
    var_6 = [var_5]
    var_7 = 'CAD'
    var_8 = '0.06'
    var_9 = [var_8]
    var_10 = 2022
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 2
    var_14 = [var_10, var_13, var_11]
    var_15 = 2023
    var_16 = [var_15, var_11, var_11]
    var_17 = 12
    var_18 = '60'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '-0.02'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 7
    var_14 = [var_10, var_13, var_11]
    var_15 = 2021
    var_16 = [var_15, var_11, var_11]
    var_17 = 1
    var_18 = '-10'
    var_19 = [var_18]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dcfc_act_act_basic. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_multi_year. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_act_same_day. Retrieved 3/8 statements.
# Partially parsed test_dcfc_act_act_one_day. Retrieved 5/12 statements.
# Partially parsed test_dcfc_act_act_non_leap_year. Retrieved 5/12 statements.
# Partially parsed test_dcfc_act_act_cross_year_boundary. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_full_leap_year. Retrieved 4/9 statements.
# Partially parsed test_dcfc_act_act_full_non_leap_year. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16942884946478'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.17216108990194'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08243131970956'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.32625945055768'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 366
    var_8 = [var_7]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 365
    var_8 = [var_7]

def test_case_0():
    var_0 = 2019
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2020
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '1'
    var_8 = [var_7]
    var_9 = 365
    var_10 = [var_9]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2021
    var_4 = [var_3, var_1, var_1]
    var_5 = '1'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2020
    var_4 = [var_3, var_1, var_1]
    var_5 = '1'
    var_6 = [var_5]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_nl_365_standard_period. Retrieved 7/13 statements.
# Partially parsed test_dcfc_nl_365_leap_day_in_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_nl_365_longer_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_nl_365_another_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_nl_365_same_start_and_asof. Retrieved 3/8 statements.
# Partially parsed test_dcfc_nl_365_one_day_period. Retrieved 5/12 statements.
# Partially parsed test_dcfc_nl_365_period_with_leap_day_excluded. Retrieved 7/14 statements.
# Partially parsed test_dcfc_nl_365_period_including_leap_day. Retrieved 7/14 statements.
# Partially parsed test_dcfc_nl_365_period_spanning_leap_year. Retrieved 5/12 statements.
# Partially parsed test_dcfc_nl_365_asof_before_start. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16986301369863'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.16986301369863'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08219178082192'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.32602739726027'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '365'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]
    var_7 = '2'
    var_8 = [var_7]
    var_9 = '365'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]
    var_7 = '2'
    var_8 = [var_7]
    var_9 = '365'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2019
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2020
    var_5 = [var_4, var_1, var_2]
    var_6 = '365'
    var_7 = [var_6]
    var_8 = [var_6]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = '0'
    var_6 = [var_5]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_interest_calculates_correctly. Retrieved 14/28 statements.
# Partially parsed test_interest_without_end_uses_asof. Retrieved 12/25 statements.
# Partially parsed test_interest_returns_zero_when_dates_out_of_order. Retrieved 14/28 statements.
# Partially parsed test_interest_with_freq_passed_to_fraction_method. Retrieved 15/32 statements.
# Partially parsed test_interest_with_zero_fraction_returns_zero_interest. Retrieved 11/25 statements.


def test_case_0():
    var_0 = 'Actual/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = 6
    var_13 = 30
    var_14 = 12
    var_15 = 31
    var_16 = '25'
    var_17 = [var_16]

def test_case_0():
    var_0 = 'Actual/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = '2000'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '0.1'
    var_9 = [var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = 3
    var_13 = 31
    var_14 = '50'
    var_15 = [var_14]

def test_case_0():
    var_0 = 'Actual/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2023
    var_11 = 6
    var_12 = 30
    var_13 = 1
    var_14 = 12
    var_15 = 31
    var_16 = '0'
    var_17 = [var_16]

def test_case_0():
    var_0 = None
    var_1 = 'Actual/360'
    var_2 = set()
    var_3 = set()
    var_4 = '1500'
    var_5 = [var_4]
    var_6 = 'GBP'
    var_7 = '0.04'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = 9
    var_12 = 30
    var_13 = 12
    var_14 = 31
    var_15 = '2'
    var_16 = [var_15]
    var_17 = '18'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'Actual/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0'
    var_4 = [var_3]
    var_5 = '5000'
    var_6 = [var_5]
    var_7 = 'JPY'
    var_8 = '0.02'
    var_9 = [var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = 12
    var_13 = 31
    var_14 = [var_3]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dcfc_30_e_360_basic_examples. Retrieved 18/38 statements.
# Partially parsed test_dcfc_30_e_360_start_day_31_adjustment. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_e_360_asof_day_31_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_360_both_days_31_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_360_no_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_360_cross_year. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_e_360_leap_year_feb_29. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_360_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_e_360_negative_days. Retrieved 13/20 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_0, var_1, var_2]
    var_8 = 29
    var_9 = [var_4, var_5, var_8]
    var_10 = 10
    var_11 = 31
    var_12 = [var_0, var_10, var_11]
    var_13 = 11
    var_14 = 30
    var_15 = [var_4, var_13, var_14]
    var_16 = 1
    var_17 = [var_4, var_5, var_16]
    var_18 = 2009
    var_19 = 5
    var_20 = [var_18, var_19, var_11]
    var_21 = 14
    var_22 = '0.16666666666667'
    var_23 = [var_22]
    var_24 = '0.16944444444444'
    var_25 = [var_24]
    var_26 = '1.08333333333333'
    var_27 = [var_26]
    var_28 = '1.33055555555556'
    var_29 = [var_28]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 30
    var_9 = var_5 - var_8
    var_10 = var_4 - var_1
    var_11 = var_8 * var_10
    var_12 = var_9 + var_11
    var_13 = 360
    var_14 = var_0 - var_0
    var_15 = var_13 * var_14
    var_16 = var_12 + var_15
    var_17 = [var_16]
    var_18 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = var_2 - var_2
    var_9 = var_4 - var_1
    var_10 = var_2 * var_9
    var_11 = var_8 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_0, var_4, var_2]
    var_7 = 30
    var_8 = var_7 - var_7
    var_9 = var_4 - var_1
    var_10 = var_7 * var_9
    var_11 = var_8 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_0, var_4, var_2]
    var_7 = var_2 - var_2
    var_8 = 30
    var_9 = var_4 - var_1
    var_10 = var_8 * var_9
    var_11 = var_7 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = var_2 - var_2
    var_9 = 30
    var_10 = var_5 - var_1
    var_11 = var_9 * var_10
    var_12 = var_8 + var_11
    var_13 = 360
    var_14 = var_4 - var_0
    var_15 = var_13 * var_14
    var_16 = var_12 + var_15
    var_17 = [var_16]
    var_18 = [var_13]

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = var_4 - var_2
    var_8 = 30
    var_9 = var_1 - var_1
    var_10 = var_8 * var_9
    var_11 = var_7 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = 0
    var_7 = [var_6]

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = var_4 - var_2
    var_8 = 30
    var_9 = var_1 - var_1
    var_10 = var_8 * var_9
    var_11 = var_7 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_date_range_with_single_day. Retrieved 3/9 statements.
# Partially parsed test_get_date_range_with_multiple_days. Retrieved 6/15 statements.
# Partially parsed test_get_date_range_with_zero_days. Retrieved 3/7 statements.
# Partially parsed test_get_date_range_with_leap_year. Retrieved 6/14 statements.
# Partially parsed test_get_date_range_with_month_crossing. Retrieved 5/13 statements.
# Partially parsed test_get_date_range_with_year_crossing. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 5
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]
    var_8 = 3
    var_9 = [var_0, var_1, var_8]
    var_10 = 4
    var_11 = [var_0, var_1, var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = []

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_1]
    var_6 = [var_0, var_1, var_2]
    var_7 = 29
    var_8 = [var_0, var_1, var_7]
    var_9 = 1
    var_10 = [var_0, var_4, var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 3
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_1, var_2]
    var_8 = [var_0, var_4, var_1]
    var_9 = [var_0, var_4, var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = 2
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_0, var_1, var_2]
    var_9 = 31
    var_10 = [var_0, var_1, var_9]
    var_11 = [var_4, var_5, var_5]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_exact_main_name. Retrieved 4/6 statements.
# Partially parsed test_find_exact_alt_name. Retrieved 5/7 statements.
# Partially parsed test_find_stripped_uppercase_main_name. Retrieved 5/7 statements.
# Partially parsed test_find_stripped_uppercase_alt_name. Retrieved 6/8 statements.
# Partially parsed test_find_case_insensitive_main_name. Retrieved 5/7 statements.
# Partially parsed test_find_case_insensitive_alt_name. Retrieved 6/8 statements.
# Partially parsed test_find_with_whitespace_main_name. Retrieved 5/7 statements.
# Partially parsed test_find_with_whitespace_alt_name. Retrieved 6/8 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = []
    var_4 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = [var_2]
    var_4 = []
    var_5 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = []
    var_4 = ' act/act '
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACTUAL/ACTUAL'
    var_3 = [var_2]
    var_4 = []
    var_5 = ' actual/actual '
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Unknown'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = []
    var_4 = 'act/act'
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACTUAL/ACTUAL'
    var_3 = [var_2]
    var_4 = []
    var_5 = 'actual/actual'
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = []
    var_4 = '  ACT/ACT  '
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACTUAL/ACTUAL'
    var_3 = [var_2]
    var_4 = []
    var_5 = '  ACTUAL/ACTUAL  '
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = ''
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '   '
    var_2 = var_0.find(var_1)
    assert var_2 is None



# Parsed testcases at query #6
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_altn
    var_2 = bool(var_0._buffer_altn == {})
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_coupon_basic_calculation. Retrieved 14/29 statements.
# Partially parsed test_coupon_with_eom_adjustment. Retrieved 15/30 statements.
# Partially parsed test_coupon_zero_fraction. Retrieved 12/27 statements.
# Partially parsed test_coupon_fraction_greater_than_one. Retrieved 14/29 statements.
# Partially parsed test_coupon_negative_rate. Retrieved 13/28 statements.
# Partially parsed test_coupon_principal_zero. Retrieved 14/29 statements.
# Partially parsed test_coupon_freq_as_int. Retrieved 13/27 statements.
# Partially parsed test_coupon_eom_31_invalid_month. Retrieved 16/31 statements.
# Partially parsed test_coupon_asof_equals_start. Retrieved 12/27 statements.
# Partially parsed test_coupon_asof_equals_end. Retrieved 12/27 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 6
    var_14 = 30
    var_15 = [var_10, var_13, var_14]
    var_16 = 2021
    var_17 = [var_16, var_11, var_11]
    var_18 = '2'
    var_19 = [var_18]
    var_20 = None
    var_21 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = '2000'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '0.03'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = 31
    var_13 = [var_10, var_11, var_12]
    var_14 = 3
    var_15 = 15
    var_16 = [var_10, var_14, var_15]
    var_17 = 7
    var_18 = [var_10, var_17, var_12]
    var_19 = '4'
    var_20 = [var_19]
    var_21 = 31
    var_22 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0'
    var_4 = [var_3]
    var_5 = '500'
    var_6 = [var_5]
    var_7 = 'GBP'
    var_8 = '0.02'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = [var_10, var_11, var_11]
    var_14 = 7
    var_15 = [var_10, var_14, var_11]
    var_16 = '2'
    var_17 = [var_16]
    var_18 = None
    var_19 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '1.5'
    var_4 = [var_3]
    var_5 = '1500'
    var_6 = [var_5]
    var_7 = 'JPY'
    var_8 = '0.01'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 12
    var_14 = 31
    var_15 = [var_10, var_13, var_14]
    var_16 = 2021
    var_17 = [var_16, var_11, var_11]
    var_18 = '1'
    var_19 = [var_18]
    var_20 = None
    var_21 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.3'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '-0.02'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 4
    var_14 = [var_10, var_13, var_11]
    var_15 = 7
    var_16 = [var_10, var_15, var_11]
    var_17 = '2'
    var_18 = [var_17]
    var_19 = None
    var_20 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.6'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 6
    var_14 = 30
    var_15 = [var_10, var_13, var_14]
    var_16 = 2021
    var_17 = [var_16, var_11, var_11]
    var_18 = '2'
    var_19 = [var_18]
    var_20 = None
    var_21 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.4'
    var_4 = [var_3]
    var_5 = '1200'
    var_6 = [var_5]
    var_7 = 'CAD'
    var_8 = '0.04'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 5
    var_14 = [var_10, var_13, var_11]
    var_15 = 10
    var_16 = [var_10, var_15, var_11]
    var_17 = 2
    var_18 = None
    var_19 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.33'
    var_4 = [var_3]
    var_5 = '800'
    var_6 = [var_5]
    var_7 = 'AUD'
    var_8 = '0.025'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 2
    var_12 = 29
    var_13 = [var_10, var_11, var_12]
    var_14 = 4
    var_15 = 30
    var_16 = [var_10, var_14, var_15]
    var_17 = 8
    var_18 = 31
    var_19 = [var_10, var_17, var_18]
    var_20 = '4'
    var_21 = [var_20]
    var_22 = 31
    var_23 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.0'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = [var_10, var_11, var_11]
    var_14 = 7
    var_15 = [var_10, var_14, var_11]
    var_16 = '2'
    var_17 = [var_16]
    var_18 = None
    var_19 = [var_3]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '1.0'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 7
    var_14 = [var_10, var_13, var_11]
    var_15 = [var_10, var_13, var_11]
    var_16 = '2'
    var_17 = [var_16]
    var_18 = None
    var_19 = [var_3]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_register_successfully_adds_new_dcc. Retrieved 6/12 statements.
# Partially parsed test_register_raises_error_on_duplicate_main_name. Retrieved 9/18 statements.
# Partially parsed test_register_raises_error_on_duplicate_altname_in_main_buffer. Retrieved 9/18 statements.
# Partially parsed test_register_raises_error_on_duplicate_altname_in_alt_buffer. Retrieved 10/19 statements.
# Partially parsed test_register_adds_all_altnames_to_alt_buffer. Retrieved 8/14 statements.
# Partially parsed test_register_does_not_affect_existing_registry_when_error_occurs. Retrieved 10/22 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'Other'
    var_7 = {var_6}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'SecondDCC'
    var_7 = {var_1}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = 'AltName'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'SecondDCC'
    var_8 = {var_2}
    var_9 = set()
    var_10 = '0.3'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Alt1'
    var_2 = 'Alt2'
    var_3 = 'Alt3'
    var_4 = {var_1, var_2, var_3}
    var_5 = 'TestDCC'
    var_6 = set()
    var_7 = '0.5'
    var_8 = [var_7]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = 'Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'SecondDCC'
    var_8 = {var_1}
    var_9 = set()
    var_10 = '0.3'
    var_11 = [var_10]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_basic_examples. Retrieved 18/42 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31_adjustment. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31_adjustment. Retrieved 7/28 statements.
# Partially parsed test_dcfc_30_e_plus_360_both_days_31_adjustment. Retrieved 7/29 statements.
# Partially parsed test_dcfc_30_e_plus_360_no_adjustment. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_plus_360_cross_year. Retrieved 14/21 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_e_plus_360_leap_year_feb29. Retrieved 13/20 statements.
# Partially parsed test_dcfc_30_e_plus_360_leap_year_feb29_asof_31_adjustment. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.16666666666667'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_2]
    var_11 = 29
    var_12 = [var_4, var_5, var_11]
    var_13 = '0.16944444444444'
    var_14 = [var_13]
    var_15 = 10
    var_16 = 31
    var_17 = [var_0, var_15, var_16]
    var_18 = 11
    var_19 = 30
    var_20 = [var_4, var_18, var_19]
    var_21 = '1.08333333333333'
    var_22 = [var_21]
    var_23 = 1
    var_24 = [var_4, var_5, var_23]
    var_25 = 2009
    var_26 = 5
    var_27 = [var_25, var_26, var_16]
    var_28 = '1.33333333333333'
    var_29 = [var_28]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = 30
    var_8 = var_5 - var_7
    var_9 = var_4 - var_1
    var_10 = var_7 * var_9
    var_11 = var_8 + var_10
    var_12 = 360
    var_13 = var_0 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 3
    var_8 = [var_0, var_7, var_1]
    var_9 = 360
    var_10 = [var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_2]
    var_6 = 30
    var_7 = [var_0, var_1, var_6]
    var_8 = 4
    var_9 = [var_0, var_8, var_1]
    var_10 = 360
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = var_5 - var_2
    var_8 = var_4 - var_1
    var_9 = var_2 * var_8
    var_10 = var_7 + var_9
    var_11 = 360
    var_12 = var_0 - var_0
    var_13 = var_11 * var_12
    var_14 = var_10 + var_13
    var_15 = [var_14]
    var_16 = [var_11]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = var_2 - var_2
    var_8 = 30
    var_9 = var_5 - var_1
    var_10 = var_8 * var_9
    var_11 = var_7 + var_10
    var_12 = 360
    var_13 = var_4 - var_0
    var_14 = var_12 * var_13
    var_15 = var_11 + var_14
    var_16 = [var_15]
    var_17 = [var_12]

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = [var_4]

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = var_4 - var_2
    var_7 = 30
    var_8 = var_1 - var_1
    var_9 = var_7 * var_8
    var_10 = var_6 + var_9
    var_11 = 360
    var_12 = var_0 - var_0
    var_13 = var_11 * var_12
    var_14 = var_10 + var_13
    var_15 = [var_14]
    var_16 = [var_11]

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 4
    var_8 = 1
    var_9 = [var_0, var_7, var_8]
    var_10 = 30
    var_11 = 360
    var_12 = [var_11]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_same_year_annual. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_semiannual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semiannual_before_midyear. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semiannual_early_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_start_midyear. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semiannual_december_start. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semiannual_december_end. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_with_eom_override. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_monthly_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_biannual_frequency. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_edge_case_negative_year. Retrieved 3/7 statements.
# Partially parsed test_last_payment_date_invalid_date_handling. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_frequency_zero_division. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_frequency_decimal. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 1
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = 1
    var_7 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = 7
    var_9 = [var_3, var_8, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 8
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = 7
    var_9 = [var_3, var_8, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 4
    var_5 = 30
    var_6 = [var_3, var_4, var_5]
    var_7 = 2
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 4
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = 1
    var_9 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 10
    var_5 = 6
    var_6 = [var_3, var_4, var_5]
    var_7 = 4
    var_8 = [var_3, var_1, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 12
    var_2 = 9
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 4
    var_6 = [var_4, var_1, var_5]
    var_7 = 1
    var_8 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2016
    var_5 = 1
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = 2
    var_9 = 2015
    var_10 = [var_9, var_1, var_2]

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 31
    var_6 = [var_4, var_1, var_5]
    var_7 = 2
    var_8 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = [var_4, var_5, var_2]
    var_7 = 1
    var_8 = 15
    var_9 = 15
    var_10 = [var_4, var_1, var_9]

def test_case_0():
    var_0 = 2014
    var_1 = 3
    var_2 = 10
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = 20
    var_7 = [var_4, var_5, var_6]
    var_8 = 12
    var_9 = [var_4, var_5, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = 6
    var_9 = 8
    var_10 = [var_4, var_9, var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = 2
    var_3 = [var_2, var_0, var_0]
    var_4 = 1
    var_5 = [var_0, var_0, var_0]

def test_case_0():
    var_0 = 2014
    var_1 = 2
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = 1
    var_9 = 28
    var_10 = [var_0, var_1, var_9]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 0
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = '1'
    var_8 = [var_7]
    var_9 = [var_3, var_1, var_1]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcfc_nl_365_handles_leap_day_correctly. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16986301369863'
    var_8 = [var_7]
    var_9 = 14



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dcfc_30_360_german_basic. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_german_leap_year. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_31_day_start. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_feb_start. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_360_german_start_31_adjusted. Retrieved 8/26 statements.
# Partially parsed test_dcfc_30_360_german_asof_31_adjusted. Retrieved 8/26 statements.
# Partially parsed test_dcfc_30_360_german_feb_last_day_start. Retrieved 10/26 statements.
# Partially parsed test_dcfc_30_360_german_feb_last_day_asof_not_end. Retrieved 9/27 statements.
# Partially parsed test_dcfc_30_360_german_feb_last_day_asof_is_end. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16666666666667'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.16944444444444'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08333333333333'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.33055555555556'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = 30
    var_8 = 30
    var_9 = 360
    var_10 = [var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 3
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 30
    var_6 = [var_0, var_4, var_5]
    var_7 = 5
    var_8 = [var_0, var_7, var_2]
    var_9 = 30
    var_10 = 360
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 30
    var_8 = 30
    var_9 = var_8 - var_7
    var_10 = 30
    var_11 = 360
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = 3
    var_8 = [var_0, var_7, var_2]
    var_9 = 30
    var_10 = 30
    var_11 = 360
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = 30
    var_8 = 360
    var_9 = [var_8]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_register_new_dcc_successfully. Retrieved 6/11 statements.
# Partially parsed test_register_duplicate_main_name_raises_error. Retrieved 9/18 statements.
# Partially parsed test_register_duplicate_altname_raises_error. Retrieved 10/19 statements.
# Partially parsed test_register_altname_conflict_with_main_name_raises_error. Retrieved 9/18 statements.
# Partially parsed test_register_main_name_conflict_with_altname_raises_error. Retrieved 9/18 statements.
# Partially parsed test_register_multiple_altnames_successfully. Retrieved 7/13 statements.
# Partially parsed test_register_empty_altnames_successfully. Retrieved 6/12 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'Alt'
    var_7 = {var_6}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = 'AltName'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'SecondDCC'
    var_8 = {var_2}
    var_9 = set()
    var_10 = '0.3'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'MainDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'AnotherDCC'
    var_7 = {var_1}
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'FirstDCC'
    var_2 = 'AltName'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = set()
    var_8 = set()
    var_9 = '0.3'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'NonExistent'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_365_a_leap_year. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_a_another_long_period. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_a_same_day. Retrieved 3/8 statements.
# Partially parsed test_dcfc_act_365_a_one_day. Retrieved 5/12 statements.
# Partially parsed test_dcfc_act_365_a_non_leap_year. Retrieved 6/13 statements.
# Partially parsed test_dcfc_act_365_a_leap_year_full. Retrieved 6/13 statements.
# Partially parsed test_dcfc_act_365_a_leap_day_in_range. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_365_a_no_leap_day_in_range. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16986301369863'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.17213114754098'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08196721311475'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.32513661202186'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 365
    var_8 = [var_7]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = '364'
    var_7 = [var_6]
    var_8 = 365
    var_9 = [var_8]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = '365'
    var_7 = [var_6]
    var_8 = 366
    var_9 = [var_8]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]
    var_7 = '2'
    var_8 = [var_7]
    var_9 = 366
    var_10 = [var_9]

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]
    var_7 = '1'
    var_8 = [var_7]
    var_9 = 365
    var_10 = [var_9]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_register_raises_error_when_main_name_already_registered. Retrieved 8/17 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Act/Act'
    var_1 = set()
    var_2 = set()
    var_3 = '0.1'
    var_4 = [var_3]
    var_5 = set()
    var_6 = set()
    var_7 = '0.2'
    var_8 = [var_7]
    var_9 = module_0.DCCRegistryMachinery()
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dcfc_30_e_360_start_day_31_adjusts_to_30. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '1.08333333333333'
    var_10 = [var_9]
    var_11 = 14



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dcfc_30_e_360_example1. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_e_360_example2. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_360_example3. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_360_example4. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_360_start_day_31. Retrieved 7/28 statements.
# Partially parsed test_dcfc_30_e_360_asof_day_31. Retrieved 7/28 statements.
# Partially parsed test_dcfc_30_e_360_both_days_31. Retrieved 6/28 statements.
# Partially parsed test_dcfc_30_e_360_no_adjustment. Retrieved 6/26 statements.
# Partially parsed test_dcfc_30_e_360_negative_days. Retrieved 6/28 statements.
# Partially parsed test_dcfc_30_e_360_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_e_360_leap_year. Retrieved 6/26 statements.
# Partially parsed test_dcfc_30_e_360_year_boundary. Retrieved 7/29 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16666666666667'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.16944444444444'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 11
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.08333333333333'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 5
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = '1.33055555555556'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 30
    var_9 = [var_0, var_1, var_8]
    var_10 = 360
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 30
    var_9 = [var_0, var_4, var_8]
    var_10 = 360
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_0, var_4, var_2]
    var_7 = 30
    var_8 = [var_0, var_1, var_7]
    var_9 = [var_0, var_4, var_7]
    var_10 = 360
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 360
    var_9 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_0, var_4, var_2]
    var_7 = [var_0, var_1, var_2]
    var_8 = [var_0, var_4, var_2]
    var_9 = 30
    var_10 = 360
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = 0
    var_7 = [var_6]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = 30
    var_8 = 360
    var_9 = [var_8]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = 30
    var_9 = [var_0, var_1, var_8]
    var_10 = [var_4, var_5, var_8]
    var_11 = 360
    var_12 = [var_11]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_last_day_of_month_true. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_leap. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_december. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_april. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30



# Parsed testcases at query #20
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_strips_and_uppercases_name_as_last_resort. Retrieved 5/7 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = []
    var_4 = '  act/act  '
    var_5 = var_0.find(var_4)



# Parsed testcases at query #22
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_register_raises_type_error_when_altname_already_registered. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test1'
    var_1 = 'Alt1'
    var_2 = {var_1}
    var_3 = set()
    var_4 = '0'
    var_5 = [var_4]
    var_6 = 'Test2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = [var_4]
    var_10 = module_0.DCCRegistryMachinery()
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #24
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_register_raises_type_error_when_altname_conflict. Retrieved 22/43 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Currency'
    var_1 = 'code'
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]
    var_5 = 'DCC'
    var_6 = 'name'
    var_7 = 'altnames'
    var_8 = 'currencies'
    var_9 = 'calculate_fraction_method'
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = module_0.DCCRegistryMachinery()
    var_12 = 'Act/Act'
    var_13 = 'Actual/Actual'
    var_14 = {var_13}
    var_15 = set()
    var_16 = [var_3]
    var_17 = 'Act/360'
    var_18 = 'Actual/360'
    var_19 = {var_18}
    var_20 = set()
    var_21 = [var_3]
    var_22 = 'NewDCC'
    var_23 = {var_18}
    var_24 = set()
    var_25 = [var_3]
    var_26 = bool(False)
    assert var_26 is True



