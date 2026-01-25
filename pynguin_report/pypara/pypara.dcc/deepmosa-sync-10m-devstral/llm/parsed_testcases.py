####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_last_day_of_month_for_january_31st. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_for_february_28th_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_for_february_29th_leap_year. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_for_april_30th. Retrieved 3/5 statements.
# Partially parsed test_is_not_last_day_of_month_for_january_30th. Retrieved 3/5 statements.
# Partially parsed test_is_not_last_day_of_month_for_february_27th_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_is_not_last_day_of_month_for_april_29th. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31

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

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 27

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 29



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_register_successful_registration. Retrieved 6/12 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_duplicate_alternative_name. Retrieved 9/18 statements.
# Partially parsed test_register_alternative_name_conflict_with_main. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TEST'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.5
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 0.5
    var_5 = [var_4]
    var_6 = set()
    var_7 = set()
    var_8 = [var_4]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'TEST2'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = 'Test2'
    var_8 = set()
    var_9 = set()
    var_10 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = set()
    var_3 = set()
    var_4 = 0.5
    var_5 = [var_4]
    var_6 = 'Test2'
    var_7 = 'TEST1'
    var_8 = {var_7}
    var_9 = set()
    var_10 = [var_4]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dcfc_nl_365. Retrieved 17/41 statements.


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
    var_22 = '0.16986301369863'
    var_23 = [var_22]
    var_24 = [var_22]
    var_25 = '1.08219178082192'
    var_26 = [var_25]
    var_27 = '1.32602739726027'
    var_28 = [var_27]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_existing_dcc_by_main_name. Retrieved 6/10 statements.
# Partially parsed test_find_existing_dcc_by_alternative_name. Retrieved 6/10 statements.
# Partially parsed test_find_existing_dcc_case_insensitive. Retrieved 7/11 statements.
# Partially parsed test_find_existing_dcc_with_whitespace. Retrieved 7/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'act/act'
    var_7 = var_0.find(var_6)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = '  Act/Act  '
    var_7 = var_0.find(var_6)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_existing_dcc_by_main_name. Retrieved 7/9 statements.
# Partially parsed test_find_existing_dcc_by_alternative_name. Retrieved 7/9 statements.
# Partially parsed test_find_with_whitespace_and_case_insensitivity. Retrieved 8/10 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = 0.5
    var_5 = lambda s, e, r: var_4
    var_6 = [var_1, var_3, var_5]
    var_7 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = 0.5
    var_5 = lambda s, e, r: var_4
    var_6 = [var_1, var_3, var_5]
    var_7 = var_0.find(var_2)

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
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = 0.5
    var_5 = lambda s, e, r: var_4
    var_6 = [var_1, var_3, var_5]
    var_7 = ' act/act '
    var_8 = var_0.find(var_7)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_returns_correct_dcc. Retrieved 12/14 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = [var_1]
    var_3 = 1
    var_4 = lambda s, e, r: var_3
    var_5 = lambda p, r, s, e, r: p
    var_6 = [var_1, var_2, var_4, var_5]
    var_7 = 'test'
    var_8 = var_0.find(var_7)
    var_9 = 'TEST'
    var_10 = var_0.find(var_9)
    var_11 = ' test '
    var_12 = var_0.find(var_11)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_coupon_basic_case. Retrieved 14/22 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/24 statements.
# Partially parsed test_coupon_zero_principal. Retrieved 13/21 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 14/22 statements.
# Partially parsed test_coupon_same_start_and_asof. Retrieved 13/21 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 1000
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 6
    var_14 = [var_10, var_13, var_11]
    var_15 = 2021
    var_16 = [var_15, var_11, var_11]
    var_17 = 2
    var_18 = 25

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 1000
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = 15
    var_13 = [var_10, var_11, var_12]
    var_14 = 6
    var_15 = [var_10, var_14, var_12]
    var_16 = 2021
    var_17 = [var_16, var_11, var_12]
    var_18 = 2
    var_19 = 15
    var_20 = 25

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 0
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 6
    var_14 = [var_10, var_13, var_11]
    var_15 = 2021
    var_16 = [var_15, var_11, var_11]
    var_17 = 2

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 1000
    var_7 = 'USD'
    var_8 = '0.0'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 6
    var_14 = [var_10, var_13, var_11]
    var_15 = 2021
    var_16 = [var_15, var_11, var_11]
    var_17 = 2
    var_18 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 360
    var_4 = lambda s, a, e, f: Decimal((a - s).days / var_3)
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 1000
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
    var_17 = 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dcfc_30_360_german_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_german_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_example_4. Retrieved 8/14 statements.


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

# Partially parsed test_last_payment_date_annual_january_start. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_annual_january_start_same_year. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_semi_annual_january_start. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_january_start_august_asof. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_january_start_april_asof. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_june_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_july_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_december_start. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semi_annual_december_start_same_year. Retrieved 6/10 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 6/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = set()
    var_5 = set()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_has_leap_day_with_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_without_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_leap_day_not_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_multiple_leap_years_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_no_leap_years_in_range. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 2020
    var_1 = 3
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 12
    var_5 = 31
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2024
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2023
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_interest_basic_case. Retrieved 12/32 statements.
# Partially parsed test_interest_without_end_date. Retrieved 10/29 statements.
# Partially parsed test_interest_zero_fraction. Retrieved 12/26 statements.
# Partially parsed test_interest_with_frequency. Retrieved 14/38 statements.


def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = 1000
    var_6 = '0.05'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = [var_6]
    var_14 = 151
    var_15 = [var_14]
    var_16 = [var_3]

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = 1000
    var_6 = '0.05'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = [var_6]
    var_12 = 151
    var_13 = [var_12]
    var_14 = [var_3]

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = 1000
    var_6 = '0.05'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = 2022
    var_11 = 12
    var_12 = 31
    var_13 = 0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = None
    var_4 = 360
    var_5 = [var_4]
    var_6 = [var_4]
    var_7 = 1000
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = 6
    var_13 = 12
    var_14 = 31
    var_15 = '2'
    var_16 = [var_15]
    var_17 = [var_8]
    var_18 = 151
    var_19 = [var_18]
    var_20 = [var_4]
    var_21 = [var_15]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_returns_correct_dcc. Retrieved 18/20 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_1)
    var_6 = 'test'
    var_7 = var_0.find(var_6)
    var_8 = 'TEST'
    var_9 = var_0.find(var_8)
    var_10 = ' Test '
    var_11 = var_0.find(var_10)
    var_12 = var_0.find(var_2)
    var_13 = 'testalt'
    var_14 = var_0.find(var_13)
    var_15 = 'TESTALT'
    var_16 = var_0.find(var_15)
    var_17 = ' TestAlt '
    var_18 = var_0.find(var_17)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dcfc_30_360_us_basic_cases. Retrieved 18/42 statements.


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
    var_28 = '1.33333333333333'
    var_29 = [var_28]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_empty_range. Retrieved 2/6 statements.
# Partially parsed test_single_day_range. Retrieved 3/9 statements.
# Partially parsed test_multi_day_range. Retrieved 6/15 statements.
# Partially parsed test_year_boundary. Retrieved 8/17 statements.


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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dcfc_30_e_360_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_360_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_example_4. Retrieved 8/14 statements.


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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dcfc_act_act_icma. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 9
    var_5 = 10
    var_6 = [var_0, var_4, var_5]
    var_7 = 2020
    var_8 = [var_7, var_1, var_2]
    var_9 = '0.5245901639'
    var_10 = [var_9]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_register_raises_typeerror_for_duplicate_altname. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'ALT1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'Test2'
    var_8 = {var_2}
    var_9 = set()
    var_10 = [var_5]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dcfc_act_act_basic. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_cross_year. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.16942884946478'
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
    var_9 = '0.17216108990194'
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
    var_9 = '1.08243131970956'
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
    var_9 = '1.32625945055768'
    var_10 = [var_9]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dcfc_act_act_predicate_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 28
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_4. Retrieved 8/14 statements.


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
    var_9 = '1.33333333333333'
    var_10 = [var_9]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor_initializes_buffers. Retrieved 11/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '_buffer_main'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0._buffer_main
    var_5 = var_0._buffer_main
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = '_buffer_altn'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = var_0._buffer_altn
    var_11 = var_0._buffer_altn
    var_12 = len(var_11)
    assert var_12 == 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_register_altname_conflict. Retrieved 8/17 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 0
    var_5 = [var_4]
    var_6 = 'Test2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = [var_4]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dcfc_act_act_predicate. Retrieved 15/25 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 1
    var_8 = 0
    var_9 = [var_8, var_8]
    var_10 = 1
    var_11 = var_9[var_10]
    var_12 = var_11 + var_10
    var_13 = 0
    var_14 = var_9[var_13]
    var_15 = 1
    var_16 = var_14 + var_15
    var_17 = bool(var_9[0] == 32 and var_9[1] == 1)
    assert var_17 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_init_initializes_buffers. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_example2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example4. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 8/15 statements.


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
    var_9 = '1.33333333333333'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.02777777777778'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 1
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = 2
    var_9 = [var_4, var_8, var_5]
    var_10 = 14



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_DCCRegistryMachinery_constructor_initializes_buffers. Retrieved 11/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '_buffer_main'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0._buffer_main
    var_5 = var_0._buffer_main
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = '_buffer_altn'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = var_0._buffer_altn
    var_11 = var_0._buffer_altn
    var_12 = len(var_11)
    assert var_12 == 0



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_init_initializes_main_buffer. Retrieved 4/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_main
    var_3 = len(var_2)
    assert var_3 == 0



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_nl_365. Retrieved 17/41 statements.


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
    var_22 = '0.16986301369863'
    var_23 = [var_22]
    var_24 = [var_22]
    var_25 = '1.08219178082192'
    var_26 = [var_25]
    var_27 = '1.32602739726027'
    var_28 = [var_27]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_register_successful. Retrieved 6/12 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_duplicate_alternative_name. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.5
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 0.5
    var_5 = [var_4]
    var_6 = set()
    var_7 = set()
    var_8 = [var_4]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = 'Test2'
    var_8 = {var_2}
    var_9 = set()
    var_10 = [var_5]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dcfc_30_360_us_basic_cases. Retrieved 18/42 statements.
# Partially parsed test_dcfc_30_360_us_last_day_of_month. Retrieved 9/21 statements.
# Partially parsed test_dcfc_30_360_us_d2_adjustment. Retrieved 8/20 statements.
# Partially parsed test_dcfc_30_360_us_d1_adjustment. Retrieved 7/19 statements.
# Partially parsed test_dcfc_30_360_us_invalid_date_range. Retrieved 5/15 statements.


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
    var_11 = [var_0, var_1, var_2]
    var_12 = 29
    var_13 = [var_4, var_5, var_12]
    var_14 = [var_4, var_5, var_12]
    var_15 = '0.16944444444444'
    var_16 = [var_15]
    var_17 = 10
    var_18 = 31
    var_19 = [var_0, var_17, var_18]
    var_20 = 11
    var_21 = 30
    var_22 = [var_4, var_20, var_21]
    var_23 = [var_4, var_20, var_21]
    var_24 = '1.08333333333333'
    var_25 = [var_24]
    var_26 = 1
    var_27 = [var_4, var_5, var_26]
    var_28 = 2009
    var_29 = 5
    var_30 = [var_28, var_29, var_18]
    var_31 = [var_28, var_29, var_18]
    var_32 = '1.33333333333333'
    var_33 = [var_32]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 14
    var_9 = '0.0'
    var_10 = [var_9]
    var_11 = [var_0, var_1, var_2]
    var_12 = 3
    var_13 = [var_0, var_12, var_2]
    var_14 = [var_0, var_12, var_2]
    var_15 = '0.16666666666667'
    var_16 = [var_15]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 14
    var_9 = '0.0'
    var_10 = [var_9]
    var_11 = 31
    var_12 = [var_0, var_1, var_11]
    var_13 = [var_0, var_4, var_5]
    var_14 = [var_0, var_4, var_5]
    var_15 = [var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_1]
    var_6 = [var_0, var_4, var_1]
    var_7 = 14
    var_8 = '0.0'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_2]
    var_11 = [var_0, var_4, var_4]
    var_12 = [var_0, var_4, var_4]
    var_13 = '0.00277777777778'
    var_14 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 31
    var_5 = [var_0, var_2, var_4]
    var_6 = [var_0, var_1, var_2]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = [var_0, var_1, var_2]
    var_10 = [var_0, var_1, var_2]
    var_11 = [var_0, var_2, var_4]
    var_12 = [var_7]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_existing_main_name. Retrieved 6/8 statements.
# Partially parsed test_find_existing_alternative_name. Retrieved 6/8 statements.
# Partially parsed test_find_stripped_uppercase_name. Retrieved 9/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = [var_2, var_3]
    var_5 = [var_1, var_4]
    var_6 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = [var_2, var_3]
    var_5 = [var_1, var_4]
    var_6 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Nonexistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = [var_2, var_3]
    var_5 = [var_1, var_4]
    var_6 = ' test '
    var_7 = var_0.find(var_6)
    var_8 = 'alt1'
    var_9 = var_0.find(var_8)



# Parsed testcases at query #5
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

# Partially parsed test_last_day_of_month. Retrieved 16/41 statements.
# Partially parsed test_not_last_day_of_month. Retrieved 16/41 statements.
# Partially parsed test_leap_year_february. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = 3
    var_8 = [var_0, var_7, var_2]
    var_9 = 4
    var_10 = 30
    var_11 = [var_0, var_9, var_10]
    var_12 = 5
    var_13 = [var_0, var_12, var_2]
    var_14 = 6
    var_15 = [var_0, var_14, var_10]
    var_16 = 7
    var_17 = [var_0, var_16, var_2]
    var_18 = 8
    var_19 = [var_0, var_18, var_2]
    var_20 = 9
    var_21 = [var_0, var_20, var_10]
    var_22 = 10
    var_23 = [var_0, var_22, var_2]
    var_24 = 11
    var_25 = [var_0, var_24, var_10]
    var_26 = 12
    var_27 = [var_0, var_26, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 27
    var_6 = [var_0, var_4, var_5]
    var_7 = 3
    var_8 = [var_0, var_7, var_2]
    var_9 = 4
    var_10 = 29
    var_11 = [var_0, var_9, var_10]
    var_12 = 5
    var_13 = [var_0, var_12, var_2]
    var_14 = 6
    var_15 = [var_0, var_14, var_10]
    var_16 = 7
    var_17 = [var_0, var_16, var_2]
    var_18 = 8
    var_19 = [var_0, var_18, var_2]
    var_20 = 9
    var_21 = [var_0, var_20, var_10]
    var_22 = 10
    var_23 = [var_0, var_22, var_2]
    var_24 = 11
    var_25 = [var_0, var_24, var_10]
    var_26 = 12
    var_27 = [var_0, var_26, var_2]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 28
    var_5 = [var_0, var_1, var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_example_4. Retrieved 8/14 statements.


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
    var_9 = '1.33333333333333'
    var_10 = [var_9]



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'Test1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'Test2'
    var_8 = {var_7}
    var_9 = set()
    var_10 = [var_5]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_4. Retrieved 8/14 statements.


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
    var_9 = '1.33333333333333'
    var_10 = [var_9]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 7/19 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 14/34 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_with_eom. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_1, var_1]
    var_8 = [var_3, var_1, var_1]
    var_9 = [var_3, var_4, var_5]
    var_10 = [var_3, var_1, var_1]
    var_11 = 9
    var_12 = [var_0, var_4, var_11]
    var_13 = 4
    var_14 = [var_3, var_4, var_13]
    var_15 = [var_0, var_4, var_11]

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
    var_10 = [var_0, var_1, var_1]
    var_11 = 8
    var_12 = [var_3, var_11, var_5]
    var_13 = [var_3, var_8, var_1]
    var_14 = [var_0, var_1, var_1]
    var_15 = 4
    var_16 = 30
    var_17 = [var_3, var_15, var_16]
    var_18 = [var_3, var_1, var_1]
    var_19 = 2012
    var_20 = 15
    var_21 = [var_19, var_4, var_20]
    var_22 = 2016
    var_23 = 6
    var_24 = [var_22, var_1, var_23]
    var_25 = [var_3, var_4, var_20]
    var_26 = [var_19, var_4, var_20]
    var_27 = [var_3, var_4, var_5]
    var_28 = [var_3, var_4, var_20]

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
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 4
    var_6 = 30
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_0, var_1, var_2]



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_eom_assignment. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 1
    var_8 = None



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_register_success. Retrieved 6/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 10/23 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 10/23 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = 0.5
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt1'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = 'TestAlt2'
    var_8 = {var_7}
    var_9 = 'EUR'
    var_10 = 0.6
    var_11 = [var_10]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = 'Test2'
    var_8 = {var_2}
    var_9 = 'EUR'
    var_10 = 0.6
    var_11 = [var_10]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__is_last_day_of_month__true_for_jan_31. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__false_for_jan_30. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__true_for_feb_28_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__false_for_feb_27_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__true_for_feb_29_leap_year. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__false_for_feb_28_leap_year. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__true_for_apr_30. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__false_for_apr_29. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__true_for_dec_31. Retrieved 3/5 statements.
# Partially parsed test__is_last_day_of_month__false_for_dec_30. Retrieved 3/5 statements.


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
    var_0 = 2023
    var_1 = 2
    var_2 = 27

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
    var_1 = 4
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 29

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 30



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_1. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_1, var_1]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 1
    var_8 = None



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

# Partially parsed test_dcfc_30_360_us_predicate. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = '0.027777777777777777'
    var_9 = [var_8]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dcfc_act_act_basic_calculation. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_another_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_invalid_date_range. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.16942884946478'
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
    var_9 = '0.17216108990194'
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
    var_9 = '1.08243131970956'
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
    var_9 = '1.32625945055768'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2019
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dcfc_30_360_german_with_standard_dates. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_german_with_leap_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_year_boundary. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_invalid_date_order. Retrieved 6/10 statements.
# Partially parsed test_dcfc_30_360_german_with_start_day_31. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_german_with_asof_day_31_and_end_not_asof. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_with_february_last_day_and_end_not_asof. Retrieved 7/13 statements.


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

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2007
    var_5 = 12
    var_6 = [var_4, var_5, var_2]
    var_7 = '0'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.02777777777778'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 31
    var_6 = [var_4, var_2, var_5]
    var_7 = 2
    var_8 = [var_4, var_7, var_2]
    var_9 = 14
    var_10 = '0.08333333333333'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2007
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 28
    var_5 = [var_0, var_1, var_4]
    var_6 = 3
    var_7 = [var_0, var_6, var_2]
    var_8 = 14
    var_9 = '0.05555555555556'
    var_10 = [var_9]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dcfc_act_365_a_without_leap_day. Retrieved 5/12 statements.
# Partially parsed test_dcfc_act_365_a_with_leap_day. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_365_a_full_year_no_leap. Retrieved 5/10 statements.
# Partially parsed test_dcfc_act_365_a_full_year_with_leap. Retrieved 5/10 statements.
# Partially parsed test_dcfc_act_365_a_partial_year_with_leap. Retrieved 6/13 statements.
# Partially parsed test_dcfc_act_365_a_partial_year_without_leap. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = '9'
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
    var_7 = [var_0, var_4, var_5]
    var_8 = '2'
    var_9 = [var_8]
    var_10 = '366'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = '1'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = '1'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = 30
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = '181'
    var_8 = [var_7]
    var_9 = '366'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = 30
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = '180'
    var_8 = [var_7]
    var_9 = '365'
    var_10 = [var_9]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_start. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_end. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_multiple_leap_days_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_no_leap_days_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_same_start_and_end_date. Retrieved 3/6 statements.
# Partially parsed test_has_leap_day_with_same_start_and_end_date_no_leap_day. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 12
    var_5 = 31
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = 29
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2021
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 5/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = [var_4]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_partial_year. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_early_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_mid_year_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_start. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_start_end_year. Retrieved 6/10 statements.


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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_find_returns_correct_dcc. Retrieved 10/12 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'AltTest'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_1)
    var_6 = var_0.find(var_2)
    var_7 = 'test'
    var_8 = var_0.find(var_7)
    var_9 = ' alttest '
    var_10 = var_0.find(var_9)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_interest_calculates_accrued_interest_correctly. Retrieved 12/29 statements.
# Partially parsed test_interest_uses_asof_as_end_when_end_is_none. Retrieved 10/26 statements.
# Partially parsed test_interest_returns_zero_when_asof_is_before_start. Retrieved 12/26 statements.
# Partially parsed test_interest_returns_zero_when_asof_is_after_end. Retrieved 13/27 statements.


def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = 1000
    var_6 = '0.05'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = 151
    var_14 = [var_13]
    var_15 = [var_3]

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = 1000
    var_6 = '0.05'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 151
    var_12 = [var_11]
    var_13 = [var_3]

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = 1000
    var_6 = '0.05'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 6
    var_10 = 1
    var_11 = 12
    var_12 = 31
    var_13 = 0

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = 1000
    var_6 = '0.05'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = 2024
    var_11 = 6
    var_12 = 12
    var_13 = 31
    var_14 = 0



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dcfc_act_act_icma. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 9
    var_5 = 10
    var_6 = [var_0, var_4, var_5]
    var_7 = 2020
    var_8 = [var_7, var_1, var_2]
    var_9 = '0.5245901639'
    var_10 = [var_9]



