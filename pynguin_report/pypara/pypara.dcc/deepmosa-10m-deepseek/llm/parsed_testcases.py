####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_register_success. Retrieved 8/14 statements.
# Partially parsed test_register_duplicate_name. Retrieved 8/21 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 10/23 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = var_0.find(var_1)
    var_8 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = set()
    var_3 = 'USD'
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = set()
    var_7 = 'EUR'
    var_8 = '0.6'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'TestDCC2'
    var_8 = {var_2}
    var_9 = 'EUR'
    var_10 = '0.6'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 6/12 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 10/19 statements.
# Partially parsed test_register_duplicate_name_in_alt_names. Retrieved 9/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = set()
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
    var_1 = 'TestDCC2'
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dcfc_act_365_l_with_non_leap_year. Retrieved 5/9 statements.
# Partially parsed test_dcfc_act_365_l_with_leap_year. Retrieved 5/9 statements.
# Partially parsed test_dcfc_act_365_l_with_single_day_in_non_leap_year. Retrieved 4/8 statements.
# Partially parsed test_dcfc_act_365_l_with_single_day_in_leap_year. Retrieved 4/8 statements.
# Partially parsed test_dcfc_act_365_l_with_same_day. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = '0.99726027397260'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = '0.99726775956284'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '0.00273972602740'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '0.00273224043716'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = '0.0'
    var_5 = [var_4]



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dcfc_act_365_a. Retrieved 22/52 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.16986301369863'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_2]
    var_11 = 29
    var_12 = [var_4, var_5, var_11]
    var_13 = '0.17213114754098'
    var_14 = [var_13]
    var_15 = 10
    var_16 = 31
    var_17 = [var_0, var_15, var_16]
    var_18 = 11
    var_19 = 30
    var_20 = [var_4, var_18, var_19]
    var_21 = '1.08196721311475'
    var_22 = [var_21]
    var_23 = 1
    var_24 = [var_4, var_5, var_23]
    var_25 = 2009
    var_26 = 5
    var_27 = [var_25, var_26, var_16]
    var_28 = '1.32513661202186'
    var_29 = [var_28]
    var_30 = 2020
    var_31 = [var_30, var_23, var_23]
    var_32 = [var_30, var_1, var_16]
    var_33 = '0.99726775956284'
    var_34 = [var_33]
    var_35 = 2019
    var_36 = [var_35, var_23, var_23]
    var_37 = [var_35, var_1, var_16]
    var_38 = '0.99726027397260'
    var_39 = [var_38]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dcfc_30_e_360_basic_case. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_360_leap_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_31st_day_adjustment. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_multi_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_360_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_e_360_31st_asof_adjustment. Retrieved 7/13 statements.


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
    var_10 = '1.33055555555556'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = '0'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 14
    var_9 = '0.22222222222222'
    var_10 = [var_9]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dcfc_30_360_isda_basic_case. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_leap_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_31_day_month_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_multi_year_case. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_isda_start_31_adjustment. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_isda_asof_31_adjustment. Retrieved 7/13 statements.


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
    var_0 = 2007
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

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = 14
    var_9 = '0.08333333333333'
    var_10 = [var_9]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic_calculation. Retrieved 7/13 statements.
# Partially parsed test_dcfc_act_365_a_leap_year_calculation. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_a_multi_year_calculation. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_a_long_period_calculation. Retrieved 8/14 statements.
# Partially parsed test_dcfc_act_365_a_same_day. Retrieved 3/6 statements.
# Partially parsed test_dcfc_act_365_a_one_day. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.16986301369863'
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
    var_9 = '0.17213114754098'
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
    var_9 = '1.08196721311475'
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
    var_9 = '1.32513661202186'
    var_10 = [var_9]

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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_calculate_daily_fraction_valid_dates. Retrieved 8/16 statements.
# Partially parsed test_calculate_daily_fraction_asof_before_start. Retrieved 8/16 statements.
# Partially parsed test_calculate_daily_fraction_asof_equals_start. Retrieved 8/16 statements.
# Partially parsed test_calculate_daily_fraction_asof_equals_end. Retrieved 7/15 statements.
# Partially parsed test_calculate_daily_fraction_asof_after_end. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'ACT/ACT'
    var_1 = set()
    var_2 = set()
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = 2
    var_9 = [var_5, var_6, var_8]
    var_10 = 3
    var_11 = [var_5, var_6, var_10]
    var_12 = [var_3]

def test_case_0():
    var_0 = 'ACT/ACT'
    var_1 = set()
    var_2 = set()
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_5, var_6, var_6]
    var_10 = 3
    var_11 = [var_5, var_6, var_10]
    var_12 = [var_3]

def test_case_0():
    var_0 = 'ACT/ACT'
    var_1 = set()
    var_2 = set()
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = [var_5, var_6, var_6]
    var_9 = 3
    var_10 = [var_5, var_6, var_9]
    var_11 = '0.00'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'ACT/ACT'
    var_1 = set()
    var_2 = set()
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = 3
    var_9 = [var_5, var_6, var_8]
    var_10 = [var_5, var_6, var_8]
    var_11 = [var_3]

def test_case_0():
    var_0 = 'ACT/ACT'
    var_1 = set()
    var_2 = set()
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = 4
    var_9 = [var_5, var_6, var_8]
    var_10 = 3
    var_11 = [var_5, var_6, var_10]
    var_12 = '0.00'
    var_13 = [var_12]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_same_year_annual_frequency. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_mid_year. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_early_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_mid_year_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_start. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_end. Retrieved 6/10 statements.


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

# Partially parsed test_30_360_isda_asof_day_31_adjustment. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 11
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = '0.08333333333333'
    var_9 = [var_8]



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 8/16 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Act/Act'
    var_1 = set()
    var_2 = set()
    var_3 = 1
    var_4 = [var_3]
    var_5 = set()
    var_6 = set()
    var_7 = 2
    var_8 = [var_7]
    var_9 = module_0.DCCRegistryMachinery()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_last_payment_date_eom_false. Retrieved 7/11 statements.


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
    var_9 = [var_3, var_4, var_5]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dcfc_30_360_german_with_example1. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_german_with_example2. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_example3. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_example4. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_31st_start_day. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_german_with_feb_end_start_day. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_german_with_31st_asof_day. Retrieved 6/11 statements.
# Partially parsed test_dcfc_30_360_german_with_feb_end_asof_day. Retrieved 8/14 statements.


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
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = 14
    var_8 = '0.08333333333333'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 14
    var_8 = '0.08333333333333'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 31
    var_5 = [var_0, var_1, var_4]
    var_6 = 14
    var_7 = '0.04444444444444'
    var_8 = [var_7]

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
    var_9 = 14
    var_10 = '0.11944444444444'
    var_11 = [var_10]



# Parsed testcases at query #15
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_existing_dcc. Retrieved 4/6 statements.
# Partially parsed test_find_stripped_and_uppercased_dcc. Retrieved 5/7 statements.
# Partially parsed test_find_alternative_name. Retrieved 5/7 statements.
# Partially parsed test_find_stripped_and_uppercased_alternative_name. Retrieved 6/8 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistingDCC'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = ' act/act '
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = ' actual/actual '
    var_6 = var_0.find(var_5)



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_range_across_leap_year. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_range_before_leap_year. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_range_after_leap_year. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_single_day_range_on_leap_day. Retrieved 3/6 statements.
# Partially parsed test_has_leap_day_with_single_day_range_not_on_leap_day. Retrieved 3/6 statements.


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
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2021
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2019
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

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dcfc_act_act_with_invalid_date_range. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2007
    var_5 = 12
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_0, var_1, var_2]
    var_8 = '0'
    var_9 = [var_8]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dcfc_30_360_us_basic_case. Retrieved 6/10 statements.
# Partially parsed test_dcfc_30_360_us_leap_year_case. Retrieved 7/11 statements.
# Partially parsed test_dcfc_30_360_us_month_end_case. Retrieved 7/11 statements.
# Partially parsed test_dcfc_30_360_us_long_period_case. Retrieved 7/11 statements.
# Partially parsed test_dcfc_30_360_us_same_date. Retrieved 3/6 statements.
# Partially parsed test_dcfc_30_360_us_31_to_30_adjustment. Retrieved 6/10 statements.
# Partially parsed test_dcfc_30_360_us_both_dates_last_day_of_month. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.1666666666666666666666666667'
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
    var_8 = '0.1694444444444444444444444444'
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
    var_8 = '1.083333333333333333333333333'
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
    var_8 = '1.333333333333333333333333333'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = '0.08333333333333333333333333333'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 29
    var_6 = [var_0, var_4, var_5]
    var_7 = '0.08333333333333333333333333333'
    var_8 = [var_7]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_coupon_calculation. Retrieved 15/29 statements.
# Partially parsed test_coupon_calculation_with_eom. Retrieved 15/29 statements.
# Partially parsed test_coupon_calculation_with_different_frequency. Retrieved 15/29 statements.
# Partially parsed test_coupon_calculation_with_zero_fraction. Retrieved 14/28 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 12
    var_16 = 31
    var_17 = [var_9, var_15, var_16]
    var_18 = 2
    var_19 = None
    var_20 = '25'
    var_21 = [var_20]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 12
    var_16 = 31
    var_17 = [var_9, var_15, var_16]
    var_18 = 2
    var_19 = 15
    var_20 = '25'
    var_21 = [var_20]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 12
    var_16 = 31
    var_17 = [var_9, var_15, var_16]
    var_18 = 4
    var_19 = None
    var_20 = '12.5'
    var_21 = [var_20]

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()
    var_3 = '0'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 12
    var_16 = 31
    var_17 = [var_9, var_15, var_16]
    var_18 = 2
    var_19 = None
    var_20 = [var_3]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_register_raises_error_when_name_already_registered. Retrieved 12/21 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = {var_4}
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = 'Actual/360'
    var_9 = {var_8}
    var_10 = 'EUR'
    var_11 = {var_10}
    var_12 = '0.6'
    var_13 = [var_12]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_calculate_daily_fraction_asof_minus_1_less_than_start. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = 12
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = '1.0'
    var_8 = [var_7]
    var_9 = 'Test'
    var_10 = set()
    var_11 = set()
    var_12 = [var_7]
    var_13 = [var_7]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_returns_dcc_when_name_matches_case_insensitive. Retrieved 6/8 statements.
# Partially parsed test_find_returns_dcc_when_name_matches_with_whitespace. Retrieved 6/8 statements.
# Partially parsed test_find_returns_dcc_when_name_matches_case_insensitive_with_whitespace. Retrieved 6/8 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = None
    var_4 = [var_1, var_2, var_3, var_3]
    var_5 = 'act/act'
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = None
    var_4 = [var_1, var_2, var_3, var_3]
    var_5 = '  Act/Act  '
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = None
    var_4 = [var_1, var_2, var_3, var_3]
    var_5 = '  act/act  '
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_start. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_end. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_leap_day_in_multiple_years. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2020
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
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2024
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dcfc_30_360_german_predicate_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_has_leap_day. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dcfc_30_e_360_with_start_day_not_31. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 28
    var_7 = [var_4, var_5, var_6]
    var_8 = '0.16666666666667'
    var_9 = [var_8]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_calculate_daily_fraction_asof_minus_1_not_less_than_start. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = 'Test'
    var_8 = set()
    var_9 = set()
    var_10 = [var_1]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example_1. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_isda_example_2. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_isda_example_3. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_isda_example_4. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_isda_asof_day_31. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_isda_leap_year. Retrieved 6/11 statements.
# Partially parsed test_dcfc_30_360_isda_year_change. Retrieved 7/12 statements.


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
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = 14
    var_8 = '0.07777777777778'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 14
    var_8 = '0.08333333333333'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = 14
    var_7 = '0.00277777777778'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.08333333333333'
    var_9 = [var_8]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_coupon_calculation. Retrieved 16/29 statements.
# Partially parsed test_coupon_calculation_with_eom. Retrieved 16/29 statements.
# Partially parsed test_coupon_calculation_with_different_frequency. Retrieved 16/29 statements.
# Partially parsed test_coupon_calculation_with_zero_fraction. Retrieved 16/29 statements.


def test_case_0():
    var_0 = '30/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 12
    var_16 = 31
    var_17 = [var_9, var_15, var_16]
    var_18 = '2'
    var_19 = [var_18]
    var_20 = None
    var_21 = 25

def test_case_0():
    var_0 = '30/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = 31
    var_12 = [var_9, var_10, var_11]
    var_13 = 6
    var_14 = 30
    var_15 = [var_9, var_13, var_14]
    var_16 = 12
    var_17 = [var_9, var_16, var_11]
    var_18 = '2'
    var_19 = [var_18]
    var_20 = 31
    var_21 = 25

def test_case_0():
    var_0 = '30/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 12
    var_16 = 31
    var_17 = [var_9, var_15, var_16]
    var_18 = '4'
    var_19 = [var_18]
    var_20 = None
    var_21 = 12.5

def test_case_0():
    var_0 = '30/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.0'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 12
    var_16 = 31
    var_17 = [var_9, var_15, var_16]
    var_18 = '2'
    var_19 = [var_18]
    var_20 = None
    var_21 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_existing_dcc. Retrieved 4/6 statements.
# Partially parsed test_find_with_stripped_and_uppercase. Retrieved 5/7 statements.
# Partially parsed test_find_with_alternative_names. Retrieved 5/7 statements.
# Partially parsed test_find_with_stripped_and_uppercase_alternative_names. Retrieved 6/8 statements.


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
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = []
    var_4 = 'act/act'
    var_5 = var_0.find(var_4)

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
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = [var_2]
    var_4 = []
    var_5 = 'actual/actual'
    var_6 = var_0.find(var_5)



# Parsed testcases at query #4
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



# Parsed testcases at query #5
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

# Partially parsed test_is_last_day_of_month_true. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_false. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_april. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_june. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_september. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_november. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_december. Retrieved 3/5 statements.


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

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 9
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 11
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 6/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 9/18 statements.
# Partially parsed test_register_duplicate_main_name_as_alt_name. Retrieved 8/17 statements.


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
    var_5 = 'TestDCC2'
    var_6 = {var_0}
    var_7 = set()
    var_8 = [var_3]
    var_9 = module_0.DCCRegistryMachinery()
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_same_year_annual_frequency. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_mid_year. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_early_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_mid_year_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_late_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_start. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_end_of_year. Retrieved 6/10 statements.


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

# Partially parsed test_dcfc_30_360_us. Retrieved 18/42 statements.


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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coupon_with_annual_frequency. Retrieved 15/26 statements.
# Partially parsed test_coupon_with_semiannual_frequency. Retrieved 14/25 statements.
# Partially parsed test_coupon_with_quarterly_frequency. Retrieved 14/25 statements.
# Partially parsed test_coupon_with_eom_adjustment. Retrieved 17/28 statements.
# Partially parsed test_coupon_with_asof_equal_to_start. Retrieved 12/23 statements.
# Partially parsed test_coupon_with_asof_equal_to_end. Retrieved 12/23 statements.


def test_case_0():
    var_0 = '30/360'
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
    var_13 = 2021
    var_14 = 6
    var_15 = 30
    var_16 = [var_13, var_14, var_15]
    var_17 = 2022
    var_18 = [var_17, var_11, var_11]
    var_19 = 1
    var_20 = '25.00'
    var_21 = [var_20]

def test_case_0():
    var_0 = '30/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 9
    var_14 = 30
    var_15 = [var_10, var_13, var_14]
    var_16 = 2021
    var_17 = [var_16, var_11, var_11]
    var_18 = 2
    var_19 = '12.50'
    var_20 = [var_19]

def test_case_0():
    var_0 = '30/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.125'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 4
    var_14 = 15
    var_15 = [var_10, var_13, var_14]
    var_16 = 7
    var_17 = [var_10, var_16, var_11]
    var_18 = 4
    var_19 = '6.25'
    var_20 = [var_19]

def test_case_0():
    var_0 = '30/360'
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
    var_12 = 31
    var_13 = [var_10, var_11, var_12]
    var_14 = 2021
    var_15 = 6
    var_16 = 30
    var_17 = [var_14, var_15, var_16]
    var_18 = 2022
    var_19 = [var_18, var_11, var_12]
    var_20 = 1
    var_21 = 31
    var_22 = '25.00'
    var_23 = [var_22]

def test_case_0():
    var_0 = '30/360'
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
    var_17 = '0.00'
    var_18 = [var_17]

def test_case_0():
    var_0 = '30/360'
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
    var_17 = '50.00'
    var_18 = [var_17]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_last_payment_date. Retrieved 17/57 statements.


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
    var_11 = [var_0, var_1, var_1]
    var_12 = [var_3, var_4, var_5]
    var_13 = 2
    var_14 = 7
    var_15 = [var_3, var_14, var_1]
    var_16 = [var_0, var_1, var_1]
    var_17 = 8
    var_18 = [var_3, var_17, var_5]
    var_19 = [var_3, var_14, var_1]
    var_20 = [var_0, var_1, var_1]
    var_21 = 4
    var_22 = 30
    var_23 = [var_3, var_21, var_22]
    var_24 = [var_3, var_1, var_1]
    var_25 = 6
    var_26 = [var_0, var_25, var_1]
    var_27 = [var_3, var_21, var_22]
    var_28 = [var_0, var_25, var_1]
    var_29 = 2008
    var_30 = [var_29, var_14, var_14]
    var_31 = 10
    var_32 = [var_3, var_31, var_25]
    var_33 = [var_3, var_14, var_14]
    var_34 = 9
    var_35 = [var_0, var_4, var_34]
    var_36 = [var_3, var_4, var_21]
    var_37 = [var_0, var_4, var_34]
    var_38 = 2012
    var_39 = 15
    var_40 = [var_38, var_4, var_39]
    var_41 = 2016
    var_42 = [var_41, var_1, var_25]
    var_43 = [var_3, var_4, var_39]
    var_44 = [var_38, var_4, var_39]
    var_45 = [var_3, var_4, var_5]
    var_46 = [var_3, var_4, var_39]



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

# Partially parsed test_dcfc_30_e_plus_360_example_1. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_2. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_3. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_4. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_with_start_day_31. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_with_asof_day_31. Retrieved 8/13 statements.


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
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 28
    var_7 = [var_4, var_5, var_6]
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
    var_8 = '0.16944444444444'
    var_9 = [var_8]
    var_10 = 14



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

# Partially parsed test_dcfc_30_360_german_with_regular_dates. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_german_with_leap_year_date. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_with_31st_day_start. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_with_feb_start. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_german_with_end_not_equal_to_asof. Retrieved 10/16 statements.
# Partially parsed test_dcfc_30_360_german_with_last_day_of_feb_start. Retrieved 7/13 statements.


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
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = [var_4, var_5, var_6]
    var_8 = 3
    var_9 = 31
    var_10 = [var_4, var_8, var_9]
    var_11 = 14
    var_12 = '0.16944444444444'
    var_13 = [var_12]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 14
    var_8 = '0.08333333333333'
    var_9 = [var_8]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dcfc_30_e_360. Retrieved 20/60 statements.


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
    var_28 = '1.33055555555556'
    var_29 = [var_28]
    var_30 = 2020
    var_31 = [var_30, var_23, var_16]
    var_32 = 3
    var_33 = [var_30, var_32, var_16]
    var_34 = [var_8]
    var_35 = [var_30, var_23, var_19]
    var_36 = [var_30, var_32, var_16]
    var_37 = [var_8]
    var_38 = [var_30, var_23, var_19]
    var_39 = [var_30, var_32, var_19]
    var_40 = [var_8]
    var_41 = [var_30, var_23, var_16]
    var_42 = [var_30, var_32, var_19]
    var_43 = [var_8]



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dcfc_act_365_l_with_leap_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_365_l_with_non_leap_year. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_365_l_with_multiple_years. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_365_l_with_long_period. Retrieved 8/13 statements.


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
    var_9 = '0.17213114754098'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 14
    var_8 = '0.16939890710383'
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
    var_8 = 14
    var_9 = '1.08196721311475'
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
    var_9 = '1.32876712328767'
    var_10 = [var_9]



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------

# Partially parsed test_register_successfully_registers_dcc. Retrieved 6/12 statements.
# Partially parsed test_register_raises_error_when_dcc_is_already_registered. Retrieved 6/14 statements.
# Partially parsed test_register_raises_error_when_altname_is_already_registered. Retrieved 8/21 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = 1
    var_6 = [var_5]
    var_7 = var_0._buffer_main['TestDCC']
    var_8 = var_0._buffer_altn['TestAlt']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = 1
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = 1
    var_6 = [var_5]
    var_7 = 'TestDCC2'
    var_8 = {var_2}
    var_9 = [var_5]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_last_payment_date_returns_start_date_when_invalid. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 1
    var_8 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dcfc_nl_365_with_leap_day. Retrieved 6/13 statements.
# Partially parsed test_dcfc_nl_365_without_leap_day. Retrieved 7/14 statements.
# Partially parsed test_dcfc_nl_365_same_day. Retrieved 4/9 statements.
# Partially parsed test_dcfc_nl_365_multi_year_period. Retrieved 8/15 statements.
# Partially parsed test_dcfc_nl_365_long_period. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = '0.00273972602740'
    var_8 = [var_7]
    var_9 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = '0.16986301369863'
    var_9 = [var_8]
    var_10 = 14

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = '0'
    var_7 = [var_6]

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
    var_9 = '1.08219178082192'
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
    var_9 = '1.32602739726027'
    var_10 = [var_9]
    var_11 = 14



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dcfc_30_360_isda_predicate_evaluates_to_false. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/5 statements.
# Partially parsed test_is_last_day_of_month_february_not_leap_year. Retrieved 3/5 statements.


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
    var_1 = 2
    var_2 = 29



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_calculate_daily_fraction_asof_minus_1_less_than_start. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = 'test'
    var_8 = set()
    var_9 = set()
    var_10 = [var_1]
    var_11 = [var_1]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dcfc_30_e_360_does_not_modify_asof_when_not_31st. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_coupon_calculation. Retrieved 15/31 statements.


def test_case_0():
    var_0 = '30/360'
    var_1 = 'Bond Basis'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = 360
    var_5 = [var_4]
    var_6 = '1000'
    var_7 = [var_6]
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 6
    var_14 = 30
    var_15 = [var_10, var_13, var_14]
    var_16 = 12
    var_17 = 31
    var_18 = [var_10, var_16, var_17]
    var_19 = 2
    var_20 = '25.00'
    var_21 = [var_20]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_registered_returns_true_when_name_is_in_main_or_alt_buffer. Retrieved 6/12 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.0'
    var_6 = [var_5]



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 11
    var_5 = 30
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = '0.08333333333333'
    var_9 = [var_8]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dcfc_act_365_l_with_leap_year. Retrieved 5/10 statements.
# Partially parsed test_dcfc_act_365_l_with_non_leap_year. Retrieved 5/10 statements.
# Partially parsed test_dcfc_act_365_l_with_year_boundary. Retrieved 6/11 statements.
# Partially parsed test_dcfc_act_365_l_with_single_day_leap_year. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = 29
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = '0.16939890710383'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = 28
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = '0.15890410958904'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = '0.08688524590164'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = '0.00273224043716'
    var_7 = [var_6]



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_coupon_calculation. Retrieved 13/23 statements.
# Partially parsed test_coupon_with_eom. Retrieved 15/25 statements.
# Partially parsed test_coupon_before_start_date. Retrieved 14/24 statements.
# Partially parsed test_coupon_after_end_date. Retrieved 11/21 statements.
# Partially parsed test_coupon_on_payment_date. Retrieved 12/22 statements.


def test_case_0():
    var_0 = 'Actual/Actual'
    var_1 = 'Act/Act'
    var_2 = {var_1}
    var_3 = lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    var_4 = 1000
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2020
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 6
    var_11 = 30
    var_12 = [var_7, var_10, var_11]
    var_13 = 2021
    var_14 = [var_13, var_8, var_8]
    var_15 = 2
    var_16 = [var_15]
    var_17 = 25

def test_case_0():
    var_0 = 'Actual/Actual'
    var_1 = 'Act/Act'
    var_2 = {var_1}
    var_3 = lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    var_4 = 1000
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2020
    var_8 = 1
    var_9 = 31
    var_10 = [var_7, var_8, var_9]
    var_11 = 6
    var_12 = 30
    var_13 = [var_7, var_11, var_12]
    var_14 = 2021
    var_15 = [var_14, var_8, var_9]
    var_16 = 2
    var_17 = [var_16]
    var_18 = 31
    var_19 = 25

def test_case_0():
    var_0 = 'Actual/Actual'
    var_1 = 'Act/Act'
    var_2 = {var_1}
    var_3 = lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    var_4 = 1000
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2020
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 2019
    var_11 = 12
    var_12 = 31
    var_13 = [var_10, var_11, var_12]
    var_14 = 2021
    var_15 = [var_14, var_8, var_8]
    var_16 = 2
    var_17 = [var_16]
    var_18 = 0

def test_case_0():
    var_0 = 'Actual/Actual'
    var_1 = 'Act/Act'
    var_2 = {var_1}
    var_3 = lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    var_4 = 1000
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2020
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 2021
    var_11 = 2
    var_12 = [var_10, var_8, var_11]
    var_13 = [var_10, var_8, var_8]
    var_14 = [var_11]
    var_15 = 0

def test_case_0():
    var_0 = 'Actual/Actual'
    var_1 = 'Act/Act'
    var_2 = {var_1}
    var_3 = lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    var_4 = 1000
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2020
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 7
    var_11 = [var_7, var_10, var_8]
    var_12 = 2021
    var_13 = [var_12, var_8, var_8]
    var_14 = 2
    var_15 = [var_14]
    var_16 = 25



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dcfc_act_act_with_non_leap_year. Retrieved 4/9 statements.
# Partially parsed test_dcfc_act_act_with_leap_year. Retrieved 4/9 statements.
# Partially parsed test_dcfc_act_act_spanning_leap_and_non_leap_years. Retrieved 6/11 statements.
# Partially parsed test_dcfc_act_act_with_single_day. Retrieved 4/9 statements.
# Partially parsed test_dcfc_act_act_with_full_year. Retrieved 4/9 statements.
# Partially parsed test_dcfc_act_act_with_full_leap_year. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_3, var_1]
    var_5 = '0.16164383561643836'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_3, var_1]
    var_5 = '0.16393442622950818'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2019
    var_1 = 12
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2020
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = '0.16939890710382514'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '0.0027397260273972603'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2021
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2022
    var_4 = [var_3, var_1, var_1]
    var_5 = '1.0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2021
    var_4 = [var_3, var_1, var_1]
    var_5 = '1.0'
    var_6 = [var_5]



