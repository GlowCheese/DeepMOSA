####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_register_success. Retrieved 6/12 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_alt_name. Retrieved 10/19 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = {}
    var_3 = {}
    var_4 = 0.5
    var_5 = [var_4]
    var_6 = {}
    var_7 = {}
    var_8 = 0.6
    var_9 = [var_8]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = 'Test2'
    var_8 = {var_2}
    var_9 = {}
    var_10 = 0.6
    var_11 = [var_10]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_has_leap_day_with_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_without_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_leap_day_exactly_at_start. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_leap_day_exactly_at_end. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_multiple_leap_years_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_no_leap_years_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_same_start_and_end_date_on_leap_day. Retrieved 3/6 statements.
# Partially parsed test_has_leap_day_with_same_start_and_end_date_not_on_leap_day. Retrieved 3/6 statements.


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
    var_0 = 2016
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
    var_3 = 2022
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
    var_0 = 2021
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_last_day_of_february_non_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_february_leap_year. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_january. Retrieved 3/5 statements.
# Partially parsed test_last_day_of_april. Retrieved 3/5 statements.
# Partially parsed test_not_last_day_of_month. Retrieved 3/5 statements.


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
    var_1 = 1
    var_2 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30

def test_case_0():
    var_0 = 2023
    var_1 = 3
    var_2 = 15



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 10/22 statements.
# Partially parsed test_last_payment_date_start_date_before_asof. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_same_month_different_year. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_multi_year_period. Retrieved 9/17 statements.


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
    var_11 = [var_0, var_1, var_2]
    var_12 = 31
    var_13 = [var_9, var_1, var_12]
    var_14 = [var_9, var_1, var_2]



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

# Partially parsed test_dcfc_act_365_l. Retrieved 18/42 statements.


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
    var_9 = '0.16939890710383'
    var_10 = [var_9]
    var_11 = [var_0, var_1, var_2]
    var_12 = 29
    var_13 = [var_4, var_5, var_12]
    var_14 = [var_4, var_5, var_12]
    var_15 = '0.17213114754098'
    var_16 = [var_15]
    var_17 = 10
    var_18 = 31
    var_19 = [var_0, var_17, var_18]
    var_20 = 11
    var_21 = 30
    var_22 = [var_4, var_20, var_21]
    var_23 = [var_4, var_20, var_21]
    var_24 = '1.08196721311475'
    var_25 = [var_24]
    var_26 = 1
    var_27 = [var_4, var_5, var_26]
    var_28 = 2009
    var_29 = 5
    var_30 = [var_28, var_29, var_18]
    var_31 = [var_28, var_29, var_18]
    var_32 = '1.32876712328767'
    var_33 = [var_32]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_date_range_empty. Retrieved 2/6 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 3/9 statements.
# Partially parsed test_get_date_range_multiple_days. Retrieved 6/15 statements.
# Partially parsed test_get_date_range_year_boundary. Retrieved 8/17 statements.
# Partially parsed test_get_date_range_month_boundary. Retrieved 6/15 statements.


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
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = 2
    var_4 = 3
    var_5 = 31



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coupon_basic_case. Retrieved 15/25 statements.
# Partially parsed test_coupon_with_eom. Retrieved 15/25 statements.
# Partially parsed test_coupon_zero_principal. Retrieved 14/24 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 15/25 statements.
# Partially parsed test_coupon_high_frequency. Retrieved 14/24 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.5
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.10'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = [var_9, var_12, var_10]
    var_14 = 12
    var_15 = 31
    var_16 = [var_9, var_14, var_15]
    var_17 = 1
    var_18 = None
    var_19 = 50

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.25
    var_4 = [var_3]
    var_5 = 2000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = 15
    var_12 = [var_9, var_10, var_11]
    var_13 = 4
    var_14 = [var_9, var_13, var_11]
    var_15 = 7
    var_16 = [var_9, var_15, var_11]
    var_17 = 2
    var_18 = 15
    var_19 = 50

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.5
    var_4 = [var_3]
    var_5 = 0
    var_6 = 'USD'
    var_7 = '0.10'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = [var_9, var_12, var_10]
    var_14 = 12
    var_15 = 31
    var_16 = [var_9, var_14, var_15]
    var_17 = 1
    var_18 = None

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.5
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.00'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = [var_9, var_12, var_10]
    var_14 = 12
    var_15 = 31
    var_16 = [var_9, var_14, var_15]
    var_17 = 1
    var_18 = None
    var_19 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.1
    var_4 = [var_3]
    var_5 = 5000
    var_6 = 'USD'
    var_7 = '0.08'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 2
    var_13 = [var_9, var_12, var_10]
    var_14 = 3
    var_15 = [var_9, var_14, var_10]
    var_16 = 12
    var_17 = None
    var_18 = 400



# Parsed testcases at query #11
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

# Partially parsed test_dcfc_30_360_us_basic_cases. Retrieved 18/42 statements.
# Partially parsed test_dcfc_30_360_us_edge_cases. Retrieved 13/37 statements.
# Partially parsed test_dcfc_30_360_us_same_dates. Retrieved 6/18 statements.
# Partially parsed test_dcfc_30_360_us_year_boundaries. Retrieved 11/29 statements.


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
    var_0 = 2020
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
    var_12 = 29
    var_13 = [var_0, var_4, var_12]
    var_14 = [var_0, var_4, var_12]
    var_15 = '0.02777777777778'
    var_16 = [var_15]
    var_17 = [var_0, var_4, var_12]
    var_18 = 3
    var_19 = [var_0, var_18, var_2]
    var_20 = [var_0, var_18, var_2]
    var_21 = '0.08333333333333'
    var_22 = [var_21]
    var_23 = [var_0, var_4, var_12]
    var_24 = 4
    var_25 = 30
    var_26 = [var_0, var_24, var_25]
    var_27 = [var_0, var_24, var_25]
    var_28 = [var_21]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = [var_0, var_1, var_1]
    var_5 = 14
    var_6 = '0.0'
    var_7 = [var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_0, var_8, var_9]
    var_11 = [var_0, var_8, var_9]
    var_12 = [var_0, var_8, var_9]
    var_13 = [var_6]

def test_case_0():
    var_0 = 2019
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2020
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = 14
    var_9 = '0.0'
    var_10 = [var_9]
    var_11 = [var_0, var_1, var_2]
    var_12 = 2
    var_13 = 28
    var_14 = [var_4, var_12, var_13]
    var_15 = [var_4, var_12, var_13]
    var_16 = '0.02777777777778'
    var_17 = [var_16]
    var_18 = [var_0, var_1, var_2]
    var_19 = 29
    var_20 = [var_4, var_12, var_19]
    var_21 = [var_4, var_12, var_19]
    var_22 = [var_16]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_9. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_has_leap_day_with_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_without_leap_year_in_range. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_start. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_end. Retrieved 4/7 statements.
# Partially parsed test_has_leap_day_with_multiple_leap_years_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_no_leap_years_in_range. Retrieved 5/8 statements.
# Partially parsed test_has_leap_day_with_single_day_range_leap_day. Retrieved 3/6 statements.
# Partially parsed test_has_leap_day_with_single_day_range_non_leap_day. Retrieved 3/6 statements.


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
    var_0 = 2016
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/14 statements.


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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_register_altname_conflict. Retrieved 8/16 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 1
    var_5 = [var_4]
    var_6 = 'Test2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = [var_4]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_init_buffer_main_is_dict. Retrieved 2/3 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_existing_main_name. Retrieved 5/7 statements.
# Partially parsed test_find_existing_alt_name. Retrieved 5/7 statements.
# Partially parsed test_find_stripped_uppercase_name. Retrieved 12/14 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'AltTest'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'AltTest'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'AltTest'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = ' test '
    var_6 = var_0.find(var_5)
    var_7 = 'TEST'
    var_8 = var_0.find(var_7)
    var_9 = ' altTest '
    var_10 = var_0.find(var_9)
    var_11 = 'ALTTTEST'
    var_12 = var_0.find(var_11)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_register_successful. Retrieved 6/12 statements.
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
    var_6 = [var_5]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = 'TestAlt2'
    var_8 = {var_7}
    var_9 = {}
    var_10 = 0.6
    var_11 = [var_10]

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = {}
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = 'Test2'
    var_8 = {var_2}
    var_9 = {}
    var_10 = 0.6
    var_11 = [var_10]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_last_day_of_month. Retrieved 9/21 statements.
# Partially parsed test_not_last_day_of_month. Retrieved 9/20 statements.
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

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = 28
    var_5 = [var_0, var_1, var_4]



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 7/19 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 14/34 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_same_start_and_asof_month. Retrieved 6/10 statements.


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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dcfc_30_360_german_with_standard_dates. Retrieved 7/12 statements.
# Partially parsed test_dcfc_30_360_german_with_leap_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_year_end. Retrieved 8/13 statements.
# Partially parsed test_dcfc_30_360_german_with_long_period. Retrieved 8/13 statements.


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_existing_main_name. Retrieved 5/7 statements.
# Partially parsed test_find_existing_alternative_name. Retrieved 5/7 statements.
# Partially parsed test_find_case_insensitive. Retrieved 6/8 statements.
# Partially parsed test_find_with_whitespace. Retrieved 6/8 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_1)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_2)

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
    var_4 = [var_1, var_3]
    var_5 = 'act/act'
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = '  Act/Act  '
    var_6 = var_0.find(var_5)



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_register_altname_conflict. Retrieved 8/15 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 'Test2'
    var_6 = {var_2}
    var_7 = set()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coupon_basic_case. Retrieved 14/24 statements.
# Partially parsed test_coupon_with_eom. Retrieved 15/25 statements.
# Partially parsed test_coupon_zero_fraction. Retrieved 13/23 statements.
# Partially parsed test_coupon_full_period. Retrieved 15/25 statements.
# Partially parsed test_coupon_partial_period. Retrieved 15/25 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.10'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = [var_9, var_12, var_10]
    var_14 = 2021
    var_15 = [var_14, var_10, var_10]
    var_16 = 1
    var_17 = None
    var_18 = 50

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.10'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = 15
    var_12 = [var_9, var_10, var_11]
    var_13 = 4
    var_14 = [var_9, var_13, var_11]
    var_15 = 2021
    var_16 = [var_15, var_10, var_11]
    var_17 = 4
    var_18 = 15
    var_19 = 25

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.10'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = [var_9, var_10, var_10]
    var_13 = 2021
    var_14 = [var_13, var_10, var_10]
    var_15 = 1
    var_16 = None
    var_17 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.10'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 12
    var_13 = 31
    var_14 = [var_9, var_12, var_13]
    var_15 = 2021
    var_16 = [var_15, var_10, var_10]
    var_17 = 1
    var_18 = None
    var_19 = 100

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '0.75'
    var_4 = [var_3]
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.10'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 9
    var_13 = 30
    var_14 = [var_9, var_12, var_13]
    var_15 = 2021
    var_16 = [var_15, var_10, var_10]
    var_17 = 1
    var_18 = None
    var_19 = 75



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_calculate_fraction_with_valid_dates. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_with_invalid_dates. Retrieved 10/19 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 12
    var_10 = 31
    var_11 = [var_4]

def test_case_0():
    var_0 = 'Test'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 2023
    var_7 = 12
    var_8 = 31
    var_9 = 6
    var_10 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcfc_act_365_a. Retrieved 18/42 statements.


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
    var_24 = '0.17213114754098'
    var_25 = [var_24]
    var_26 = '1.08196721311475'
    var_27 = [var_26]
    var_28 = '1.32513661202186'
    var_29 = [var_28]



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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcfc_act_act. Retrieved 18/42 statements.


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
    var_22 = '0.16942884946478'
    var_23 = [var_22]
    var_24 = '0.17216108990194'
    var_25 = [var_24]
    var_26 = '1.08243131970956'
    var_27 = [var_26]
    var_28 = '1.32625945055768'
    var_29 = [var_28]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example_1. Retrieved 7/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_2. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_3. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example_4. Retrieved 8/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 6/12 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_day. Retrieved 3/7 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_year. Retrieved 4/8 statements.
# Partially parsed test_dcfc_30_e_plus_360_leap_year. Retrieved 7/13 statements.


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
    var_0 = 2020
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_1]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '360'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 31
    var_5 = [var_0, var_1, var_4]
    var_6 = 2
    var_7 = [var_0, var_6, var_1]
    var_8 = [var_0, var_1, var_2]
    var_9 = '360'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2021
    var_4 = [var_3, var_1, var_1]
    var_5 = '1'
    var_6 = [var_5]

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
    var_9 = '360'
    var_10 = [var_9]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_interest_with_valid_dates. Retrieved 12/29 statements.
# Partially parsed test_interest_with_asof_equal_to_end. Retrieved 11/27 statements.
# Partially parsed test_interest_with_asof_before_start. Retrieved 12/26 statements.
# Partially parsed test_interest_with_asof_after_end. Retrieved 13/27 statements.
# Partially parsed test_interest_with_frequency. Retrieved 13/33 statements.


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
    var_10 = 12
    var_11 = 31
    var_12 = 364
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

def test_case_0():
    var_0 = 'ACT/360'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = 360
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = 1000
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2023
    var_10 = 1
    var_11 = 6
    var_12 = 12
    var_13 = 31
    var_14 = '2'
    var_15 = [var_14]
    var_16 = 151
    var_17 = [var_16]
    var_18 = [var_3]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_calculate_fraction_invalid_dates. Retrieved 8/16 statements.
# Partially parsed test_calculate_fraction_valid_dates. Retrieved 8/16 statements.
# Partially parsed test_calculate_fraction_with_freq. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.5
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = 3

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.5
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_3]

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = 0.5
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = 0.25
    var_10 = [var_9]
    var_11 = [var_9]



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dcfc_act_act_icma. Retrieved 9/27 statements.


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
    var_11 = 1
    var_12 = [var_7, var_11, var_11]
    var_13 = [var_7, var_11, var_11]
    var_14 = [var_7, var_11, var_11]
    var_15 = [var_7, var_11, var_11]
    var_16 = [var_7, var_11, var_2]
    var_17 = [var_7, var_11, var_1]
    var_18 = '0.5'
    var_19 = [var_18]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dcfc_act_365_a. Retrieved 18/42 statements.


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
    var_24 = '0.17213114754098'
    var_25 = [var_24]
    var_26 = '1.08196721311475'
    var_27 = [var_26]
    var_28 = '1.32513661202186'
    var_29 = [var_28]



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dcfc_30_360_german_predicate_true. Retrieved 7/12 statements.


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

# Partially parsed test_coupon_standard_case. Retrieved 14/22 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/24 statements.
# Partially parsed test_coupon_asof_before_start. Retrieved 15/23 statements.
# Partially parsed test_coupon_asof_after_end. Retrieved 14/22 statements.
# Partially parsed test_coupon_frequency_4. Retrieved 14/22 statements.


def test_case_0():
    var_0 = 'Test'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = {var_2}
    var_4 = 360
    var_5 = lambda s, a, e, f: Decimal((a - s).days / var_4)
    var_6 = [var_0, var_1, var_3, var_5]
    var_7 = 1000
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
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = {var_2}
    var_4 = 360
    var_5 = lambda s, a, e, f: Decimal((a - s).days / var_4)
    var_6 = [var_0, var_1, var_3, var_5]
    var_7 = 1000
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
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = {var_2}
    var_4 = 360
    var_5 = lambda s, a, e, f: Decimal((a - s).days / var_4)
    var_6 = [var_0, var_1, var_3, var_5]
    var_7 = 1000
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 2019
    var_14 = 12
    var_15 = [var_13, var_14, var_11]
    var_16 = 2021
    var_17 = [var_16, var_11, var_11]
    var_18 = 1
    var_19 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = {var_2}
    var_4 = 360
    var_5 = lambda s, a, e, f: Decimal((a - s).days / var_4)
    var_6 = [var_0, var_1, var_3, var_5]
    var_7 = 1000
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 2022
    var_14 = [var_13, var_11, var_11]
    var_15 = 2021
    var_16 = [var_15, var_11, var_11]
    var_17 = 1
    var_18 = 0

def test_case_0():
    var_0 = 'Test'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = {var_2}
    var_4 = 360
    var_5 = lambda s, a, e, f: Decimal((a - s).days / var_4)
    var_6 = [var_0, var_1, var_3, var_5]
    var_7 = 1000
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = 4
    var_14 = [var_10, var_13, var_11]
    var_15 = 2021
    var_16 = [var_15, var_11, var_11]
    var_17 = 4
    var_18 = 12.5



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_30_e_plus_360_predicate_at_line_30. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dcc_registry_machinery_initialization. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_altn



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_coupon_basic_case. Retrieved 14/22 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/24 statements.
# Partially parsed test_coupon_zero_principal. Retrieved 13/21 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 14/22 statements.
# Partially parsed test_coupon_asof_before_start. Retrieved 15/23 statements.
# Partially parsed test_coupon_asof_after_end. Retrieved 15/23 statements.


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
    var_12 = [var_10, var_11, var_11]
    var_13 = 6
    var_14 = 15
    var_15 = [var_10, var_13, var_14]
    var_16 = 2021
    var_17 = [var_16, var_11, var_14]
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
    var_8 = '0'
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
    var_13 = 2019
    var_14 = 6
    var_15 = [var_13, var_14, var_11]
    var_16 = 2021
    var_17 = [var_16, var_11, var_11]
    var_18 = 2
    var_19 = 0

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
    var_13 = 2022
    var_14 = 6
    var_15 = [var_13, var_14, var_11]
    var_16 = 2021
    var_17 = [var_16, var_11, var_11]
    var_18 = 2
    var_19 = 0



# Parsed testcases at query #30
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 5
    var_6 = var_3.day
    assert var_6 == 15

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 30
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 2
    var_6 = var_3.day
    assert var_6 == 28

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

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2020
    var_5 = var_3.month
    assert var_5 == 2
    var_6 = var_3.day
    assert var_6 == 29

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2021
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2021
    var_5 = var_3.month
    assert var_5 == 2
    var_6 = var_3.day
    assert var_6 == 28



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 8/17 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.5
    var_6 = [var_5]
    var_7 = set()
    var_8 = set()
    var_9 = [var_5]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dcfc_act_act_basic_calculation. Retrieved 7/12 statements.
# Partially parsed test_dcfc_act_act_leap_year. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_long_period. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_different_years. Retrieved 8/13 statements.
# Partially parsed test_dcfc_act_act_invalid_date_range. Retrieved 6/10 statements.


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
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2022
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = '0'
    var_8 = [var_7]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 7/19 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 14/34 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_start_after_asof. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_invalid_date_handling. Retrieved 5/9 statements.


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

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_1, var_1]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/9 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/8 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_august. Retrieved 7/11 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_april. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_june_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_annual_frequency_december_start. Retrieved 6/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_january. Retrieved 8/12 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december. Retrieved 6/10 statements.


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



