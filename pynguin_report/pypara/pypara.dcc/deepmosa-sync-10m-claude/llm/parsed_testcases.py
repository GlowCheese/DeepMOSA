####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year_not_last. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_30_days. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_last_day. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 27
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dcfc_act_act_icma_basic. Retrieved 7/16 statements.
# Partially parsed test_dcfc_act_act_icma_same_dates. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_with_freq. Retrieved 7/16 statements.
# Partially parsed test_dcfc_act_act_icma_full_period. Retrieved 4/12 statements.
# Partially parsed test_dcfc_act_act_icma_one_day. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_half_year. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_act_icma_with_none_freq. Retrieved 7/16 statements.


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

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = 12
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = '0'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = 30
    var_5 = [var_0, var_3, var_4]
    var_6 = 2021
    var_7 = [var_6, var_1, var_1]
    var_8 = '2'
    var_9 = [var_8]
    var_10 = '0'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2021
    var_4 = [var_3, var_1, var_1]
    var_5 = [var_3, var_1, var_1]
    var_6 = '1'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 366
    var_6 = [var_0, var_1, var_5]
    var_7 = '0'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 7
    var_4 = [var_0, var_3, var_1]
    var_5 = 2021
    var_6 = [var_5, var_1, var_1]
    var_7 = '0.49'
    var_8 = [var_7]
    var_9 = '0.51'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = 30
    var_5 = [var_0, var_3, var_4]
    var_6 = 2021
    var_7 = [var_6, var_1, var_1]
    var_8 = None
    var_9 = '1'
    var_10 = [var_9]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_register_success. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 10/19 statements.
# Partially parsed test_register_conflict_with_existing_main_name. Retrieved 9/18 statements.
# Partially parsed test_register_empty_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_multiple_dcc. Retrieved 11/19 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Test/Alt1'
    var_3 = 'Test/Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = var_0._buffer_main['Test/DCC']
    var_9 = var_0._buffer_altn['Test/Alt1']
    var_10 = var_0._buffer_altn['Test/Alt2']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = set()
    var_7 = set()
    var_8 = '0.6'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'Test/Alt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'Test/DCC2'
    var_8 = {var_2}
    var_9 = set()
    var_10 = '0.6'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'Test/DCC2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = '0.6'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = var_0._buffer_main['Test/DCC']
    var_7 = var_0._buffer_altn
    var_8 = len(var_7)
    assert var_8 == 0

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'Test/Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'Test/DCC2'
    var_8 = 'Test/Alt2'
    var_9 = {var_8}
    var_10 = set()
    var_11 = '0.6'
    var_12 = [var_11]
    var_13 = var_0._buffer_main['Test/DCC1']
    var_14 = var_0._buffer_main['Test/DCC2']
    var_15 = var_0._buffer_altn['Test/Alt1']
    var_16 = var_0._buffer_altn['Test/Alt2']



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

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_august. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_april. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_before_asof. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_december_start. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_semi_annual_december. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_semi_annual_december_year_end. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_with_eom_parameter. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_eom_adjustment. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_with_decimal_frequency. Retrieved 5/13 statements.
# Partially parsed test_last_payment_date_quarterly_frequency_decimal. Retrieved 6/14 statements.


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
    var_5 = 12
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 2
    var_6 = 28
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_1, var_2]

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
    var_0 = 2008
    var_1 = 7
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 10
    var_5 = 6
    var_6 = [var_3, var_4, var_5]
    var_7 = 4
    var_8 = [var_7]
    var_9 = [var_3, var_1, var_1]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_register_raises_typeerror_when_dcc_name_already_registered. Retrieved 6/15 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Act/Act'
    var_1 = set()
    var_2 = set()
    var_3 = set()
    var_4 = set()
    var_5 = module_0.DCCRegistryMachinery()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'already registered'



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_act_act_one_day_non_leap. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_one_day_leap. Retrieved 5/13 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '365'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '366'
    var_8 = [var_7]



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




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_last_payment_date_predicate_false. Retrieved 7/18 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_register_raises_error_when_altname_already_registered. Retrieved 10/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 'Test2'
    var_7 = 'Alt3'
    var_8 = {var_2, var_7}
    var_9 = set()
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Day count convention 'Test2' is already registered"



# Parsed testcases at query #13
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True



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

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_last_day. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 4
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dcfc_act_act_predicate_false. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = 27
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_5, var_6]
    var_9 = '0'
    var_10 = [var_9]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_with_exact_name. Retrieved 9/14 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 10/15 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 10/15 statements.
# Partially parsed test_find_with_alternative_name_stripped_uppercase. Retrieved 11/16 statements.
# Partially parsed test_find_nonexistent_name. Retrieved 10/15 statements.
# Partially parsed test_find_case_insensitive. Retrieved 10/15 statements.
# Partially parsed test_find_with_whitespace. Retrieved 10/15 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'altnames'
    var_5 = 'Act/Act'
    var_6 = []
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'altnames'
    var_5 = 'ACT/ACT'
    var_6 = []
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = '  act/act  '
    var_10 = var_0.find(var_9)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'altnames'
    var_5 = 'Actual/Actual'
    var_6 = 'Act/Act'
    var_7 = [var_6]
    var_8 = {var_3: var_5, var_4: var_7}
    var_9 = [var_1, var_2, var_8]
    var_10 = var_0.find(var_6)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'altnames'
    var_5 = 'Actual/Actual'
    var_6 = 'Act/Act'
    var_7 = [var_6]
    var_8 = {var_3: var_5, var_4: var_7}
    var_9 = [var_1, var_2, var_8]
    var_10 = '  act/act  '
    var_11 = var_0.find(var_10)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'altnames'
    var_5 = 'Act/Act'
    var_6 = []
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = 'Nonexistent/Convention'
    var_10 = var_0.find(var_9)
    assert var_10 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'altnames'
    var_5 = 'ACT/ACT'
    var_6 = []
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = 'act/act'
    var_10 = var_0.find(var_9)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'altnames'
    var_5 = 'ACT/ACT'
    var_6 = []
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = '   ACT/ACT   '
    var_10 = var_0.find(var_9)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dcfc_30_e_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_start_day_31. Retrieved 14/22 statements.
# Partially parsed test_dcfc_30_e_360_asof_day_31. Retrieved 14/22 statements.
# Partially parsed test_dcfc_30_e_360_both_day_31. Retrieved 13/21 statements.
# Partially parsed test_dcfc_30_e_360_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_360_year_difference. Retrieved 13/21 statements.


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
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 15
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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 30
    var_8 = var_7 - var_2
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
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_2]
    var_6 = 30
    var_7 = var_6 - var_6
    var_8 = var_4 - var_1
    var_9 = var_6 * var_8
    var_10 = var_7 + var_9
    var_11 = 360
    var_12 = var_0 - var_0
    var_13 = var_11 * var_12
    var_14 = var_10 + var_13
    var_15 = [var_14]
    var_16 = [var_11]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 0
    var_6 = [var_5]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2010
    var_5 = [var_4, var_1, var_2]
    var_6 = var_2 - var_2
    var_7 = 30
    var_8 = var_1 - var_1
    var_9 = var_7 * var_8
    var_10 = var_6 + var_9
    var_11 = 360
    var_12 = var_4 - var_0
    var_13 = var_11 * var_12
    var_14 = var_10 + var_13
    var_15 = [var_14]
    var_16 = [var_11]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_a_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_over_year. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_act_365_a_one_day. Retrieved 5/14 statements.
# Partially parsed test_dcfc_act_365_a_with_freq_parameter. Retrieved 6/14 statements.


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
    var_7 = '365'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = 30
    var_5 = [var_0, var_3, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '0'
    var_9 = [var_8]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dcfc_act_365_l_basic. Retrieved 18/45 statements.
# Partially parsed test_dcfc_act_365_l_same_day. Retrieved 4/12 statements.
# Partially parsed test_dcfc_act_365_l_one_day. Retrieved 5/15 statements.
# Partially parsed test_dcfc_act_365_l_leap_year. Retrieved 6/16 statements.
# Partially parsed test_dcfc_act_365_l_with_freq_parameter. Retrieved 8/18 statements.


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
    var_22 = '0.16939890710383'
    var_23 = [var_22]
    var_24 = '0.17213114754098'
    var_25 = [var_24]
    var_26 = '1.08196721311475'
    var_27 = [var_26]
    var_28 = '1.32876712328767'
    var_29 = [var_28]

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
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '365'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = '28'
    var_8 = [var_7]
    var_9 = '366'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = '2'
    var_9 = [var_8]
    var_10 = 14
    var_11 = '0.16939890710383'
    var_12 = [var_11]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_august. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_april. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_frequency_june_start. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_frequency_december. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_end. Retrieved 6/12 statements.


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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_30_days. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_last. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 4
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_30. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_31. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_first_day. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_with_stripped_uppercase_fallback. Retrieved 3/8 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '  act/act  '
    var_2 = var_0.find(var_1)
    var_3 = var_2.name
    assert var_3 == 'ACT/ACT'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_start. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_with_leap_day_at_end. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_multiple_leap_years_in_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_leap_day_before_range. Retrieved 5/10 statements.
# Partially parsed test_has_leap_day_leap_day_after_range. Retrieved 4/9 statements.
# Partially parsed test_has_leap_day_single_day_leap_day. Retrieved 3/8 statements.
# Partially parsed test_has_leap_day_single_day_non_leap_day. Retrieved 3/8 statements.
# Partially parsed test_has_leap_day_century_leap_year. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]

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
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2021
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]

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
    var_3 = 2
    var_4 = 28
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2019
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2000
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 1
    var_6 = [var_0, var_4, var_5]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_same_day. Retrieved 4/11 statements.
# Partially parsed test_dcfc_30_360_us_one_day_difference. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_360_us_month_end_handling. Retrieved 7/16 statements.
# Partially parsed test_dcfc_30_360_us_with_freq_parameter. Retrieved 7/17 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 16
    var_5 = [var_0, var_1, var_4]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '360'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 29
    var_6 = [var_0, var_4, var_5]
    var_7 = '29'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 6
    var_4 = 30
    var_5 = [var_0, var_3, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '180'
    var_9 = [var_8]
    var_10 = '360'
    var_11 = [var_10]



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dcfc_30_e_360_asof_day_not_31. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]
    var_8 = '60'
    var_9 = [var_8]
    var_10 = '360'
    var_11 = [var_10]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 5/9 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_altname_conflicts_with_existing_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_altname_conflicts_with_existing_altname. Retrieved 9/18 statements.
# Partially parsed test_register_multiple_dcc_sequential. Retrieved 10/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_0._buffer_main['Test/DCC']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0
    var_7 = [var_6]
    var_8 = var_0._buffer_main['Test/DCC']
    var_9 = var_0._buffer_altn['Alt1']
    var_10 = var_0._buffer_altn['Alt2']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = [var_4]
    var_6 = set()
    var_7 = set()
    var_8 = [var_4]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = [var_4]
    var_6 = 'Test/DCC2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = [var_4]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'SharedAlt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'Test/DCC2'
    var_8 = {var_2}
    var_9 = set()
    var_10 = [var_5]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'Test/DCC2'
    var_8 = 'Alt2'
    var_9 = {var_8}
    var_10 = set()
    var_11 = [var_5]
    var_12 = var_0._buffer_main['Test/DCC1']
    var_13 = var_0._buffer_main['Test/DCC2']
    var_14 = var_0._buffer_altn['Alt1']
    var_15 = var_0._buffer_altn['Alt2']



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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_nl_365_basic. Retrieved 7/14 statements.
# Partially parsed test_dcfc_nl_365_leap_day. Retrieved 8/15 statements.
# Partially parsed test_dcfc_nl_365_longer_period. Retrieved 8/15 statements.
# Partially parsed test_dcfc_nl_365_another_period. Retrieved 8/15 statements.
# Partially parsed test_dcfc_nl_365_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_nl_365_one_day. Retrieved 5/13 statements.
# Partially parsed test_dcfc_nl_365_with_freq_parameter. Retrieved 6/15 statements.


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
    var_9 = '0.16986301369863'
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
    var_9 = '1.08219178082192'
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
    var_9 = '1.32602739726027'
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
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '365'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '14'
    var_8 = [var_7]
    var_9 = '365'
    var_10 = [var_9]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_with_exact_main_name. Retrieved 5/12 statements.
# Partially parsed test_find_with_exact_alternative_name. Retrieved 5/12 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/12 statements.
# Partially parsed test_find_with_nonexistent_name. Retrieved 5/12 statements.
# Partially parsed test_find_with_lowercase_variant. Retrieved 5/12 statements.
# Partially parsed test_find_with_whitespace_and_case_variation. Retrieved 6/13 statements.
# Partially parsed test_find_alternative_name_with_case_and_whitespace. Retrieved 6/13 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = var_0.find(var_1)

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
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = '  act/act  '
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = 'NonExistent'
    var_5 = var_0.find(var_4)
    assert var_5 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360 US'
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = '30/360 us'
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/365F'
    var_2 = 'Act/365 Fixed'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = '  act/365f  '
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/365F'
    var_2 = 'Act/365 Fixed'
    var_3 = [var_2]
    var_4 = [var_1, var_3]
    var_5 = '  act/365 fixed  '
    var_6 = var_0.find(var_5)



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_has_leap_day_with_leap_day_in_range. Retrieved 4/10 statements.
# Partially parsed test_has_leap_day_without_leap_day_in_range. Retrieved 4/10 statements.
# Partially parsed test_has_leap_day_multiple_leap_years_with_leap_day. Retrieved 4/10 statements.
# Partially parsed test_has_leap_day_before_leap_day. Retrieved 4/10 statements.
# Partially parsed test_has_leap_day_after_leap_day. Retrieved 5/11 statements.
# Partially parsed test_has_leap_day_exact_leap_day_range. Retrieved 3/9 statements.
# Partially parsed test_has_leap_day_spanning_multiple_leap_years. Retrieved 5/11 statements.
# Partially parsed test_has_leap_day_no_leap_years_in_range. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = [var_0, var_4, var_2]

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
    var_4 = 2024
    var_5 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = 28
    var_5 = [var_0, var_3, var_4]

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
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2019
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 5/9 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name_raises_error. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_altname_raises_error. Retrieved 10/19 statements.
# Partially parsed test_register_altname_conflicts_with_main_name_raises_error. Retrieved 9/18 statements.
# Partially parsed test_register_multiple_valid_dccs. Retrieved 11/19 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = var_0._buffer_main['Test/DCC']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'ALT1'
    var_3 = 'ALT2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = var_0._buffer_main['Test/DCC']
    var_9 = var_0._buffer_altn['ALT1']
    var_10 = var_0._buffer_altn['ALT2']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = set()
    var_7 = set()
    var_8 = '0.6'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'SHARED'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'Test/DCC2'
    var_8 = {var_2}
    var_9 = set()
    var_10 = '0.6'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = 'Test/DCC2'
    var_7 = {var_1}
    var_8 = set()
    var_9 = '0.6'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'ALT1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'Test/DCC2'
    var_8 = 'ALT2'
    var_9 = {var_8}
    var_10 = set()
    var_11 = '0.6'
    var_12 = [var_11]
    var_13 = var_0._buffer_main['Test/DCC1']
    var_14 = var_0._buffer_main['Test/DCC2']
    var_15 = var_0._buffer_altn['ALT1']
    var_16 = var_0._buffer_altn['ALT2']



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_register_duplicate_main_name_raises_typeerror. Retrieved 6/14 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = set()
    var_3 = set()
    var_4 = set()
    var_5 = set()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Day count convention 'Act/Act' is already registered"



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dcfc_act_365_l. Retrieved 25/71 statements.


def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = [var_0, var_1, var_1]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = [var_0, var_1, var_1]
    var_8 = 2
    var_9 = [var_0, var_1, var_8]
    var_10 = [var_0, var_1, var_8]
    var_11 = '1'
    var_12 = [var_11]
    var_13 = '365'
    var_14 = [var_13]
    var_15 = 2007
    var_16 = 12
    var_17 = 28
    var_18 = [var_15, var_16, var_17]
    var_19 = 2008
    var_20 = [var_19, var_8, var_17]
    var_21 = 14
    var_22 = '0.16939890710383'
    var_23 = [var_22]
    var_24 = [var_15, var_16, var_17]
    var_25 = 29
    var_26 = [var_19, var_8, var_25]
    var_27 = '0.17213114754098'
    var_28 = [var_27]
    var_29 = 10
    var_30 = 31
    var_31 = [var_15, var_29, var_30]
    var_32 = 11
    var_33 = 30
    var_34 = [var_19, var_32, var_33]
    var_35 = '1.08196721311475'
    var_36 = [var_35]
    var_37 = [var_19, var_8, var_1]
    var_38 = 2009
    var_39 = 5
    var_40 = [var_38, var_39, var_30]
    var_41 = '1.32876712328767'
    var_42 = [var_41]
    var_43 = 2020
    var_44 = [var_43, var_1, var_1]
    var_45 = [var_43, var_1, var_8]
    var_46 = [var_11]
    var_47 = '366'
    var_48 = [var_47]
    var_49 = 2019
    var_50 = [var_49, var_1, var_1]
    var_51 = [var_49, var_1, var_8]
    var_52 = [var_11]
    var_53 = [var_13]



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

# Partially parsed test_init_creates_empty_buffers. Retrieved 7/9 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = var_0._buffer_main
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = var_0._buffer_altn
    var_5 = var_0._buffer_altn
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_interest_calculates_accrued_interest_correctly. Retrieved 12/26 statements.
# Partially parsed test_interest_uses_asof_as_end_when_end_is_none. Retrieved 11/24 statements.
# Partially parsed test_interest_with_frequency_parameter. Retrieved 14/30 statements.
# Partially parsed test_interest_with_zero_rate. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = [var_3]
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 6
    var_11 = 30
    var_12 = [var_7, var_10, var_11]
    var_13 = 12
    var_14 = 31
    var_15 = [var_7, var_13, var_14]
    var_16 = '0.5'
    var_17 = [var_16]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = [var_3]
    var_5 = '0.1'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 3
    var_11 = 31
    var_12 = [var_7, var_10, var_11]
    var_13 = None
    var_14 = '0.25'
    var_15 = [var_14]

def test_case_0():
    var_0 = []
    var_1 = 'Test DCC'
    var_2 = set()
    var_3 = set()
    var_4 = '5000'
    var_5 = [var_4]
    var_6 = '0.02'
    var_7 = [var_6]
    var_8 = 2024
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = 9
    var_12 = 30
    var_13 = [var_8, var_11, var_12]
    var_14 = 12
    var_15 = 31
    var_16 = [var_8, var_14, var_15]
    var_17 = '4'
    var_18 = [var_17]
    var_19 = '0.75'
    var_20 = [var_19]
    var_21 = var_0[0][3]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 6
    var_11 = 30
    var_12 = [var_7, var_10, var_11]
    var_13 = 12
    var_14 = 31
    var_15 = [var_7, var_13, var_14]
    var_16 = [var_5]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coupon_basic. Retrieved 12/26 statements.
# Partially parsed test_coupon_with_eom. Retrieved 16/30 statements.
# Partially parsed test_coupon_annual_frequency. Retrieved 13/27 statements.
# Partially parsed test_coupon_quarterly_frequency. Retrieved 14/28 statements.
# Partially parsed test_coupon_with_decimal_frequency. Retrieved 14/29 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = [var_3]
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 6
    var_11 = 15
    var_12 = [var_7, var_10, var_11]
    var_13 = 2015
    var_14 = [var_13, var_8, var_8]
    var_15 = 2
    var_16 = '0.5'
    var_17 = [var_16]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '5000'
    var_4 = [var_3]
    var_5 = '0.10'
    var_6 = [var_5]
    var_7 = 2012
    var_8 = 12
    var_9 = 15
    var_10 = [var_7, var_8, var_9]
    var_11 = 2015
    var_12 = 31
    var_13 = [var_11, var_8, var_12]
    var_14 = 2016
    var_15 = 1
    var_16 = 6
    var_17 = [var_14, var_15, var_16]
    var_18 = 2
    var_19 = 15
    var_20 = '0.25'
    var_21 = [var_20]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '10000'
    var_4 = [var_3]
    var_5 = '0.02'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 12
    var_11 = 9
    var_12 = [var_7, var_10, var_11]
    var_13 = 2015
    var_14 = 4
    var_15 = [var_13, var_10, var_14]
    var_16 = 1
    var_17 = '0.75'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = [var_3]
    var_5 = '0.08'
    var_6 = [var_5]
    var_7 = 2008
    var_8 = 7
    var_9 = [var_7, var_8, var_8]
    var_10 = 2015
    var_11 = 10
    var_12 = 6
    var_13 = [var_10, var_11, var_12]
    var_14 = 2016
    var_15 = 1
    var_16 = [var_14, var_15, var_15]
    var_17 = 4
    var_18 = '0.33'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '7500'
    var_4 = [var_3]
    var_5 = '0.06'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 6
    var_9 = 1
    var_10 = [var_7, var_8, var_9]
    var_11 = 2015
    var_12 = 4
    var_13 = 30
    var_14 = [var_11, var_12, var_13]
    var_15 = 2016
    var_16 = [var_15, var_9, var_9]
    var_17 = '2'
    var_18 = [var_17]
    var_19 = '0.5'
    var_20 = [var_19]



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_date_range. Retrieved 6/17 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 2/8 statements.
# Partially parsed test_get_date_range_two_days. Retrieved 3/11 statements.
# Partially parsed test_get_date_range_different_months. Retrieved 5/15 statements.
# Partially parsed test_get_date_range_different_years. Retrieved 6/15 statements.


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
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_4]
    var_6 = [var_0, var_1, var_2]
    var_7 = 31
    var_8 = [var_0, var_1, var_7]
    var_9 = [var_0, var_4, var_1]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_0, var_1, var_2]
    var_9 = [var_4, var_5, var_5]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_register_valid_dcc. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 9/18 statements.
# Partially parsed test_register_altname_conflicts_with_existing_altname. Retrieved 10/19 statements.
# Partially parsed test_register_multiple_valid_dcc. Retrieved 15/23 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/360'
    var_2 = 'Test360'
    var_3 = 'TestDCC'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = var_0._buffer_main['Test/360']
    var_9 = var_0._buffer_altn['Test360']
    var_10 = var_0._buffer_altn['TestDCC']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = set()
    var_7 = set()
    var_8 = '0.6'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = set()
    var_8 = set()
    var_9 = '0.6'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'Actual/365'
    var_8 = {var_2}
    var_9 = set()
    var_10 = '0.6'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'Actual/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'Act/365'
    var_8 = 'Actual/365'
    var_9 = {var_8}
    var_10 = set()
    var_11 = '0.6'
    var_12 = [var_11]
    var_13 = var_0._buffer_main
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_0._buffer_altn
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_0._buffer_main['Act/360']
    var_18 = var_0._buffer_main['Act/365']



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_register_raises_error_when_altname_already_registered. Retrieved 8/16 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'DCC1'
    var_1 = 'ALT1'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 'DCC2'
    var_5 = {var_1}
    var_6 = set()
    var_7 = module_0.DCCRegistryMachinery()
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'already registered'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_register_valid_dcc. Retrieved 6/10 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 6/13 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 8/15 statements.
# Partially parsed test_register_altname_conflicts_with_main_name. Retrieved 7/14 statements.
# Partially parsed test_register_multiple_altnames. Retrieved 11/15 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Test/Alternative'
    var_3 = 'T/D'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = var_0._buffer_main['Test/DCC']
    var_7 = var_0._buffer_altn['Test/Alternative']
    var_8 = var_0._buffer_altn['T/D']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = set()
    var_5 = set()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'Test/Alternative'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 'Test/DCC2'
    var_6 = {var_2}
    var_7 = set()
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = set()
    var_3 = set()
    var_4 = 'Test/DCC2'
    var_5 = {var_1}
    var_6 = set()
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = 'Alt3'
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = var_0._buffer_main['Test/DCC']
    var_8 = var_0._buffer_altn['Alt1']
    var_9 = var_0._buffer_altn['Alt2']
    var_10 = var_0._buffer_altn['Alt3']
    var_11 = var_0._buffer_main
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_0._buffer_altn
    var_14 = len(var_13)
    assert var_14 == 3



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_last_payment_date_annual_frequency. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_annual_frequency_same_year. Retrieved 4/10 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_august. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_april. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_frequency_june_start. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_quarterly_frequency. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_annual_frequency_december. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_semi_annual_frequency_december_31. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_with_eom_parameter. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_before_first_payment. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_eom_adjustment. Retrieved 5/11 statements.


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
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 12
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 28
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_1, var_2]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_coupon_basic. Retrieved 14/26 statements.
# Partially parsed test_coupon_with_eom. Retrieved 15/27 statements.
# Partially parsed test_coupon_annual_frequency. Retrieved 13/25 statements.
# Partially parsed test_coupon_semi_annual. Retrieved 16/28 statements.


def test_case_0():
    var_0 = 'Actual/Actual'
    var_1 = set()
    var_2 = set()
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '1000'
    var_6 = [var_5]
    var_7 = '0.05'
    var_8 = [var_7]
    var_9 = 2014
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 6
    var_13 = 15
    var_14 = [var_9, var_12, var_13]
    var_15 = 2015
    var_16 = [var_15, var_10, var_10]
    var_17 = 2
    var_18 = None
    var_19 = '25'
    var_20 = [var_19]

def test_case_0():
    var_0 = '30/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.25'
    var_4 = [var_3]
    var_5 = '10000'
    var_6 = [var_5]
    var_7 = '0.04'
    var_8 = [var_7]
    var_9 = 2014
    var_10 = 1
    var_11 = 31
    var_12 = [var_9, var_10, var_11]
    var_13 = 3
    var_14 = 15
    var_15 = [var_9, var_13, var_14]
    var_16 = 7
    var_17 = [var_9, var_16, var_11]
    var_18 = 4
    var_19 = 31
    var_20 = '100'
    var_21 = [var_20]

def test_case_0():
    var_0 = 'Actual/365'
    var_1 = set()
    var_2 = set()
    var_3 = '0.75'
    var_4 = [var_3]
    var_5 = '5000'
    var_6 = [var_5]
    var_7 = '0.06'
    var_8 = [var_7]
    var_9 = 2014
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 10
    var_13 = [var_9, var_12, var_10]
    var_14 = 2015
    var_15 = [var_14, var_10, var_10]
    var_16 = 1
    var_17 = None
    var_18 = '225'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'Actual/360'
    var_1 = set()
    var_2 = set()
    var_3 = '0.3'
    var_4 = [var_3]
    var_5 = '2000'
    var_6 = [var_5]
    var_7 = '0.08'
    var_8 = [var_7]
    var_9 = 2012
    var_10 = 12
    var_11 = 15
    var_12 = [var_9, var_10, var_11]
    var_13 = 2015
    var_14 = 31
    var_15 = [var_13, var_10, var_14]
    var_16 = 2016
    var_17 = 6
    var_18 = [var_16, var_17, var_11]
    var_19 = 2
    var_20 = 15
    var_21 = '48'
    var_22 = [var_21]



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

# Partially parsed test_has_leap_day_no_leap_day_in_range. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 2021
    var_1 = 3
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 12
    var_5 = 31
    var_6 = [var_0, var_4, var_5]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_a_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_long_period_1. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_long_period_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_act_365_a_one_day. Retrieved 5/14 statements.
# Partially parsed test_dcfc_act_365_a_with_freq_parameter. Retrieved 8/17 statements.


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
    var_7 = '365'
    var_8 = [var_7]

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
    var_9 = 14
    var_10 = '0.16986301369863'
    var_11 = [var_10]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_register_raises_typeerror_when_altname_already_registered. Retrieved 10/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 'Test2'
    var_7 = 'Alt3'
    var_8 = {var_2, var_7}
    var_9 = set()
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'already registered'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_day. Retrieved 3/9 statements.
# Partially parsed test_dcfc_act_act_one_day_non_leap. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_one_day_leap. Retrieved 5/13 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '365'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '366'
    var_8 = [var_7]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dcfc_act_act_icma_basic. Retrieved 7/16 statements.
# Partially parsed test_dcfc_act_act_icma_same_dates. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_with_freq. Retrieved 7/21 statements.
# Partially parsed test_dcfc_act_act_icma_one_day_period. Retrieved 4/12 statements.
# Partially parsed test_dcfc_act_act_icma_half_period. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_act_icma_with_freq_4. Retrieved 7/16 statements.


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

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 2020
    var_6 = [var_5, var_1, var_2]
    var_7 = '0'
    var_8 = [var_7]

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
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = [var_0, var_1, var_1]
    var_6 = '1'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 7
    var_4 = [var_0, var_3, var_1]
    var_5 = 2020
    var_6 = [var_5, var_1, var_1]
    var_7 = 2
    var_8 = '0.50'
    var_9 = [var_8]

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
    var_9 = '4'
    var_10 = [var_9]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_next_payment_date_annual_frequency. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_annual_frequency_with_eom. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_semi_annual_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_quarterly_frequency. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_monthly_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_with_eom_february. Retrieved 4/11 statements.
# Partially parsed test_next_payment_date_eom_invalid_day. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_with_decimal_frequency. Retrieved 5/12 statements.


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
    var_3 = 4
    var_4 = None
    var_5 = [var_0, var_3, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = None
    var_5 = 2
    var_6 = [var_0, var_5, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1]
    var_5 = 2015
    var_6 = [var_5, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = None
    var_6 = 7
    var_7 = [var_0, var_6, var_1]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_a_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_year_span. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_act_365_a_one_day. Retrieved 5/14 statements.
# Partially parsed test_dcfc_act_365_a_with_freq_parameter. Retrieved 6/16 statements.


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
    var_7 = '365'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_3, var_1]
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '31'
    var_8 = [var_7]
    var_9 = '365'
    var_10 = [var_9]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_coupon_basic_annual_frequency. Retrieved 11/23 statements.
# Partially parsed test_coupon_with_eom. Retrieved 14/26 statements.
# Partially parsed test_coupon_semi_annual_frequency. Retrieved 13/25 statements.
# Partially parsed test_coupon_quarterly_frequency. Retrieved 14/26 statements.
# Partially parsed test_coupon_with_decimal_frequency. Retrieved 11/24 statements.
# Partially parsed test_coupon_with_none_eom. Retrieved 14/26 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = [var_3]
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 6
    var_11 = [var_7, var_10, var_8]
    var_12 = 2015
    var_13 = [var_12, var_8, var_8]
    var_14 = 1
    var_15 = '25'
    var_16 = [var_15]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = [var_3]
    var_5 = '0.04'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = 31
    var_10 = [var_7, var_8, var_9]
    var_11 = 3
    var_12 = 15
    var_13 = [var_7, var_11, var_12]
    var_14 = 7
    var_15 = [var_7, var_14, var_9]
    var_16 = 2
    var_17 = 31
    var_18 = '20'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '5000'
    var_4 = [var_3]
    var_5 = '0.06'
    var_6 = [var_5]
    var_7 = 2012
    var_8 = 12
    var_9 = 15
    var_10 = [var_7, var_8, var_9]
    var_11 = 2016
    var_12 = 1
    var_13 = 6
    var_14 = [var_11, var_12, var_13]
    var_15 = [var_11, var_13, var_9]
    var_16 = 2
    var_17 = '1125'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '10000'
    var_4 = [var_3]
    var_5 = '0.08'
    var_6 = [var_5]
    var_7 = 2008
    var_8 = 7
    var_9 = [var_7, var_8, var_8]
    var_10 = 2015
    var_11 = 10
    var_12 = 6
    var_13 = [var_10, var_11, var_12]
    var_14 = 2016
    var_15 = 1
    var_16 = [var_14, var_15, var_8]
    var_17 = 4
    var_18 = '80'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '3000'
    var_4 = [var_3]
    var_5 = '0.03'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 9
    var_11 = [var_7, var_10, var_8]
    var_12 = 2015
    var_13 = [var_12, var_8, var_8]
    var_14 = '1'
    var_15 = [var_14]
    var_16 = '45'
    var_17 = [var_16]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1500'
    var_4 = [var_3]
    var_5 = '0.07'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 6
    var_9 = 1
    var_10 = [var_7, var_8, var_9]
    var_11 = 2015
    var_12 = 4
    var_13 = 30
    var_14 = [var_11, var_12, var_13]
    var_15 = [var_11, var_8, var_9]
    var_16 = 1
    var_17 = None
    var_18 = '346.5'
    var_19 = [var_18]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_calculate_daily_fraction_basic. Retrieved 9/20 statements.
# Partially parsed test_calculate_daily_fraction_with_different_fractions. Retrieved 12/34 statements.
# Partially parsed test_calculate_daily_fraction_asof_equals_start. Retrieved 8/21 statements.
# Partially parsed test_calculate_daily_fraction_with_freq. Retrieved 10/24 statements.
# Partially parsed test_calculate_daily_fraction_returns_decimal. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 3
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = [var_0, var_1, var_1]
    var_9 = 3
    var_10 = [var_0, var_1, var_9]
    var_11 = [var_0, var_5, var_6]
    var_12 = '0.05'
    var_13 = [var_12]
    var_14 = '0.15'
    var_15 = [var_14]
    var_16 = 'Test DCC'
    var_17 = set()
    var_18 = set()
    var_19 = [var_0, var_1, var_1]
    var_20 = [var_0, var_1, var_9]
    var_21 = [var_0, var_5, var_6]
    var_22 = '0.1'
    var_23 = [var_22]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_3, var_4, var_4]
    var_7 = 12
    var_8 = 31
    var_9 = [var_3, var_7, var_8]
    var_10 = '0.05'
    var_11 = [var_10]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 3
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '4'
    var_12 = [var_11]
    var_13 = '0'
    var_14 = [var_13]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 3
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_same_dates. Retrieved 4/12 statements.
# Partially parsed test_dcfc_30_360_us_one_day_difference. Retrieved 6/16 statements.
# Partially parsed test_dcfc_30_360_us_month_end_adjustment. Retrieved 7/17 statements.
# Partially parsed test_dcfc_30_360_us_year_change. Retrieved 7/17 statements.
# Partially parsed test_dcfc_30_360_us_d2_31_with_d1_30. Retrieved 6/16 statements.
# Partially parsed test_dcfc_30_360_us_multiple_months. Retrieved 5/15 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = '0'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 16
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = '1'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 29
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = '29'
    var_9 = [var_8]
    var_10 = '360'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = [var_4, var_5, var_5]
    var_8 = '1'
    var_9 = [var_8]
    var_10 = '360'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 31
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_3, var_1]
    var_5 = [var_0, var_3, var_1]
    var_6 = '60'
    var_7 = [var_6]
    var_8 = '360'
    var_9 = [var_8]



