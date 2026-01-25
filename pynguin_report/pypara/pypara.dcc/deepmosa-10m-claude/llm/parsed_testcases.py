####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 5/9 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name_raises_error. Retrieved 8/17 statements.
# Partially parsed test_register_duplicate_altname_raises_error. Retrieved 10/19 statements.
# Partially parsed test_register_altname_conflicts_with_main_name_raises_error. Retrieved 9/18 statements.
# Partially parsed test_register_multiple_dcc_success. Retrieved 11/19 statements.


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
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
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
    var_2 = 'SharedAlt'
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
    var_2 = 'Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = 'Test/DCC2'
    var_8 = 'Alt2'
    var_9 = {var_8}
    var_10 = set()
    var_11 = '0.6'
    var_12 = [var_11]
    var_13 = var_0._buffer_main['Test/DCC1']
    var_14 = var_0._buffer_main['Test/DCC2']
    var_15 = var_0._buffer_altn['Alt1']
    var_16 = var_0._buffer_altn['Alt2']



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 5/9 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_altname_conflicts_with_main_name. Retrieved 8/17 statements.
# Partially parsed test_register_altname_conflicts_with_existing_altname. Retrieved 9/18 statements.
# Partially parsed test_register_multiple_dcc_no_conflicts. Retrieved 14/22 statements.


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
    var_1 = 'Main/Name'
    var_2 = 'Alt/Name1'
    var_3 = 'Alt/Name2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0
    var_7 = [var_6]
    var_8 = var_0._buffer_main['Main/Name']
    var_9 = var_0._buffer_altn['Alt/Name1']
    var_10 = var_0._buffer_altn['Alt/Name2']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Duplicate/DCC'
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
    var_1 = 'First/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = [var_4]
    var_6 = 'Second/DCC'
    var_7 = {var_1}
    var_8 = set()
    var_9 = [var_4]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'First/DCC'
    var_2 = 'Shared/Alt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'Second/DCC'
    var_8 = {var_2}
    var_9 = set()
    var_10 = [var_5]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'DCC/One'
    var_2 = 'Alt1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = [var_5]
    var_7 = 'DCC/Two'
    var_8 = 'Alt2'
    var_9 = {var_8}
    var_10 = set()
    var_11 = [var_5]
    var_12 = var_0._buffer_main
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = var_0._buffer_altn
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_0._buffer_main['DCC/One']
    var_17 = var_0._buffer_main['DCC/Two']



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dcfc_act_act_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_act_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_act_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_act_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_act_same_day. Retrieved 3/10 statements.
# Partially parsed test_dcfc_act_act_one_day_non_leap_year. Retrieved 5/14 statements.
# Partially parsed test_dcfc_act_act_one_day_leap_year. Retrieved 5/14 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_360_isda_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_360_isda_both_30th. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_360_isda_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_isda_one_year_apart. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_isda_with_freq_parameter. Retrieved 6/15 statements.


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
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 15
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_5]
    var_8 = 360
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 29
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_5]
    var_8 = 360
    var_9 = [var_8]

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
    var_4 = 2009
    var_5 = [var_4, var_1, var_2]
    var_6 = [var_1]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_4]
    var_7 = 30
    var_8 = [var_7]
    var_9 = 360
    var_10 = [var_9]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_register_successful. Retrieved 7/11 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 7/16 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 9/18 statements.
# Partially parsed test_register_main_name_conflicts_with_existing_altname. Retrieved 8/17 statements.
# Partially parsed test_register_multiple_altnames. Retrieved 10/14 statements.
# Partially parsed test_register_empty_altnames. Retrieved 7/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = 'Test/Alt1'
    var_3 = 'Test/Alt2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0
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
    var_2 = 'Test/Alt'
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
    var_2 = 'Test/Alt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0
    var_6 = [var_5]
    var_7 = set()
    var_8 = set()
    var_9 = [var_5]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Primary'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = 'Alt3'
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = 0
    var_8 = [var_7]
    var_9 = var_0._buffer_altn
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_0._buffer_altn['Alt1']
    var_12 = var_0._buffer_altn['Alt2']
    var_13 = var_0._buffer_altn['Alt3']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_0._buffer_main['Test/DCC']
    var_7 = var_0._buffer_altn
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dcfc_act_365_l_same_day. Retrieved 3/10 statements.
# Partially parsed test_dcfc_act_365_l_one_day. Retrieved 4/13 statements.
# Partially parsed test_dcfc_act_365_l_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_l_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_l_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_l_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_l_leap_year_divisor. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_365_l_non_leap_year_divisor. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_365_l_multiple_days. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = [var_0, var_1, var_1]
    var_5 = 0
    var_6 = [var_5]

def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = [var_1]
    var_7 = 365
    var_8 = [var_7]

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
    var_9 = '1.32876712328767'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = 364
    var_8 = [var_7]
    var_9 = 366
    var_10 = [var_9]

def test_case_0():
    var_0 = 2007
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]
    var_6 = [var_0, var_3, var_4]
    var_7 = 364
    var_8 = [var_7]
    var_9 = 365
    var_10 = [var_9]

def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 10
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_3]
    var_6 = 9
    var_7 = [var_6]
    var_8 = 365
    var_9 = [var_8]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_last_payment_date_basic_annual. Retrieved 5/11 statements.
# Partially parsed test_last_payment_date_same_year. Retrieved 4/10 statements.
# Partially parsed test_last_payment_date_semi_annual. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_august. Retrieved 7/13 statements.
# Partially parsed test_last_payment_date_semi_annual_april. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_june_start. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_quarterly. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_december_start. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_semi_annual_december. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_semi_annual_december_end_year. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_with_eom_parameter. Retrieved 6/12 statements.
# Partially parsed test_last_payment_date_february_eom. Retrieved 8/14 statements.
# Partially parsed test_last_payment_date_before_start_date. Retrieved 5/11 statements.


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
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = 3
    var_6 = 15
    var_7 = [var_4, var_5, var_6]
    var_8 = 2
    var_9 = 28
    var_10 = [var_4, var_8, var_9]

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_1, var_2]



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

# Partially parsed test_find_with_exact_name. Retrieved 4/8 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/9 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 5/9 statements.
# Partially parsed test_find_with_alternative_name_stripped_uppercase. Retrieved 6/10 statements.
# Partially parsed test_find_nonexistent_name. Retrieved 5/9 statements.
# Partially parsed test_find_case_insensitive. Retrieved 5/9 statements.
# Partially parsed test_find_with_whitespace_handling. Retrieved 5/9 statements.


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
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = []
    var_4 = '  act/act  '
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = []
    var_5 = var_0.find(var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = []
    var_5 = '  act/act  '
    var_6 = var_0.find(var_5)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = []
    var_4 = 'Nonexistent'
    var_5 = var_0.find(var_4)
    assert var_5 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360 US'
    var_2 = []
    var_3 = []
    var_4 = '30/360 us'
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30E/360'
    var_2 = []
    var_3 = []
    var_4 = '   30e/360   '
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = var_0.find(var_1)
    assert var_2 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_calculate_daily_fraction_basic. Retrieved 9/20 statements.
# Partially parsed test_calculate_daily_fraction_with_different_values. Retrieved 9/24 statements.
# Partially parsed test_calculate_daily_fraction_asof_minus_1_before_start. Retrieved 9/20 statements.
# Partially parsed test_calculate_daily_fraction_with_freq. Retrieved 10/24 statements.
# Partially parsed test_calculate_daily_fraction_negative_result. Retrieved 9/24 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0.1'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 5
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_4, var_5]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0.05'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 10
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '4'
    var_12 = [var_11]
    var_13 = '0.2'
    var_14 = [var_13]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '-0.1'
    var_12 = [var_11]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_coupon_basic. Retrieved 12/26 statements.
# Partially parsed test_coupon_with_eom. Retrieved 14/28 statements.
# Partially parsed test_coupon_different_frequencies. Retrieved 12/26 statements.
# Partially parsed test_coupon_zero_rate. Retrieved 11/23 statements.
# Partially parsed test_coupon_large_principal. Retrieved 11/25 statements.


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
    var_15 = 1
    var_16 = '0.5'
    var_17 = [var_16]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '5000'
    var_4 = [var_3]
    var_5 = '0.03'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = 31
    var_10 = [var_7, var_8, var_9]
    var_11 = 7
    var_12 = 15
    var_13 = [var_7, var_11, var_12]
    var_14 = 2015
    var_15 = [var_14, var_8, var_9]
    var_16 = 2
    var_17 = 31
    var_18 = '0.25'
    var_19 = [var_18]

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
    var_10 = 3
    var_11 = 15
    var_12 = [var_7, var_10, var_11]
    var_13 = 4
    var_14 = [var_7, var_13, var_8]
    var_15 = 4
    var_16 = '0.1'
    var_17 = [var_16]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 6
    var_11 = 15
    var_12 = [var_7, var_10, var_11]
    var_13 = 2015
    var_14 = [var_13, var_8, var_8]
    var_15 = 1
    var_16 = [var_5]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000000'
    var_4 = [var_3]
    var_5 = '0.04'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 9
    var_11 = [var_7, var_10, var_8]
    var_12 = 2015
    var_13 = [var_12, var_8, var_8]
    var_14 = 1
    var_15 = '0.75'
    var_16 = [var_15]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dcfc_act_act_icma_basic. Retrieved 7/16 statements.
# Partially parsed test_dcfc_act_act_icma_same_dates. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_asof_equals_end. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_with_frequency. Retrieved 8/21 statements.
# Partially parsed test_dcfc_act_act_icma_one_day_period. Retrieved 4/12 statements.
# Partially parsed test_dcfc_act_act_icma_half_period. Retrieved 7/16 statements.


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
    var_4 = 2020
    var_5 = [var_4, var_1, var_2]
    var_6 = [var_4, var_1, var_2]
    var_7 = '1'
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
    var_11 = '0.5245901639'
    var_12 = [var_11]
    var_13 = [var_9]

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
    var_4 = 2
    var_5 = [var_0, var_3, var_4]
    var_6 = 2020
    var_7 = [var_6, var_1, var_1]
    var_8 = '0.49'
    var_9 = [var_8]
    var_10 = '0.51'
    var_11 = [var_10]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_with_exact_name. Retrieved 4/12 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/11 statements.
# Partially parsed test_find_case_insensitive. Retrieved 5/11 statements.
# Partially parsed test_find_with_whitespace. Retrieved 5/11 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = var_0.find(var_1)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.name
    assert var_5 == 'Act/Act'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == 'Actual/Actual'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = '  act/act  '
    var_4 = var_0.find(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == 'ACT/ACT'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'NonExistent/Convention'
    var_2 = var_0.find(var_1)
    assert var_2 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360 ISDA'
    var_2 = []
    var_3 = '30/360 isda'
    var_4 = var_0.find(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == '30/360 ISDA'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACTUAL/365'
    var_2 = []
    var_3 = '   actual/365   '
    var_4 = var_0.find(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == 'ACTUAL/365'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcc_registry_machinery_initialization. Retrieved 7/9 statements.


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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_next_payment_date_annual_frequency. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_annual_frequency_with_eom. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_semi_annual_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_quarterly_frequency. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_monthly_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_with_decimal_frequency. Retrieved 5/12 statements.
# Partially parsed test_next_payment_date_eom_invalid_day. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_eom_february. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_eom_none. Retrieved 8/16 statements.


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
    var_2 = [var_0, var_1, var_1]
    var_3 = '2'
    var_4 = [var_3]
    var_5 = None
    var_6 = 7
    var_7 = [var_0, var_6, var_1]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 7
    var_6 = [var_0, var_5, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2015
    var_5 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 3
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = None
    var_6 = 9
    var_7 = [var_0, var_6, var_2]
    var_8 = 10
    var_9 = 1
    var_10 = [var_0, var_8, var_9]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dcc_registry_machinery_init. Retrieved 7/9 statements.


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_same_date. Retrieved 4/11 statements.
# Partially parsed test_dcfc_30_360_us_one_day. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_360_us_end_of_month_adjustment. Retrieved 7/16 statements.
# Partially parsed test_dcfc_30_360_us_month_transition. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_360_us_year_transition. Retrieved 7/16 statements.


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
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = '30'
    var_7 = [var_6]
    var_8 = '360'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 1
    var_6 = [var_4, var_5, var_2]
    var_7 = '30'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dcc_act_act_predicate. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'Act/Act'
    var_4 = 'Actual/Actual'
    var_5 = 'Actual/Actual (ISDA)'
    var_6 = {var_4, var_5}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_last_payment_date_predicate. Retrieved 17/59 statements.


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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_57. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 15
    var_2 = [var_0, var_0, var_1]
    var_3 = 10
    var_4 = [var_0, var_0, var_3]
    var_5 = 1



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dcfc_30_360_us_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_us_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_us_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_us_one_day_difference. Retrieved 5/14 statements.
# Partially parsed test_dcfc_30_360_us_month_difference. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_360_us_year_difference. Retrieved 5/14 statements.
# Partially parsed test_dcfc_30_360_us_last_day_of_february_non_leap. Retrieved 7/16 statements.
# Partially parsed test_dcfc_30_360_us_day_31_adjustment. Retrieved 7/16 statements.


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
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '360'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = '30'
    var_7 = [var_6]
    var_8 = '360'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = [var_4, var_1, var_2]
    var_6 = '360'
    var_7 = [var_6]
    var_8 = [var_6]

def test_case_0():
    var_0 = 2007
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_dates. Retrieved 3/9 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 5/13 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_year. Retrieved 4/12 statements.
# Partially parsed test_dcfc_30_e_plus_360_with_freq_parameter. Retrieved 6/15 statements.


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
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_1]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '360'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = '30'
    var_6 = [var_5]
    var_7 = '360'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2009
    var_4 = [var_3, var_1, var_1]
    var_5 = '360'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_3, var_1]
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '30'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dcfc_30_360_us_predicate_line_42. Retrieved 8/16 statements.


def test_case_0():
    var_0 = '\n    Test that the predicate at line 42 (if d1 == 31:) evaluates to True\n    when start day is 31.\n    '
    var_1 = 2008
    var_2 = 1
    var_3 = 31
    var_4 = [var_1, var_2, var_3]
    var_5 = 2
    var_6 = 29
    var_7 = [var_1, var_5, var_6]
    var_8 = [var_1, var_5, var_6]
    var_9 = '0.08333333333333'
    var_10 = [var_9]
    var_11 = 14



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dcfc_act_act_icma_basic. Retrieved 7/16 statements.
# Partially parsed test_dcfc_act_act_icma_same_dates. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_with_frequency. Retrieved 8/18 statements.
# Partially parsed test_dcfc_act_act_icma_full_period. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_icma_one_day. Retrieved 6/16 statements.


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
    var_11 = '0.2622950820'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 2020
    var_5 = [var_4, var_1, var_2]
    var_6 = [var_4, var_1, var_2]
    var_7 = '1'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 10
    var_6 = [var_0, var_1, var_5]
    var_7 = '1'
    var_8 = [var_7]
    var_9 = '8'
    var_10 = [var_9]



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

# Partially parsed test_is_last_day_of_month_true_for_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_false_for_non_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_30_days. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_31. Retrieved 3/9 statements.


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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dcfc_30_360_german_example_1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_german_example_2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example_3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example_4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_same_date. Retrieved 4/9 statements.
# Partially parsed test_dcfc_30_360_german_one_month. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_360_german_with_freq_parameter. Retrieved 7/16 statements.
# Partially parsed test_dcfc_30_360_german_start_day_31. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_german_asof_day_31. Retrieved 8/17 statements.
# Partially parsed test_dcfc_30_360_german_feb_last_day_not_end. Retrieved 8/17 statements.


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
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '12'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '1'
    var_9 = [var_8]
    var_10 = '12'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 15
    var_6 = [var_0, var_4, var_5]
    var_7 = '15'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 31
    var_5 = [var_0, var_1, var_4]
    var_6 = 2
    var_7 = 28
    var_8 = [var_0, var_6, var_7]
    var_9 = '15'
    var_10 = [var_9]
    var_11 = '360'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 29
    var_6 = [var_0, var_4, var_5]
    var_7 = 3
    var_8 = [var_0, var_7, var_2]
    var_9 = '45'
    var_10 = [var_9]
    var_11 = '360'
    var_12 = [var_11]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_last_payment_date_predicate. Retrieved 17/59 statements.


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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #32
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
    var_1 = 1
    var_2 = module_0._construct_date(var_0, var_1, var_1)
    var_3 = var_2.year
    assert var_3 == 2023
    var_4 = var_2.month
    assert var_4 == 1
    var_5 = var_2.day
    assert var_5 == 1

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 12
    var_6 = var_3.day
    assert var_6 == 31

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
    var_1 = 4
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 4
    var_6 = var_3.day
    assert var_6 == 30

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 9
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 9
    var_6 = var_3.day
    assert var_6 == 30

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 0
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = -2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = -5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = -15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_date_range. Retrieved 6/17 statements.
# Partially parsed test_get_date_range_single_day. Retrieved 2/8 statements.
# Partially parsed test_get_date_range_two_days. Retrieved 3/11 statements.
# Partially parsed test_get_date_range_across_months. Retrieved 6/17 statements.
# Partially parsed test_get_date_range_across_years. Retrieved 7/17 statements.


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
    var_5 = 3
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_1, var_2]
    var_8 = 31
    var_9 = [var_0, var_1, var_8]
    var_10 = [var_0, var_4, var_1]
    var_11 = [var_0, var_4, var_4]

def test_case_0():
    var_0 = 2022
    var_1 = 12
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_0, var_1, var_2]
    var_9 = 31
    var_10 = [var_0, var_1, var_9]
    var_11 = [var_4, var_5, var_5]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_last_payment_date_line_54_predicate_true. Retrieved 12/26 statements.


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
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_3, var_4, var_5]
    var_11 = 2
    var_12 = 7
    var_13 = [var_3, var_12, var_1]
    var_14 = 2008
    var_15 = [var_14, var_12, var_12]
    var_16 = 10
    var_17 = 6
    var_18 = [var_3, var_16, var_17]
    var_19 = 4
    var_20 = [var_3, var_12, var_12]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_date. Retrieved 3/9 statements.
# Partially parsed test_dcfc_act_act_one_day. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_non_leap_year. Retrieved 5/13 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '366'
    var_8 = [var_7]

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



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_calculate_daily_fraction_basic. Retrieved 9/20 statements.
# Partially parsed test_calculate_daily_fraction_with_different_values. Retrieved 13/29 statements.
# Partially parsed test_calculate_daily_fraction_asof_minus_1_before_start. Retrieved 9/20 statements.
# Partially parsed test_calculate_daily_fraction_with_freq. Retrieved 12/25 statements.
# Partially parsed test_calculate_daily_fraction_negative_result. Retrieved 13/29 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0'
    var_12 = [var_11]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = '0.2'
    var_3 = [var_2]
    var_4 = '0.3'
    var_5 = [var_4]
    var_6 = 'Test DCC'
    var_7 = set()
    var_8 = set()
    var_9 = 2024
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 5
    var_13 = [var_9, var_10, var_12]
    var_14 = 12
    var_15 = 31
    var_16 = [var_9, var_14, var_15]
    var_17 = '0.1'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = 5
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_4, var_5]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0.5'
    var_12 = [var_11]

def test_case_0():
    var_0 = []
    var_1 = 'Test DCC'
    var_2 = set()
    var_3 = set()
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 5
    var_8 = [var_4, var_5, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_4, var_9, var_10]
    var_12 = '2'
    var_13 = [var_12]
    var_14 = len(var_0)
    assert var_14 == 2
    var_15 = var_0[0][3]
    var_16 = var_0[1][3]
    var_17 = '0'
    var_18 = [var_17]

def test_case_0():
    var_0 = '0.7'
    var_1 = [var_0]
    var_2 = '0.3'
    var_3 = [var_2]
    var_4 = 0
    var_5 = [var_4]
    var_6 = 'Test DCC'
    var_7 = set()
    var_8 = set()
    var_9 = 2024
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = 5
    var_13 = [var_9, var_10, var_12]
    var_14 = 12
    var_15 = 31
    var_16 = [var_9, var_14, var_15]
    var_17 = '-0.4'
    var_18 = [var_17]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_last_payment_date_predicate_line_1_false. Retrieved 17/59 statements.


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



# Parsed testcases at query #41
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
    var_1 = 1
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 1
    var_6 = var_3.day
    assert var_6 == 31

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 25
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 12
    var_6 = var_3.day
    assert var_6 == 25

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
    var_1 = 4
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 4
    var_6 = var_3.day
    assert var_6 == 30

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 6
    var_6 = var_3.day
    assert var_6 == 30

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 0
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = -2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = -5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = -15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = module_0._construct_date(var_0, var_1, var_1)
    var_3 = var_2.year
    assert var_3 == 2023
    var_4 = var_2.month
    assert var_4 == 1
    var_5 = var_2.day
    assert var_5 == 1

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 12
    var_6 = var_3.day
    assert var_6 == 31



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dcfc_30_360_us_predicate_line_38. Retrieved 7/17 statements.
# Partially parsed test_dcfc_30_360_us_predicate_line_38_with_d1_31. Retrieved 6/16 statements.
# Partially parsed test_dcfc_30_360_us_predicate_line_38_false_condition. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = '30'
    var_9 = [var_8]
    var_10 = '360'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = [var_0, var_4, var_2]
    var_7 = '30'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_0, var_4, var_5]
    var_8 = '46'
    var_9 = [var_8]
    var_10 = '360'
    var_11 = [var_10]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_calculate_fraction_valid_dates. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_asof_equals_start. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_equals_end. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_before_start. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_asof_after_end. Retrieved 9/19 statements.
# Partially parsed test_calculate_fraction_with_freq_parameter. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 6
    var_7 = 15
    var_8 = [var_3, var_6, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = '0.5'
    var_13 = [var_12]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_3, var_4, var_4]
    var_7 = 12
    var_8 = 31
    var_9 = [var_3, var_7, var_8]
    var_10 = '0.25'
    var_11 = [var_10]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 12
    var_7 = 31
    var_8 = [var_3, var_6, var_7]
    var_9 = [var_3, var_6, var_7]
    var_10 = '0.75'
    var_11 = [var_10]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 6
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = 1
    var_8 = [var_3, var_7, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = '0'
    var_13 = [var_12]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2025
    var_7 = [var_6, var_4, var_4]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 6
    var_7 = 15
    var_8 = [var_3, var_6, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = '2'
    var_13 = [var_12]
    var_14 = '0.6'
    var_15 = [var_14]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_coupon_basic. Retrieved 12/26 statements.
# Partially parsed test_coupon_with_eom. Retrieved 14/28 statements.
# Partially parsed test_coupon_semi_annual. Retrieved 15/29 statements.
# Partially parsed test_coupon_quarterly. Retrieved 14/28 statements.
# Partially parsed test_coupon_decimal_freq. Retrieved 12/27 statements.
# Partially parsed test_coupon_zero_fraction. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test'
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
    var_15 = 1
    var_16 = '0.5'
    var_17 = [var_16]

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '5000'
    var_4 = [var_3]
    var_5 = '0.03'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = 31
    var_10 = [var_7, var_8, var_9]
    var_11 = 7
    var_12 = 15
    var_13 = [var_7, var_11, var_12]
    var_14 = 2015
    var_15 = [var_14, var_8, var_9]
    var_16 = 2
    var_17 = 31
    var_18 = '0.25'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '10000'
    var_4 = [var_3]
    var_5 = '0.06'
    var_6 = [var_5]
    var_7 = 2012
    var_8 = 12
    var_9 = 15
    var_10 = [var_7, var_8, var_9]
    var_11 = 2015
    var_12 = 31
    var_13 = [var_11, var_8, var_12]
    var_14 = 2016
    var_15 = 6
    var_16 = [var_14, var_15, var_9]
    var_17 = 2
    var_18 = 15
    var_19 = '0.48'
    var_20 = [var_19]

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '2000'
    var_4 = [var_3]
    var_5 = '0.04'
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
    var_18 = '0.24'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'Test'
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
    var_15 = '1'
    var_16 = [var_15]
    var_17 = '0.5'
    var_18 = [var_17]

def test_case_0():
    var_0 = 'Test'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = [var_3]
    var_5 = '0.05'
    var_6 = [var_5]
    var_7 = 2014
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = [var_7, var_8, var_8]
    var_11 = 2015
    var_12 = [var_11, var_8, var_8]
    var_13 = 1
    var_14 = '0'
    var_15 = [var_14]



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_calculate_daily_fraction_predicate_false. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 5
    var_7 = [var_3, var_4, var_6]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0'
    var_12 = [var_11]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_360_isda_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 5/11 statements.
# Partially parsed test_dcfc_30_360_isda_asof_day_31_start_day_30. Retrieved 5/11 statements.
# Partially parsed test_dcfc_30_360_isda_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_isda_one_month_difference. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_360_isda_one_year_difference. Retrieved 5/11 statements.
# Partially parsed test_dcfc_30_360_isda_returns_decimal. Retrieved 3/9 statements.


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
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 29
    var_6 = [var_0, var_4, var_5]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]

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
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '12'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = [var_4, var_1, var_2]
    var_6 = '1'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_register_new_dcc. Retrieved 4/11 statements.
# Partially parsed test_register_dcc_with_altnames. Retrieved 6/14 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 6/14 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 8/16 statements.
# Partially parsed test_register_altname_conflicts_with_main_name. Retrieved 7/15 statements.
# Partially parsed test_register_multiple_dcc. Retrieved 9/20 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC'
    var_2 = set()
    var_3 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC2'
    var_2 = 'ALT1'
    var_3 = 'ALT2'
    var_4 = {var_2, var_3}
    var_5 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC3'
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
    var_1 = 'Test/DCC4'
    var_2 = 'ALTNAME'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 'Test/DCC5'
    var_6 = {var_2}
    var_7 = set()
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC6'
    var_2 = set()
    var_3 = set()
    var_4 = 'Test/DCC7'
    var_5 = {var_1}
    var_6 = set()
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'already registered'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC8'
    var_2 = 'ALT8'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 'Test/DCC9'
    var_6 = 'ALT9'
    var_7 = {var_6}
    var_8 = set()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dcfc_nl_365_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_nl_365_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_longer_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_over_year. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_same_day. Retrieved 4/10 statements.
# Partially parsed test_dcfc_nl_365_one_day. Retrieved 6/15 statements.
# Partially parsed test_dcfc_nl_365_with_none_freq. Retrieved 8/16 statements.


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
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = '1'
    var_7 = [var_6]
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
    var_7 = None
    var_8 = 14
    var_9 = '0.16986301369863'
    var_10 = [var_9]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_with_exact_name. Retrieved 4/12 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 5/11 statements.
# Partially parsed test_find_nonexistent_convention. Retrieved 5/11 statements.
# Partially parsed test_find_with_lowercase_input. Retrieved 5/11 statements.
# Partially parsed test_find_with_whitespace_and_case_variations. Retrieved 5/11 statements.
# Partially parsed test_find_returns_same_object. Retrieved 6/12 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = var_0.find(var_1)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.name
    assert var_5 == 'Act/Act'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/ACT'
    var_2 = []
    var_3 = '  act/act  '
    var_4 = var_0.find(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == 'ACT/ACT'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = var_0.find(var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == 'Actual/Actual'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = 'NonExistent/Convention'
    var_4 = var_0.find(var_3)
    assert var_4 is None

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '30/360 US'
    var_2 = []
    var_3 = '30/360 us'
    var_4 = var_0.find(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == '30/360 US'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACT/360'
    var_2 = []
    var_3 = '  act/360  '
    var_4 = var_0.find(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.name
    assert var_6 == 'ACT/360'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = []
    var_3 = var_0.find(var_1)
    var_4 = 'testdcc'
    var_5 = var_0.find(var_4)
    var_6 = bool(var_3 is var_5)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_next_payment_date_annual_frequency_no_eom. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_annual_frequency_with_eom. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_semi_annual_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_quarterly_frequency. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_monthly_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_with_eom_invalid_day. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_with_eom_february. Retrieved 4/9 statements.
# Partially parsed test_next_payment_date_decimal_frequency. Retrieved 5/10 statements.
# Partially parsed test_next_payment_date_different_start_date. Retrieved 6/11 statements.
# Partially parsed test_next_payment_date_leap_year. Retrieved 7/12 statements.


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
    var_4 = 2015
    var_5 = [var_4, var_1, var_2]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 30
    var_4 = 2015
    var_5 = [var_4, var_1, var_3]

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2.0
    var_4 = None
    var_5 = 7
    var_6 = [var_0, var_5, var_1]

def test_case_0():
    var_0 = 2020
    var_1 = 6
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = None
    var_6 = 2021
    var_7 = [var_6, var_1, var_2]

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dcfc_act_365_l_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_l_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_l_over_year. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_l_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_l_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_act_365_l_one_day. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_365_l_non_leap_year. Retrieved 6/15 statements.
# Partially parsed test_dcfc_act_365_l_with_freq_parameter. Retrieved 7/17 statements.


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
    var_9 = '1.32876712328767'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 16
    var_5 = [var_0, var_1, var_4]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '366'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2007
    var_1 = 2
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 16
    var_5 = [var_0, var_1, var_4]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '365'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 15
    var_5 = [var_0, var_1, var_4]
    var_6 = '4'
    var_7 = [var_6]
    var_8 = '14'
    var_9 = [var_8]
    var_10 = '366'
    var_11 = [var_10]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dcfc_act_365_a_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_act_365_a_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_long_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_extended_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_act_365_a_same_day. Retrieved 3/10 statements.
# Partially parsed test_dcfc_act_365_a_one_day_difference. Retrieved 5/14 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '365'
    var_8 = [var_7]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_coupon_basic. Retrieved 19/39 statements.
# Partially parsed test_coupon_with_eom. Retrieved 21/38 statements.
# Partially parsed test_coupon_quarterly_frequency. Retrieved 19/36 statements.
# Partially parsed test_coupon_without_eom. Retrieved 21/38 statements.


def test_case_0():
    var_0 = 'Money'
    var_1 = 'amount'
    var_2 = 'currency'
    var_3 = [var_1, var_2]
    var_4 = 'Currency'
    var_5 = 'code'
    var_6 = [var_5]
    var_7 = 'Test DCC'
    var_8 = set()
    var_9 = set()
    var_10 = '1000'
    var_11 = [var_10]
    var_12 = 'USD'
    var_13 = '0.05'
    var_14 = [var_13]
    var_15 = 2014
    var_16 = 1
    var_17 = [var_15, var_16, var_16]
    var_18 = 6
    var_19 = [var_15, var_18, var_16]
    var_20 = 2015
    var_21 = [var_20, var_16, var_16]
    var_22 = 1
    var_23 = '25'
    var_24 = [var_23]

def test_case_0():
    var_0 = 'Money'
    var_1 = 'amount'
    var_2 = 'currency'
    var_3 = [var_1, var_2]
    var_4 = 'Currency'
    var_5 = 'code'
    var_6 = [var_5]
    var_7 = 'Test DCC'
    var_8 = set()
    var_9 = set()
    var_10 = '2000'
    var_11 = [var_10]
    var_12 = 'USD'
    var_13 = '0.1'
    var_14 = [var_13]
    var_15 = 2014
    var_16 = 1
    var_17 = 31
    var_18 = [var_15, var_16, var_17]
    var_19 = 3
    var_20 = 15
    var_21 = [var_15, var_19, var_20]
    var_22 = 7
    var_23 = [var_15, var_22, var_17]
    var_24 = 2
    var_25 = 31

def test_case_0():
    var_0 = 'Money'
    var_1 = 'amount'
    var_2 = 'currency'
    var_3 = [var_1, var_2]
    var_4 = 'Currency'
    var_5 = 'code'
    var_6 = [var_5]
    var_7 = 'Test DCC'
    var_8 = set()
    var_9 = set()
    var_10 = '5000'
    var_11 = [var_10]
    var_12 = 'EUR'
    var_13 = '0.02'
    var_14 = [var_13]
    var_15 = 2014
    var_16 = 1
    var_17 = 15
    var_18 = [var_15, var_16, var_17]
    var_19 = 4
    var_20 = 10
    var_21 = [var_15, var_19, var_20]
    var_22 = [var_15, var_20, var_17]
    var_23 = 4

def test_case_0():
    var_0 = 'Money'
    var_1 = 'amount'
    var_2 = 'currency'
    var_3 = [var_1, var_2]
    var_4 = 'Currency'
    var_5 = 'code'
    var_6 = [var_5]
    var_7 = 'Test DCC'
    var_8 = set()
    var_9 = set()
    var_10 = '10000'
    var_11 = [var_10]
    var_12 = 'GBP'
    var_13 = '0.04'
    var_14 = [var_13]
    var_15 = 2013
    var_16 = 6
    var_17 = 15
    var_18 = [var_15, var_16, var_17]
    var_19 = 2014
    var_20 = 12
    var_21 = [var_19, var_20, var_17]
    var_22 = 2015
    var_23 = [var_22, var_16, var_17]
    var_24 = 2
    var_25 = None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dcfc_30_e_plus_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_plus_360_start_day_31. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_asof_day_31. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_day. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_month. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_plus_360_one_year. Retrieved 5/13 statements.


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
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = '46'
    var_8 = [var_7]
    var_9 = '360'
    var_10 = [var_9]

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
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = '30'
    var_7 = [var_6]
    var_8 = '360'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = [var_4, var_1, var_2]
    var_6 = '360'
    var_7 = [var_6]
    var_8 = [var_6]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_act_365_l_predicate_false. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = [var_4, var_5, var_2]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_calculate_fraction_valid_dates. Retrieved 10/20 statements.
# Partially parsed test_calculate_fraction_with_freq. Retrieved 11/22 statements.
# Partially parsed test_calculate_fraction_asof_equals_start. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_equals_end. Retrieved 8/18 statements.
# Partially parsed test_calculate_fraction_asof_before_start. Retrieved 9/19 statements.
# Partially parsed test_calculate_fraction_asof_after_end. Retrieved 9/19 statements.
# Partially parsed test_calculate_fraction_none_freq. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 6
    var_7 = 15
    var_8 = [var_3, var_6, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = '0.5'
    var_13 = [var_12]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 3
    var_7 = 15
    var_8 = [var_3, var_6, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = '2'
    var_13 = [var_12]
    var_14 = '0.5'
    var_15 = [var_14]

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
    var_10 = '0.75'
    var_11 = [var_10]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 12
    var_7 = 31
    var_8 = [var_3, var_6, var_7]
    var_9 = [var_3, var_6, var_7]
    var_10 = '1.0'
    var_11 = [var_10]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 6
    var_5 = 1
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_3, var_5, var_5]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2024
    var_7 = [var_6, var_4, var_4]
    var_8 = 12
    var_9 = 31
    var_10 = [var_3, var_8, var_9]
    var_11 = '0'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 4
    var_7 = 15
    var_8 = [var_3, var_6, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = None
    var_13 = '0.333'
    var_14 = [var_13]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dcfc_30_360_isda_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_360_isda_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_31. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_360_isda_start_day_30_asof_day_31. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_isda_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_isda_one_month_difference. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_360_isda_one_year_difference. Retrieved 5/13 statements.


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
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 15
    var_6 = [var_0, var_4, var_5]
    var_7 = [var_5]
    var_8 = 360
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 31
    var_6 = [var_0, var_4, var_5]
    var_7 = 0
    var_8 = [var_7]
    var_9 = 360
    var_10 = [var_9]

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
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = 30
    var_7 = [var_6]
    var_8 = 360
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = [var_4, var_1, var_2]
    var_6 = 360
    var_7 = [var_6]
    var_8 = [var_6]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dcfc_act_365_a_predicate_line_24. Retrieved 23/45 statements.


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 2008
    var_5 = 2
    var_6 = [var_4, var_5, var_2]
    var_7 = 62
    var_8 = [var_7]
    var_9 = 365
    var_10 = [var_9]
    var_11 = [var_0, var_1, var_2]
    var_12 = 29
    var_13 = [var_4, var_5, var_12]
    var_14 = 63
    var_15 = [var_14]
    var_16 = 366
    var_17 = [var_16]
    var_18 = 10
    var_19 = 31
    var_20 = [var_0, var_18, var_19]
    var_21 = 11
    var_22 = 30
    var_23 = [var_4, var_21, var_22]
    var_24 = 396
    var_25 = [var_24]
    var_26 = [var_16]
    var_27 = 1
    var_28 = [var_4, var_5, var_27]
    var_29 = 2009
    var_30 = 5
    var_31 = [var_29, var_30, var_19]
    var_32 = 485
    var_33 = [var_32]
    var_34 = [var_16]
    var_35 = result1.as_tuple()[var_5]
    assert var_35 == -3
    var_36 = result2.as_tuple()[var_5]
    assert var_36 == -3
    var_37 = result3.as_tuple()[var_5]
    assert var_37 == -3
    var_38 = result4.as_tuple()[var_5]
    assert var_38 == -3



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_register_raises_typeerror_when_altname_already_registered. Retrieved 10/18 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test/DCC1'
    var_2 = 'ALT1'
    var_3 = 'ALT2'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 'Test/DCC2'
    var_7 = 'ALT3'
    var_8 = {var_2, var_7}
    var_9 = set()
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Test/DCC2'
    var_12 = 'already registered'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_with_exact_name. Retrieved 4/12 statements.
# Partially parsed test_find_with_stripped_uppercase_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_alternative_name. Retrieved 5/11 statements.
# Partially parsed test_find_with_alternative_name_stripped_uppercase. Retrieved 6/12 statements.
# Partially parsed test_find_nonexistent_name. Retrieved 5/11 statements.
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
    var_1 = 'Actual/Actual'
    var_2 = 'Act/Act'
    var_3 = [var_2]
    var_4 = '  act/act  '
    var_5 = var_0.find(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = []
    var_3 = 'NonExistent/Convention'
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dcc_registry_machinery_initialization. Retrieved 7/9 statements.


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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_register_raises_error_when_altname_already_registered. Retrieved 8/17 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_register_raises_typeerror_when_altname_already_registered. Retrieved 10/18 statements.


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
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_calculate_fraction_predicate_false. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 6
    var_7 = 15
    var_8 = [var_3, var_6, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = '0.5'
    var_13 = [var_12]



# Parsed testcases at query #23
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



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_calculate_fraction_predicate_false. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'Test DCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 6
    var_7 = 15
    var_8 = [var_3, var_6, var_7]
    var_9 = 12
    var_10 = 31
    var_11 = [var_3, var_9, var_10]
    var_12 = '0.5'
    var_13 = [var_12]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dcfc_30_360_german_example1. Retrieved 7/15 statements.
# Partially parsed test_dcfc_30_360_german_example2. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example3. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_example4. Retrieved 8/16 statements.
# Partially parsed test_dcfc_30_360_german_same_date. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_360_german_one_day. Retrieved 6/15 statements.
# Partially parsed test_dcfc_30_360_german_day_31_adjustment. Retrieved 7/16 statements.
# Partially parsed test_dcfc_30_360_german_end_of_february. Retrieved 8/18 statements.


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
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = '0'
    var_5 = [var_4]

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
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = 3
    var_7 = 31
    var_8 = [var_0, var_6, var_7]
    var_9 = '1'
    var_10 = [var_9]
    var_11 = '360'
    var_12 = [var_11]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dcfc_act_act_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_act_act_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_act_act_same_dates. Retrieved 3/8 statements.
# Partially parsed test_dcfc_act_act_one_day. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_non_leap_year. Retrieved 5/13 statements.
# Partially parsed test_dcfc_act_act_freq_parameter_ignored. Retrieved 4/11 statements.


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
    var_3 = '0'
    var_4 = [var_3]

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
    var_5 = '2'
    var_6 = [var_5]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dcfc_30_e_360_example1. Retrieved 7/14 statements.
# Partially parsed test_dcfc_30_e_360_example2. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_example3. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_example4. Retrieved 8/15 statements.
# Partially parsed test_dcfc_30_e_360_start_day_31. Retrieved 14/22 statements.
# Partially parsed test_dcfc_30_e_360_asof_day_31. Retrieved 14/22 statements.
# Partially parsed test_dcfc_30_e_360_both_day_31. Retrieved 13/21 statements.
# Partially parsed test_dcfc_30_e_360_same_dates. Retrieved 4/10 statements.
# Partially parsed test_dcfc_30_e_360_one_day_difference. Retrieved 5/13 statements.
# Partially parsed test_dcfc_30_e_360_one_month_difference. Retrieved 6/14 statements.
# Partially parsed test_dcfc_30_e_360_one_year_difference. Retrieved 5/13 statements.


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
    var_0 = 2008
    var_1 = 1
    var_2 = 28
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
    var_4 = 2
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
    var_4 = 16
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_1]
    var_7 = 360
    var_8 = [var_7]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_0, var_4, var_2]
    var_6 = 30
    var_7 = [var_6]
    var_8 = 360
    var_9 = [var_8]

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2009
    var_5 = [var_4, var_1, var_2]
    var_6 = 360
    var_7 = [var_6]
    var_8 = [var_6]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dcc_registry_machinery_constructor. Retrieved 3/5 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True
    var_5 = var_0._buffer_main
    var_6 = var_0._buffer_altn



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_register_successful. Retrieved 6/12 statements.
# Partially parsed test_register_duplicate_main_name. Retrieved 6/14 statements.
# Partially parsed test_register_duplicate_altname. Retrieved 8/16 statements.
# Partially parsed test_register_altname_conflicts_with_main_name. Retrieved 7/15 statements.
# Partially parsed test_register_multiple_altnames. Retrieved 7/12 statements.


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TEST/DCC'
    var_2 = 'Test'
    var_3 = 'TestDCC'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = var_0._buffer_main['TEST/DCC']
    var_7 = var_0._buffer_altn['Test']
    var_8 = var_0._buffer_altn['TestDCC']

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TEST/DCC'
    var_2 = set()
    var_3 = set()
    var_4 = set()
    var_5 = set()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TEST/DCC'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TEST/DCC1'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 'TEST/DCC2'
    var_6 = {var_2}
    var_7 = set()
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'TEST/DCC2'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TEST/DCC1'
    var_2 = set()
    var_3 = set()
    var_4 = 'TEST/DCC2'
    var_5 = {var_1}
    var_6 = set()
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'TEST/DCC2'

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TEST/DCC'
    var_2 = 'Alt1'
    var_3 = 'Alt2'
    var_4 = 'Alt3'
    var_5 = {var_2, var_3, var_4}
    var_6 = set()
    var_7 = var_0._buffer_altn['Alt1']
    var_8 = var_0._buffer_altn['Alt2']
    var_9 = var_0._buffer_altn['Alt3']



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_is_last_day_of_month_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_not_last_day. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_february_non_leap_year. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_april_30. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_december_31. Retrieved 3/9 statements.
# Partially parsed test_is_last_day_of_month_first_day. Retrieved 3/9 statements.


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

def test_case_0():
    var_0 = 2024
    var_1 = 3
    var_2 = 1
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_last_payment_date_basic_annual. Retrieved 5/12 statements.
# Partially parsed test_last_payment_date_same_year. Retrieved 4/11 statements.
# Partially parsed test_last_payment_date_semi_annual. Retrieved 7/14 statements.
# Partially parsed test_last_payment_date_semi_annual_august. Retrieved 7/14 statements.
# Partially parsed test_last_payment_date_semi_annual_april. Retrieved 6/13 statements.
# Partially parsed test_last_payment_date_june_start. Retrieved 6/13 statements.
# Partially parsed test_last_payment_date_quarterly. Retrieved 6/13 statements.
# Partially parsed test_last_payment_date_december_start. Retrieved 6/13 statements.
# Partially parsed test_last_payment_date_semi_annual_december. Retrieved 8/15 statements.
# Partially parsed test_last_payment_date_semi_annual_december_year_end. Retrieved 6/13 statements.
# Partially parsed test_last_payment_date_with_decimal_frequency. Retrieved 5/13 statements.
# Partially parsed test_last_payment_date_with_explicit_eom. Retrieved 5/12 statements.
# Partially parsed test_last_payment_date_eom_adjustment. Retrieved 6/13 statements.


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
    var_8 = [var_4, var_1, var_6]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dcfc_nl_365_basic. Retrieved 7/15 statements.
# Partially parsed test_dcfc_nl_365_with_leap_day. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_longer_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_extended_period. Retrieved 8/16 statements.
# Partially parsed test_dcfc_nl_365_same_date. Retrieved 4/11 statements.
# Partially parsed test_dcfc_nl_365_single_day. Retrieved 6/15 statements.
# Partially parsed test_dcfc_nl_365_with_freq_parameter. Retrieved 8/17 statements.


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
    var_9 = '0.16986301369863'
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
    var_10 = '0.16986301369863'
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
    var_10 = '1.08219178082192'
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
    var_10 = '1.32602739726027'
    var_11 = [var_10]

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
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = 29
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = '1'
    var_8 = [var_7]
    var_9 = '365'
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
    var_8 = '4'
    var_9 = [var_8]
    var_10 = 14
    var_11 = '0.16986301369863'
    var_12 = [var_11]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2015
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = 1



