####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


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
    var_18 = 3
    var_19 = 15
    var_20 = '0'
    var_21 = '28'
    var_22 = 360
    var_23 = 2006
    var_24 = 6
    var_25 = 20
    var_26 = var_25 - var_19
    var_27 = var_1 - var_24
    var_28 = var_12 * var_27
    var_29 = var_26 + var_28
    var_30 = var_3 - var_23
    var_31 = var_22 * var_30
    var_32 = var_29 + var_31



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16942884946478'
    var_8 = 29
    var_9 = '0.17216108990194'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08243131970956'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32625945055768'
    var_19 = 2020
    var_20 = '0'
    var_21 = 2019
    var_22 = '1'
    var_23 = '365'
    var_24 = 3
    var_25 = '366'
    var_26 = '364'
    var_27 = 2015
    var_28 = 2017
    var_29 = '2'



# Parsed testcases at query #3
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = '\n    Test the register method of DCCRegistryMachinery class.\n    '
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC'
    var_3 = 'Test/Alternative'
    var_4 = 'TestAlt'
    var_5 = {var_3, var_4}
    var_6 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = "\n    Test that registering a DCC with a name that's already registered raises TypeError.\n    "
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Duplicate/DCC'
    var_3 = set()
    var_4 = set()
    var_5 = set()
    var_6 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = "\n    Test that registering a DCC with an alternative name that's already registered raises TypeError.\n    "
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'First/DCC'
    var_3 = 'Shared/Alt'
    var_4 = {var_3}
    var_5 = set()
    var_6 = 'Second/DCC'
    var_7 = {var_3}
    var_8 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = '\n    Test that registering a DCC with an alternative name that conflicts with an existing main name raises TypeError.\n    '
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'First/DCC'
    var_3 = set()
    var_4 = set()
    var_5 = 'Second/DCC'
    var_6 = {var_2}
    var_7 = set()

import pypara.dcc as module_0

def test_case_0():
    var_0 = '\n    Test registering a DCC with associated currencies.\n    '
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'USD'
    var_3 = 'EUR'
    var_4 = 'Currency/DCC'
    var_5 = 'CurrencyAlt'
    var_6 = {var_5}



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_30_360_isda function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16666666666667'
    var_8 = 29
    var_9 = '0.16944444444444'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08333333333333'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.33333333333333'
    var_19 = 6
    var_20 = 15
    var_21 = 0
    var_22 = 360
    var_23 = 3
    var_24 = 60
    var_25 = 2006
    var_26 = 720
    var_27 = -90



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the dcfc_act_365_a function with various date ranges.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = '0.17213114754098'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08196721311475'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32513661202186'
    var_19 = '0'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act_icma function.'
    var_1 = 2019
    var_2 = 3
    var_3 = 2
    var_4 = 9
    var_5 = 10
    var_6 = 2020
    var_7 = '0.5245901639'
    var_8 = '0'
    var_9 = '1'
    var_10 = '0.2622950820'
    var_11 = 4
    var_12 = '0.5'
    var_13 = 28
    var_14 = 1
    var_15 = 12
    var_16 = 31
    var_17 = 2018
    var_18 = 6
    var_19 = 30
    var_20 = '2'
    var_21 = 15



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test the interest method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'T-DCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2023
    var_9 = 1
    var_10 = 11
    var_11 = 12
    var_12 = 31
    var_13 = 10
    var_14 = 360
    var_15 = 0
    var_16 = 2
    var_17 = 10000
    var_18 = 4



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act_icma function.'
    var_1 = 2019
    var_2 = 3
    var_3 = 2
    var_4 = 9
    var_5 = 10
    var_6 = 2020
    var_7 = '0.5245901639'
    var_8 = 1
    var_9 = 12
    var_10 = 31
    var_11 = 0
    var_12 = 6
    var_13 = 15
    var_14 = 7
    var_15 = 2021
    var_16 = '0.49'
    var_17 = '0.51'
    var_18 = 30
    var_19 = 28
    var_20 = 8
    var_21 = 20
    var_22 = 4



# Parsed testcases at query #9
#--------------------------


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
    var_17 = '1.33055555555556'
    var_18 = 15
    var_19 = 0
    var_20 = 16
    var_21 = 360
    var_22 = 3
    var_23 = 60



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16942884946478'
    var_8 = 29
    var_9 = '0.17216108990194'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08243131970956'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32625945055768'
    var_19 = 2010
    var_20 = 6
    var_21 = 15
    var_22 = '0'
    var_23 = 16
    var_24 = '1'
    var_25 = '365'
    var_26 = '366'
    var_27 = '2'
    var_28 = 2015



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test the dcfc_act_365_l function with various date ranges.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16939890710383'
    var_8 = 29
    var_9 = '0.17213114754098'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08196721311475'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32876712328767'
    var_19 = 2020
    var_20 = 6
    var_21 = 15
    var_22 = '0'
    var_23 = 2019
    var_24 = 16
    var_25 = '1'
    var_26 = '365'
    var_27 = '366'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test the dcfc_30_360_isda day count fraction function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16666666666667'
    var_8 = 29
    var_9 = '0.16944444444444'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08333333333333'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.33333333333333'
    var_19 = 6
    var_20 = 15
    var_21 = '0'
    var_22 = '1'
    var_23 = False
    var_24 = 2006
    var_25 = var_11 - var_15
    var_26 = var_2 - var_15
    var_27 = var_13 * var_26
    var_28 = var_25 + var_27
    var_29 = 360
    var_30 = var_4 - var_24
    var_31 = var_29 * var_30
    var_32 = var_28 + var_31
    var_33 = None
    var_34 = '2'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test the coupon method of DCC class.'
    var_1 = 'Test/360'
    var_2 = 'T/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2014
    var_9 = 1
    var_10 = 6
    var_11 = 15
    var_12 = 12
    var_13 = 31
    var_14 = 2
    var_15 = 7
    var_16 = 31
    var_17 = 2013
    var_18 = 4
    var_19 = 0
    var_20 = 10000



# Parsed testcases at query #14
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = '\n    Test the register method of DCCRegistryMachinery class.\n    '
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/DCC1'
    var_3 = 'Test1'
    var_4 = 'T1'
    var_5 = {var_3, var_4}
    var_6 = set()
    var_7 = set()
    var_8 = set()
    var_9 = 'Test/DCC3'
    var_10 = {var_3}
    var_11 = set()
    var_12 = 'Test/DCC4'
    var_13 = 'Test4'
    var_14 = 'T4'
    var_15 = {var_13, var_14}
    var_16 = set()
    var_17 = 'Test/DCC5'
    var_18 = 'Test5'
    var_19 = 'T5'
    var_20 = {var_18, var_19}
    var_21 = set()
    var_22 = 'Test/DCC6'
    var_23 = set()
    var_24 = set()
    var_25 = var_1.registry
    var_26 = var_1.table



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test DCC.calculate_daily_fraction method.'
    var_1 = 'Test DCC'
    var_2 = 'Test'
    var_3 = 'TDCC'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 2020
    var_7 = 1
    var_8 = 12
    var_9 = 31
    var_10 = 2
    var_11 = 365
    var_12 = 10
    var_13 = 7
    var_14 = 3
    var_15 = 15
    var_16 = 14
    var_17 = 5
    var_18 = 20
    var_19 = 19
    var_20 = None



# Parsed testcases at query #16
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Test the find method of DCCRegistryMachinery class.'
    var_1 = module_0.DCCRegistryMachinery()
    var_2 = 'Test/Convention'
    var_3 = 'TC'
    var_4 = 'Test Conv'
    var_5 = {var_3, var_4}
    var_6 = set()
    var_7 = var_1.find(var_2)
    var_8 = var_1.find(var_3)
    var_9 = var_1.find(var_4)
    var_10 = '  Test/Convention  '
    var_11 = var_1.find(var_10)
    var_12 = 'test/convention'
    var_13 = var_1.find(var_12)
    var_14 = '  test/convention  '
    var_15 = var_1.find(var_14)
    var_16 = 'NonExistent/Convention'
    var_17 = var_1.find(var_16)
    assert var_17 is None
    var_18 = ''
    var_19 = var_1.find(var_18)
    assert var_19 is None



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test the coupon method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'TDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 1000
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2020
    var_9 = 1
    var_10 = 3
    var_11 = 4
    var_12 = 4
    var_13 = None
    var_14 = 31
    var_15 = 2
    var_16 = 29
    var_17 = 10000
    var_18 = 10



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    Unit tests for DCC.calculate_daily_fraction method.\n    '
    var_1 = 'TEST'
    var_2 = set()
    var_3 = set()
    var_4 = 2017
    var_5 = 1
    var_6 = 10
    var_7 = 2
    var_8 = 9
    var_9 = 3
    var_10 = 5



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16986301369863'
    var_7 = 29
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = 0
    var_18 = 365



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act_icma function.'
    var_1 = 2019
    var_2 = 3
    var_3 = 2
    var_4 = 9
    var_5 = 10
    var_6 = 2020
    var_7 = '0.5245901639'
    var_8 = 1
    var_9 = 12
    var_10 = 31
    var_11 = 0
    var_12 = 6
    var_13 = 30
    var_14 = 29
    var_15 = 2021
    var_16 = 15
    var_17 = 2018
    var_18 = 5
    var_19 = 11
    var_20 = 4



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act_icma function.'
    var_1 = 2019
    var_2 = 3
    var_3 = 2
    var_4 = 9
    var_5 = 10
    var_6 = 2020
    var_7 = '0.5245901639'
    var_8 = 1
    var_9 = '0.2622950820'
    var_10 = 0
    var_11 = 6
    var_12 = 30
    var_13 = 12
    var_14 = 31
    var_15 = 4
    var_16 = '0.1311475410'
    var_17 = 29
    var_18 = None



# Parsed testcases at query #22
#--------------------------


import datetime as module_0

def test_case_0():
    var_0 = 'Test the calculate_daily_fraction method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'TEST'
    var_3 = 'T'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 2020
    var_7 = 1
    var_8 = 2
    var_9 = 12
    var_10 = 31
    var_11 = 365
    var_12 = 5
    var_13 = 0
    var_14 = 6
    var_15 = 15
    var_16 = 20
    var_17 = None
    var_18 = module_0.timedelta()
    var_19 = 2021
    var_20 = 3
    var_21 = module_0.timedelta()



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test the dcfc_act_365_a day count fraction calculator.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = '0.17213114754098'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08196721311475'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32513661202186'
    var_19 = 6
    var_20 = 15
    var_21 = '0'
    var_22 = '364'
    var_23 = '365'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test DCC.calculate_fraction method with various scenarios.'
    var_1 = 'Test DCC'
    var_2 = 'TDCC'
    var_3 = 'TestDCC'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 2020
    var_7 = 1
    var_8 = 10
    var_9 = 0
    var_10 = 5
    var_11 = 4
    var_12 = 9
    var_13 = 15
    var_14 = 2
    var_15 = 28
    var_16 = 29
    var_17 = 3



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_nl_365 function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08219178082192'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32602739726027'
    var_18 = 6
    var_19 = 15
    var_20 = 0
    var_21 = 16
    var_22 = 365
    var_23 = 364



# Parsed testcases at query #26
#--------------------------


import datetime as module_0

def test_case_0():
    var_0 = 'Test the calculate_daily_fraction method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'TEST'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 2020
    var_6 = 1
    var_7 = 3
    var_8 = 12
    var_9 = 31
    var_10 = 365
    var_11 = 10
    var_12 = 2
    var_13 = 28
    var_14 = 29
    var_15 = 6
    var_16 = 15
    var_17 = None
    var_18 = module_0.timedelta()



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_365_a function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = '0.17213114754098'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08196721311475'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32513661202186'
    var_19 = 2020
    var_20 = 6
    var_21 = 15
    var_22 = '0'
    var_23 = 2019
    var_24 = 16
    var_25 = '1'
    var_26 = '365'
    var_27 = '366'
    var_28 = None
    var_29 = '2'
    var_30 = '364'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test the interest method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'TDCC'
    var_3 = 'Test'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 1000
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = 2023
    var_10 = 1
    var_11 = 2
    var_12 = 12
    var_13 = 31
    var_14 = None
    var_15 = 0
    var_16 = 100000
    var_17 = '0.10'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test DCC.calculate_fraction method.'
    var_1 = 'Test DCC'
    var_2 = 'Test'
    var_3 = 'DCC_Test'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 2020
    var_7 = 1
    var_8 = 12
    var_9 = 31
    var_10 = 6
    var_11 = 30
    var_12 = '181'
    var_13 = '365'
    var_14 = 2019
    var_15 = 2021
    var_16 = 3
    var_17 = '4'
    var_18 = '90'
    var_19 = 15



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_nl_365 function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08219178082192'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32602739726027'
    var_18 = 0
    var_19 = 365
    var_20 = 3



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16942884946478'
    var_8 = 29
    var_9 = '0.17216108990194'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08243131970956'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32625945055768'
    var_19 = 2010
    var_20 = 6
    var_21 = 15
    var_22 = '0'
    var_23 = 16
    var_24 = '1'
    var_25 = '365'
    var_26 = '364'
    var_27 = '366'
    var_28 = 3
    var_29 = '2'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16942884946478'
    var_8 = 29
    var_9 = '0.17216108990194'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08243131970956'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32625945055768'
    var_19 = 2020
    var_20 = 6
    var_21 = 15
    var_22 = '0'
    var_23 = 16
    var_24 = '1'
    var_25 = '366'
    var_26 = 2019
    var_27 = '365'



# Parsed testcases at query #3
#--------------------------


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
    var_17 = '1.33055555555556'
    var_18 = 2020
    var_19 = 15
    var_20 = '0'
    var_21 = '29'
    var_22 = '360'
    var_23 = var_7 - var_19
    var_24 = var_4 - var_14
    var_25 = var_12 * var_24
    var_26 = var_23 + var_25
    var_27 = 360
    var_28 = var_18 - var_18
    var_29 = var_27 * var_28
    var_30 = var_26 + var_29
    var_31 = 2019
    var_32 = var_19 - var_19
    var_33 = var_14 - var_1
    var_34 = var_12 * var_33
    var_35 = var_32 + var_34
    var_36 = var_18 - var_31
    var_37 = var_27 * var_36
    var_38 = var_35 + var_37
    var_39 = 3
    var_40 = var_12 - var_12
    var_41 = var_39 - var_14
    var_42 = var_12 * var_41
    var_43 = var_40 + var_42
    var_44 = var_18 - var_18
    var_45 = var_27 * var_44
    var_46 = var_43 + var_45
    var_47 = '2'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the interest method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'T-DCC'
    var_3 = 'TDCC'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 1000
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = 2023
    var_10 = 1
    var_11 = 2
    var_12 = 12
    var_13 = 31
    var_14 = 365
    var_15 = 0
    var_16 = 5000
    var_17 = 'EUR'
    var_18 = '0.10'
    var_19 = 4
    var_20 = 90
    var_21 = 10000
    var_22 = 'GBP'
    var_23 = '0.03'
    var_24 = 7
    var_25 = 181



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_nl_365 function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08219178082192'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32602739726027'
    var_18 = 6
    var_19 = 15
    var_20 = '0'
    var_21 = 3
    var_22 = 92
    var_23 = 365
    var_24 = None



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16986301369863'
    var_7 = 29
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = 0
    var_18 = 365



# Parsed testcases at query #7
#--------------------------


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
    var_18 = 6
    var_19 = 15
    var_20 = '0'
    var_21 = '1.00'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test the calculate_daily_fraction method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'TDCC'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 2020
    var_6 = 1
    var_7 = 2
    var_8 = 12
    var_9 = 31
    var_10 = 365
    var_11 = 5
    var_12 = 11
    var_13 = 3



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_30_360_us function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16666666666667'
    var_8 = 29
    var_9 = '0.16944444444444'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08333333333333'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.33333333333333'
    var_19 = 6
    var_20 = 15
    var_21 = '0'
    var_22 = 3
    var_23 = '1'
    var_24 = '12'
    var_25 = '60'
    var_26 = '360'
    var_27 = '31'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test the coupon method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'TEST'
    var_3 = 'T'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '1000'
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = 2020
    var_10 = 1
    var_11 = 6
    var_12 = 15
    var_13 = 12
    var_14 = 31
    var_15 = 2
    var_16 = '10000'
    var_17 = '0.10'
    var_18 = 4
    var_19 = '2'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_act function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16942884946478'
    var_8 = 29
    var_9 = '0.17216108990194'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08243131970956'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32625945055768'
    var_19 = 2020
    var_20 = '0'
    var_21 = '1'
    var_22 = '366'
    var_23 = 2019
    var_24 = '365'
    var_25 = '2'
    var_26 = None
    var_27 = 3



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_30_360_isda function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16666666666667'
    var_8 = 29
    var_9 = '0.16944444444444'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08333333333333'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.33333333333333'
    var_19 = 3
    var_20 = 6
    var_21 = 15
    var_22 = 0



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_30_360_isda function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16666666666667'
    var_8 = 29
    var_9 = '0.16944444444444'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08333333333333'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.33333333333333'
    var_19 = 2020
    var_20 = 6
    var_21 = 15
    var_22 = '0'
    var_23 = 16
    var_24 = '1'
    var_25 = '360'
    var_26 = var_8 - var_13
    var_27 = var_5 - var_15
    var_28 = var_13 * var_27
    var_29 = var_26 + var_28
    var_30 = 360
    var_31 = var_19 - var_19
    var_32 = var_30 * var_31
    var_33 = var_29 + var_32
    var_34 = 3
    var_35 = var_13 - var_13
    var_36 = var_34 - var_15
    var_37 = var_13 * var_36
    var_38 = var_35 + var_37
    var_39 = var_19 - var_19
    var_40 = var_30 * var_39
    var_41 = var_38 + var_40
    var_42 = 2018
    var_43 = 2021
    var_44 = var_21 - var_21
    var_45 = var_20 - var_20
    var_46 = var_13 * var_45
    var_47 = var_44 + var_46
    var_48 = var_43 - var_42
    var_49 = var_30 * var_48
    var_50 = var_47 + var_49
    var_51 = '2'



# Parsed testcases at query #14
#--------------------------


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
    var_17 = '1.33055555555556'
    var_18 = 2020
    var_19 = 6
    var_20 = 15
    var_21 = '0'
    var_22 = 3



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test the coupon method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'test'
    var_3 = 't'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 1000
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = 2014
    var_10 = 1
    var_11 = 6
    var_12 = 15
    var_13 = 2015
    var_14 = 1
    var_15 = None
    var_16 = 2
    var_17 = 15
    var_18 = '2'
    var_19 = 0
    var_20 = 10000
    var_21 = 'EUR'
    var_22 = '0.03'
    var_23 = 2020
    var_24 = 30
    var_25 = 2021
    var_26 = 2
    var_27 = None



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test the dcfc_act_365_l function with various date ranges.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16939890710383'
    var_8 = 29
    var_9 = '0.17213114754098'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08196721311475'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32876712328767'
    var_19 = 2020
    var_20 = 0
    var_21 = 2019
    var_22 = 365
    var_23 = 366
    var_24 = 364
    var_25 = None



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test the coupon method of DCC class.'
    var_1 = 'Test DCC'
    var_2 = 'T-DCC'
    var_3 = 'TDCC'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '1000'
    var_7 = 'USD'
    var_8 = '0.05'
    var_9 = 2014
    var_10 = 1
    var_11 = 6
    var_12 = 15
    var_13 = 12
    var_14 = 31
    var_15 = 2
    var_16 = None
    var_17 = 7
    var_18 = 31
    var_19 = 4
    var_20 = '0'
    var_21 = '1000000'
    var_22 = '0.10'
    var_23 = 'EUR'



# Parsed testcases at query #18
#--------------------------


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
    var_17 = '1.33055555555556'
    var_18 = 3
    var_19 = 15
    var_20 = 0
    var_21 = None



# Parsed testcases at query #19
#--------------------------


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
    var_17 = '1.33055555555556'
    var_18 = 6
    var_19 = 15
    var_20 = '0'
    var_21 = '0.16111111111111'
    var_22 = '0.07777777777778'
    var_23 = 3
    var_24 = '0.08055555555556'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_act_365_l function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16939890710383'
    var_8 = 29
    var_9 = '0.17213114754098'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08196721311475'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.32876712328767'
    var_19 = 2020
    var_20 = 6
    var_21 = 15
    var_22 = 0
    var_23 = 2019
    var_24 = 3
    var_25 = 16
    var_26 = 365
    var_27 = 366
    var_28 = None



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_30_360_german function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16666666666667'
    var_8 = 29
    var_9 = '0.16944444444444'
    var_10 = 10
    var_11 = 31
    var_12 = 11
    var_13 = 30
    var_14 = '1.08333333333333'
    var_15 = 1
    var_16 = 2009
    var_17 = 5
    var_18 = '1.33055555555556'
    var_19 = 6
    var_20 = 15
    var_21 = '0'
    var_22 = 3
    var_23 = '2'
    var_24 = 2010
    var_25 = '4'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for dcfc_nl_365 function.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08219178082192'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32602739726027'
    var_18 = 6
    var_19 = 15
    var_20 = '0'
    var_21 = 16
    var_22 = '1'
    var_23 = 365
    var_24 = 3



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test the dcfc_nl_365 function with various date ranges.'
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 14
    var_7 = '0.16986301369863'
    var_8 = 29
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08219178082192'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32602739726027'
    var_18 = 2020
    var_19 = '0'
    var_20 = '0.00273972602740'
    var_21 = '2'



# Parsed testcases at query #24
#--------------------------


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
    var_17 = '1.33055555555556'
    var_18 = 6
    var_19 = 15
    var_20 = '0'
    var_21 = 3
    var_22 = 20



# Parsed testcases at query #25
#--------------------------


import datetime as module_0

def test_case_0():
    var_0 = 'Test the interest calculation method of DCC class.'
    var_1 = 'Test/365'
    var_2 = 'T/365'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '1000'
    var_6 = 'USD'
    var_7 = '0.05'
    var_8 = 2023
    var_9 = 1
    var_10 = 7
    var_11 = 2
    var_12 = 12
    var_13 = 31
    var_14 = 182
    var_15 = 365
    var_16 = '0'
    var_17 = module_0.timedelta()
    var_18 = '10000'
    var_19 = '0.10'
    var_20 = '2'



