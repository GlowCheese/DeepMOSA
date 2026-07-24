####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16666666666667'
    var_6 = 14
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
    var_18 = 2023
    var_19 = '0'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_act function with various date ranges including \n    leap years and non-leap years.\n    '
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = '0.16942884946478'
    var_7 = 14
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
    var_19 = 2017
    var_20 = '0'



# Parsed testcases at query #3
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'ACTUAL/360'
    var_2 = 'ACT/360_ALT'
    var_3 = 'ACT/360'
    var_4 = var_0.find(var_3)
    var_5 = var_0.find(var_1)
    var_6 = var_0.find(var_2)
    var_7 = '30/360 ISDA'
    var_8 = '30/360'
    var_9 = var_0.find(var_8)
    var_10 = var_0.find(var_7)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_30_360_isda function using the provided doctest examples.\n    '
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
    var_19 = 2023
    var_20 = '0.07777777777778'



# Parsed testcases at query #5
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_a function with various date scenarios.\n    The function relies on _get_actual_day_count and _has_leap_day.\n    '
    var_1 = 'start'
    var_2 = 'asof'
    var_3 = 'end'
    var_4 = 'expected'
    var_5 = 'leap_day'
    var_6 = 2007
    var_7 = 12
    var_8 = 28
    var_9 = 2008
    var_10 = 2
    var_11 = '0.16986301369863'
    var_12 = False
    var_13 = 29
    var_14 = '0.17260273972603'
    var_15 = True
    var_16 = 10
    var_17 = 31
    var_18 = 11
    var_19 = 30
    var_20 = '1.08493150684932'
    var_21 = 2009
    var_22 = 5
    var_23 = '1.32876712328767'
    var_24 = 'end'
    var_25 = 'start'
    var_26 = var_1 - var_3
    var_27 = var_26.days
    var_28 = 'start'
    var_29 = 'asof'
    var_30 = 'end'
    var_31 = module_0.dcfc_act_365_a(var_1, var_3, var_5)
    var_32 = '1.00000000000000'
    var_33 = 'expected'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_a function with various date scenarios.\n    The function logic relies on _get_actual_day_count and _has_leap_day.\n    '
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = 29
    var_7 = 10
    var_8 = 31
    var_9 = 11
    var_10 = 30
    var_11 = 1
    var_12 = 2009
    var_13 = 5
    var_14 = '62'
    var_15 = '365'
    var_16 = var_1 / var_3
    var_17 = '63'
    var_18 = '366'
    var_19 = var_6 / var_8
    var_20 = '426'
    var_21 = var_11 / var_12
    var_22 = '485'
    var_23 = 62
    var_24 = False



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 6
    var_3 = 12
    var_4 = 31
    var_5 = '1000.00'
    var_6 = '0.05'
    var_7 = 'ACT/ACT'
    var_8 = 'ACT/ACT/ICMA'
    var_9 = {var_8}
    var_10 = set()
    var_11 = '0.5'
    var_12 = '25.00'
    var_13 = None
    var_14 = '2'
    var_15 = 2024
    var_16 = '0'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_nl_365 function with various date ranges to ensure\n    correct handling of leap years and non-leap years according to the \n    NL/365 convention.\n    '
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



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16666666666667'
    var_6 = 14
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
    var_18 = 2023
    var_19 = '0'
    var_20 = 3



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the dcfc_act_365_l function with various date scenarios \n    to ensure correct day count fraction calculation for the \n    'Actual/365 Leap Year' convention.\n    "
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = '62'
    var_7 = '366'
    var_8 = 14
    var_9 = 29
    var_10 = '63'
    var_11 = 1
    var_12 = 2009
    var_13 = 5
    var_14 = 31
    var_15 = '120'
    var_16 = '365'
    var_17 = 10
    var_18 = 11
    var_19 = 30
    var_20 = '395'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16942mun42884946478'
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
    var_19 = 2023
    var_20 = 365



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Mock Convention'
    var_1 = 'Mock'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 2023
    var_5 = 1
    var_6 = 6
    var_7 = 2024
    var_8 = '365'
    var_9 = '0'
    var_10 = 2022
    var_11 = 12
    var_12 = 31
    var_13 = 2
    var_14 = 'FreqTest'
    var_15 = set()
    var_16 = set()
    var_17 = '2'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_a function with various date ranges, \n    including leap and non-leap year scenarios.\n    '
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = '0.16986301369863'
    var_7 = 14
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
    var_19 = 2023
    var_20 = '0'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'MockConvention'
    var_1 = 'Mock'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 2023
    var_5 = 1
    var_6 = 6
    var_7 = 12
    var_8 = 31
    var_9 = '2'
    var_10 = 364
    var_11 = 365
    var_12 = 2022
    var_13 = 0
    var_14 = 2024



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
    var_6 = '0.16942884946478'
    var_7 = 29
    var_8 = '0.17216108990194'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08243131970956'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32625945055768'
    var_18 = 2023



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_l function with various date ranges including leap years.\n    '
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = '0.16939890710383'
    var_7 = 29
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32876712328767'
    var_18 = 1e-14

def test_case_0():
    var_0 = "\n    Specifically tests the logic for the denominator 366 vs 365 \n    based on the 'asof' year being a leap year.\n    "
    var_1 = 2008
    var_2 = 1
    var_3 = 2
    var_4 = 366
    var_5 = 1e-14
    var_6 = 2007
    var_7 = 365



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_nl_365 function with various date ranges, \n    including leap year and non-leap year scenarios.\n    '
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
    var_18 = '0'



# Parsed testcases at query #7
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_a function using the provided doctest examples.\n    The function relies on _get_actual_day_count and _has_leap_day.\n    '
    var_1 = 'start'
    var_2 = 'asof'
    var_3 = 'end'
    var_4 = 'expected'
    var_5 = 2007
    var_6 = 12
    var_7 = 28
    var_8 = 2008
    var_9 = 2
    var_10 = '0.16986301369863'
    var_11 = 29
    var_12 = '0.17260273972603'
    var_13 = 10
    var_14 = 31
    var_15 = 11
    var_16 = 30
    var_17 = '1.08493150684932'
    var_18 = 1
    var_19 = 2009
    var_20 = 5
    var_21 = '1.32876712328767'
    var_22 = 'start'
    var_23 = 'asof'
    var_24 = 'end'
    var_25 = module_0.dcfc_act_365_a(var_1, var_3, var_5)



# Parsed testcases at query #8
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
    var_18 = 2023
    var_19 = 3
    var_20 = '0.08333333333333'
    var_21 = 2024
    var_22 = '0.16388888888889'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'Actual/365'
    var_2 = 'A/365'
    var_3 = {var_2}
    var_4 = 2023
    var_5 = 1
    var_6 = 6
    var_7 = 12
    var_8 = 31
    var_9 = 364
    var_10 = 365
    var_11 = 2022
    var_12 = 0
    var_13 = 2024
    var_14 = 2



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_a function with various date ranges, \n    including leap years and non-leap years, to verify the \n    day count fraction calculation.\n    '
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = '0.16986301369863'
    var_7 = 1e-14
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
    var_19 = 2023
    var_20 = '0.0'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Unit test for dcfc_30_360_isda function using provided doctest examples.\n    '
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = '0.16666666666667'
    var_7 = 14
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
    var_19 = 2023
    var_20 = '0.08333333333333'



