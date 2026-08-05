####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
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
    var_8 = '1.0'
    var_9 = 'AsOf Convention'
    var_10 = set()
    var_11 = set()
    var_12 = '151'
    var_13 = 2022
    var_14 = '0'
    var_15 = 2025
    var_16 = 'Freq Convention'
    var_17 = set()
    var_18 = set()
    var_19 = '2.5'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_a function with various date scenarios including leap years.\n    Assumes _get_actual_day_count and _has_leap_day are available in the scope \n    as they are internal dependencies of the provided code.\n    '
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



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_l function with various date scenarios \n    to ensure correct day count fraction calculation based on leap years.\n    '
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
    var_19 = 2023
    var_20 = '0'



# Parsed testcases at query #5
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
    var_17 = '1.33055555555556'
    var_18 = 2023
    var_19 = '0'
    var_20 = '30'
    var_21 = '360'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_nl_365 function with various date ranges including \n    leap year and non-leap year scenarios.\n    '
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
    var_18 = 2023



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the dcfc_act_365_l function with various date scenarios including leap and non-leap years.\n    The convention uses 366 days if the 'asof' year is a leap year, otherwise 365.\n    "
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 1
    var_6 = 27
    var_7 = '30'
    var_8 = '366'
    var_9 = 14
    var_10 = 2
    var_11 = '0.16939890710383'
    var_12 = '1'
    var_13 = '365'
    var_14 = 3
    var_15 = '2'



# Parsed testcases at query #8
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'start'
    var_1 = 'asof'
    var_2 = 'end'
    var_3 = 'expected'
    var_4 = 2007
    var_5 = 12
    var_6 = 28
    var_7 = 2008
    var_8 = 2
    var_9 = '0.16986301369863'
    var_10 = 29
    var_11 = '0.17213114754098'
    var_12 = 10
    var_13 = 31
    var_14 = 11
    var_15 = 30
    var_16 = '1.08196721311475'
    var_17 = 1
    var_18 = 2009
    var_19 = 5
    var_20 = '1.32513661202186'
    var_21 = 'start'
    var_22 = 'asof'
    var_23 = 'end'
    var_24 = module_0.dcfc_act_365_a(var_1, var_3, var_5)
    var_25 = 14
    var_26 = round(var_24, var_25)
    var_27 = 'expected'
    var_28 = round(var_8, var_25)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the interest method of the DCC class.\n    Verifies that the calculation follows: principal * rate * fraction.\n    '
    var_1 = '0.5'
    var_2 = 'Actual/Actual'
    var_3 = 'ACT/ACT'
    var_4 = {var_3}
    var_5 = 2023
    var_6 = 1
    var_7 = 6
    var_8 = 2024
    var_9 = '0.05'
    var_10 = '1000.00'
    var_11 = '25.00'
    var_12 = None
    var_13 = '0'



# Parsed testcases at query #10
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
    var_8 = '2'
    var_9 = '365'
    var_10 = '360'
    var_11 = '0'
    var_12 = 2022
    var_13 = 2025



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the coupon method of the DCC class.\n    The coupon method relies on interest, which in turn relies on calculate_fraction.\n    It also uses internal helpers _last_payment_date and _next_payment_date.\n    '
    var_1 = '1000.00'
    var_2 = '0.05'
    var_3 = 2023
    var_4 = 1
    var_5 = 6
    var_6 = 2024
    var_7 = 2
    var_8 = 1
    var_9 = '0.25'
    var_10 = 'Actual/Actual'
    var_11 = 'Act/Act'
    var_12 = {var_11}
    var_13 = set()
    var_14 = '12.5'
    var_15 = 7

def test_case_0():
    var_0 = 'Tests coupon with annual frequency.'
    var_1 = '1.0'
    var_2 = 'Test'
    var_3 = set()
    var_4 = set()
    var_5 = '100'
    var_6 = '0.10'
    var_7 = 2023
    var_8 = 1
    var_9 = 12
    var_10 = 31
    var_11 = 2024
    var_12 = '10.0'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_30_e_360 function with various date scenarios \n    to ensure correct day count fraction calculation.\n    '
    var_1 = 2007
    var_2 = 12
    var_3 = 28
    var_4 = 2008
    var_5 = 2
    var_6 = '0.16666666666667'
    var_7 = 14
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08333333333333'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.33055555555556'
    var_17 = 3
    var_18 = 2023
    var_19 = 6
    var_20 = 15
    var_21 = '0'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Unit tests for the dcfc_nl_365 function.\n    Validates calculation against provided doctest examples and edge cases.\n    '
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
    var_18 = 2023
    var_19 = '0'



# Parsed testcases at query #14
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
    var_8 = '0.17260273972603'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08219178082192'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32602739726027'
    var_18 = 2023
    var_19 = '0'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
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
    var_17 = '1.33055555555556'
    var_18 = 2023
    var_19 = 3
    var_20 = '0.24722222222222'



# Parsed testcases at query #2
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
    var_8 = 151
    var_9 = 360
    var_10 = 2022
    var_11 = 12
    var_12 = 31
    var_13 = 0
    var_14 = 2025
    var_15 = 'FreqTest'
    var_16 = set()
    var_17 = set()
    var_18 = 5



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Mock Convention'
    var_1 = 'Mock'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 2023
    var_5 = 1
    var_6 = 2024
    var_7 = 0
    var_8 = 2
    var_9 = 365
    var_10 = 10
    var_11 = 3



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 'Mock Convention'
    var_2 = 'Mock'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 2023
    var_6 = 1
    var_7 = 10
    var_8 = 20
    var_9 = 19
    var_10 = 2022
    var_11 = 12
    var_12 = 31
    var_13 = 0
    var_14 = 5
    var_15 = 'Freq Convention'
    var_16 = set()
    var_17 = set()
    var_18 = 2
    var_19 = '9.5'



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'MockConvention'
    var_1 = 'Mock'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 2023
    var_5 = 1
    var_6 = 6
    var_7 = 2024
    var_8 = 365
    var_9 = 2022
    var_10 = 12
    var_11 = 31
    var_12 = 0
    var_13 = 2
    var_14 = '2'



# Parsed testcases at query #7
#--------------------------


import datetime as module_0

def test_case_0():
    var_0 = 'Mock Convention'
    var_1 = 'Mock'
    var_2 = {var_1}
    var_3 = set()
    var_4 = 2023
    var_5 = 1
    var_6 = 10
    var_7 = 20
    var_8 = '0.019'
    var_9 = '0'
    var_10 = module_0.timedelta()
    var_11 = '2'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 6
    var_3 = 2024
    var_4 = '2'
    var_5 = 'Test Convention'
    var_6 = 'Test'
    var_7 = {var_6}
    var_8 = set()
    var_9 = '365'
    var_10 = '360'
    var_11 = 2022
    var_12 = 12
    var_13 = 31
    var_14 = '0'
    var_15 = 2
    var_16 = None



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
    var_13 = '1.08333333366667'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.33333333333333'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_365_a function with various date ranges including leap years.\n    '
    var_1 = 14



# Parsed testcases at query #11
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
    var_19 = 8
    var_20 = '0.08333333333333'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_act_act_icma function with various scenarios including \n    standard periods, leap years, and different frequencies.\n    '
    var_1 = 2019
    var_2 = 3
    var_3 = 2
    var_4 = 9
    var_5 = 10
    var_6 = 2020
    var_7 = '0.5245901639'
    var_8 = 2023
    var_9 = 1
    var_10 = 7
    var_11 = 2024
    var_12 = 12
    var_13 = 31
    var_14 = '0'
    var_15 = '1'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dcfc_30_360_us function with various scenarios including \n    standard dates, end of months, and leap years.\n    '
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
    var_20 = 2024
    var_21 = '359'
    var_22 = '360'
    var_23 = 3
    var_24 = '60'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'MockDCC'
    var_1 = 'Mock'
    var_2 = {var_1}
    var_3 = set()
    var_4 = '1000.00'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 31
    var_9 = 2
    var_10 = '30'
    var_11 = '360'
    var_12 = None
    var_13 = '0'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the calculate_fraction method of the DCC class.\n    Tests valid date ranges, invalid date orders (start > asof or asof > end), \n    and correct delegation to the calculation method.\n    '
    var_1 = 'MockConvention'
    var_2 = 'Mock'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 2023
    var_6 = 1
    var_7 = 6
    var_8 = 12
    var_9 = 31
    var_10 = '2'
    var_11 = '0.05'
    var_12 = 2022
    var_13 = '0'
    var_14 = 2024



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the interest calculation method of the DCC class.\n    '
    var_1 = 'Actual/360'
    var_2 = 'A/360'
    var_3 = {var_2}
    var_4 = '1000.00'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 31
    var_9 = 4
    var_10 = '30'
    var_11 = '360'
    var_12 = None
    var_13 = 2



# Parsed testcases at query #17
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
    var_17 = '1.33055555555556'
    var_18 = 2023
    var_19 = '0'
    var_20 = 3
    var_21 = '1'
    var_22 = '6'



