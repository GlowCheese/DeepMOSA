####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
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
    var_18 = 2020
    var_19 = '0.08333333333333'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = 31
    var_3 = 14
    var_4 = '0.08219178082192'
    var_5 = 2016
    var_6 = 12
    var_7 = '0.99726775956284'
    var_8 = 2015
    var_9 = '2.00000000000000'
    var_10 = 2012
    var_11 = '4.00000000000000'
    var_12 = 2
    var_13 = 29
    var_14 = '0.07650273224044'
    var_15 = 30
    var_16 = 2014
    var_17 = 3
    var_18 = '0.16721311475410'



# Parsed testcases at query #3
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestAlt1'
    var_3 = 'TestAlt2'
    var_4 = {var_2, var_3}
    var_5 = 'USD'
    var_6 = 'EUR'
    var_7 = {var_5, var_6}
    var_8 = module_0._as_ccys(var_7)
    var_9 = '0.5'
    var_10 = 'TestDCC2'
    var_11 = {var_2}
    var_12 = 'GBP'
    var_13 = {var_12}
    var_14 = module_0._as_ccys(var_13)
    var_15 = '0.7'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33333333333333'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 29
    var_7 = '0.17213114754098'
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08196721311475'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32513661202186'



# Parsed testcases at query #6
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = {var_4}
    var_6 = module_0._as_ccys(var_5)
    var_7 = '0.1'
    var_8 = var_0.find(var_1)
    var_9 = var_0.find(var_2)
    var_10 = ' act/act '
    var_11 = var_0.find(var_10)
    var_12 = 'NonExistent'
    var_13 = var_0.find(var_12)
    assert var_13 is None



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = '0.2'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33333333333333'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '1'
    var_7 = '0.5245901639'



# Parsed testcases at query #10
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
    var_17 = 2020
    var_18 = '0.00273972602740'
    var_19 = '0.15890410958904'



# Parsed testcases at query #11
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



# Parsed testcases at query #12
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
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32513661202186'



# Parsed testcases at query #13
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
    var_18 = 2017
    var_19 = '0.0'
    var_20 = '0.00273972602740'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = 1
    var_7 = '0.5245901639'
    var_8 = 12
    var_9 = 31
    var_10 = 366
    var_11 = 4
    var_12 = 7
    var_13 = 91
    var_14 = 182
    var_15 = 0
    var_16 = 6
    var_17 = 62
    var_18 = 183



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16986301369863'
    var_15 = '0.17213114754098'
    var_16 = '1.08196721311475'
    var_17 = '1.32513661202186'



# Parsed testcases at query #16
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
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32513661202186'
    var_18 = 2019
    var_19 = '1'
    var_20 = '365'
    var_21 = 2020
    var_22 = '366'



# Parsed testcases at query #17
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = 'Actual/Actual (ISDA)'
    var_4 = {var_2, var_3}
    var_5 = 'USD'
    var_6 = 'EUR'
    var_7 = {var_5, var_6}
    var_8 = module_0._as_ccys(var_7)
    var_9 = '0.5'
    var_10 = var_0.find(var_1)
    var_11 = var_0.find(var_2)
    var_12 = var_0.find(var_3)
    var_13 = {var_2, var_3}
    var_14 = {var_5, var_6}
    var_15 = module_0._as_ccys(var_14)
    var_16 = '30/360'
    var_17 = {var_1}
    var_18 = {var_5, var_6}
    var_19 = module_0._as_ccys(var_18)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33333333333333'



# Parsed testcases at query #19
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
    var_19 = '0'



# Parsed testcases at query #20
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
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32513661202186'
    var_18 = 2020
    var_19 = '0.00273224043716'
    var_20 = 2021
    var_21 = '1.00273224043716'



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = '0.5245901639'
    var_7 = 1
    var_8 = 6
    var_9 = 30
    var_10 = 12
    var_11 = 31
    var_12 = '0.5081967213'
    var_13 = 2021
    var_14 = 7
    var_15 = 2022
    var_16 = '0.0'
    var_17 = 4
    var_18 = '0.2540983607'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
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



# Parsed testcases at query #2
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
    var_18 = '0.00000000000000'
    var_19 = '0.00273224043716'



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



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'mock_dcc'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = 'USD'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 31
    var_9 = 12
    var_10 = '5'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16986301369863'
    var_15 = '1.08219178082192'
    var_16 = '1.32602739726027'



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



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33055555555556'



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
    var_17 = '1.33333333333333'
    var_18 = 3



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_dcc'
    var_1 = set()
    var_2 = set()
    var_3 = '100'
    var_4 = 'USD'
    var_5 = '0.1'
    var_6 = 2023
    var_7 = 1
    var_8 = 6
    var_9 = 30
    var_10 = 12
    var_11 = 31
    var_12 = '5'
    var_13 = '0'
    var_14 = 2022
    var_15 = 2024
    var_16 = None
    var_17 = 'EUR'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'mock'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = '0.1'
    var_8 = 2022
    var_9 = 12
    var_10 = 31
    var_11 = '0'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'SimpleDCC'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = 'USD'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 15
    var_9 = 31
    var_10 = 2
    var_11 = '0'
    var_12 = '1'
    var_13 = 30
    var_14 = '-0.05'
    var_15 = '-1000'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16986301369863'
    var_6 = 14
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
    var_17 = '1.32513661202186'
    var_18 = 2020
    var_19 = '0.0'
    var_20 = '366'



# Parsed testcases at query #13
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
    var_17 = 2020
    var_18 = '0.00273972602740'
    var_19 = 2019
    var_20 = 3



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33333333333333'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'TEST'
    var_1 = set()
    var_2 = set()
    var_3 = '100'
    var_4 = 'USD'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 7
    var_9 = 2024
    var_10 = 2
    var_11 = 365
    var_12 = 2022
    var_13 = 12
    var_14 = 31
    var_15 = '0'
    var_16 = 4
    var_17 = 4
    var_18 = 31
    var_19 = 2
    var_20 = 15
    var_21 = 1



# Parsed testcases at query #16
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
    var_19 = '0.08333333333333'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33333333333333'



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
    var_18 = '0.99722222222222'
    var_19 = 2010
    var_20 = 15
    var_21 = '0.0'



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



# Parsed testcases at query #20
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
    var_8 = '0.17213114754098'
    var_9 = 10
    var_10 = 31
    var_11 = 11
    var_12 = 30
    var_13 = '1.08196721311475'
    var_14 = 1
    var_15 = 2009
    var_16 = 5
    var_17 = '1.32513661202186'



# Parsed testcases at query #21
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = {var_4}
    var_6 = module_0._as_ccys(var_5)
    var_7 = '0.16942884946478'
    var_8 = var_0.find(var_1)
    var_9 = var_0.find(var_2)
    var_10 = '  Act/Act  '
    var_11 = var_0.find(var_10)
    var_12 = 'ACT/ACT'
    var_13 = var_0.find(var_12)
    var_14 = 'NonExistent'
    var_15 = var_0.find(var_14)
    assert var_15 is None



# Parsed testcases at query #22
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
    var_17 = 2020
    var_18 = 3
    var_19 = '0.16712328767123'
    var_20 = 2019
    var_21 = 2021
    var_22 = '1.00547945205479'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
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
    var_18 = '0.00000000000000'
    var_19 = 2020
    var_20 = '1.00000000000000'
    var_21 = 2019
    var_22 = 2021
    var_23 = '3.00000000000000'
    var_24 = 3
    var_25 = '0.00273224043716'
    var_26 = '0.00273972602740'



# Parsed testcases at query #2
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
    var_18 = 2020
    var_19 = '0.00273224043716'
    var_20 = 2021
    var_21 = '1.00000000000000'
    var_22 = '1.00273972602740'



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



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'TEST'
    var_1 = set()
    var_2 = set()
    var_3 = '100'
    var_4 = 'USD'
    var_5 = '0.05'
    var_6 = 2023
    var_7 = 1
    var_8 = 31
    var_9 = 12
    var_10 = 30
    var_11 = 364
    var_12 = '0'
    var_13 = 2022
    var_14 = 2024
    var_15 = 'EUR'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16986301369863'
    var_15 = '1.08219178082192'
    var_16 = '1.32602739726027'



# Parsed testcases at query #6
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'Actual/Actual'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = '0.5'
    var_6 = var_0.find(var_1)
    var_7 = var_0.find(var_2)
    var_8 = 'SomeOtherName'
    var_9 = {var_8}
    var_10 = 'EUR'



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
    var_9 = 3
    var_10 = 31
    var_11 = '0.08333333333333'
    var_12 = 10
    var_13 = 11
    var_14 = 30
    var_15 = '1.08333333333333'
    var_16 = 1
    var_17 = 2009
    var_18 = 5
    var_19 = '1.33055555555556'
    var_20 = 15
    var_21 = '0.13888888888889'



# Parsed testcases at query #8
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = 'TestAlt1'
    var_2 = 'TestAlt2'
    var_3 = {var_1, var_2}
    var_4 = 'USD'
    var_5 = 'EUR'
    var_6 = {var_4, var_5}
    var_7 = module_0._as_ccys(var_6)
    var_8 = '0.5'
    var_9 = module_0.DCCRegistryMachinery()
    var_10 = 'TestDCC2'
    var_11 = {var_1}
    var_12 = 'GBP'
    var_13 = {var_12}
    var_14 = module_0._as_ccys(var_13)
    var_15 = '0.3'
    var_16 = 'TestAlt3'
    var_17 = {var_16}
    var_18 = 'JPY'
    var_19 = {var_18}
    var_20 = module_0._as_ccys(var_19)
    var_21 = '0.1'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'ACT/365'
    var_1 = set()
    var_2 = set()
    var_3 = 365
    var_4 = '100'
    var_5 = 'USD'
    var_6 = '0.1'
    var_7 = 2023
    var_8 = 1
    var_9 = 2
    var_10 = '0.0273972602739726'
    var_11 = '0'
    var_12 = 3
    var_13 = 2022
    var_14 = 12
    var_15 = 31
    var_16 = 4
    var_17 = 'EUR'



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'TestDCC'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 3
    var_6 = '0.1'
    var_7 = 2
    var_8 = '0.0'



# Parsed testcases at query #13
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'MockDCC'
    var_1 = 'mock'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = {var_3}
    var_5 = module_0._as_ccys(var_4)
    var_6 = '1000'
    var_7 = '0.05'
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 30
    var_12 = 12
    var_13 = 31
    var_14 = '25.0'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33055555555556'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33055555555556'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = 1
    var_7 = '0.5245901639'
    var_8 = 2021
    var_9 = 0
    var_10 = 4
    var_11 = 91
    var_12 = 366
    var_13 = 28



# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'mock_dcc'
    var_1 = set()
    var_2 = set()
    var_3 = '1000'
    var_4 = 'USD'
    var_5 = module_0.Currency(var_4)
    var_6 = '0.05'
    var_7 = 2023
    var_8 = 1
    var_9 = '0'
    var_10 = module_0.Currency(var_4)
    var_11 = 2
    var_12 = '1'
    var_13 = module_0.Currency(var_4)
    var_14 = 3
    var_15 = '2'
    var_16 = module_0.Currency(var_4)
    var_17 = module_0.Currency(var_4)
    var_18 = 4



# Parsed testcases at query #19
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = {var_4}
    var_6 = module_0._as_ccys(var_5)
    var_7 = '1.0'
    var_8 = 'TestDCC2'
    var_9 = {var_2}
    var_10 = 'EUR'
    var_11 = {var_10}
    var_12 = module_0._as_ccys(var_11)
    var_13 = '2.0'



# Parsed testcases at query #20
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'SampleDCC'
    var_1 = 'Sample'
    var_2 = 'Test'
    var_3 = {var_1, var_2}
    var_4 = 'USD'
    var_5 = {var_4}
    var_6 = module_0._as_ccys(var_5)
    var_7 = '1000.00'
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = 30
    var_12 = 12
    var_13 = 31
    var_14 = '0.05'
    var_15 = '2'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = 9
    var_4 = 10
    var_5 = 2020
    var_6 = 1
    var_7 = '0.5245901639'
    var_8 = 12
    var_9 = 31
    var_10 = '0'
    var_11 = '1'
    var_12 = 4
    var_13 = 7
    var_14 = '0.5'
    var_15 = 28
    var_16 = 8
    var_17 = '0.0163934426'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'TEST'
    var_1 = set()
    var_2 = set()
    var_3 = 2023
    var_4 = 1
    var_5 = 12
    var_6 = 31
    var_7 = '0.5'
    var_8 = 2
    var_9 = '0.0'
    var_10 = 2022
    var_11 = 2024
    var_12 = 'DAYCOUNT'
    var_13 = set()
    var_14 = set()
    var_15 = 3
    var_16 = '1'



# Parsed testcases at query #23
#--------------------------


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
    var_10 = '1.0'
    var_11 = 4
    var_12 = 7
    var_13 = '0.5'
    var_14 = '0.0081967213'
    var_15 = 28
    var_16 = 8
    var_17 = '0.0163934426'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 29
    var_6 = 10
    var_7 = 31
    var_8 = 11
    var_9 = 30
    var_10 = 1
    var_11 = 2009
    var_12 = 5
    var_13 = 14
    var_14 = '0.16666666666667'
    var_15 = '0.16944444444444'
    var_16 = '1.08333333333333'
    var_17 = '1.33333333333333'



