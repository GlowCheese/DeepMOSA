####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_19 = '0.08333333333333'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16942884946478'
    var_6 = 14
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
    var_19 = '0.0'
    var_20 = 2019
    var_21 = '1.0'



# Parsed testcases at query #3
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestDCCAlt1'
    var_3 = 'TestDCCAlt2'
    var_4 = {var_2, var_3}
    var_5 = 'USD'
    var_6 = {var_5}
    var_7 = module_0._as_ccys(var_6)
    var_8 = '0.5'
    var_9 = 'TestDCCAlt3'
    var_10 = {var_9}
    var_11 = 'EUR'
    var_12 = {var_11}
    var_13 = module_0._as_ccys(var_12)
    var_14 = '0.3'
    var_15 = 'ConflictDCC'
    var_16 = {var_2}
    var_17 = 'GBP'
    var_18 = {var_17}
    var_19 = module_0._as_ccys(var_18)
    var_20 = '0.4'



# Parsed testcases at query #4
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
    var_18 = '1.02777777777778'



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Test1'
    var_2 = 'TEST1'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.5
    var_6 = 'Test2'
    var_7 = 'TEST2'
    var_8 = 'ALTERNATIVE'
    var_9 = {var_7, var_8}
    var_10 = set()
    var_11 = var_0.find(var_1)
    var_12 = var_0.find(var_6)
    var_13 = var_0.find(var_8)
    var_14 = ' test1 '
    var_15 = var_0.find(var_14)
    var_16 = '  test2  '
    var_17 = var_0.find(var_16)
    var_18 = 'NonExistent'
    var_19 = var_0.find(var_18)
    assert var_19 is None
    var_20 = ''
    var_21 = var_0.find(var_20)
    assert var_21 is None



# Parsed testcases at query #7
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
    var_19 = '0'
    var_20 = '0.00273972602740'
    var_21 = '1.0'
    var_22 = 2016
    var_23 = '1.00273972602740'
    var_24 = 6
    var_25 = '0.50068493150685'
    var_26 = None



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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'Simple'
    var_1 = 'simple'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = {var_3}
    var_5 = module_0._as_ccys(var_4)
    var_6 = 2020
    var_7 = 1
    var_8 = 10
    var_9 = 5
    var_10 = 4
    var_11 = 9
    var_12 = 3
    var_13 = 0
    var_14 = 8
    var_15 = 2019
    var_16 = 12
    var_17 = 31



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 6
    var_3 = 12
    var_4 = 31
    var_5 = 2
    var_6 = 'TEST'
    var_7 = 'TEST_ALT'
    var_8 = {var_7}
    var_9 = 'USD'
    var_10 = module_0.Currency(var_9)
    var_11 = {var_10}
    var_12 = 152
    var_13 = 366
    var_14 = 2019
    var_15 = 2021
    var_16 = None



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'TEST'
    var_1 = 'TEST_ALT'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = 1000
    var_7 = module_0.Currency(var_3)
    var_8 = '0.05'
    var_9 = 2020
    var_10 = 1
    var_11 = 12
    var_12 = 31
    var_13 = '0'
    var_14 = 6
    var_15 = 30
    var_16 = '1'
    var_17 = 2019
    var_18 = 2021



# Parsed testcases at query #15
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = '0.5'
    var_6 = var_0.find(var_1)
    var_7 = var_0.find(var_2)
    var_8 = 'AnotherAlt'
    var_9 = {var_8}
    var_10 = 'EUR'
    var_11 = '0.3'
    var_12 = 'NewDCC'
    var_13 = {var_2}
    var_14 = 'GBP'
    var_15 = '0.4'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16942884946478'
    var_6 = 14
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
    var_18 = 2010
    var_19 = '0.00000000000000'
    var_20 = '1.00000000000000'
    var_21 = 2012
    var_22 = 2013
    var_23 = '3.00000000000000'



# Parsed testcases at query #17
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
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'



# Parsed testcases at query #18
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
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'



# Parsed testcases at query #19
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
    var_10 = '0.0'
    var_11 = 29
    var_12 = '425'
    var_13 = '731'
    var_14 = '1'
    var_15 = 6
    var_16 = 30
    var_17 = 2021
    var_18 = '2'
    var_19 = '212'
    var_20 = '366'
    var_21 = '1.0'
    var_22 = 15
    var_23 = '15'



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
    var_18 = '0.00000000000000'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_18 = 2020
    var_19 = '0'
    var_20 = '1'
    var_21 = 6
    var_22 = '0.5'



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
    var_19 = '0'
    var_20 = '0.00273972602740'
    var_21 = 2019
    var_22 = '1.00000000000000'
    var_23 = 3
    var_24 = '60'
    var_25 = '366'
    var_26 = '1'
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
    var_18 = '0.00000000000000'
    var_19 = '0.02777777777778'



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'simple'
    var_1 = {var_0}
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = {var_3}
    var_5 = 1000
    var_6 = module_0.Currency(var_2)
    var_7 = '0.05'
    var_8 = 2020
    var_9 = 1
    var_10 = 31
    var_11 = 0
    var_12 = module_0.Currency(var_2)
    var_13 = 15
    var_14 = 14
    var_15 = 30
    var_16 = 2



# Parsed testcases at query #5
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
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'



# Parsed testcases at query #6
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'TestDCCAlt'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = {var_4}
    var_6 = module_0._as_ccys(var_5)
    var_7 = '0.5'
    var_8 = var_0.find(var_1)
    var_9 = var_0.find(var_2)
    var_10 = 'AnotherAlt'
    var_11 = {var_10}
    var_12 = 'EUR'
    var_13 = {var_12}
    var_14 = module_0._as_ccys(var_13)
    var_15 = '0.3'
    var_16 = 'NewDCC'
    var_17 = {var_2}
    var_18 = 'GBP'
    var_19 = {var_18}
    var_20 = module_0._as_ccys(var_19)
    var_21 = '0.4'
    var_22 = var_0.registry
    var_23 = len(var_22)
    assert var_23 == 1



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


import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'TestDCC'
    var_2 = 'AltTestDCC'
    var_3 = {var_2}
    var_4 = 'USD'
    var_5 = '0.5'
    var_6 = 'AnotherAlt'
    var_7 = {var_6}
    var_8 = 'EUR'
    var_9 = '0.3'
    var_10 = 'NewDCC'
    var_11 = {var_2}
    var_12 = 'GBP'
    var_13 = '0.4'



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'TestAlt'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = '0.5'
    var_7 = 1000
    var_8 = module_0.Currency(var_3)
    var_9 = '0.10'
    var_10 = 2020
    var_11 = 1
    var_12 = 6
    var_13 = 12
    var_14 = 31
    var_15 = 'Test2'
    var_16 = 'TestAlt2'
    var_17 = {var_16}
    var_18 = 'EUR'
    var_19 = module_0.Currency(var_18)
    var_20 = {var_19}
    var_21 = '0.25'
    var_22 = 0
    var_23 = module_0.Currency(var_3)
    var_24 = '0'



# Parsed testcases at query #10
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
    var_8 = 10
    var_9 = 31
    var_10 = 11
    var_11 = 30
    var_12 = '1.08219178082192'
    var_13 = 1
    var_14 = 2009
    var_15 = 5
    var_16 = '1.32602739726027'
    var_17 = '1.0'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 6
    var_3 = 12
    var_4 = 31
    var_5 = 2
    var_6 = 'ACT/ACT'
    var_7 = {var_6}
    var_8 = 'USD'
    var_9 = 182
    var_10 = 366
    var_11 = 2019
    var_12 = 2021
    var_13 = 0



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = 'Test'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = '0.5'
    var_7 = 1000
    var_8 = module_0.Currency(var_3)
    var_9 = '0.05'
    var_10 = 2020
    var_11 = 1
    var_12 = 6
    var_13 = 12
    var_14 = 31
    var_15 = 2
    var_16 = None
    var_17 = 25
    var_18 = module_0.Currency(var_3)



# Parsed testcases at query #13
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
    var_18 = '0.0'
    var_19 = 3
    var_20 = '0.08333333333333'



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'TEST'
    var_1 = 'TEST_ALT'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = 2020
    var_7 = 1
    var_8 = 10
    var_9 = 5
    var_10 = '0.1111111111111111111111111111'
    var_11 = 2
    var_12 = 2019
    var_13 = 12
    var_14 = 31
    var_15 = 11



# Parsed testcases at query #15
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
    var_10 = '0.0'
    var_11 = 6
    var_12 = 2021
    var_13 = '0.4109589041'
    var_14 = '1.0'



# Parsed testcases at query #16
#--------------------------


import pypara.dcc as module_0

def test_case_0():
    var_0 = 'TestDCC'
    var_1 = 'Test'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = {var_3}
    var_5 = module_0._as_ccys(var_4)
    var_6 = 2020
    var_7 = 1
    var_8 = 15
    var_9 = 31
    var_10 = '0.4838709677419355'
    var_11 = '0'
    var_12 = '1'
    var_13 = 2019
    var_14 = 12
    var_15 = 2
    var_16 = 'TestDCCFreq'
    var_17 = 'TestFreq'
    var_18 = {var_17}
    var_19 = {var_3}
    var_20 = module_0._as_ccys(var_19)
    var_21 = '2'
    var_22 = '0.967741935483871'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = '0.16939890710383'
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
    var_17 = '1.32876712328767'



# Parsed testcases at query #18
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'TEST'
    var_1 = 'TEST_ALT'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = 1000
    var_7 = '0.10'
    var_8 = 2020
    var_9 = 1
    var_10 = 6
    var_11 = 12
    var_12 = 31
    var_13 = 50
    var_14 = 0
    var_15 = '0.00'
    var_16 = 'ACT/360'
    var_17 = 'ACTUAL/360'
    var_18 = {var_17}
    var_19 = module_0.Currency(var_3)
    var_20 = {var_19}
    var_21 = 7
    var_22 = 50.27777777777778



# Parsed testcases at query #19
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'MOCK'
    var_1 = 'MOCK_ALT'
    var_2 = {var_1}
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = {var_4}
    var_6 = 1000
    var_7 = module_0.Currency(var_3)
    var_8 = '0.10'
    var_9 = 2020
    var_10 = 1
    var_11 = 6
    var_12 = 12
    var_13 = 31
    var_14 = 50
    var_15 = module_0.Currency(var_3)
    var_16 = 'MOCK2'
    var_17 = 'MOCK_ALT2'
    var_18 = {var_17}
    var_19 = 'EUR'
    var_20 = module_0.Currency(var_19)
    var_21 = {var_20}
    var_22 = 25
    var_23 = module_0.Currency(var_19)
    var_24 = 'MOCK_ZERO'
    var_25 = 'MOCK_ALT_ZERO'
    var_26 = {var_25}
    var_27 = 'GBP'
    var_28 = module_0.Currency(var_27)
    var_29 = {var_28}
    var_30 = 0
    var_31 = module_0.Currency(var_27)
    var_32 = 2019
    var_33 = module_0.Currency(var_3)



# Parsed testcases at query #20
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



# Parsed testcases at query #21
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
    var_19 = '0'
    var_20 = '0.00273972602740'
    var_21 = '1.0'
    var_22 = 2016
    var_23 = '1.00273972602740'



# Parsed testcases at query #22
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
    var_19 = '0'
    var_20 = '0.00277777777778'
    var_21 = '0.08333333333333'
    var_22 = 2021
    var_23 = '1.0'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 1000
    var_1 = 'USD'
    var_2 = '0.05'
    var_3 = 2020
    var_4 = 1
    var_5 = 3
    var_6 = 6
    var_7 = 2
    var_8 = None
    var_9 = 'MOCK'
    var_10 = set()
    var_11 = set()
    var_12 = 360
    var_13 = 7



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
    var_18 = 2020
    var_19 = '0.0'
    var_20 = 15
    var_21 = '0.14166666666667'
    var_22 = '0.13888888888889'



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = 2008
    var_4 = 2
    var_5 = 14
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
    var_18 = 2020
    var_19 = '0.00273972602740'
    var_20 = 2019
    var_21 = 3
    var_22 = '0.00273224043716'



# Parsed testcases at query #27
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
    var_10 = 2021
    var_11 = '0.4958904110'
    var_12 = '1.0000000000'
    var_13 = 12
    var_14 = 31



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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



