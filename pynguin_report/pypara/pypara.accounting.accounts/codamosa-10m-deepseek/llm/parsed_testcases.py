####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = 'Different Name'
    var_5 = '999'
    var_6 = '1001'
    var_7 = 'Invalid'
    var_8 = 'Self Parent'
    var_9 = '1001'
    var_10 = 'Bank Account'



# Parsed testcases at query #2
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = []
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = []
    var_8 = '999'
    var_9 = '9999'
    var_10 = 'Invalid'
    var_11 = 'Self Parent'
    var_12 = 'Different Name'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Liabilities'
    var_5 = (var_3, var_4)
    var_6 = '3'
    var_7 = 'Equities'
    var_8 = (var_6, var_7)
    var_9 = '4'
    var_10 = 'Revenues'
    var_11 = (var_9, var_10)
    var_12 = '5'
    var_13 = 'Expenses'
    var_14 = (var_12, var_13)
    var_15 = [var_2, var_5, var_8, var_11, var_14]



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 'Equities'
    var_8 = 'Revenues'
    var_9 = 'Expenses'



# Parsed testcases at query #5
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.__iter__()
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = '1'
    var_5 = 'Assets'
    var_6 = '2'
    var_7 = 'Liabilities'
    var_8 = '3'
    var_9 = 'Equities'
    var_10 = '4'
    var_11 = 'Revenues'
    var_12 = '5'
    var_13 = 'Expenses'
    var_14 = '1000'
    var_15 = 'Liquidity'
    var_16 = '1001'
    var_17 = 'Bank Account'
    var_18 = var_0.__iter__()
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 7



# Parsed testcases at query #6
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '999'
    var_7 = '2000'
    var_8 = 'Non-existent Parent'
    var_9 = '1'
    var_10 = 'Same as Parent'
    var_11 = '1'
    var_12 = '1000'
    var_13 = 'Conflict'
    var_14 = '2'
    var_15 = '2000'
    var_16 = 'New Parent'
    var_17 = '2001'
    var_18 = 'New Child'



# Parsed testcases at query #7
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '9999'
    var_7 = '2000'
    var_8 = 'Invalid Account'
    var_9 = 'Self Parent'
    var_10 = 'Different Name'
    var_11 = list(var_0)
    var_12 = len(var_11)
    assert var_12 == 7



# Parsed testcases at query #8
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = []
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = []
    var_8 = '999'
    var_9 = '9999'
    var_10 = 'Invalid Account'
    var_11 = 'Self Parent'
    var_12 = 'Different Name'
    var_13 = var_0._accounts
    var_14 = len(var_13)
    assert var_14 == 7
    var_15 = var_0._subaccounts
    var_16 = len(var_15)
    assert var_16 == 2



# Parsed testcases at query #9
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = '1000'
    var_7 = 'Liquidity'
    var_8 = '1001'
    var_9 = 'Bank Account'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '1001'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #12
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = 0
    var_5 = '1001'
    var_6 = 'Bank Account'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Different Name'
    var_7 = 'Self Parent'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'



# Parsed testcases at query #2
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Different Name'
    var_7 = '9999'
    var_8 = '2000'
    var_9 = 'Invalid Parent'
    var_10 = var_0.add(var_2, var_4, var_9)
    var_11 = 'Self Parent'



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = 'Assets'
    var_7 = 'Liabilities'
    var_8 = 'Equities'
    var_9 = 'Revenues'
    var_10 = 'Expenses'
    var_11 = [var_6, var_7, var_8, var_9, var_10]
    var_12 = list(var_0)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = {}
    var_15 = module_0.COA(rootspec=var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 'Equities'
    var_8 = 'Revenues'
    var_9 = 'Expenses'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '1001'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #17
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #20
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1001'
    var_7 = 'Invalid Account'
    var_8 = '9999'
    var_9 = '2000'
    var_10 = 'Non-Existent Parent Account'
    var_11 = '1002'
    var_12 = 'Conflict Account'
    var_13 = '1001'
    var_14 = '1002'
    var_15 = 'Different Name'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '1001'



# Parsed testcases at query #23
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #30
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #31
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1'
    var_7 = 'Invalid Account'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1'
    var_12 = '1000'
    var_13 = 'Different Name'
    var_14 = var_0.accounts
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 7



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #36
#--------------------------




# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #38
#--------------------------




# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 'Equities'
    var_8 = 'Revenues'
    var_9 = 'Expenses'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #41
#--------------------------




# Parsed testcases at query #42
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '1000'



# Parsed testcases at query #44
#--------------------------




# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #47
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()



# Parsed testcases at query #48
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1000'
    var_7 = '1001'
    var_8 = 'Conflict Account'
    var_9 = '9999'
    var_10 = '2000'
    var_11 = 'Invalid Parent'
    var_12 = '1'
    var_13 = 'Self Parent'



# Parsed testcases at query #49
#--------------------------




# Parsed testcases at query #50
#--------------------------




####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #2
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #3
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = 'Invalid Account'
    var_5 = '9999'
    var_6 = '1001'
    var_7 = 'Invalid Parent'
    var_8 = '1002'
    var_9 = 'Conflict Account'
    var_10 = 'Different Name'
    var_11 = '1001'
    var_12 = 'Bank Account'



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = []
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = []
    var_8 = 'Different Name'
    var_9 = '999'
    var_10 = '1002'
    var_11 = 'Invalid Account'
    var_12 = 'Self Parent'



# Parsed testcases at query #11
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Different Name'
    var_7 = 'Self Parent'
    var_8 = '9999'
    var_9 = '9998'
    var_10 = 'Invalid Parent'



# Parsed testcases at query #12
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #15
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '3'
    var_7 = '3000'
    var_8 = 'Equity Sub-Account'



# Parsed testcases at query #16
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1001'
    var_3 = 'Bank Account'
    var_4 = '2'
    var_5 = 'Different Name'
    var_6 = '1002'
    var_7 = '999'
    var_8 = '1003'
    var_9 = 'Invalid Account'
    var_10 = '1001'
    var_11 = 'Same Code'
    var_12 = var_0.add(var_2, var_8, var_11)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 'Equities'
    var_8 = 'Revenues'
    var_9 = 'Expenses'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equities'
    var_6 = '4'
    var_7 = 'Revenues'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #19
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 0



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'



# Parsed testcases at query #22
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 0



# Parsed testcases at query #23
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 0



# Parsed testcases at query #24
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = 'Different Name'
    var_5 = 'Self Parent'
    var_6 = '999'
    var_7 = '1001'
    var_8 = 'Invalid Parent'
    var_9 = '1001'
    var_10 = 'Bank Account'



# Parsed testcases at query #25
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'



# Parsed testcases at query #26
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1001'
    var_7 = 'Invalid Account'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = '1001'
    var_13 = 'Conflict Name'
    var_14 = '2'
    var_15 = '3'
    var_16 = '4'
    var_17 = '5'
    var_18 = [var_11, var_14, var_15, var_16, var_17, var_12, var_4]
    var_19 = 'Assets'
    var_20 = 'Liabilities'
    var_21 = 'Equities'
    var_22 = 'Revenues'
    var_23 = 'Expenses'
    var_24 = [var_19, var_20, var_21, var_22, var_23, var_13, var_5]



# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '2'
    var_5 = '2000'
    var_6 = 'Loans'
    var_7 = '1001'
    var_8 = 'Bank Account'
    var_9 = '1002'
    var_10 = 'Cash'



# Parsed testcases at query #29
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'



# Parsed testcases at query #30
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = 'Different Name'
    var_7 = 'Self Parent'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'



