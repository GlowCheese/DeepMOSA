####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_10 = '1002'
    var_11 = 'Cash'
    var_12 = var_0.structure
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 5



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #3
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = [(code, acct.name) for (code, acct) in var_0]
    var_2 = '1'
    var_3 = 'Assets'
    var_4 = '2'
    var_5 = 'Liabilities'
    var_6 = '3'
    var_7 = 'Equities'
    var_8 = '4'
    var_9 = 'Revenues'
    var_10 = '5'
    var_11 = 'Expenses'



# Parsed testcases at query #4
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = 'Assets'
    var_5 = '2'
    var_6 = 'Liabilities'
    var_7 = '3'
    var_8 = 'Equities'
    var_9 = '4'
    var_10 = 'Revenues'
    var_11 = '5'
    var_12 = 'Expenses'



# Parsed testcases at query #5
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
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1002'
    var_12 = 'Cash'
    var_13 = '2'
    var_14 = '1002'
    var_15 = 'Different Parent'



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
    var_6 = '1'
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



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
    var_6 = '1002'
    var_7 = 'Cash Account'
    var_8 = '9999'
    var_9 = 'NonExistent'
    var_10 = var_0.nodify(var_3)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #9
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
    var_7 = '1002'
    var_8 = 'Invalid Parent'
    var_9 = '1'
    var_10 = 'Invalid Self Parent'
    var_11 = '1002'
    var_12 = 'Cash'
    var_13 = '2'
    var_14 = '1002'
    var_15 = 'Cash'



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


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



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
    var_6 = '1'
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #16
#--------------------------




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
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 'Equities'
    var_8 = 'Revenues'
    var_9 = 'Expenses'



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
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



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = '999'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #29
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #32
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



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


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



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
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = '1000'
    var_12 = 'Liquidity'
    var_13 = '9999'



# Parsed testcases at query #40
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



# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #44
#--------------------------




# Parsed testcases at query #45
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



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 1



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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #49
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



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #51
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



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #54
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



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '999'



# Parsed testcases at query #56
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #57
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
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1'
    var_12 = '1000'
    var_13 = 'Different Name'



# Parsed testcases at query #58
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #59
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = 'boguscode'



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #62
#--------------------------




# Parsed testcases at query #63
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Same Parent and Code'



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    assert var_2 == 0
    var_3 = '4'
    var_4 = '5'
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 'Equities'
    var_8 = 'Revenues'
    var_9 = 'Expenses'
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = '1000'
    var_12 = 'Liquidity'
    var_13 = '9999'



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = '1'
    assert var_0 == 0
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 'Equities'
    var_8 = 'Revenues'
    var_9 = 'Expenses'



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #72
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #73
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



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #75
#--------------------------




# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'boguscode'



# Parsed testcases at query #78
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



# Parsed testcases at query #79
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



# Parsed testcases at query #80
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



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #83
#--------------------------




# Parsed testcases at query #84
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #85
#--------------------------




# Parsed testcases at query #86
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 'Assets'
    var_1 = 'Liabilities'
    var_2 = 'Equities'
    var_3 = 'Revenues'
    var_4 = 'Expenses'
    var_5 = [var_0, var_1, var_2, var_3, var_4]



# Parsed testcases at query #88
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #92
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #94
#--------------------------




# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #96
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
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #97
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



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #99
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



# Parsed testcases at query #100
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



# Parsed testcases at query #101
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



# Parsed testcases at query #102
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '2'
    var_7 = '1000'
    var_8 = 'Different Name'
    var_9 = '999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1'
    var_13 = 'Same Parent and Code'



# Parsed testcases at query #103
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



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'boguscode'



# Parsed testcases at query #105
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #106
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #107
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



# Parsed testcases at query #108
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #109
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #110
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '999'



# Parsed testcases at query #111
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #112
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'boguscode'



# Parsed testcases at query #113
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #114
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



# Parsed testcases at query #115
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #116
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #117
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



# Parsed testcases at query #118
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
    var_7 = 'Invalid Parent'
    var_8 = 'Same Parent and Code'



# Parsed testcases at query #119
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



# Parsed testcases at query #120
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #121
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #122
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #123
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #124
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #125
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



# Parsed testcases at query #126
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #127
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



# Parsed testcases at query #128
#--------------------------




# Parsed testcases at query #129
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #130
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #131
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #132
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



# Parsed testcases at query #133
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #134
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Test Account'
    var_4 = '9999'
    var_5 = '1001'
    var_6 = 'Invalid Parent'
    var_7 = 'Same Code'
    var_8 = 'Different Name'
    var_9 = '1001'
    var_10 = 'Sub Test Account'
    var_11 = var_0.structure
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = 0



# Parsed testcases at query #135
#--------------------------




# Parsed testcases at query #136
#--------------------------




# Parsed testcases at query #137
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



# Parsed testcases at query #138
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
    var_1 = [(code, acct.name) for (code, acct) in var_0]
    var_2 = '1'
    var_3 = 'Assets'
    var_4 = '2'
    var_5 = 'Liabilities'
    var_6 = '3'
    var_7 = 'Equities'
    var_8 = '4'
    var_9 = 'Revenues'
    var_10 = '5'
    var_11 = 'Expenses'



# Parsed testcases at query #3
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
    var_7 = 'Duplicate Code'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Orphan Account'
    var_11 = '1002'
    var_12 = 'Another Account'
    var_13 = '2'
    var_14 = '1002'
    var_15 = 'Inconsistent Account'
    var_16 = '1000'
    var_17 = 'Self Parent'



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #6
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '9999'
    var_5 = '1001'
    var_6 = 'Invalid Parent'
    var_7 = '1'
    var_8 = 'Same Code'
    var_9 = '1001'
    var_10 = 'Bank Account'
    var_11 = '1000'
    var_12 = '1001'
    var_13 = 'Different Name'
    var_14 = '1002'
    var_15 = 'Cash Account'



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
    var_7 = '1002'
    var_8 = 'Invalid Parent'
    var_9 = '1001'
    var_10 = 'Same Parent and Code'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '999'



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
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1'
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #13
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
    var_11 = '1001'
    var_12 = 'Different Name'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #15
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #16
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
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #17
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
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #18
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = var_0.structure
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 5
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = var_10.children
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_10.children[var_9]
    var_14 = var_13.children
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = var_13.children[var_9]
    var_17 = var_16.children
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = 1
    var_20 = var_7[var_19]
    var_21 = '2'
    var_22 = var_20.children
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = 2
    var_25 = var_7[var_24]
    var_26 = '3'
    var_27 = var_25.children
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = 3
    var_30 = var_7[var_29]
    var_31 = '4'
    var_32 = var_30.children
    var_33 = len(var_32)
    assert var_33 == 0
    var_34 = 4
    var_35 = var_7[var_34]
    var_36 = '5'
    var_37 = var_35.children
    var_38 = len(var_37)
    assert var_38 == 0



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



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
    var_6 = '1000'
    var_7 = '1001'
    var_8 = 'Different Name'
    var_9 = '1001'
    var_10 = 'Same Code'
    var_11 = '9999'
    var_12 = '1002'
    var_13 = 'Non-existent Parent'



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
    var_6 = '9999'
    var_7 = '1002'
    var_8 = 'Invalid'
    var_9 = '1001'
    var_10 = 'Invalid'
    var_11 = '1000'
    var_12 = '1001'
    var_13 = 'Different Name'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #24
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



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
    var_6 = '9999'
    var_7 = '1002'
    var_8 = 'Invalid Account'
    var_9 = '1001'
    var_10 = 'Invalid Account'
    var_11 = '1000'
    var_12 = '1001'
    var_13 = 'Different Name'



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
    var_6 = '1002'
    var_7 = 'Cash Account'
    var_8 = 0
    var_9 = 1
    var_10 = '2'



# Parsed testcases at query #27
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '2'



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #33
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]
    var_11 = '999'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



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


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '2'



# Parsed testcases at query #37
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Test Account'
    var_4 = '999'
    var_5 = '1001'
    var_6 = 'Invalid Parent'
    var_7 = '1000'
    var_8 = 'Self Parent'
    var_9 = '1001'
    var_10 = 'Another Account'
    var_11 = '1'
    var_12 = '1001'
    var_13 = 'Different Name'
    var_14 = '2000'
    var_15 = 'Level 1'
    var_16 = '2001'
    var_17 = 'Level 2'
    var_18 = '2002'
    var_19 = 'Level 3'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #40
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '9999'
    var_5 = '1001'
    var_6 = 'Invalid Parent'
    var_7 = '1'
    var_8 = 'Same Code'
    var_9 = '1'
    var_10 = '1000'
    var_11 = 'Different Name'
    var_12 = '1001'
    var_13 = 'Bank Account'
    var_14 = var_0.accounts
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 7



# Parsed testcases at query #41
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Same Parent and Code'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #43
#--------------------------




# Parsed testcases at query #44
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
    var_7 = 'Different Name'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1001'
    var_12 = 'Same Parent and Code'



# Parsed testcases at query #45
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #46
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = 'Different Name'
    var_5 = '9999'
    var_6 = '1001'
    var_7 = 'Orphan Account'
    var_8 = 'Self Parent Account'



# Parsed testcases at query #47
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '10'
    var_6 = '20'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #50
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1002'
    var_7 = 'Cash Account'
    var_8 = 0
    var_9 = 1
    var_10 = '2'



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '999'



# Parsed testcases at query #53
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = var_0.structure
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 5
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = var_10.children
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_10.children[var_9]
    var_14 = var_13.children
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = var_13.children[var_9]
    var_17 = var_16.children
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = 1
    var_20 = var_7[var_19]
    var_21 = '2'
    var_22 = var_20.children
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = 2
    var_25 = var_7[var_24]
    var_26 = '3'
    var_27 = var_25.children
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = 3
    var_30 = var_7[var_29]
    var_31 = '4'
    var_32 = var_30.children
    var_33 = len(var_32)
    assert var_33 == 0
    var_34 = 4
    var_35 = var_7[var_34]
    var_36 = '5'
    var_37 = var_35.children
    var_38 = len(var_37)
    assert var_38 == 0



# Parsed testcases at query #54
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #55
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1002'
    var_13 = 'Self Parent'



# Parsed testcases at query #56
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #58
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
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #60
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
    var_7 = 'Invalid'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #61
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
    var_7 = '1002'
    var_8 = 'Invalid Parent'
    var_9 = '1'
    var_10 = 'Same Code'
    var_11 = '1'
    var_12 = '1000'
    var_13 = 'Different Name'



# Parsed testcases at query #62
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
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 1



# Parsed testcases at query #65
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Same Code'



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #67
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1002'
    var_7 = 'Cash'
    var_8 = 0



# Parsed testcases at query #68
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #69
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
    var_7 = '1002'
    var_8 = 'Invalid Parent'
    var_9 = '1001'
    var_10 = 'Self Parent'
    var_11 = '1000'
    var_12 = 'Different Name'



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #71
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



# Parsed testcases at query #72
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Test Account'
    var_4 = 'Different Name'
    var_5 = '9999'
    var_6 = '1001'
    var_7 = 'Orphan Account'
    var_8 = 'Self Parent Account'



# Parsed testcases at query #73
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Same Parent and Code'



# Parsed testcases at query #74
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



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #76
#--------------------------




# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #78
#--------------------------




# Parsed testcases at query #79
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
    var_7 = 'Different Name'
    var_8 = '9999'
    var_9 = '1002'
    var_10 = 'Invalid Parent'
    var_11 = '1'
    var_12 = 'Same Parent and Code'
    var_13 = var_0.structure
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 5
    var_16 = 0



# Parsed testcases at query #80
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
    var_10 = 'Invalid Parent Account'
    var_11 = '1002'
    var_12 = 'Cash'
    var_13 = '2'
    var_14 = '1002'
    var_15 = 'Cash'



# Parsed testcases at query #81
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
    var_12 = '1001'
    var_13 = 'Different Name'



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #83
#--------------------------




# Parsed testcases at query #84
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



# Parsed testcases at query #85
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1002'
    var_7 = 'Cash Account'
    var_8 = 0
    var_9 = 1
    var_10 = '9999'
    var_11 = 'NonExistent'
    var_12 = var_0.nodify(var_3)



# Parsed testcases at query #86
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Same Parent and Code'



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #88
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
    var_7 = '1000'
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #89
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



# Parsed testcases at query #90
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



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #92
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #93
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



# Parsed testcases at query #94
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #95
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
    var_7 = '1002'
    var_8 = 'Invalid Account'
    var_9 = '1001'
    var_10 = 'Invalid Account'
    var_11 = '1000'
    var_12 = '1001'
    var_13 = 'Different Name'



# Parsed testcases at query #96
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '2'



# Parsed testcases at query #97
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
    var_10 = '1000'
    var_11 = 'Test Account'



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '999'



# Parsed testcases at query #99
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #100
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



# Parsed testcases at query #101
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #102
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #103
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Invalid Parent'
    var_12 = '1001'
    var_13 = 'Self Parent'



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #105
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #106
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'
    var_6 = '1002'
    var_7 = 'Cash Account'
    var_8 = 0
    var_9 = 1
    var_10 = '2'



# Parsed testcases at query #107
#--------------------------




# Parsed testcases at query #108
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #109
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



# Parsed testcases at query #110
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
    var_7 = '1002'
    var_8 = 'Invalid Parent'
    var_9 = '1001'
    var_10 = 'Same Parent and Code'
    var_11 = '1000'
    var_12 = '1001'
    var_13 = 'Different Name'



# Parsed testcases at query #111
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
    var_7 = '1002'
    var_8 = 'Invalid Parent'
    var_9 = '1'
    var_10 = 'Same Code'
    var_11 = '1002'
    var_12 = 'Cash'
    var_13 = '1'
    var_14 = '1002'
    var_15 = 'Different Name'



# Parsed testcases at query #112
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
    var_8 = 'Different Name'
    var_9 = '9999'
    var_10 = '1002'
    var_11 = 'Orphan Account'
    var_12 = '1'
    var_13 = 'Self Parent'



# Parsed testcases at query #113
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



# Parsed testcases at query #114
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #115
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '1001'
    var_5 = 'Bank Account'



# Parsed testcases at query #116
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'
    var_4 = '9999'
    var_5 = '1001'
    var_6 = 'Invalid Parent'
    var_7 = 'Same Code'
    var_8 = 'Different Name'
    var_9 = '1001'
    var_10 = 'Bank Account'



# Parsed testcases at query #117
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
    var_10 = 'Invalid Parent Account'
    var_11 = '1'
    var_12 = '1000'
    var_13 = 'Different Name'



# Parsed testcases at query #118
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
    var_7 = '1002'
    var_8 = 'Invalid Parent'
    var_9 = '1'
    var_10 = 'Invalid Account'
    var_11 = '1000'
    var_12 = '1001'
    var_13 = 'Different Name'



# Parsed testcases at query #119
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #120
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #121
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = '999'



# Parsed testcases at query #122
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



# Parsed testcases at query #123
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #124
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
    var_10 = [var_5, var_6, var_7, var_8, var_9]



# Parsed testcases at query #125
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



# Parsed testcases at query #126
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



# Parsed testcases at query #127
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



# Parsed testcases at query #128
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



# Parsed testcases at query #129
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = 'Liquidity'
    var_3 = '9999'



# Parsed testcases at query #130
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



# Parsed testcases at query #131
#--------------------------


def test_case_0():
    var_0 = 0



# Parsed testcases at query #132
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'



