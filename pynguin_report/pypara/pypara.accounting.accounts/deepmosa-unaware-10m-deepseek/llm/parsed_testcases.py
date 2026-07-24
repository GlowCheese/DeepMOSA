####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_6 = 0
    var_7 = '9999'
    var_8 = 'Fake'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #3
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = [code for (code, _) in var_1]
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'
    var_9 = '1000'
    var_10 = 'Liquidity'
    var_11 = '1001'
    var_12 = 'Bank Account'
    var_13 = list(var_0)
    var_14 = len(var_13)
    assert var_14 == 7
    var_15 = [code for (code, _) in var_13]
    var_16 = [var_4, var_5, var_6, var_7, var_8, var_9, var_11]
    var_17 = [code for (code, _) in var_13]
    var_18 = [Code(c) for c in var_16]
    var_19 = iter(var_0)
    var_20 = next(var_19)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = 0
    var_23 = var_20[var_22]
    var_24 = 1
    var_25 = var_20[var_24]



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
    var_5 = (var_3, var_4)
    var_6 = '2'
    var_7 = 'Liabilities'
    var_8 = (var_6, var_7)
    var_9 = '3'
    var_10 = 'Equities'
    var_11 = (var_9, var_10)
    var_12 = '4'
    var_13 = 'Revenues'
    var_14 = (var_12, var_13)
    var_15 = '5'
    var_16 = 'Expenses'
    var_17 = (var_15, var_16)
    var_18 = [var_5, var_8, var_11, var_14, var_17]
    var_19 = [code for (code, _) in var_1]
    var_20 = 'A'
    var_21 = 'Custom Assets'
    var_22 = 'L'
    var_23 = 'Custom Liabilities'
    var_24 = 'E'
    var_25 = 'Custom Equities'
    var_26 = 'R'
    var_27 = 'Custom Revenues'
    var_28 = 'X'
    var_29 = 'Custom Expenses'
    var_30 = module_0.COA()
    var_31 = '1000'
    var_32 = 'Liquidity'
    var_33 = '1001'
    var_34 = 'Bank Account'
    var_35 = list(var_30)
    var_36 = len(var_35)
    assert var_36 == 7
    var_37 = [var_3, var_6, var_9, var_12, var_15, var_31, var_33]
    var_38 = iter(var_0)
    var_39 = '__next__'
    var_40 = hasattr(var_38, var_39)
    var_41 = module_0.COA()
    var_42 = list(var_41)
    var_43 = list(var_41)
    var_44 = iter(var_41)
    var_45 = iter(var_41)
    var_46 = list(var_44)
    var_47 = list(var_45)



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = '1'
    var_11 = 'Self Parent'
    var_12 = var_0.add(var_2, var_7, var_11)
    var_13 = '2'
    var_14 = '2000'
    var_15 = 'Some Liability'
    var_16 = '3'
    var_17 = '2000'
    var_18 = 'Different Name'
    var_19 = var_0.add(var_2, var_11, var_18)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'A'
    var_1 = 'L'
    var_2 = 'E'
    var_3 = 'R'
    var_4 = 'X'
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = '4'
    var_9 = '5'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #10
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
    var_7 = '9999'
    var_8 = 'Non-existent parent'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self parent'
    var_11 = '2000'
    var_12 = 'Account A'
    var_13 = '2'
    var_14 = '2000'
    var_15 = 'Different Parent'
    var_16 = var_0.add(var_2, var_4, var_15)
    var_17 = '2000'
    var_18 = 'Different Name'



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Non-existent Parent'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = module_0.COA()
    var_12 = 'First Child'
    var_13 = var_11.add(var_8, var_9, var_12)
    var_14 = '2'
    var_15 = '1000'
    var_16 = 'Different Parent'
    var_17 = var_11.add(var_2, var_4, var_16)
    var_18 = module_0.COA()
    var_19 = 'Original Name'
    var_20 = '1'
    var_21 = '1000'
    var_22 = 'Different Name'
    var_23 = var_18.add(var_2, var_4, var_22)
    var_24 = module_0.COA()
    var_25 = '4'
    var_26 = '4000'
    var_27 = 'Service Revenue'
    var_28 = '5'
    var_29 = '5000'
    var_30 = 'Office Supplies'



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = '1'
    var_11 = 'Self Parent'
    var_12 = var_0.add(var_2, var_7, var_11)
    var_13 = '2'
    var_14 = '2000'
    var_15 = 'Current Liabilities'
    var_16 = '3'
    var_17 = '2000'
    var_18 = 'Different Name'
    var_19 = var_0.add(var_2, var_11, var_18)
    var_20 = '4'
    var_21 = '4000'
    var_22 = 'Sales Revenue'
    var_23 = '5'
    var_24 = '5000'
    var_25 = 'Operating Expenses'
    var_26 = [code for (code, _) in var_0]
    var_27 = var_0.structure
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 5
    var_30 = 0



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Non-existent Parent'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = '2'
    var_12 = 'Different Parent'
    var_13 = 'Different Name'
    var_14 = var_0.subaccounts(var_8)
    var_15 = [code for (code, _) in var_0]



# Parsed testcases at query #14
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
    var_11 = '1'
    var_12 = 'Self Parent'
    var_13 = var_0.add(var_2, var_9, var_12)
    var_14 = '2'
    var_15 = 'Different Parent'
    var_16 = 'Different Name'
    var_17 = list(var_0)



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
    var_6 = '9999'
    var_7 = '9998'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = 'Different Name'



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = module_0.COA()
    var_12 = 'First Account'
    var_13 = var_11.add(var_8, var_9, var_12)
    var_14 = '2'
    var_15 = '1000'
    var_16 = 'Different Parent Account'
    var_17 = var_11.add(var_2, var_4, var_16)
    var_18 = '1'
    var_19 = '1000'
    var_20 = 'Different Name'
    var_21 = var_11.add(var_2, var_4, var_20)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



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
    var_6 = 0
    var_7 = '100101'
    var_8 = 'Savings Account'
    var_9 = '100102'
    var_10 = 'Checking Account'
    var_11 = '4'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '4000'



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    assert var_9 == 1
    var_10 = 'Self Parent'
    var_11 = module_0.COA()
    var_12 = '2'
    var_13 = 'Different Name'
    var_14 = list(var_0)
    var_15 = [code for (code, _) in var_14]



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
    var_6 = '2'
    var_7 = 'Different Name'
    var_8 = '9999'
    var_9 = 'Self Parent'
    var_10 = '9998'
    var_11 = 'New Account'



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Non-existent Parent Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = module_0.COA()
    var_11 = 'Liquidity'
    var_12 = '2'
    var_13 = '1000'
    var_14 = 'Liquidity'
    var_15 = var_10.add(var_2, var_4, var_14)
    var_16 = module_0.COA()
    var_17 = '1'
    var_18 = '1000'
    var_19 = 'Different Name'
    var_20 = var_16.add(var_2, var_4, var_19)
    var_21 = '1'
    var_22 = 'Self Parent'
    var_23 = var_0.add(var_2, var_18, var_22)



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = 'Different Name'
    var_12 = '2'
    var_13 = '2000'
    var_14 = 'Current Liabilities'
    var_15 = '3'
    var_16 = '3000'
    var_17 = 'Retained Earnings'
    var_18 = '4'
    var_19 = '4000'
    var_20 = 'Sales Revenue'
    var_21 = '5'
    var_22 = '5000'
    var_23 = 'Operating Expenses'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'



# Parsed testcases at query #25
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.structure
    var_3 = list(var_2)
    var_4 = module_0.COA()
    var_5 = var_4.toplevel
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 5
    var_8 = module_0.COA()
    var_9 = '1'
    var_10 = '1000'
    var_11 = 'Liquidity'
    var_12 = '1001'
    var_13 = 'Bank Account'
    var_14 = 0
    var_15 = module_0.COA()
    var_16 = '10'
    var_17 = 'A1'
    var_18 = '101'
    var_19 = 'A2'
    var_20 = '1011'
    var_21 = 'A3'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #27
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
    var_9 = 'Self Parent'
    var_10 = '999'
    var_11 = '9999'
    var_12 = 'Invalid Parent'
    var_13 = [code for (code, _) in var_0]



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = '1000'



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
    var_6 = '999'
    var_7 = '9999'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = '2000'
    var_12 = 'First Name'
    var_13 = '2'
    var_14 = '2000'
    var_15 = 'Different Name'
    var_16 = var_0.add(var_2, var_4, var_15)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = [var_1, var_2, var_3, var_4, var_5]



# Parsed testcases at query #32
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
    var_13 = '2'
    var_14 = '2000'
    var_15 = 'Long Term Liabilities'
    var_16 = '3'
    var_17 = '3000'
    var_18 = 'Retained Earnings'
    var_19 = '4'
    var_20 = '4000'
    var_21 = 'Sales Revenue'
    var_22 = '5'
    var_23 = '5000'
    var_24 = 'Operating Expenses'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '1001'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1001'
    var_2 = '1'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #2
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = [code for (code, _) in var_1]
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'
    var_9 = '1000'
    var_10 = 'Liquidity'
    var_11 = '1001'
    var_12 = 'Bank Account'
    var_13 = list(var_0)
    var_14 = len(var_13)
    assert var_14 == 7
    var_15 = [code for (code, _) in var_13]
    var_16 = module_0.COA()
    var_17 = '1100'
    var_18 = 'First Added'
    var_19 = '1200'
    var_20 = 'Second Added'
    var_21 = list(var_16)
    var_22 = len(var_21)
    assert var_22 == 7
    var_23 = -2
    var_24 = var_21[var_23:]
    var_25 = [code for (code, _) in var_24]



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
    var_6 = module_0.COA()
    var_7 = '999'
    var_8 = '1000'
    var_9 = 'Test Account'
    var_10 = var_6.add(var_2, var_4, var_9)
    var_11 = 'Self Parent'
    var_12 = module_0.COA()
    var_13 = 'First Account'
    var_14 = var_12.add(var_9, var_10, var_13)
    var_15 = '2'
    var_16 = '1000'
    var_17 = 'Different Account'
    var_18 = var_12.add(var_2, var_4, var_17)
    var_19 = module_0.COA()
    var_20 = 'Original Name'
    var_21 = '1'
    var_22 = '1000'
    var_23 = 'Different Name'
    var_24 = var_19.add(var_2, var_4, var_23)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = {var_0, var_1, var_2, var_3, var_4}
    var_6 = 'A'
    var_7 = 'L'



# Parsed testcases at query #5
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
    var_6 = '2'
    var_7 = 'Different Name'
    var_8 = 'Different Name'
    var_9 = 'Self Parent'
    var_10 = '9999'
    var_11 = '9998'
    var_12 = 'Invalid Parent'
    var_13 = var_0.add(var_2, var_4, var_12)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '1001'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #10
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
    var_7 = '9999'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = '1002'
    var_12 = 'Account A'
    var_13 = '2'
    var_14 = '1002'
    var_15 = 'Different Account'
    var_16 = var_0.add(var_2, var_4, var_15)



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
    var_6 = '2'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '1000'



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
    var_6 = '9999'
    var_7 = '9998'
    var_8 = 'Invalid Account'
    var_9 = var_0.add(var_2, var_4, var_8)
    var_10 = 'Self Parent'
    var_11 = '2'
    var_12 = 'Different Name'
    var_13 = 'Different Name'
    var_14 = '4'
    var_15 = '4000'
    var_16 = 'Sales Revenue'
    var_17 = var_0._accounts
    var_18 = len(var_17)
    assert var_18 == 7



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '__call__'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '101'
    var_1 = '201'
    var_2 = '1'



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1001'
    var_2 = '1'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #23
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'A'
    assert var_0 == 5
    var_1 = 'L'
    var_2 = 'E'
    var_3 = 'R'
    var_4 = 'X'
    var_5 = '1000'
    var_6 = '1001'
    var_7 = module_0.COA()
    var_8 = lambda : var_7



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'A'
    var_1 = 'L'
    var_2 = 'E'
    var_3 = 'R'
    var_4 = 'X'
    var_5 = '1001'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = '101'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'A'
    var_1 = 'L'
    var_2 = 'E'
    var_3 = 'R'
    var_4 = 'X'
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = '4'
    var_9 = '5'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'A'
    var_1 = 'L'
    var_2 = 'E'
    var_3 = 'R'
    var_4 = 'X'
    var_5 = [var_0, var_1, var_2, var_3, var_4]



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #33
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'A'
    var_6 = 'L'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '1000'
    var_2 = '1001'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = '1000'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'A'
    var_1 = 'L'
    var_2 = 'E'
    var_3 = 'R'
    var_4 = 'X'
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = '4'
    var_9 = '5'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = 'A'
    var_6 = 'Custom Assets'
    var_7 = 'L'
    var_8 = 'Custom Liabilities'
    var_9 = 'E'
    var_10 = 'Custom Equities'
    var_11 = 'R'
    var_12 = 'Custom Revenues'
    var_13 = 'X'
    var_14 = 'Custom Expenses'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = '1000'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = '4'
    var_4 = '5'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'Assets'
    var_7 = 'Liabilities'
    var_8 = 'Equities'
    var_9 = 'Revenues'
    var_10 = 'Expenses'
    var_11 = [var_6, var_7, var_8, var_9, var_10]



