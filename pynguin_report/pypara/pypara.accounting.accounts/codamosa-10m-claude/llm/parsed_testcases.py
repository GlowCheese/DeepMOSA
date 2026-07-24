####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Current Liabilities'
    var_10 = '1000'
    var_11 = 'Same Code'
    var_12 = '9999'
    var_13 = '9998'
    var_14 = 'Non-existent Parent'
    var_15 = var_1.add(var_2, var_11, var_14)
    var_16 = '1'
    var_17 = '1000'
    var_18 = 'Different Name'
    var_19 = var_1.add(var_2, var_11, var_18)
    var_20 = '2'
    var_21 = '1000'
    var_22 = 'Liquidity'
    var_23 = var_1.add(var_2, var_11, var_22)
    var_24 = '1002'
    var_25 = 'Savings Account'



# Parsed testcases at query #2
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Payables'
    var_10 = '1'
    var_11 = 'Invalid'
    var_12 = '9999'
    var_13 = 'Invalid'
    var_14 = '1'
    var_15 = '1000'
    var_16 = 'Different Name'
    var_17 = var_1.add(var_2, var_13, var_16)
    var_18 = '2'
    var_19 = '1000'
    var_20 = 'Liquidity'
    var_21 = var_1.add(var_2, var_13, var_20)
    var_22 = '1002'
    var_23 = 'Cash'
    var_24 = '1003'
    var_25 = 'Receivables'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '100'
    var_2 = 'Custom Assets'
    var_3 = '200'
    var_4 = 'Custom Liabilities'
    var_5 = '300'
    var_6 = 'Custom Equities'
    var_7 = '400'
    var_8 = 'Custom Revenues'
    var_9 = '500'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #5
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the __iter__ method of COA class.'
    var_1 = module_0.COA()
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = 'code'
    var_5 = 'name'
    var_6 = 'type'
    var_7 = [code for (code, _) in var_2]
    var_8 = [account.name for (_, account) in var_2]
    var_9 = [account.type for (_, account) in var_2]

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test __iter__ includes both root and sub-accounts.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = list(var_1)
    var_8 = len(var_7)
    assert var_8 == 7
    var_9 = [code for (code, _) in var_7]

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test that iteration can be called multiple times.'
    var_1 = module_0.COA()
    var_2 = list(var_1)
    var_3 = list(var_1)
    var_4 = len(var_2)
    var_5 = len(var_3)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test that __iter__ returns an iterator.'
    var_1 = module_0.COA()
    var_2 = iter(var_1)
    var_3 = '1'
    var_4 = '2'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol __call__ method.'
    var_1 = '1'
    var_2 = '1000'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom implementation.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts.__call__ with custom root specification.'
    var_1 = '10'
    var_2 = 'Assets Custom'
    var_3 = '20'
    var_4 = 'Liabilities Custom'
    var_5 = '30'
    var_6 = 'Equities Custom'
    var_7 = '40'
    var_8 = 'Revenues Custom'
    var_9 = '50'
    var_10 = 'Expenses Custom'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ can be invoked multiple times.'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '100'
    var_2 = 'Custom Assets'
    var_3 = '200'
    var_4 = 'Custom Liabilities'
    var_5 = '300'
    var_6 = 'Custom Equities'
    var_7 = '400'
    var_8 = 'Custom Revenues'
    var_9 = '500'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom implementation.'
    var_1 = '10'
    var_2 = 'My Assets'
    var_3 = '20'
    var_4 = 'My Liabilities'
    var_5 = '30'
    var_6 = 'My Equities'
    var_7 = '40'
    var_8 = 'My Revenues'
    var_9 = '50'
    var_10 = 'My Expenses'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol is properly defined and callable.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1100'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that modifies the COA.'
    var_1 = '1000'
    var_2 = '1'
    var_3 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can return a customized COA.'
    var_1 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = '100'
    var_2 = '200'
    var_3 = '300'
    var_4 = '400'
    var_5 = '500'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Liquidity'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a simple COA returning function.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a complex COA structure.'
    var_1 = '1000'
    var_2 = '1001'
    var_3 = '2000'

def test_case_0():
    var_0 = 'Test that any callable returning COA satisfies ReadChartOfAccounts protocol.'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times with consistent results.'
    var_1 = '1'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = 'D'
    var_5 = 'E'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test the __call__ method of ReadChartOfAccounts protocol.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom COA configuration.'
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'
    var_5 = 'E'
    var_6 = 'Custom Equities'
    var_7 = 'R'
    var_8 = 'Custom Revenues'
    var_9 = 'X'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'
    var_1 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts is callable.'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts with a custom implementation that returns a customized COA.\n    '
    var_1 = '1000'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts __call__ can be invoked multiple times.'
    var_1 = 0
    assert var_1 == 2

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts __call__ returns COA with default root accounts.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom COA configuration.'
    var_1 = '10'
    var_2 = '20'
    var_3 = '30'
    var_4 = '40'
    var_5 = '50'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA initialization.'
    var_1 = '10'
    var_2 = '20'
    var_3 = '30'
    var_4 = '40'
    var_5 = '50'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    var_2 = [var_1]



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts returning COA with custom rootspec.'
    var_1 = '10'
    var_2 = 'My Assets'
    var_3 = '20'
    var_4 = 'My Liabilities'
    var_5 = '30'
    var_6 = 'My Equities'
    var_7 = '40'
    var_8 = 'My Revenues'
    var_9 = '50'
    var_10 = 'My Expenses'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that returns a customized COA.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1'
    var_2 = '5'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ can return a custom COA.'
    var_1 = '1000'
    var_2 = '1001'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'
    var_11 = '1'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Test Account'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that returns modified COA.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'L'
    var_4 = 'My Liabilities'
    var_5 = 'E'
    var_6 = 'My Equities'
    var_7 = 'R'
    var_8 = 'My Revenues'
    var_9 = 'X'
    var_10 = 'My Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 1
    assert var_1 == 2



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom implementation.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts callable can be invoked multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts returns a COA with root accounts.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts protocol with lambda implementation.\n    '
    var_1 = module_0.COA()
    var_2 = lambda : var_1
    var_3 = '1'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times independently.\n    '
    var_1 = '1000'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = 'A'



# Parsed testcases at query #40
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol __call__ method.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'
    var_7 = module_0.COA()
    var_8 = lambda : var_7



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts protocol with custom rootspec.\n    '
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'
    var_5 = 'E'
    var_6 = 'Custom Equities'
    var_7 = 'R'
    var_8 = 'Custom Revenues'
    var_9 = 'X'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times independently.\n    '
    var_1 = '1'
    var_2 = '5'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol is correctly implemented.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #43
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '1100'
    var_8 = 'Receivables'
    var_9 = '2000'
    var_10 = 'Invalid Account'
    var_11 = '9999'
    var_12 = '3000'
    var_13 = 'Invalid Parent'
    var_14 = var_1.add(var_2, var_10, var_13)
    var_15 = '2'
    var_16 = '1000'
    var_17 = 'Liquidity'
    var_18 = var_1.add(var_2, var_10, var_17)
    var_19 = '1'
    var_20 = '1000'
    var_21 = 'Different Name'
    var_22 = var_1.add(var_2, var_10, var_21)
    var_23 = '2'
    var_24 = '2000'
    var_25 = 'Long-term Debt'
    var_26 = '3'
    var_27 = '3000'
    var_28 = 'Capital'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test the __call__ method of ReadChartOfAccounts protocol.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'

def test_case_0():
    var_0 = 'Test __call__ method returning an empty COA with only root accounts.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test __call__ method with a complex account structure.'
    var_1 = '1001'
    var_2 = '1000'
    var_3 = '1002'
    var_4 = '5001'
    var_5 = '5000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts is callable.'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 2

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return COA with custom root specification.\n    '
    var_1 = '100'
    var_2 = 'Current Assets'
    var_3 = '200'
    var_4 = 'Current Liabilities'
    var_5 = '300'
    var_6 = 'Owner Equity'
    var_7 = '400'
    var_8 = 'Operating Revenues'
    var_9 = '500'
    var_10 = 'Operating Expenses'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation.'
    var_1 = '100'
    var_2 = 'Custom Assets'
    var_3 = '200'
    var_4 = 'Custom Liabilities'
    var_5 = '300'
    var_6 = 'Custom Equities'
    var_7 = '400'
    var_8 = 'Custom Revenues'
    var_9 = '500'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can return custom COA instances.\n    '
    var_1 = '1100'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'Test the __call__ method of ReadChartOfAccounts protocol.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol is properly defined.'
    var_1 = '1'
    var_2 = '2'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a more complex COA structure.'
    var_1 = '1001'
    var_2 = '4000'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA implementation.'
    var_1 = '1000'
    var_2 = '1'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1'



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol works with custom COA implementations.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts protocol with a custom implementation.\n    '
    var_1 = 'A'
    var_2 = 'L'
    var_3 = 'E'
    var_4 = 'R'
    var_5 = 'X'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that calling ReadChartOfAccounts returns a valid COA object.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom account setup.'
    var_1 = '1000'
    var_2 = '1001'
    var_3 = '1100'



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can return COA with custom rootspec.'
    var_1 = '10'
    var_2 = 'My Assets'
    var_3 = '20'
    var_4 = 'My Liabilities'
    var_5 = '30'
    var_6 = 'My Equities'
    var_7 = '40'
    var_8 = 'My Revenues'
    var_9 = '50'
    var_10 = 'My Expenses'



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = '1'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be implemented with custom root specifications.\n    '
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol works with custom implementations.\n    '
    var_1 = 'A'
    var_2 = 'L'
    var_3 = 'E'
    var_4 = 'R'
    var_5 = 'X'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called multiple times.\n    '
    var_1 = '1'



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    var_2 = [var_1]

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts returns different COA instances on each call.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a more complex COA.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = '1001'
    var_4 = '1002'
    var_5 = '1100'
    var_6 = '1101'



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts returns a valid COA with default accounts.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts protocol with a custom implementation that returns a customized COA.\n    '
    var_1 = '1000'

def test_case_0():
    var_0 = '\n    Test that any callable returning COA complies with ReadChartOfAccounts protocol.\n    '



# Parsed testcases at query #60
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Accounts Payable'
    var_10 = '1'
    var_11 = 'Invalid'
    var_12 = '9999'
    var_13 = '1002'
    var_14 = 'Invalid Parent'
    var_15 = var_1.add(var_2, var_11, var_14)
    var_16 = '1000'
    var_17 = '1001'
    var_18 = 'Different Name'
    var_19 = var_1.add(var_2, var_11, var_18)
    var_20 = '1002'
    var_21 = 'Savings Account'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '

def test_case_0():
    var_0 = '\n    Test that functions implementing ReadChartOfAccounts protocol are recognized.\n    '
    var_1 = '1000'

def test_case_0():
    var_0 = '\n    Test that multiple different implementations can satisfy the protocol.\n    '
    var_1 = '1'
    var_2 = '10'



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times and returns new instances.\n    '

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return COA with custom root specifications.\n    '
    var_1 = '100'
    var_2 = 'Custom Assets'
    var_3 = '200'
    var_4 = 'Custom Liabilities'
    var_5 = '300'
    var_6 = 'Custom Equities'
    var_7 = '400'
    var_8 = 'Custom Revenues'
    var_9 = '500'
    var_10 = 'Custom Expenses'



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = '100'
    var_2 = '200'
    var_3 = '300'
    var_4 = '400'
    var_5 = '500'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1'



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts returns a valid COA with root accounts.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'B'
    var_4 = 'My Liabilities'
    var_5 = 'C'
    var_6 = 'My Equities'
    var_7 = 'D'
    var_8 = 'My Revenues'
    var_9 = 'E'
    var_10 = 'My Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that modifies the COA.'
    var_1 = '1000'
    var_2 = '1'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation returning pre-configured COA.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'
    var_1 = '1'
    var_2 = '5'



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that modifies COA.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that adds accounts.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol is properly implemented.'



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = '1000'
    var_7 = 'Test Account'



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be checked at runtime.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol works with different implementations.'
    var_1 = '10'
    var_2 = '20'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #74
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '1002'
    var_8 = 'Savings'
    var_9 = '2'
    var_10 = '2000'
    var_11 = 'Accounts Payable'
    var_12 = '3'
    var_13 = '3000'
    var_14 = 'Retained Earnings'
    var_15 = '1'
    var_16 = 'Invalid'
    var_17 = '9999'
    var_18 = '9998'
    var_19 = 'Invalid'
    var_20 = var_1.add(var_2, var_16, var_19)
    var_21 = '1'
    var_22 = '1000'
    var_23 = 'Different Name'
    var_24 = var_1.add(var_2, var_16, var_23)



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom root specification.'
    var_1 = 'A'
    var_2 = 'L'
    var_3 = 'E'
    var_4 = 'R'
    var_5 = 'X'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specifications.'
    var_1 = '10'
    var_2 = '20'
    var_3 = '30'
    var_4 = '40'
    var_5 = '50'



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ works with custom rootspec.'
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be used as type hint.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = '1000'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = 0
    var_2 = 1
    var_3 = var_1 + var_2
    assert var_3 == 5
    var_4 = 'name'
    var_5 = 'code'
    var_6 = 'type'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts protocol with a custom implementation that adds accounts.\n    '
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = '\n    Test that any callable returning COA satisfies ReadChartOfAccounts protocol.\n    '



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test the __call__ method of ReadChartOfAccounts protocol.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #3
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test COA.add method for adding accounts to chart of accounts.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Long-term Debt'
    var_10 = '9999'
    var_11 = '9000'
    var_12 = 'Invalid Parent'
    var_13 = var_1.add(var_2, var_3, var_12)
    var_14 = '1000'
    var_15 = 'Self Parent'
    var_16 = var_1.add(var_2, var_11, var_15)
    var_17 = '1'
    var_18 = '1000'
    var_19 = 'Different Name'
    var_20 = var_1.add(var_2, var_15, var_19)
    var_21 = '2'
    var_22 = '1000'
    var_23 = 'Liquidity'
    var_24 = var_1.add(var_2, var_15, var_23)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts.__call__ returns a COA instance.\n    '

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts.__call__ works with custom rootspec.\n    '
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol is satisfied by callable implementations.\n    '
    var_1 = '1000'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts.__call__ can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that modifies the COA.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom COA configuration.'
    var_1 = '100'
    var_2 = '200'
    var_3 = '300'
    var_4 = '400'
    var_5 = '500'
    var_6 = '1'
    var_7 = '2'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom COA configuration.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times independently.\n    '

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return COA with custom root specifications.\n    '
    var_1 = '10'
    var_2 = '20'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'



# Parsed testcases at query #10
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '1002'
    var_8 = 'Savings'
    var_9 = '1000'
    var_10 = 'Self Parent'
    var_11 = '9999'
    var_12 = '9998'
    var_13 = 'Non-existent Parent'
    var_14 = var_1.add(var_2, var_10, var_13)
    var_15 = '1'
    var_16 = '1000'
    var_17 = 'Different Name'
    var_18 = var_1.add(var_2, var_10, var_17)
    var_19 = '2'
    var_20 = '1000'
    var_21 = 'Liquidity'
    var_22 = var_1.add(var_2, var_10, var_21)
    var_23 = '2'
    var_24 = '2000'
    var_25 = 'Debt'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test the __call__ method of ReadChartOfAccounts protocol.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with default COA initialization.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '100'
    var_2 = '200'
    var_3 = '300'
    var_4 = '400'
    var_5 = '500'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1100'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts with a custom COA configuration.\n    '
    var_1 = '100'
    var_2 = 'Assets Custom'
    var_3 = '200'
    var_4 = 'Liabilities Custom'
    var_5 = '300'
    var_6 = 'Equities Custom'
    var_7 = '400'
    var_8 = 'Revenues Custom'
    var_9 = '500'
    var_10 = 'Expenses Custom'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times and returns independent instances.\n    '
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = '10'
    var_2 = 'Total Assets'
    var_3 = '20'
    var_4 = 'Total Liabilities'
    var_5 = '30'
    var_6 = 'Total Equities'
    var_7 = '40'
    var_8 = 'Total Revenues'
    var_9 = '50'
    var_10 = 'Total Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts callable returns a COA instance.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts callable can return COA with custom rootspec.'
    var_1 = '10'
    var_2 = '20'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts callable can be called multiple times.'
    var_1 = 0
    assert var_1 == 2

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts callable can return COA with added accounts.'
    var_1 = '1000'
    var_2 = '1001'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol is properly defined and callable.'
    var_1 = []



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root account specifications.'
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'L'
    var_4 = 'My Liabilities'
    var_5 = 'E'
    var_6 = 'My Equities'
    var_7 = 'R'
    var_8 = 'My Revenues'
    var_9 = 'X'
    var_10 = 'My Expenses'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts.__call__ with custom rootspec.'
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'
    var_5 = 'E'
    var_6 = 'Custom Equities'
    var_7 = 'R'
    var_8 = 'Custom Revenues'
    var_9 = 'X'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = 'A'
    var_2 = 'All Assets'
    var_3 = 'L'
    var_4 = 'All Liabilities'
    var_5 = 'E'
    var_6 = 'All Equities'
    var_7 = 'R'
    var_8 = 'All Revenues'
    var_9 = 'X'
    var_10 = 'All Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that adds sub-accounts.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom COA configuration.'
    var_1 = '100'
    var_2 = '200'
    var_3 = '300'
    var_4 = '400'
    var_5 = '500'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called multiple times.\n    '

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol works with custom rootspec.\n    '
    var_1 = '10'
    var_2 = '20'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can return custom COA instances.\n    '
    var_1 = '1000'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 1
    assert var_1 == 2



# Parsed testcases at query #24
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Long-term Debt'
    var_10 = '1'
    var_11 = 'Invalid'
    var_12 = '9999'
    var_13 = '9998'
    var_14 = 'Invalid'
    var_15 = var_1.add(var_2, var_11, var_14)
    var_16 = '1'
    var_17 = '1000'
    var_18 = 'Different Name'
    var_19 = var_1.add(var_2, var_11, var_18)
    var_20 = '1002'
    var_21 = 'Cash'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that different implementations can satisfy the ReadChartOfAccounts protocol.\n    '
    var_1 = '1'
    var_2 = '10'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = 'A'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test the __call__ method of ReadChartOfAccounts protocol.'
    var_1 = '1'
    var_2 = '1000'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol works with runtime checking.'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'L'
    var_4 = 'My Liabilities'
    var_5 = 'E'
    var_6 = 'My Equities'
    var_7 = 'R'
    var_8 = 'My Revenues'
    var_9 = 'X'
    var_10 = 'My Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = 'A'
    var_7 = 'L'
    var_8 = 'E'
    var_9 = 'R'
    var_10 = 'X'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = '10'
    var_7 = '20'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '100'
    var_2 = '200'
    var_3 = '300'
    var_4 = '400'
    var_5 = '500'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that returns a modified COA.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return a custom configured COA.\n    '
    var_1 = '1000'
    var_2 = '1'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol is correctly implemented.\n    '



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can return an empty COA with only root accounts.'
    var_1 = '1'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ can return custom COA instances.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom COA initialization.'
    var_1 = '10'
    var_2 = '20'
    var_3 = '30'
    var_4 = '40'
    var_5 = '50'



# Parsed testcases at query #40
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Long-term Debt'
    var_10 = '9999'
    var_11 = '9001'
    var_12 = 'Invalid'
    var_13 = var_1.add(var_2, var_3, var_12)
    var_14 = '1000'
    var_15 = 'Self Parent'
    var_16 = var_1.add(var_2, var_11, var_15)
    var_17 = '1'
    var_18 = '1000'
    var_19 = 'Different Name'
    var_20 = var_1.add(var_2, var_15, var_19)
    var_21 = '1002'
    var_22 = 'Savings Account'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that modifies COA.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 1
    assert var_1 == 2



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts implementation with custom rootspec.'
    var_1 = '100'
    var_2 = '200'
    var_3 = '300'
    var_4 = '400'
    var_5 = '500'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol is properly defined and usable.'

def test_case_0():
    var_0 = 'Test that a ReadChartOfAccounts implementation returns a properly populated COA.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = '1001'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts is callable and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom rootspec.'
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'
    var_5 = 'E'
    var_6 = 'Custom Equities'
    var_7 = 'R'
    var_8 = 'Custom Revenues'
    var_9 = 'X'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '10'
    var_2 = 'Total Assets'
    var_3 = '20'
    var_4 = 'Total Liabilities'
    var_5 = '30'
    var_6 = 'Total Equities'
    var_7 = '40'
    var_8 = 'Total Revenues'
    var_9 = '50'
    var_10 = 'Total Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol is properly defined.'
    var_1 = '__call__'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1000'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that modifies COA.'
    var_1 = '1000'
    var_2 = '1'

def test_case_0():
    var_0 = 'Test that an object can be used as ReadChartOfAccounts if it has __call__ returning COA.'
    var_1 = '1'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts with a custom implementation.\n    '
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'L'
    var_4 = 'My Liabilities'
    var_5 = 'E'
    var_6 = 'My Equities'
    var_7 = 'R'
    var_8 = 'My Revenues'
    var_9 = 'X'
    var_10 = 'My Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = 'A'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts is callable and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom COA initialization.'
    var_1 = '10'
    var_2 = 'Current Assets'
    var_3 = '20'
    var_4 = 'Current Liabilities'
    var_5 = '30'
    var_6 = 'Shareholders Equity'
    var_7 = '40'
    var_8 = 'Operating Revenues'
    var_9 = '50'
    var_10 = 'Operating Expenses'

def test_case_0():
    var_0 = 'Test that a function complies with ReadChartOfAccounts protocol.'
    var_1 = '1001'
    var_2 = '1'



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts.__call__ with a custom COA.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Custom Account'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that a function implementing ReadChartOfAccounts protocol works correctly.\n    '
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.\n    '
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Test Account'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation that modifies the COA.'
    var_1 = '1000'
    var_2 = '1'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Test Account'



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts protocol with custom COA initialization.\n    '
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'
    var_5 = 'E'
    var_6 = 'Custom Equities'
    var_7 = 'R'
    var_8 = 'Custom Revenues'
    var_9 = 'X'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times and returns independent instances.\n    '



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol works with custom implementations.\n    '
    var_1 = '1000'
    var_2 = '1'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times independently.\n    '



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom initialization.'
    var_1 = 'A'
    var_2 = 'L'



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom COA implementation.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that any callable returning COA satisfies ReadChartOfAccounts protocol.'



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can return custom COA configurations.\n    '
    var_1 = '1100'
    var_2 = '1101'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = '1'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return a COA with custom rootspec.\n    '
    var_1 = '10'
    var_2 = 'Current Assets'
    var_3 = '20'
    var_4 = 'Current Liabilities'
    var_5 = '30'
    var_6 = 'Owner Equity'
    var_7 = '40'
    var_8 = 'Operating Revenues'
    var_9 = '50'
    var_10 = 'Operating Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.\n    '
    var_1 = 0
    assert var_1 == 2
    var_2 = '1'



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'Test the __call__ method of ReadChartOfAccounts protocol.'
    var_1 = '1'
    var_2 = '1000'
    var_3 = '2'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'
    var_7 = '1100'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom implementation.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return customized COA instances.\n    '
    var_1 = '1000'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 1
    assert var_1 == 2



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol enforces __call__ method.\n    '
    var_1 = '10'
    var_2 = '20'
    var_3 = '30'
    var_4 = '40'
    var_5 = '50'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times independently.\n    '
    var_1 = '1'
    var_2 = '1000'
    var_3 = 'Test Account'



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol works with custom COA configurations.\n    '
    var_1 = '1100'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times independently.\n    '
    var_1 = '1'
    var_2 = '1200'
    var_3 = 'Savings'



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be invoked multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test ReadChartOfAccounts protocol with custom COA configuration.\n    '
    var_1 = '10'
    var_2 = '20'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times independently.\n    '
    var_1 = '1000'



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts callable returns a valid COA.'
    var_1 = '1'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom initialized COA.'
    var_1 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #68
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Long-term Debt'
    var_10 = '3'
    var_11 = '3000'
    var_12 = 'Retained Earnings'
    var_13 = '1'
    var_14 = 'Invalid'
    var_15 = '9999'
    var_16 = '9998'
    var_17 = 'Invalid'
    var_18 = var_1.add(var_2, var_14, var_17)
    var_19 = '1'
    var_20 = '1000'
    var_21 = 'Different Name'
    var_22 = var_1.add(var_2, var_14, var_21)
    var_23 = [code for (code, _) in var_1]



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom COA initialization.'
    var_1 = '10'
    var_2 = '20'
    var_3 = '30'
    var_4 = '40'
    var_5 = '50'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'
    var_1 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.\n    '
    var_1 = '1'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return COA with custom rootspec.\n    '
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can return an empty COA with only root accounts.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times.\n    '
    var_1 = '1'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can return COA with custom rootspec.\n    '
    var_1 = '10'
    var_2 = 'Custom Assets'
    var_3 = '20'
    var_4 = 'Custom Liabilities'
    var_5 = '30'
    var_6 = 'Custom Equities'
    var_7 = '40'
    var_8 = 'Custom Revenues'
    var_9 = '50'
    var_10 = 'Custom Expenses'



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts is callable and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = '10'
    var_2 = 'Current Assets'
    var_3 = '20'
    var_4 = 'Current Liabilities'
    var_5 = '30'
    var_6 = 'Owner Equity'
    var_7 = '40'
    var_8 = 'Operating Revenues'
    var_9 = '50'
    var_10 = 'Operating Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts.__call__ returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts.__call__ with custom rootspec.'
    var_1 = 'A'
    var_2 = 'Custom Assets'
    var_3 = 'L'
    var_4 = 'Custom Liabilities'
    var_5 = 'E'
    var_6 = 'Custom Equities'
    var_7 = 'R'
    var_8 = 'Custom Revenues'
    var_9 = 'X'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts.__call__ that creates and returns COA with subaccounts.'
    var_1 = '1001'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol works with custom rootspec.\n    '
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'L'
    var_4 = 'My Liabilities'
    var_5 = 'E'
    var_6 = 'My Equities'
    var_7 = 'R'
    var_8 = 'My Revenues'
    var_9 = 'X'
    var_10 = 'My Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called multiple times.\n    '
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with custom COA configuration.'
    var_1 = 'A'
    var_2 = 'L'
    var_3 = 'E'
    var_4 = 'R'
    var_5 = 'X'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 2



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = '1000'



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts protocol with a custom implementation.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3



# Parsed testcases at query #81
#--------------------------


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test the add method of COA class.'
    var_1 = module_0.COA()
    var_2 = '1'
    var_3 = '1000'
    var_4 = 'Liquidity'
    var_5 = '1001'
    var_6 = 'Bank Account'
    var_7 = '2'
    var_8 = '2000'
    var_9 = 'Current Liabilities'
    var_10 = '3000'
    var_11 = 'Invalid'
    var_12 = '9999'
    var_13 = '9000'
    var_14 = 'Non-existent Parent'
    var_15 = var_1.add(var_2, var_11, var_14)
    var_16 = '2'
    var_17 = '1000'
    var_18 = 'Liquidity'
    var_19 = var_1.add(var_2, var_11, var_18)
    var_20 = '1'
    var_21 = '1000'
    var_22 = 'Different Name'
    var_23 = var_1.add(var_2, var_11, var_22)
    var_24 = '4'
    var_25 = '4000'
    var_26 = 'Sales Revenue'



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol works with different implementations.'
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'L'
    var_4 = 'My Liabilities'
    var_5 = 'E'
    var_6 = 'My Equities'
    var_7 = 'R'
    var_8 = 'My Revenues'
    var_9 = 'X'
    var_10 = 'My Expenses'
    var_11 = '1000'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    var_2 = [var_1]
    var_3 = '1'
    var_4 = '1001'
    var_5 = 'Account 1'



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = 'A'
    var_2 = 'My Assets'
    var_3 = 'L'
    var_4 = 'My Liabilities'
    var_5 = 'E'
    var_6 = 'My Equities'
    var_7 = 'R'
    var_8 = 'My Revenues'
    var_9 = 'X'
    var_10 = 'My Expenses'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = '1'



# Parsed testcases at query #84
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom COA configuration.'
    var_1 = '1000'
    var_2 = '1001'

def test_case_0():
    var_0 = 'Test that implementations comply with ReadChartOfAccounts protocol.'



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can return customized COA instances.'
    var_1 = '1100'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with custom root specification.'
    var_1 = '100'
    var_2 = 'Custom Assets'
    var_3 = '200'
    var_4 = 'Custom Liabilities'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times independently.'



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns a COA instance.'
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts can be called multiple times.'
    var_1 = 0
    assert var_1 == 3

def test_case_0():
    var_0 = 'Test ReadChartOfAccounts with a custom rootspec.'
    var_1 = '10'
    var_2 = 'Current Assets'
    var_3 = '20'
    var_4 = 'Current Liabilities'
    var_5 = '30'
    var_6 = 'Owner Equity'
    var_7 = '40'
    var_8 = 'Sales Revenue'
    var_9 = '50'
    var_10 = 'Operating Expenses'



# Parsed testcases at query #88
#--------------------------


def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts protocol can return custom COA instances.\n    '
    var_1 = '100'
    var_2 = 'Custom Assets'
    var_3 = '200'
    var_4 = 'Custom Liabilities'
    var_5 = '300'
    var_6 = 'Custom Equities'
    var_7 = '400'
    var_8 = 'Custom Revenues'
    var_9 = '500'
    var_10 = 'Custom Expenses'

def test_case_0():
    var_0 = '\n    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.\n    '
    var_1 = '1'
    var_2 = '5'



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = '\n    Test the __call__ method of ReadChartOfAccounts protocol.\n    '
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'
    var_6 = '1000'



