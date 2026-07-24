####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 9/19 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 15/52 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Assets'
    var_2 = '2000'
    var_3 = 'Liabilities'
    var_4 = '3000'
    var_5 = 'Equity'
    var_6 = '4000'
    var_7 = 'Income'
    var_8 = '5000'
    var_9 = 'Expenses'
    var_10 = '1'
    var_11 = '2'
    var_12 = '3'
    var_13 = '4'
    var_14 = '5'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 3/7 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = 'Parent Account'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 7/32 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0._subaccounts
    var_4 = bool(var_0._subaccounts == {})
    assert var_4 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Assets'
    var_2 = '2'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_new_account. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account_with_same_details. Retrieved 4/9 statements.
# Partially parsed test_add_account_with_same_parent_and_code. Retrieved 4/9 statements.
# Partially parsed test_add_account_with_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_same_parent_and_code_but_different_name. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account 1'
    var_4 = 'Test Account 2'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_nodify_creates_node_with_account_and_empty_children. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_account_and_subaccounts. Retrieved 4/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 'Assets'
    var_3 = module_0.COA()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 9/19 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = '2000'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_add_with_nonexistent_parent. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '100'
    var_3 = 'Test Account'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_nodify_creates_node_with_correct_account. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_subaccounts. Retrieved 4/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #14
#--------------------------

# Failed to parse test_call_returns_coa_instance.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/23 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Current Assets'
    var_2 = '2000'
    var_3 = 'Current Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0._subaccounts
    var_4 = bool(var_0._subaccounts == {})
    assert var_4 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Assets'
    var_2 = '2'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '10000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_parent_account_not_defined. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '100'
    var_3 = 'New Account'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_successful_creation. Retrieved 4/8 statements.
# Partially parsed test_add_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_self_parent. Retrieved 3/6 statements.
# Partially parsed test_add_existing_account_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_existing_account_inconsistent. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account 1'
    var_4 = 'Test Account 2'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 9/21 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/42 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0.accounts
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'
    var_2 = '200'
    var_3 = 'Liabilities'
    var_4 = '300'
    var_5 = 'Equity'
    var_6 = '400'
    var_7 = 'Income'
    var_8 = '500'
    var_9 = 'Expenses'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0._subaccounts
    var_4 = bool(var_0._subaccounts == {})
    assert var_4 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/26 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Current Assets'
    var_2 = '2000'
    var_3 = 'Current Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '67890'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_add_successful_creation. Retrieved 4/8 statements.
# Partially parsed test_add_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_self_parent. Retrieved 3/6 statements.
# Partially parsed test_add_existing_account_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_existing_account_inconsistent. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_existing_account_with_inconsistent_parent. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'
    var_4 = '3'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_existing_account_with_inconsistent_information. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = 'Different Name'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_successful_creation. Retrieved 4/8 statements.
# Partially parsed test_add_parent_not_defined. Retrieved 4/8 statements.
# Partially parsed test_add_account_is_its_own_parent. Retrieved 3/7 statements.
# Partially parsed test_add_existing_account_consistent. Retrieved 4/9 statements.
# Partially parsed test_add_existing_account_inconsistent. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Sub-Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Test Sub-Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Sub-Account'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Sub-Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Sub-Account'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_existing_account_inconsistent_parent. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account 1'
    var_4 = '2'
    var_5 = '1.1'
    var_6 = 'Sub Account 1'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = '2'
    var_5 = '1.1'
    var_6 = 'Test Account'



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 6/28 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Cash'
    var_2 = '2'
    var_3 = 'Revenue'
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account 1'
    var_4 = '2'
    var_5 = '1.1'
    var_6 = 'Sub Account 1'



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa_instance.




# Parsed testcases at query #37
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = '789'
    var_5 = 'Test COA'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 6/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'
    var_4 = '3'
    var_5 = 'Different Parent'



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa_instance.




# Parsed testcases at query #42
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_existing_account_inconsistent_parent. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = '2'
    var_5 = '1.1'
    var_6 = 'Test Account'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #46
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '10000'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '54321'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_add_successful_creation. Retrieved 4/8 statements.
# Partially parsed test_add_existing_account_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_parent_not_defined. Retrieved 4/8 statements.
# Partially parsed test_add_inconsistent_existing_account. Retrieved 6/12 statements.
# Partially parsed test_add_self_parent. Retrieved 3/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = '2'
    var_5 = 'Different Name'
    var_6 = bool(False)
    assert var_6 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'COA001'
    var_1 = 'Test COA'
    var_2 = 'ACC001'
    var_3 = 'Parent Account'
    var_4 = 'SUB001'
    var_5 = 'Sub Account'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 3/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_add_new_account. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account_with_same_details. Retrieved 4/9 statements.
# Partially parsed test_add_account_with_invalid_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_self_as_parent. Retrieved 3/7 statements.
# Partially parsed test_add_account_with_inconsistent_details. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account 1'
    var_4 = 'Test Account 2'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_parent_instance_exists. Retrieved 4/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_parent_instance_exists. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/24 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 4/18 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    var_3 = bool(var_1)
    assert var_3 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = '3'



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_nodify_creates_node_with_correct_account. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_empty_children_list. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_subaccounts_as_children. Retrieved 4/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'PARENT'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = 'SUB'
    var_4 = 'Sub Account'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 3/5 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/31 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = bool(var_1)
    assert var_3 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equity'
    var_6 = '4'
    var_7 = 'Income'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '1001'
    var_1 = 'Cash'
    var_2 = '1000'
    var_3 = 'Assets'
    var_4 = 'GAAP'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_coa_constructor_initializes_root_accounts. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_default_rootspec. Retrieved 1/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = bool(var_1)
    assert var_4 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 9/19 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/42 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Assets'
    var_2 = '2000'
    var_3 = 'Custom Liabilities'
    var_4 = '3000'
    var_5 = 'Custom Equity'
    var_6 = '4000'
    var_7 = 'Custom Revenue'
    var_8 = '5000'
    var_9 = 'Custom Expenses'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Assets'
    var_2 = '2000'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/23 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '100'
    var_1 = 'Custom Asset'
    var_2 = '200'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/23 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/44 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 7/32 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Assets'
    var_2 = '2000'
    var_3 = 'Liabilities'
    var_4 = '3000'
    var_5 = 'Equities'
    var_6 = '4000'
    var_7 = 'Incomes'
    var_8 = '5000'
    var_9 = 'Expenses'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Assets'
    var_2 = '2000'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_with_nonexistent_parent. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '100'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_subaccount_constructor_with_valid_args. Retrieved 6/11 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = 'COA1'
    var_5 = 'Test COA'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/44 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0.accounts
    var_4 = var_0.toplevel
    var_5 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equity'
    var_6 = '4'
    var_7 = 'Income'
    var_8 = '5'
    var_9 = 'Expenses'

def test_case_0():
    var_0 = '10'
    var_1 = 'Current Assets'
    var_2 = '20'
    var_3 = 'Current Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/23 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 9/24 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Current Assets'
    var_2 = '2000'
    var_3 = 'Current Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1._accounts
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor_with_valid_args. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/46 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0._subaccounts
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_0.rootspec
    assert var_5 is None

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equity'
    var_6 = '4'
    var_7 = 'Income'
    var_8 = '5'
    var_9 = 'Expenses'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/44 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0.accounts
    var_4 = var_0.toplevel
    var_5 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equity'
    var_6 = '4'
    var_7 = 'Income'
    var_8 = '5'
    var_9 = 'Expenses'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Assets'
    var_2 = '20'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_add_new_account. Retrieved 4/9 statements.
# Partially parsed test_add_existing_account. Retrieved 4/10 statements.
# Partially parsed test_add_account_inconsistent_data. Retrieved 7/14 statements.
# Partially parsed test_add_account_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_self_parent. Retrieved 3/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = '2'
    var_5 = '1.1'
    var_6 = 'Different Account'
    var_7 = bool(False)
    assert var_7 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa_instance.




# Parsed testcases at query #36
#--------------------------

# Partially parsed test_parent_instance_not_none. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_nodify_creates_node_with_account_and_empty_children. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_account_and_subaccounts. Retrieved 6/19 statements.
# Partially parsed test_nodify_creates_nested_nodes. Retrieved 7/23 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '11'
    var_3 = 'Sub Account 1'
    var_4 = '12'
    var_5 = 'Sub Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '11'
    var_3 = 'Sub Account 1'
    var_4 = '111'
    var_5 = 'Sub Sub Account 1'
    var_6 = 0



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 9/13 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0.accounts
    var_4 = var_0.toplevel
    var_5 = None
    var_6 = var_0.structure
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 4/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Cash'
    var_2 = '2'
    var_3 = 'Bank'
    var_4 = []



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.
# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 6/21 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'PARENT'
    var_1 = 'Parent Account'
    var_2 = 'Main COA'
    var_3 = module_0.COA(var_2)
    var_4 = 'SUB'
    var_5 = 'Sub Account'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'COA001'
    var_1 = 'Test COA'
    var_2 = 'ACC001'
    var_3 = 'Parent Account'
    var_4 = 'SUB001'
    var_5 = 'Test SubAccount'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_nodify_creates_node_with_correct_account. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_empty_children_for_leaf_account. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_subaccounts. Retrieved 4/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #45
#--------------------------

# Partially parsed test_parent_instance_not_none. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/23 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Current Assets'
    var_2 = '2000'
    var_3 = 'Current Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 5/14 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/24 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    var_3 = var_0._accounts[var_2]
    var_4 = var_3.name

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test COA'
    var_1 = module_0.COA(var_0)
    var_2 = '1234'
    var_3 = 'Parent Account'
    var_4 = '5678'
    var_5 = 'Sub Account'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/24 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 6/26 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = str(var_1)

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = '2000'
    var_3 = 'Custom Liability'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = str(var_0)
    var_5 = var_1.find(var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 4/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 11/51 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '3'
    var_5 = 'Equity'
    var_6 = '4'
    var_7 = 'Income'
    var_8 = '5'
    var_9 = 'Expenses'
    var_10 = None



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = 'Parent Account'
    var_3 = 'Test COA'
    var_4 = module_0.COA(var_3)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = []



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_parent_instance_exists. Retrieved 4/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



