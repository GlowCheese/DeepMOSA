####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_1 = 'Custom Assets'
    var_2 = '2000'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_add_successful_creation. Retrieved 4/8 statements.
# Partially parsed test_add_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_same_parent_and_code. Retrieved 3/6 statements.
# Partially parsed test_add_existing_account_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_existing_account_inconsistent. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'First Name'
    var_4 = 'Second Name'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/20 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Assets'
    var_2 = '2'
    var_3 = 'Custom Liabilities'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 'Assets'
    var_3 = module_0.COA()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_nodify_creates_node_with_account_and_children. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_nested_children. Retrieved 4/11 statements.


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parent_instance_not_none. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 4/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Assets'
    var_2 = '2'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_add_new_account. Retrieved 4/9 statements.
# Partially parsed test_add_existing_account. Retrieved 4/12 statements.
# Partially parsed test_add_inconsistent_account. Retrieved 7/14 statements.
# Partially parsed test_add_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_self_parent. Retrieved 3/7 statements.


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
    var_4 = '1'
    var_5 = '1.1'
    var_6 = 'Different Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Test Account'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/18 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/42 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Assets'
    var_2 = '2000'
    var_3 = 'Custom Liabilities'
    var_4 = '3000'
    var_5 = 'Custom Equity'
    var_6 = '4000'
    var_7 = 'Custom Income'
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

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '67890'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 6/26 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'

def test_case_0():
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 4/9 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/22 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    var_3 = str(var_1)

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Assets'
    var_2 = module_0.COA()
    var_3 = '1010'
    var_4 = 'Cash'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_add_successful_creation. Retrieved 4/7 statements.
# Partially parsed test_add_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_same_parent_and_code. Retrieved 3/7 statements.
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
    var_2 = '999.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Account'

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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = '2'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'COA1'
    var_1 = 'Test COA'
    var_2 = module_0.COA()
    var_3 = 'PARENT'
    var_4 = 'Parent Account'
    var_5 = 'SUB'
    var_6 = 'Sub Account'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_add_new_account. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_invalid_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_same_parent_and_code. Retrieved 3/6 statements.
# Partially parsed test_add_account_with_inconsistent_data. Retrieved 5/10 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = 'Different Name'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Test SubAccount'
    var_2 = 'Parent Account'
    var_3 = module_0.COA()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 7/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = '2000'
    var_2 = 'Cash'
    var_3 = 'Accounts Receivable'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.COA()
    var_6 = lambda : var_5



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_existing_account_with_inconsistent_info. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account 1'
    var_4 = '1'
    var_5 = '1.1'
    var_6 = 'Different Name'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'
    var_4 = '3'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Test SubAccount'
    var_2 = 'Test Account'
    var_3 = 'COA1'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #38
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



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account 1'
    var_4 = '2'
    var_5 = '1.1'
    var_6 = 'Test Account 1'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #43
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
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Account'

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '123'
    var_2 = 'Parent Account'
    var_3 = '456'
    var_4 = 'Sub Account'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0



# Parsed testcases at query #2
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
    var_1 = 'Custom Asset'
    var_2 = '2000'
    var_3 = 'Custom Liability'
    var_4 = '3000'
    var_5 = 'Custom Equity'
    var_6 = '4000'
    var_7 = 'Custom Revenue'
    var_8 = '5000'
    var_9 = 'Custom Expense'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = '2000'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '67890'
    var_3 = 'Parent Account'
    var_4 = 'COA1'
    var_5 = module_0.COA()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_subaccount_success. Retrieved 4/8 statements.
# Partially parsed test_add_existing_subaccount_success. Retrieved 4/8 statements.
# Partially parsed test_add_subaccount_invalid_parent. Retrieved 4/8 statements.
# Partially parsed test_add_subaccount_same_as_parent. Retrieved 3/7 statements.
# Partially parsed test_add_subaccount_inconsistent_data. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'
    var_4 = 'Different Name'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_nodify_creates_node_with_account_and_empty_children. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_account_and_subaccounts. Retrieved 6/19 statements.
# Partially parsed test_nodify_creates_nested_node_structure. Retrieved 7/23 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account 1'
    var_4 = '1.2'
    var_5 = 'Sub Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account 1'
    var_4 = '1.1.1'
    var_5 = 'Sub Sub Account 1'
    var_6 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '1000'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_add_new_subaccount. Retrieved 4/11 statements.
# Partially parsed test_add_existing_subaccount_with_same_details. Retrieved 4/9 statements.
# Partially parsed test_add_subaccount_with_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_subaccount_with_same_parent_and_code. Retrieved 3/6 statements.
# Partially parsed test_add_subaccount_with_inconsistent_details. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'
    var_4 = 'Different Name'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 8/25 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/46 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 7/32 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Assets'
    var_2 = '20'
    var_3 = 'Custom Liabilities'
    var_4 = '30'
    var_5 = 'Custom Equity'
    var_6 = '40'
    var_7 = 'Custom Income'
    var_8 = '50'
    var_9 = 'Custom Expenses'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Assets'
    var_2 = '20'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_parent_instance_exists. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/10 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/44 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 7/32 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = 'Custom Equity'
    var_6 = '4'
    var_7 = 'Custom Income'
    var_8 = '5'
    var_9 = 'Custom Expense'

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 3/5 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/31 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5

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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Cash'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/10 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/27 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = str(var_1)

def test_case_0():
    var_0 = '100'
    var_1 = 'Custom Assets'
    var_2 = '200'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'COA001'
    var_1 = 'Test COA'
    var_2 = module_0.COA()
    var_3 = 'ACC001'
    var_4 = 'Parent Account'
    var_5 = 'SUB001'
    var_6 = 'Sub Account'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 'Assets'
    var_3 = module_0.COA()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_parentinstance_is_not_none. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = var_0.accounts
    var_5 = var_0.accounts
    var_6 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_nodify_creates_node_with_account_and_empty_children. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_node_with_account_and_children. Retrieved 6/19 statements.
# Partially parsed test_nodify_creates_node_with_nested_children. Retrieved 7/23 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account 1'
    var_4 = '1.2'
    var_5 = 'Sub Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account 1'
    var_4 = '1.1.1'
    var_5 = 'Sub Account 1.1'
    var_6 = 0



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 15/47 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 9/32 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = 'Custom Equity'
    var_6 = '4'
    var_7 = 'Custom Income'
    var_8 = '5'
    var_9 = 'Custom Expense'
    var_10 = '1'
    var_11 = '2'
    var_12 = '3'
    var_13 = '4'
    var_14 = '5'

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = '5'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #24
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
    var_0 = '10'
    var_1 = 'Current Assets'
    var_2 = '20'
    var_3 = 'Current Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 3/18 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 0



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'COA001'
    var_1 = 'Main COA'
    var_2 = module_0.COA()
    var_3 = 'ACC001'
    var_4 = 'Parent Account'
    var_5 = 'SUB001'
    var_6 = 'Sub Account'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '1234'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 10/27 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/42 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = var_0.accounts
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = '4'
    var_9 = '5'

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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 15/48 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

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
    var_10 = '1'
    var_11 = '2'
    var_12 = '3'
    var_13 = '4'
    var_14 = '5'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_parent_instance_not_none. Retrieved 4/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #34
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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_parent_instance_not_none. Retrieved 4/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Test SubAccount'
    var_2 = '123'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 10/20 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/41 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = var_0._subaccounts
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = '4'
    var_9 = '5'

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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Cash'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 8/18 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/42 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 6/26 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 5
    var_3 = '1'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = '5'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = '2000'
    var_3 = 'Custom Liability'
    var_4 = '3000'
    var_5 = 'Custom Equity'
    var_6 = '4000'
    var_7 = 'Custom Income'
    var_8 = '5000'
    var_9 = 'Custom Expense'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = '3'
    var_4 = '4'
    var_5 = '5'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_parent_instance_not_none. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'test_coa'
    var_1 = module_0.COA(var_0)
    var_2 = 'test_account'
    var_3 = 'Test Account'
    var_4 = '12345'
    var_5 = 'Test SubAccount'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_coa_constructor_initializes_root_accounts. Retrieved 7/55 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Assets'
    var_2 = '2'
    var_3 = 'Custom Liabilities'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_parent_account_not_defined. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '001'
    var_3 = 'Test Account'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/30 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 5

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = '3'
    var_5 = '4'
    var_6 = '5'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'COA1'
    var_5 = 'Test COA'
    var_6 = module_0.COA()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa_instance. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



