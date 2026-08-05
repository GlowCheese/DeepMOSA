####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_raises_error_on_invalid_logic. Retrieved 2/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = '1'

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = 'Duplicate'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/14 statements.
# Partially parsed test_nodify_handles_leaf_node. Retrieved 2/5 statements.
# Partially parsed test_nodify_recursive_structure. Retrieved 6/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child'
    var_4 = '1.1.1'
    var_5 = 'Grandchild'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_type_property_delegation. Retrieved 1/5 statements.
# Partially parsed test_subaccount_coa_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_properties_delegate_to_parent. Retrieved 1/8 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings Sub-account'

def test_case_0():
    var_0 = 'Checking Sub-account'

def test_case_0():
    var_0 = 'Immutable'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Savings Sub-account'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Savings'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_read_chart_of_accounts_success.




# Parsed testcases at query #8
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-Account'
    var_2 = 'ASSET'
    var_3 = 'COA_ROOT'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '101'
    var_2 = 'Cash'
    var_3 = 'Assets'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_existing_account_inconsistent_data_raises_error. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '\n    Ensures that when an account with the same code already exists but has \n    different parent, name, or code, a ValueError is raised. \n    This forces the predicate at line 27 to evaluate to False.\n    '
    var_1 = '1'
    var_2 = '101'
    var_3 = 'Assets'
    var_4 = 'Cash'
    var_5 = 'Different Name'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_self_parent_raises_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_nonexistent_parent_raises_error. Retrieved 5/14 statements.
# Partially parsed test_coa_add_existing_account_returns_same_instance. Retrieved 4/13 statements.
# Partially parsed test_coa_add_existing_account_mismatch_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self Parent Error'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '99'
    var_3 = '11'
    var_4 = 'No Parent Error'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Original Name'
    var_4 = 'Different Name'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_add_raises_error_on_inconsistent_account_data. Retrieved 7/19 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Current Assets'
    var_4 = '111'
    var_5 = 'Cash'
    var_6 = 'Different Name'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_value. Retrieved 3/10 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = {var_0: var_1}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_coa. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'account1'
    var_1 = 'account2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_type_property_access. Retrieved 1/6 statements.
# Partially parsed test_subaccount_coa_property_access. Retrieved 1/6 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 'account_2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_properties_delegation. Retrieved 1/8 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Checking Sub-Account'

def test_case_0():
    var_0 = 'Immutable'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_integrity. Retrieved 7/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = var_0._accounts
    var_4 = len(var_3)
    var_5 = len(var_2)
    var_6 = var_0._subaccounts



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_error_self_parenting. Retrieved 3/11 statements.
# Partially parsed test_coa_add_error_missing_parent. Retrieved 5/14 statements.
# Partially parsed test_coa_add_existing_account_same_info. Retrieved 4/13 statements.
# Partially parsed test_coa_add_existing_account_mismatch_info. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self Parent'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '99'
    var_3 = '991'
    var_4 = 'Orphan'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Original Name'
    var_4 = 'Different Name'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_properties_access_parent_attributes. Retrieved 1/8 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Checking Sub-Account'

def test_case_0():
    var_0 = 'Immutable'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/18 statements.
# Partially parsed test_coa_constructor_empty_rootspec_behavior. Retrieved 5/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets Account'
    var_2 = '200'
    var_3 = 'Liabilities Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = dict(var_1)
    var_3 = len(var_2)
    var_4 = '1'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 7/25 statements.
# Partially parsed test_nodify_with_no_children_returns_leaf_node. Retrieved 2/9 statements.
# Partially parsed test_nodify_handles_multiple_siblings. Retrieved 6/19 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = '1101'
    var_5 = 'Petty Cash'
    var_6 = 0

def test_case_0():
    var_0 = '2'
    var_1 = 'Liabilities'

def test_case_0():
    var_0 = '3'
    var_1 = 'Equity'
    var_2 = '31'
    var_3 = 'Retained Earnings'
    var_4 = '32'
    var_5 = 'Common Stock'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_same_code_raises_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_nonexistent_parent_raises_error. Retrieved 5/14 statements.
# Partially parsed test_coa_add_existing_account_returns_same. Retrieved 4/13 statements.
# Partially parsed test_coa_add_existing_account_mismatch_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self Parent'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '99'
    var_3 = '991'
    var_4 = 'No Parent'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = 'Different Name'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_add_with_existing_parent_should_not_raise_error. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_verifies_accounts_dict. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_iteration. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    var_3 = var_0._accounts

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    var_3 = 0
    var_4 = 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_preserves_order. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = '1'

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = [a.code for a in var_2]
    var_4 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 7/25 statements.
# Partially parsed test_nodify_with_no_children_returns_leaf_node. Retrieved 2/9 statements.
# Partially parsed test_nodify_with_multiple_siblings. Retrieved 6/21 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = '1101'
    var_5 = 'Petty Cash'
    var_6 = 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = '12'
    var_5 = 'Inventory'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 5/14 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = '2000'
    var_2 = 'Cash'
    var_3 = 'Accounts Payable'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-account'
    var_2 = 'Asset'
    var_3 = 'MainCOA'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'account1'
    var_1 = 100
    var_2 = {var_0: var_1}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_fails_when_parent_is_not_defined. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1000'
    var_2 = '1010'
    var_3 = 'Test Account'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/8 statements.
# Partially parsed test_sub_account_immutability. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '1001'
    var_1 = 'Savings Sub-Account'

def test_case_0():
    var_0 = '1001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Savings'
    var_2 = '100'
    var_3 = 'Assets'
    var_4 = module_0.COA()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'account1'
    var_1 = 'account2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_structure. Retrieved 4/18 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/17 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_type_property_returns_parent_type. Retrieved 1/6 statements.
# Partially parsed test_subaccount_coa_property_returns_parent_coa. Retrieved 1/6 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Test'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = [a.type for a in var_1]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'
    var_2 = '200'
    var_3 = 'Liabilities'

def test_case_0():
    var_0 = '999'
    var_1 = 'Custom Asset'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Savings'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '1001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0._accounts
    var_5 = var_0._subaccounts

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '200'
    var_1 = 'Liabilities'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 7/9 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 8/34 statements.
# Partially parsed test_coa_constructor_preserves_order. Retrieved 4/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts
    var_5 = iter(var_4)
    var_6 = next(var_5)

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'
    var_2 = '200'
    var_3 = 'Liabilities'
    var_4 = '10'
    var_5 = 'Assets Custom'
    var_6 = '20'
    var_7 = 'Liabilities Custom'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 4/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'
    var_2 = 'Standard'
    var_3 = module_0.COA(var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_default_naming_logic. Retrieved 3/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = 0
    var_5 = var_3[var_4]
    var_6 = var_5.code
    var_7 = 1
    var_8 = var_3[var_7]
    var_9 = var_8.code



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_rootspec_partial. Retrieved 3/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Asset'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Liability'
    var_2 = '1'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_add_fails_when_parent_is_not_defined. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '100'
    var_2 = '101'
    var_3 = 'Test Account'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 6/17 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'
    var_4 = 'Root Account'
    var_5 = 'Root'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_sub_account_type_property_returns_parent_type. Retrieved 1/6 statements.
# Partially parsed test_sub_account_coa_property_returns_parent_coa. Retrieved 1/6 statements.
# Partially parsed test_sub_account_is_frozen. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_same_code_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_missing_parent_error. Retrieved 5/14 statements.
# Partially parsed test_coa_add_existing_account_idempotency. Retrieved 4/16 statements.
# Partially parsed test_coa_add_existing_account_mismatch_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self Parent'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '99'
    var_3 = '991'
    var_4 = 'Ghost Account'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Original Name'
    var_4 = 'Different Name'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Root'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = '1'

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = [a.code for a in var_2]
    var_4 = lambda x: str(x)
    var_5 = sorted(var_3, key=var_4)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'account1'
    var_1 = 100
    var_2 = {var_0: var_1}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 7/26 statements.
# Partially parsed test_nodify_handles_leaf_nodes. Retrieved 2/10 statements.
# Partially parsed test_nodify_handles_empty_subaccounts. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = '1101'
    var_5 = 'Petty Cash'
    var_6 = 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_type_property_returns_parent_type. Retrieved 1/6 statements.
# Partially parsed test_subaccount_coa_property_returns_parent_coa. Retrieved 1/6 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/17 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Type'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Root'
    var_2 = '2'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_default_values. Retrieved 3/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = 0
    var_2 = '1'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/17 statements.
# Partially parsed test_nodify_leaf_node_has_no_children. Retrieved 2/9 statements.
# Partially parsed test_nodify_deep_structure. Retrieved 7/22 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Assets'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = '1.1.1'
    var_3 = 'Assets'
    var_4 = 'Current Assets'
    var_5 = 'Cash'
    var_6 = 0



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_type_property_returns_parent_type. Retrieved 1/6 statements.
# Partially parsed test_subaccount_coa_property_returns_parent_coa. Retrieved 1/6 statements.
# Partially parsed test_subaccount_is_immutable. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Savings Account'



# Parsed testcases at query #47
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_expected_coa.




