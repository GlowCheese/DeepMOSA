####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 11/13 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.toplevel
    var_5 = list(var_4)
    var_6 = len(var_5)
    var_7 = 0
    var_8 = var_0.accounts
    var_9 = list(var_8)[var_7]
    var_10 = var_9.parent
    assert var_10 is None

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Asset'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Liability'
    var_2 = '1'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_self_parent_raises_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_non_existent_parent_raises_error. Retrieved 5/14 statements.
# Partially parsed test_coa_add_existing_account_returns_same. Retrieved 6/21 statements.
# Partially parsed test_coa_add_existing_account_mismatch_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '99'
    var_3 = '11'
    var_4 = 'No Parent'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = {}
    var_3 = None
    var_4 = '11'
    var_5 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Original Name'
    var_4 = 'Different Name'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/14 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 2/14 statements.


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
    var_0 = '100'
    var_1 = 'Assets'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_parent_exists_so_parentinstance_is_not_none. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Assets'
    var_3 = 'Cash'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/13 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Custom Asset'

def test_case_0():
    var_0 = '999'
    var_1 = 'Custom Liability'
    var_2 = '1'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_default_naming_logic. Retrieved 3/6 statements.


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
    var_1 = 'Custom Root'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = '1'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 6/21 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = '1101'
    var_5 = 'Petty Cash'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets Custom'

def test_case_0():
    var_0 = '200'
    var_1 = 'Liabilities Custom'
    var_2 = '1'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_verifies_account_mapping. Retrieved 3/10 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = dict(var_0)
    var_2 = len(var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_coa. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Accounts Receivable'
    var_2 = 'Inventory'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_same_code_raises_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_non_existent_parent_raises_error. Retrieved 4/8 statements.
# Partially parsed test_coa_add_existing_account_returns_same. Retrieved 4/13 statements.
# Partially parsed test_coa_add_inconsistent_data_raises_error. Retrieved 5/15 statements.
# Partially parsed test_coa_add_mismatched_parent_raises_error. Retrieved 7/21 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self Parent'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = '99'
    var_3 = '991'

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

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = '11'
    var_5 = 'Cash'
    var_6 = 'Cash'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_raises_error_when_account_info_is_inconsistent. Retrieved 5/14 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Assets'
    var_3 = 'Cash'
    var_4 = 'Different Name'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 3/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'
    var_2 = module_0.COA()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_account_inconsistent_name_raises_error. Retrieved 5/16 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '101'
    var_3 = 'Cash'
    var_4 = 'Different Name'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_value. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 100
    var_2 = {var_0: var_1}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_account_inconsistent_data_raises_error. Retrieved 10/17 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Root'
    var_4 = var_0.accounts
    var_5 = iter(var_4)
    var_6 = next(var_5)
    var_7 = var_6.code
    var_8 = 'Valid SubAccount'
    var_9 = 'Invalid Name'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_coa. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 'account_2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_add_raises_value_error_on_inconsistent_account_data. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = 'Different Name'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_type_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_subaccount_coa_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Test'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 100



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_sub_account_type_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_sub_account_coa_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_sub_account_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Test'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 100



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_raises_error_when_account_details_mismatch. Retrieved 6/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '11'
    var_3 = 'Root'
    var_4 = 'Original Name'
    var_5 = 'Different Name'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 'account_2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_same_code_raises_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_non_existent_parent_raises_error. Retrieved 5/14 statements.
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
    var_2 = 'Self Parent'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '99'
    var_3 = '991'
    var_4 = 'Ghost Child'

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



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 4/10 statements.


def test_case_0():
    var_0 = '1001'
    var_1 = 'Savings Sub-Account'
    var_2 = 'Asset'
    var_3 = 'COA_001'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_coa. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Accounts Receivable'
    var_2 = 'Inventory'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_sub_account_type_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_sub_account_coa_property_delegation. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'SubAccountName'

def test_case_0():
    var_0 = 'Test'

def test_case_0():
    var_0 = 'Test'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_add_raises_value_error_on_inconsistent_account_data. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '11'
    var_2 = 'Assets'
    var_3 = 'Cash'
    var_4 = 'Different Name'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 5/16 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = '2000'
    var_2 = 'Cash'
    var_3 = 'Accounts Payable'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 5/20 statements.
# Partially parsed test_nodify_handles_account_with_no_subaccounts. Retrieved 3/8 statements.
# Partially parsed test_nodify_recursive_depth. Retrieved 6/17 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '11'
    var_2 = 'Assets'
    var_3 = 'Cash'
    var_4 = 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = []

def test_case_0():
    var_0 = '1'
    var_1 = '11'
    var_2 = '111'
    var_3 = 'Assets'
    var_4 = 'Cash'
    var_5 = 'Petty Cash'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = var_1.accounts

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Root'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_default_naming_logic. Retrieved 4/8 statements.


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
    var_1 = 'Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = 0
    var_3 = '1'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/14 statements.
# Partially parsed test_coa_add_self_parent_raises_error. Retrieved 3/10 statements.
# Partially parsed test_coa_add_non_existent_parent_raises_error. Retrieved 5/14 statements.
# Partially parsed test_coa_add_duplicate_account_returns_existing. Retrieved 4/13 statements.
# Partially parsed test_coa_add_duplicate_account_mismatch_raises_error. Retrieved 5/14 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '101'
    var_2 = 'Cash'
    var_3 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = '1'
    var_1 = '101'
    var_2 = 'Assets'
    var_3 = '999'
    var_4 = 'Ghost'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '1'
    var_1 = '101'
    var_2 = 'Cash'
    var_3 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = '101'
    var_2 = 'Assets'
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = 'coa'
    var_2 = [var_0, var_1]
    var_3 = 'name'
    var_4 = [var_3]
    var_5 = '12345'
    var_6 = 'Savings Sub-Account'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/18 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = '1'
    var_5 = var_2[0].code
    var_6 = var_2[0].name

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'
    var_2 = '200'
    var_3 = 'Liabilities'

def test_case_0():
    var_0 = '300'
    var_1 = 'Equity'
    var_2 = '1'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/9 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/18 statements.
# Partially parsed test_coa_constructor_preserves_order. Retrieved 3/4 statements.


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
    var_2 = '200'
    var_3 = 'Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = [a.type for a in var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_add_parent_exists_to_evaluate_predicate_false. Retrieved 4/14 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 4/17 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Assets'
    var_3 = 'Cash'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 7/25 statements.
# Partially parsed test_nodify_leaf_node_has_no_children. Retrieved 2/9 statements.
# Partially parsed test_nodify_with_unrelated_account. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = '111'
    var_5 = 'Petty Cash'
    var_6 = 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_default_values. Retrieved 5/7 statements.


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
    var_1 = 'Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = iter(var_1)
    var_3 = next(var_2)
    var_4 = '1'
    var_5 = var_3.code
    var_6 = var_3.name



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_fails_when_parent_is_not_defined. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1000'
    var_2 = '1100'
    var_3 = 'Cash'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 4/17 statements.
# Partially parsed test_nodify_handles_leaf_node_with_no_children. Retrieved 2/9 statements.
# Partially parsed test_nodify_handles_deeply_nested_structure. Retrieved 7/20 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '11'
    var_2 = 'Assets'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = '11'
    var_2 = '111'
    var_3 = 'Assets'
    var_4 = 'Current Assets'
    var_5 = 'Cash'
    var_6 = 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_coa_constructor_with_no_spec. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_custom_spec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_verifies_account_type_mapping. Retrieved 3/9 statements.


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
    var_1 = var_0.accounts
    var_2 = var_0.accounts



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_coa_constructor_with_no_rootspec. Retrieved 7/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_iterates_correctly. Retrieved 8/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = '1'
    var_7 = var_2[0].code

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = var_1[var_3][var_3]
    var_6 = 1
    var_7 = var_1[var_3][var_6]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_sub_account_type_property_access. Retrieved 1/5 statements.
# Partially parsed test_sub_account_coa_property_access. Retrieved 1/6 statements.
# Partially parsed test_sub_account_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 9/13 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.toplevel
    var_5 = list(var_4)
    var_6 = len(var_5)
    var_7 = var_0.toplevel
    var_8 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '200'
    var_1 = 'Liabilities'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_account_with_existing_parent_does_not_trigger_none_check. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_coa. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Accounts Receivable'
    var_2 = 'Inventory'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_sub_account_type_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_sub_account_coa_property_delegation. Retrieved 1/6 statements.
# Partially parsed test_sub_account_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_value. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 'account_2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings Sub-account'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 100
    var_2 = {var_0: var_1}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 3/18 statements.


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
    var_2 = 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 4/17 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Root'
    var_2 = '1.1'
    var_3 = 'Sub'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_coa_constructor_default_roots. Retrieved 6/11 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 5/21 statements.
# Partially parsed test_coa_constructor_empty_rootspec_behavior. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None
    var_5 = '1'
    var_6 = var_2[0].code
    var_7 = var_2[0].name

def test_case_0():
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '20'
    var_3 = 'Liabilities'
    var_4 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_error_self_parent. Retrieved 3/12 statements.
# Partially parsed test_coa_add_error_parent_not_found. Retrieved 5/15 statements.
# Partially parsed test_coa_add_existing_account_consistent. Retrieved 4/13 statements.
# Partially parsed test_coa_add_existing_account_inconsistent. Retrieved 5/16 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self'

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



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'SubAccountName'
    var_2 = 'MainCOA'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_same_code_as_parent_raises_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_non_existent_parent_raises_error. Retrieved 5/9 statements.
# Partially parsed test_coa_add_duplicate_account_returns_existing. Retrieved 4/16 statements.
# Partially parsed test_coa_add_duplicate_account_inconsistent_data_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self Parent'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
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
    var_3 = 'Original Name'
    var_4 = 'Different Name'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_add_account_ensures_parent_already_in_subaccounts_buffer. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_add_existing_account_returns_same_instance. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '11'
    var_2 = 'Cash'
    var_3 = 'Assets'



