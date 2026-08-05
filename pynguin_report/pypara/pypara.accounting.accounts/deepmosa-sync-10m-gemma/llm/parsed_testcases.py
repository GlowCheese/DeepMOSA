####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 6/22 statements.
# Partially parsed test_nodify_with_leaf_node. Retrieved 2/10 statements.
# Partially parsed test_nodify_recursively_processes_deep_hierarchy. Retrieved 8/25 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = '111'
    var_5 = 'Bank Account'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Current Assets'
    var_4 = '111'
    var_5 = 'Cash'
    var_6 = '1111'
    var_7 = 'Petty Cash'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'

def test_case_0():
    var_0 = '999'
    var_1 = 'Custom Revenue'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = bool(var_0._accounts == var_0._accounts)
    assert var_2 is True
    var_3 = var_0._subaccounts
    var_4 = bool(var_0._subaccounts == var_0._subaccounts)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_account'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_error_same_code. Retrieved 3/12 statements.
# Partially parsed test_coa_add_error_missing_parent. Retrieved 4/11 statements.
# Partially parsed test_coa_add_existing_account_consistency. Retrieved 5/19 statements.
# Partially parsed test_coa_add_error_inconsistent_data. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '101'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '999'
    var_3 = '101'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = None
    var_3 = '101'
    var_4 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '101'
    var_3 = 'Cash'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_add_fails_when_parent_is_not_defined. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '999'
    var_3 = '101'
    var_4 = 'Cash'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 6/24 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Current Assets'
    var_4 = '111'
    var_5 = 'Cash'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_mixed_rootspec. Retrieved 2/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Account'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets Only'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-account'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 5/14 statements.


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
    var_1 = 'Custom Root'

def test_case_0():
    var_0 = '999'
    var_1 = 'Only Asset Custom'
    var_2 = 1
    var_3 = 2
    var_4 = str(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_validation_of_types. Retrieved 1/4 statements.


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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_coa_constructor_default_roots. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_invalid_rootspec_type. Retrieved 5/6 statements.


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
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_coa_constructor_default_init. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 3/10 statements.
# Partially parsed test_coa_constructor_rootspec_partial. Retrieved 6/21 statements.


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
    var_2 = 0

def test_case_0():
    var_0 = 0
    var_1 = '10'
    var_2 = 'Overridden'
    var_3 = 1
    var_4 = None
    var_5 = '2'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_initializes_correct_types. Retrieved 3/5 statements.
# Partially parsed test_coa_constructor_handles_empty_rootspec_explicitly. Retrieved 5/6 statements.


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
    var_1 = var_0.accounts
    var_2 = [a.type for a in var_1]

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_preserves_order. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_raises_error_on_invalid_rootspec_format. Retrieved 4/5 statements.


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
    var_1 = 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1._accounts
    var_3 = len(var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'
    var_2 = 'Asset'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_with_existing_parent_does_not_raise_none_error. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/14 statements.
# Partially parsed test_coa_add_same_code_raises_error. Retrieved 3/10 statements.
# Partially parsed test_coa_add_non_existent_parent_raises_error. Retrieved 4/8 statements.
# Partially parsed test_coa_add_existing_account_returns_same_instance. Retrieved 4/12 statements.
# Partially parsed test_coa_add_existing_account_mismatch_raises_error. Retrieved 5/14 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Cash'
    var_3 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = 'Self Parent'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Ghost Account'

def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Cash'
    var_3 = 'Assets'

def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Assets'
    var_3 = 'Original Name'
    var_4 = 'Different Name'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #24
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Savings Sub-Account'
    var_2 = 'Asset'
    var_3 = 'Standard_COA'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_coa_constructor_default_roots. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 2/15 statements.


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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-Account'
    var_2 = 'Asset'
    var_3 = 'MainCOA'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 7/25 statements.
# Partially parsed test_nodify_with_no_children_returns_leaf_node. Retrieved 2/9 statements.
# Partially parsed test_nodify_with_multiple_siblings. Retrieved 6/23 statements.


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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 3/10 statements.
# Partially parsed test_coa_constructor_rootspec_partial. Retrieved 6/20 statements.


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
    var_2 = 0

def test_case_0():
    var_0 = 0
    var_1 = '10'
    var_2 = 'Only This One'
    var_3 = 1
    var_4 = None
    var_5 = '2'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_same_code_raises_error. Retrieved 3/12 statements.
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
    var_3 = '11'
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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings Sub-Account'
    var_2 = 'Asset'
    var_3 = 'Standard COA'



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

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #37
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_add_fails_when_parent_is_not_defined. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = '1000'
    var_3 = '1001'
    var_4 = 'New Account'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '1234'
    var_1 = 'Savings Sub-account'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_coa_constructor_default_roots. Retrieved 6/11 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/18 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/13 statements.


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

def test_case_0():
    var_0 = '30'
    var_1 = 'Equity'
    var_2 = 0



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/9 statements.
# Partially parsed test_coa_constructor_initializes_correct_types. Retrieved 4/5 statements.


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
    var_2 = [a.type for a in var_1]
    var_3 = len(var_2)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.toplevel
    var_5 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Custom Asset'

def test_case_0():
    var_0 = '200'
    var_1 = 'Custom Liability'
    var_2 = '1'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_coa_constructor_default_roots. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/10 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/15 statements.


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

def test_case_0():
    var_0 = '200'
    var_1 = 'Liabilities Only'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'account1'
    var_1 = 'account2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_add_with_existing_parent_does_not_trigger_none_error. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Asset'
    var_2 = '101'
    var_3 = 'Cash'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/18 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_properties_delegation. Retrieved 1/6 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Savings Sub-Account'

def test_case_0():
    var_0 = 'Test'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/8 statements.
# Partially parsed test_coa_constructor_rootspec_partial. Retrieved 3/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Account'

def test_case_0():
    var_0 = '10'
    var_1 = 'Liability Root'
    var_2 = '1'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 7/26 statements.
# Partially parsed test_nodify_leaf_node_has_no_children. Retrieved 4/15 statements.
# Partially parsed test_nodify_single_level_structure. Retrieved 2/10 statements.


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
    var_2 = '11'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 5/21 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'
    var_4 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_coa_constructor_no_rootspec. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 5/21 statements.
# Partially parsed test_coa_constructor_default_naming. Retrieved 5/7 statements.


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
    var_4 = '300'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = iter(var_1)
    var_3 = next(var_2)
    var_4 = '1'
    var_5 = var_3.code
    var_6 = var_3.name



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/18 statements.
# Partially parsed test_coa_constructor_empty_rootspec_behavior. Retrieved 6/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '20'
    var_3 = 'Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = iter(var_2)
    var_4 = next(var_3)
    var_5 = '1'
    var_6 = var_4.code



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_same_code_raises_error. Retrieved 3/11 statements.
# Partially parsed test_coa_add_nonexistent_parent_raises_error. Retrieved 5/14 statements.
# Partially parsed test_coa_add_duplicate_account_returns_existing. Retrieved 4/16 statements.
# Partially parsed test_coa_add_duplicate_account_mismatch_raises_error. Retrieved 5/15 statements.


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
    var_3 = '11'
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



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/17 statements.
# Partially parsed test_coa_constructor_empty_rootspec_behavior. Retrieved 6/7 statements.


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
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = iter(var_2)
    var_4 = next(var_3)
    var_5 = '1'
    var_6 = var_4.code



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_coa_constructor_with_no_rootspec. Retrieved 5/10 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 3/10 statements.
# Partially parsed test_coa_constructor_default_naming. Retrieved 4/9 statements.


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
    var_2 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = 0
    var_2 = var_0.accounts
    var_3 = '1'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings'
    var_2 = 'COA_001'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_default_initialization. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 2/9 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts
    var_5 = var_0.accounts
    var_6 = None

def test_case_0():
    var_0 = '99'
    var_1 = 'Custom Type'

def test_case_0():
    var_0 = '10'
    var_1 = 'Special Asset'
    var_2 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/18 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '11'
    var_3 = 'Cash'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '12345'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_add_does_not_raise_when_parent_exists. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '100'
    var_2 = '101'
    var_3 = 'Test Account'
    var_4 = 'Assets'
    var_5 = 'Parent Account'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_with_existing_parent. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Cash'
    var_3 = 'Assets'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_sub_account_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'SA001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 4/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Savings Sub-account'
    var_2 = 'Default COA'
    var_3 = module_0.COA(var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_nodify_returns_correct_node_structure. Retrieved 4/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 4/7 statements.
# Partially parsed test_subaccount_immutability. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Savings Sub-Account'
    var_2 = 'Asset'
    var_3 = 'MainCOA'

def test_case_0():
    var_0 = 'Liability'
    var_1 = 'SecondaryCOA'
    var_2 = 'SUB002'
    var_3 = 'Credit Sub-Account'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '1001'
    var_1 = 'Savings Sub-Account'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Savings'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_coa_constructor_default_roots. Retrieved 7/13 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_empty_rootspec_is_treated_as_none. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None
    var_5 = '1'
    var_6 = 'Asset'

def test_case_0():
    var_0 = '10'
    var_1 = 'Total Assets'
    var_2 = '20'
    var_3 = 'Total Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_add_fails_when_parent_not_defined. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '100'
    var_3 = 'New Account'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_coa_constructor_no_rootspec. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 5/20 statements.


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
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '20'
    var_3 = 'Liabilities'
    var_4 = '1'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_expected_coa. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'account_1'
    var_1 = 'account_2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #27
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
    var_4 = 'Orphan Account'

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_fails_self_parenting. Retrieved 3/11 statements.
# Partially parsed test_coa_add_fails_missing_parent. Retrieved 5/14 statements.
# Partially parsed test_coa_add_returns_existing_account_if_identical. Retrieved 4/16 statements.
# Partially parsed test_coa_add_fails_on_inconsistent_existing_account. Retrieved 5/15 statements.


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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_correct_value. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'account1'
    var_1 = 'account2'
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_raises_error_on_inconsistent_account_data. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = '1.1'
    var_2 = 'Assets'
    var_3 = 'Original Name'
    var_4 = 'Different Name'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_coa_add_success. Retrieved 4/15 statements.
# Partially parsed test_coa_add_error_self_parenting. Retrieved 3/12 statements.
# Partially parsed test_coa_add_error_missing_parent. Retrieved 5/15 statements.
# Partially parsed test_coa_add_return_existing_if_identical. Retrieved 4/16 statements.
# Partially parsed test_coa_add_error_inconsistent_data. Retrieved 5/16 statements.
# Partially parsed test_coa_add_error_mismatched_parent. Retrieved 8/23 statements.


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
    var_4 = 'Orphan Account'

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
    var_6 = '2'
    var_7 = 'Cash'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_type_property_returns_parent_type. Retrieved 1/6 statements.
# Partially parsed test_subaccount_coa_property_returns_parent_coa. Retrieved 1/6 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Savings Account'

def test_case_0():
    var_0 = 'Sub'

def test_case_0():
    var_0 = 'Sub'

def test_case_0():
    var_0 = 'Sub'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_subaccount_constructor_initialization. Retrieved 1/4 statements.
# Partially parsed test_subaccount_properties_delegation. Retrieved 1/8 statements.
# Partially parsed test_subaccount_immutability. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Sub Account Name'

def test_case_0():
    var_0 = 'Test'



