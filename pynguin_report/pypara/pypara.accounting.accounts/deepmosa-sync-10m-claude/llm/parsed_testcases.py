####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.




# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nodify_creates_node_with_account. Retrieved 2/6 statements.
# Partially parsed test_nodify_creates_node_with_children. Retrieved 4/13 statements.
# Partially parsed test_nodify_creates_nested_tree_structure. Retrieved 7/22 statements.
# Partially parsed test_nodify_returns_node_instance. Retrieved 2/8 statements.
# Partially parsed test_nodify_with_multiple_children. Retrieved 8/23 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Child'
    var_4 = '1.1.1'
    var_5 = 'Test Grandchild'
    var_6 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child 1'
    var_4 = '1.2'
    var_5 = 'Child 2'
    var_6 = '1.3'
    var_7 = 'Child 3'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'Chart of Accounts'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1100'
    var_6 = 'Cash'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_nodify_returns_coa_node_instance. Retrieved 3/20 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/5 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/15 statements.
# Partially parsed test_coa_constructor_partial_custom_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_initializes_empty_subaccounts. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_creates_root_accounts_in_order. Retrieved 2/7 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_accounts_are_frozen. Retrieved 3/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = bool(var_0._accounts is not None)
    assert var_2 is True
    var_3 = var_0._subaccounts
    var_4 = bool(var_0._subaccounts is not None)
    assert var_4 is True
    var_5 = var_0._accounts
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = bool(var_5)
    assert var_7 is True

def test_case_0():
    var_0 = '1000'
    var_1 = 'Assets'
    var_2 = '2000'
    var_3 = 'Liabilities'

def test_case_0():
    var_0 = '5000'
    var_1 = 'Custom Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = var_0._subaccounts
    var_3 = len(var_2)
    assert var_3 == 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1._accounts
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = bool(var_0)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_initializes_all_account_types. Retrieved 5/12 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/14 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_creates_root_accounts. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None
    var_5 = bool(var_4 is not None)
    assert var_5 is True

def test_case_0():
    var_0 = '100'
    var_1 = 'Total Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/22 statements.
# Partially parsed test_coa_constructor_accounts_are_root_accounts. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_creates_all_account_types. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_empty_rootspec_uses_defaults. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_none_rootspec_uses_defaults. Retrieved 5/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = bool(var_0._accounts is not None)
    assert var_2 is True
    var_3 = var_0._subaccounts
    var_4 = bool(var_0._subaccounts is not None)
    assert var_4 is True
    var_5 = var_0._accounts
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = var_0.accounts
    var_9 = list(var_8)
    var_10 = len(var_9)
    var_11 = bool(var_5)
    assert var_11 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_defined. Retrieved 4/8 statements.
# Partially parsed test_add_returns_existing_account_if_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_different_info. Retrieved 5/10 statements.
# Partially parsed test_add_updates_subaccounts_buffer. Retrieved 4/10 statements.
# Partially parsed test_add_multiple_subaccounts_to_same_parent. Retrieved 6/14 statements.


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
    var_2 = 'Test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'An account can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Test'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Parent account is not (yet) defined'

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
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Account name, code and parent do not match'

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
    var_3 = '1.2'
    var_4 = 'First Child'
    var_5 = 'Second Child'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_add_existing_account_returns_existing. Retrieved 9/28 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = module_0.COA()
    var_6 = '1'
    var_7 = '1.1'
    var_8 = 'Test Account'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/11 statements.
# Failed to parse test_read_chart_of_accounts_call_is_callable.
# Failed to parse test_read_chart_of_accounts_call_returns_consistent_type.


def test_case_0():
    var_0 = 'accounts'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/27 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Asset'
    var_2 = 'Standard COA'
    var_3 = 'Sub Account 1'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 1/11 statements.
# Failed to parse test_read_chart_of_accounts_call_returns_coa_instance.
# Partially parsed test_read_chart_of_accounts_call_multiple_invocations. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'accounts'

def test_case_0():
    var_0 = 0
    assert var_0 == 3



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/32 statements.
# Partially parsed test_subaccount_constructor_frozen. Retrieved 7/35 statements.


def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'PARENT001'
    var_2 = 'Asset'
    var_3 = 'Standard COA'
    var_4 = 'Parent Account'
    var_5 = 'Sub Account'

def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'PARENT001'
    var_2 = 'Asset'
    var_3 = 'Standard COA'
    var_4 = 'Parent Account'
    var_5 = 'Sub Account'
    var_6 = 'NEW_CODE'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/16 statements.
# Partially parsed test_subaccount_constructor_with_different_values. Retrieved 4/16 statements.
# Partially parsed test_subaccount_is_frozen. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = 'Cash'

def test_case_0():
    var_0 = 'Liability'
    var_1 = 'IFRS COA'
    var_2 = '2000'
    var_3 = 'Accounts Payable'

def test_case_0():
    var_0 = 'Equity'
    var_1 = 'Test COA'
    var_2 = '3000'
    var_3 = 'Capital'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'frozen'
    var_6 = bool('frozen' in str(type(e).__name__).lower())
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_basic_subaccount. Retrieved 4/8 statements.
# Partially parsed test_add_same_parent_and_code_raises_error. Retrieved 3/6 statements.
# Partially parsed test_add_nonexistent_parent_raises_error. Retrieved 4/8 statements.
# Partially parsed test_add_existing_account_with_matching_info_returns_existing. Retrieved 7/11 statements.
# Partially parsed test_add_existing_account_with_mismatched_info_raises_error. Retrieved 5/10 statements.
# Partially parsed test_add_multiple_subaccounts_to_same_parent. Retrieved 6/14 statements.
# Partially parsed test_add_nested_subaccounts. Retrieved 6/12 statements.
# Partially parsed test_add_account_updates_subaccounts_buffer. Retrieved 4/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test SubAccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'An account can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'Test'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Parent account is not (yet) defined'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test SubAccount'
    var_4 = var_0.accounts
    var_5 = [a for a in var_4]
    var_6 = len(var_5)
    assert var_6 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Account name, code and parent do not match'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'First Child'
    var_4 = '1.2'
    var_5 = 'Second Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.1.1'
    var_4 = 'Child'
    var_5 = 'Grandchild'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Child'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'Chart of Accounts'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1001'
    var_6 = 'Cash'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 1/11 statements.
# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Partially parsed test_read_chart_of_accounts_call_multiple_times. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'accounts'

def test_case_0():
    var_0 = 0
    assert var_0 == 2



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/20 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash Account'
    var_2 = 'Standard COA'
    var_3 = '1'
    var_4 = 'Assets'
    var_5 = 'asset'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_add_account_with_inconsistent_information_raises_error. Retrieved 11/34 statements.


def test_case_0():
    var_0 = 'Code'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = '1'
    var_7 = 'Assets'
    var_8 = '100'
    var_9 = 'Test Account'
    var_10 = 'Different Name'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Account name, code and parent do not match'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_basic_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_nested_subaccount. Retrieved 6/14 statements.
# Partially parsed test_add_duplicate_account_same_details. Retrieved 4/10 statements.
# Partially parsed test_add_parent_not_defined. Retrieved 4/8 statements.
# Partially parsed test_add_account_is_own_parent. Retrieved 3/7 statements.
# Partially parsed test_add_duplicate_account_different_name. Retrieved 7/14 statements.
# Partially parsed test_add_duplicate_account_different_parent. Retrieved 10/20 statements.
# Partially parsed test_add_multiple_accounts_same_parent. Retrieved 6/16 statements.
# Partially parsed test_add_account_in_subaccounts_buffer. Retrieved 4/10 statements.


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
    var_3 = 'First Level'
    var_4 = '1.1.1'
    var_5 = 'Second Level'

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
    var_5 = 'Parent account is not (yet) defined'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'An account can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original Name'
    var_4 = '1'
    var_5 = '1.1'
    var_6 = 'Different Name'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Account name, code and parent do not match existing chart of accounts member'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = '2'
    var_5 = '2.1'
    var_6 = 'Another Account'
    var_7 = '2'
    var_8 = '1.1'
    var_9 = 'Test Account'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Account name, code and parent do not match existing chart of accounts member'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'First Child'
    var_4 = '1.2'
    var_5 = 'Second Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_account_with_inconsistent_name_raises_error. Retrieved 10/45 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = '1'
    var_6 = 'Assets'
    var_7 = '1.1'
    var_8 = 'Current Assets'
    var_9 = 'Different Name'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'do not match existing chart of accounts member'



