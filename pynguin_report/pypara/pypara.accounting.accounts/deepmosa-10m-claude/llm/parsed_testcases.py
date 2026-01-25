####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/14 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_creates_root_accounts_in_order. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_accounts_buffer_initialized. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_subaccounts_buffer_initialized. Retrieved 4/5 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'
    var_2 = '200'
    var_3 = 'Liabilities'

def test_case_0():
    var_0 = '999'
    var_1 = 'Custom Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = [acc.code for acc in var_1]
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = var_0.toplevel

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = var_0._accounts
    var_3 = len(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = var_0._subaccounts
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1001'
    var_3 = 'Cash'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/9 statements.
# Partially parsed test_add_returns_existing_account_with_same_properties. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/7 statements.
# Partially parsed test_add_raises_error_when_parent_not_defined. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_different_properties. Retrieved 5/11 statements.
# Partially parsed test_add_updates_subaccounts_buffer. Retrieved 4/11 statements.
# Partially parsed test_add_multiple_subaccounts_to_same_parent. Retrieved 6/15 statements.
# Partially parsed test_add_nested_subaccounts. Retrieved 6/13 statements.


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
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Parent account is not (yet) defined'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'do not match existing chart of accounts member'

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
    var_3 = 'Account 1'
    var_4 = '1.2'
    var_5 = 'Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.1.1'
    var_4 = 'Child Account'
    var_5 = 'Grandchild Account'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/12 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/13 statements.
# Partially parsed test_coa_constructor_initializes_accounts_dict. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_initializes_subaccounts_dict. Retrieved 2/3 statements.
# Partially parsed test_coa_constructor_creates_root_accounts_only. Retrieved 4/7 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = var_1.find(var_2)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.type
    var_8 = var_5.name
    var_9 = bool(var_5.name == var_3)
    assert var_9 is True

def test_case_0():
    var_0 = '100'
    var_1 = 'Total Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = var_0._accounts
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 5/9 statements.
# Partially parsed test_coa_constructor_none_rootspec. Retrieved 5/9 statements.
# Partially parsed test_coa_constructor_creates_ordered_dict. Retrieved 3/5 statements.
# Partially parsed test_coa_constructor_root_accounts_are_frozen. Retrieved 1/3 statements.


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

def test_case_0():
    var_0 = '10'
    var_1 = 'My Assets'
    var_2 = '2'

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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = var_0._subaccounts

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_nodify_creates_node_with_account. Retrieved 2/6 statements.
# Partially parsed test_nodify_creates_node_with_children. Retrieved 4/13 statements.
# Partially parsed test_nodify_creates_nested_node_structure. Retrieved 7/22 statements.
# Partially parsed test_nodify_creates_node_with_multiple_children. Retrieved 6/24 statements.
# Partially parsed test_nodify_returns_node_instance. Retrieved 2/8 statements.


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
    var_3 = 'Child'
    var_4 = '1.1.1'
    var_5 = 'Grandchild'
    var_6 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child 1'
    var_4 = '1.2'
    var_5 = 'Child 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_nodify_creates_node_for_account_with_no_subaccounts. Retrieved 2/6 statements.
# Partially parsed test_nodify_creates_node_with_subaccounts. Retrieved 4/12 statements.
# Partially parsed test_nodify_creates_nested_nodes_with_multiple_levels. Retrieved 7/20 statements.
# Partially parsed test_nodify_returns_node_instance. Retrieved 2/8 statements.


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
    var_3 = '1.1.1'
    var_4 = 'Child'
    var_5 = 'Grandchild'
    var_6 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_nodify_returns_node_with_account_and_children. Retrieved 10/43 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'equity'
    var_3 = 'revenue'
    var_4 = 'expense'
    var_5 = module_0.COA()
    var_6 = '1'
    var_7 = '1.1'
    var_8 = 'Child Account'
    var_9 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_add_account_with_valid_parent. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '999'
    var_5 = 'Test Account'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_creates_root_accounts. Retrieved 4/6 statements.
# Partially parsed test_coa_constructor_all_account_types. Retrieved 2/6 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/13 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = bool(var_1)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = set()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '10'
    var_1 = 'My Assets'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/32 statements.


def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Asset'
    var_2 = 'Standard COA'
    var_3 = 'ACC000'
    var_4 = 'Parent Account'
    var_5 = 'Sub Account'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_creates_all_account_types. Retrieved 3/8 statements.
# Partially parsed test_coa_constructor_default_codes. Retrieved 3/7 statements.
# Partially parsed test_coa_constructor_creates_root_accounts_only. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts
    var_5 = list(var_4)

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = set()
    var_2 = len(var_1)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = [acc.code for (_, acc) in var_0]
    var_2 = 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None

def test_case_0():
    var_0 = '10'
    var_1 = 'My Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 5/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/15 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 2/14 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_none_rootspec. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_default_accounts_are_root. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_accounts_buffer_ordered. Retrieved 4/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts

def test_case_0():
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '20'
    var_3 = 'Liabilities'

def test_case_0():
    var_0 = '100'
    var_1 = 'My Assets'

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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'Chart of Accounts'
    var_3 = '1000'
    var_4 = 'Parent Account'
    var_5 = '1001'
    var_6 = 'Sub Account'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/14 statements.
# Partially parsed test_coa_constructor_initializes_empty_buffers. Retrieved 7/9 statements.
# Partially parsed test_coa_constructor_frozen_after_init. Retrieved 1/3 statements.


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

def test_case_0():
    var_0 = '100'
    var_1 = 'Total Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = var_0._subaccounts
    var_3 = var_0._accounts
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = var_0._subaccounts
    var_7 = len(var_6)
    assert var_7 == 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_with_valid_parent_account. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Test Sub Account'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 2/20 statements.
# Partially parsed test_subaccount_is_frozen. Retrieved 1/12 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Test Sub Account'

def test_case_0():
    var_0 = 'Test Sub Account'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool('frozen' in str(type(e)).lower() or 'frozen' in str(e).lower())
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 2/16 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Assets'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_nodify_returns_node_with_account_and_children. Retrieved 9/37 statements.


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
    var_8 = 'Child Account'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/31 statements.


def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Asset'
    var_2 = 'Standard COA'
    var_3 = 'PARENT001'
    var_4 = 'Parent Account'
    var_5 = 'Sub Account'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/21 statements.
# Partially parsed test_subaccount_constructor_frozen. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'Chart of Accounts'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1001'
    var_6 = 'Cash'

def test_case_0():
    var_0 = 'liability'
    var_1 = 'Chart of Accounts'
    var_2 = '2000'
    var_3 = 'Liabilities'
    var_4 = '2001'
    var_5 = 'Accounts Payable'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 3/5 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/37 statements.
# Partially parsed test_coa_constructor_creates_root_accounts. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_accounts_are_root_accounts. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 5/9 statements.
# Partially parsed test_coa_constructor_default_root_codes. Retrieved 1/6 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1._accounts
    var_3 = len(var_2)
    var_4 = var_1.accounts

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = len(var_1)
    assert var_2 == 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.
# Failed to parse test_read_chart_of_accounts_call_empty.
# Failed to parse test_read_chart_of_accounts_call_multiple_accounts.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_add_with_valid_parent_account. Retrieved 9/28 statements.


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
    var_8 = 'Test Sub-Account'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 3/12 statements.
# Partially parsed test_subaccount_is_frozen. Retrieved 3/14 statements.
# Partially parsed test_subaccount_properties. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = '1000'
    var_2 = 'Cash'

def test_case_0():
    var_0 = 'Liability'
    var_1 = '2000'
    var_2 = 'Accounts Payable'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Equity'
    var_1 = '3000'
    var_2 = 'Common Stock'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'accounts'



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Failed to parse test_read_chart_of_accounts_call_returns_coa_empty.
# Failed to parse test_read_chart_of_accounts_call_multiple_invocations.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/42 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/19 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 5/9 statements.
# Partially parsed test_coa_constructor_creates_ordered_dict. Retrieved 3/5 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_accounts_are_root_accounts. Retrieved 1/3 statements.


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
    var_4 = '3'
    var_5 = 'Equity'
    var_6 = '4'
    var_7 = 'Revenue'
    var_8 = '5'
    var_9 = 'Expense'

def test_case_0():
    var_0 = '100'
    var_1 = 'Custom Assets'
    var_2 = '2'
    var_3 = '3'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = var_0._subaccounts

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = 'Cash'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_new_subaccount. Retrieved 9/40 statements.
# Partially parsed test_add_duplicate_account_with_same_info. Retrieved 9/39 statements.
# Partially parsed test_add_account_same_parent_and_code_raises_error. Retrieved 8/37 statements.
# Partially parsed test_add_account_with_nonexistent_parent_raises_error. Retrieved 9/39 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'ASSET'
    var_1 = 'LIABILITY'
    var_2 = 'EQUITY'
    var_3 = 'REVENUE'
    var_4 = 'EXPENSE'
    var_5 = module_0.COA()
    var_6 = '1'
    var_7 = '1.1'
    var_8 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'ASSET'
    var_1 = 'LIABILITY'
    var_2 = 'EQUITY'
    var_3 = 'REVENUE'
    var_4 = 'EXPENSE'
    var_5 = module_0.COA()
    var_6 = '1'
    var_7 = '1.1'
    var_8 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'ASSET'
    var_1 = 'LIABILITY'
    var_2 = 'EQUITY'
    var_3 = 'REVENUE'
    var_4 = 'EXPENSE'
    var_5 = module_0.COA()
    var_6 = '1.1'
    var_7 = 'Test'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'ASSET'
    var_1 = 'LIABILITY'
    var_2 = 'EQUITY'
    var_3 = 'REVENUE'
    var_4 = 'EXPENSE'
    var_5 = module_0.COA()
    var_6 = '99'
    var_7 = '99.1'
    var_8 = 'Test'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'not (yet) defined'

def test_case_0():
    var_0 = 'ASSET'
    var_1 = 'LIABILITY'
    var_2 = 'EQUITY'
    var_3 = 'REVENUE'
    var_4 = 'EXPENSE'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/13 statements.
# Partially parsed test_subaccount_is_frozen. Retrieved 4/12 statements.
# Partially parsed test_subaccount_parent_type_property. Retrieved 4/9 statements.
# Partially parsed test_subaccount_parent_coa_property. Retrieved 4/9 statements.


def test_case_0():
    var_0 = '1001'
    var_1 = 'Test Sub Account'
    var_2 = 'Asset'
    var_3 = 'Standard COA'

def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1001'
    var_3 = 'Test Sub Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'Liability'
    var_1 = 'Standard COA'
    var_2 = '2001'
    var_3 = 'Liability Sub'

def test_case_0():
    var_0 = 'Equity'
    var_1 = 'Custom COA'
    var_2 = '3001'
    var_3 = 'Equity Sub'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_nodify_single_account_no_children. Retrieved 2/6 statements.
# Partially parsed test_nodify_account_with_children. Retrieved 6/16 statements.
# Partially parsed test_nodify_nested_hierarchy. Retrieved 7/20 statements.
# Partially parsed test_nodify_multiple_root_accounts. Retrieved 3/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.2'
    var_4 = 'Child Account 1'
    var_5 = 'Child Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.1.1'
    var_4 = 'Level 1 Account'
    var_5 = 'Level 2 Account'
    var_6 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.
# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Failed to parse test_read_chart_of_accounts_call_empty.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_initializes_accounts_buffer. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_initializes_subaccounts_buffer. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 5/9 statements.
# Partially parsed test_coa_constructor_creates_frozen_instance. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/13 statements.


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
    var_1 = var_0._accounts
    var_2 = len(var_1)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = var_0._subaccounts
    var_3 = len(var_2)
    assert var_3 == 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '100'
    var_1 = 'My Assets'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/9 statements.
# Partially parsed test_add_returns_existing_account_with_same_properties. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/7 statements.
# Partially parsed test_add_raises_error_when_parent_not_defined. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_different_properties. Retrieved 5/11 statements.
# Partially parsed test_add_updates_subaccounts_buffer. Retrieved 4/11 statements.
# Partially parsed test_add_multiple_subaccounts_to_same_parent. Retrieved 6/19 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Test Account'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'can not be the parent of itself'

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
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'do not match existing chart of accounts member'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.2'
    var_4 = 'First Child'
    var_5 = 'Second Child'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 8/16 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 8/27 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_creates_root_accounts. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_accounts_frozen. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'

def test_case_0():
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '20'
    var_3 = 'Liabilities'
    var_4 = '30'
    var_5 = 'Equity'
    var_6 = '40'
    var_7 = 'Revenue'

def test_case_0():
    var_0 = '100'
    var_1 = 'My Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/13 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 5/6 statements.
# Partially parsed test_coa_constructor_initializes_subaccounts_buffer. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.


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

def test_case_0():
    var_0 = '10'
    var_1 = 'My Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

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
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 2/14 statements.
# Partially parsed test_subaccount_constructor_frozen. Retrieved 2/16 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Test Sub Account'

def test_case_0():
    var_0 = '2000'
    var_1 = 'Frozen Test Account'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Asset'
    var_2 = 'Standard COA'
    var_3 = 'Sub Account 1'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_nodify_returns_node_with_account_and_children. Retrieved 9/47 statements.


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
    var_8 = 'SubAsset'



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.
# Failed to parse test_read_chart_of_accounts_call_empty.
# Failed to parse test_read_chart_of_accounts_call_returns_coa_type.




# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_with_valid_parent_account. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Sub-Account'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 1/11 statements.
# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Failed to parse test_read_chart_of_accounts_callable.


def test_case_0():
    var_0 = 'accounts'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 5/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_accounts_are_root_accounts. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 3/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)

def test_case_0():
    var_0 = '100'
    var_1 = 'Assets'
    var_2 = '200'
    var_3 = 'Liabilities'

def test_case_0():
    var_0 = '999'
    var_1 = 'CustomAssets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 5/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/15 statements.
# Partially parsed test_coa_constructor_accounts_in_accounts_dict. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_preserves_order. Retrieved 6/7 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)

def test_case_0():
    var_0 = '1000'
    var_1 = 'Assets'
    var_2 = '2000'
    var_3 = 'Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = 0
    var_6 = var_2[var_5]

def test_case_0():
    var_0 = '5000'
    var_1 = 'My Assets'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 7/29 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = module_0.COA()
    var_6 = '1'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/31 statements.
# Partially parsed test_subaccount_constructor_frozen. Retrieved 6/33 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = 'Parent Account'
    var_4 = '1001'
    var_5 = 'Sub Account'

def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = 'Parent Account'
    var_4 = '1001'
    var_5 = 'Sub Account'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/18 statements.
# Partially parsed test_subaccount_is_frozen. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'Test COA'
    var_1 = '1000'
    var_2 = 'Parent Account'
    var_3 = 'Asset'
    var_4 = '1001'
    var_5 = 'Sub Account'

def test_case_0():
    var_0 = 'Test COA'
    var_1 = '1000'
    var_2 = 'Parent Account'
    var_3 = 'Asset'
    var_4 = '1001'
    var_5 = 'Sub Account'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 5/6 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_root_accounts_in_buffer. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_default_root_account_names. Retrieved 3/8 statements.
# Partially parsed test_coa_constructor_default_root_account_codes. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/16 statements.
# Partially parsed test_coa_constructor_initializes_subaccounts_dict. Retrieved 4/5 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    var_3 = bool(var_1)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '10'
    var_1 = 'Fixed Assets'
    var_2 = 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = var_0._subaccounts
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/32 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = '1'
    var_4 = 'Assets'
    var_5 = 'Cash'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_nodify_creates_node_with_account_and_children. Retrieved 6/16 statements.
# Partially parsed test_nodify_creates_node_for_leaf_account. Retrieved 4/11 statements.
# Partially parsed test_nodify_creates_nested_tree_structure. Retrieved 7/19 statements.
# Partially parsed test_nodify_root_account_with_no_children. Retrieved 2/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.2'
    var_4 = 'Child Account 1'
    var_5 = 'Child Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Leaf Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.1.1'
    var_4 = 'Child Account'
    var_5 = 'Grandchild Account'
    var_6 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_add_basic_subaccount. Retrieved 4/8 statements.
# Partially parsed test_add_account_same_parent_and_code. Retrieved 3/6 statements.
# Partially parsed test_add_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_duplicate_account_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_duplicate_account_inconsistent_name. Retrieved 5/10 statements.
# Partially parsed test_add_duplicate_account_inconsistent_parent. Retrieved 5/11 statements.
# Partially parsed test_add_multiple_subaccounts. Retrieved 6/14 statements.
# Partially parsed test_add_nested_subaccounts. Retrieved 6/11 statements.
# Partially parsed test_add_account_properties. Retrieved 4/9 statements.


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
    var_2 = 'Test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Test'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Parent account is not'

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
    var_6 = 'do not match existing'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '1.1'
    var_4 = 'Test Account'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'do not match existing'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Account 1'
    var_4 = '1.2'
    var_5 = 'Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.1.1'
    var_4 = 'Child Account'
    var_5 = 'Grandchild Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/18 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 2/13 statements.
# Partially parsed test_coa_constructor_accounts_are_root_accounts. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_creates_ordered_dict. Retrieved 2/3 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 5/6 statements.


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
    var_1 = 'Assets Custom'
    var_2 = '2'
    var_3 = 'Liabilities Custom'

def test_case_0():
    var_0 = '100'
    var_1 = 'My Assets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/29 statements.


def test_case_0():
    var_0 = 'Chart1'
    var_1 = '1000'
    var_2 = 'Assets'
    var_3 = 'Asset'
    var_4 = '1001'
    var_5 = 'Cash'



# Parsed testcases at query #56
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.
# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Failed to parse test_read_chart_of_accounts_call_empty_coa.




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'accounts'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/16 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_accounts_buffer_initialized. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_subaccounts_buffer_initialized. Retrieved 4/5 statements.


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

def test_case_0():
    var_0 = '10'
    var_1 = 'My Assets'
    var_2 = 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = var_0._accounts
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = var_0._subaccounts
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_creates_root_accounts_with_default_codes. Retrieved 3/9 statements.
# Partially parsed test_coa_constructor_creates_root_accounts_with_type_names. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_is_frozen. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_0)
    assert var_5 is True

def test_case_0():
    var_0 = '10'
    var_1 = 'CustomAsset'
    var_2 = '20'
    var_3 = 'CustomLiability'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = {code: account for (code, account) in var_0}
    var_2 = var_0.accounts
    var_3 = [account.name for account in var_2]

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = len(var_1)
    assert var_2 == 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/8 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/16 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 2/14 statements.
# Partially parsed test_coa_constructor_creates_frozen_instance. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 5/7 statements.


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

def test_case_0():
    var_0 = '10'
    var_1 = 'MyAssets'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = bool(False)
    assert var_1 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._subaccounts
    var_2 = len(var_1)
    assert var_2 == 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/31 statements.
# Partially parsed test_subaccount_is_frozen. Retrieved 6/33 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'General'
    var_2 = '1000'
    var_3 = 'Bank'
    var_4 = '1001'
    var_5 = 'Checking Account'

def test_case_0():
    var_0 = 'Asset'
    var_1 = 'General'
    var_2 = '1000'
    var_3 = 'Bank'
    var_4 = '1001'
    var_5 = 'Checking Account'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_add_new_subaccount. Retrieved 4/13 statements.
# Partially parsed test_add_duplicate_account_same_details. Retrieved 4/13 statements.
# Partially parsed test_add_account_parent_not_defined. Retrieved 4/9 statements.
# Partially parsed test_add_account_parent_is_self. Retrieved 3/8 statements.
# Partially parsed test_add_duplicate_account_different_details. Retrieved 7/15 statements.
# Partially parsed test_add_account_appears_in_subaccounts. Retrieved 4/11 statements.
# Partially parsed test_add_nested_accounts. Retrieved 6/14 statements.


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
    var_8 = 'Account name, code and parent do not match'

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
    var_3 = 'Level 1'
    var_4 = '1.1.1'
    var_5 = 'Level 2'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_add_with_valid_parent_account. Retrieved 11/61 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Test that the predicate at line 18 evaluates to False when a valid parent account exists.\n    '
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = module_0.COA()
    var_7 = '1'
    var_8 = '1-1'
    var_9 = 'Test SubAccount'
    var_10 = var_6.add(var_7, var_8, var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True
    var_12 = var_10.code
    var_13 = bool(var_10.code == var_8)
    assert var_13 is True
    var_14 = var_10.name
    var_15 = bool(var_10.name == var_9)
    assert var_15 is True
    var_16 = var_10.parent.code
    var_17 = bool(var_10.parent.code == var_7)
    assert var_17 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_nodify_creates_node_with_account_and_children. Retrieved 4/12 statements.
# Partially parsed test_nodify_creates_node_without_children. Retrieved 2/5 statements.
# Partially parsed test_nodify_creates_nested_structure. Retrieved 7/19 statements.
# Partially parsed test_nodify_returns_node_instance. Retrieved 2/7 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.1.1'
    var_4 = 'Child'
    var_5 = 'Grandchild'
    var_6 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 2/16 statements.
# Partially parsed test_subaccount_constructor_frozen. Retrieved 2/18 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/11 statements.
# Failed to parse test_read_chart_of_accounts_call_is_callable.
# Failed to parse test_read_chart_of_accounts_call_no_arguments.


def test_case_0():
    var_0 = 'accounts'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_new_account. Retrieved 4/13 statements.
# Partially parsed test_add_account_self_parent. Retrieved 4/12 statements.
# Partially parsed test_add_account_nonexistent_parent. Retrieved 5/14 statements.
# Partially parsed test_add_existing_account_same_details. Retrieved 4/13 statements.
# Partially parsed test_add_existing_account_different_details. Retrieved 5/15 statements.
# Partially parsed test_add_multiple_accounts. Retrieved 6/22 statements.
# Partially parsed test_add_nested_accounts. Retrieved 6/18 statements.
# Partially parsed test_add_account_subaccounts_buffer. Retrieved 4/15 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '1.1'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '1.1'
    var_3 = 'Self Parent'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'can not be the parent of itself'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '999'
    var_3 = '999.1'
    var_4 = 'Test Account'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Parent account is not (yet) defined'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '1.1'
    var_3 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '1.1'
    var_3 = 'Cash'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'do not match existing chart of accounts member'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '1.1'
    var_3 = 'Cash'
    var_4 = '1.2'
    var_5 = 'Bank'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '1.1'
    var_3 = 'Current Assets'
    var_4 = '1.1.1'
    var_5 = 'Cash'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '1.1'
    var_3 = 'Cash'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/29 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'General Ledger'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1100'
    var_6 = 'Cash'



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Failed to parse test_read_chart_of_accounts_call_returns_empty_coa.
# Failed to parse test_read_chart_of_accounts_call_multiple_invocations.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_add_existing_account_predicate. Retrieved 9/83 statements.


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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_existing_account_predicate. Retrieved 9/40 statements.


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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/9 statements.
# Partially parsed test_add_returns_existing_account_with_same_properties. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/7 statements.
# Partially parsed test_add_raises_error_when_parent_not_defined. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_different_properties. Retrieved 5/11 statements.
# Partially parsed test_add_updates_subaccounts_buffer. Retrieved 4/11 statements.
# Partially parsed test_add_multiple_subaccounts_to_same_parent. Retrieved 6/15 statements.


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
    var_2 = 'Test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'Test Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Parent account is not (yet) defined'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'do not match existing chart of accounts member'

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
    var_3 = 'First Account'
    var_4 = '1.2'
    var_5 = 'Second Account'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/32 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = 'Parent Account'
    var_4 = '1001'
    var_5 = 'Sub Account'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_to_false. Retrieved 16/46 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 39 evaluates to False when parent already exists in _subaccounts.'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = module_0.COA()
    var_7 = None
    var_8 = '1001'
    var_9 = 'First Sub-Account'
    var_10 = var_7.code
    var_11 = '1002'
    var_12 = 'Second Sub-Account'
    var_13 = var_7.code
    var_14 = bool(var_7 in var_6._subaccounts)
    assert var_14 is True
    var_15 = var_6._subaccounts[var_7]
    var_16 = len(var_15)
    assert var_16 == 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 8/26 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = '1000'
    var_3 = 'General'
    var_4 = module_0.COA(var_3)
    var_5 = '100'
    var_6 = 'Parent'
    var_7 = 'Sub Account'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_existing_account_returns_same_account. Retrieved 9/27 statements.


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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_account_parent_already_in_subaccounts. Retrieved 11/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '\n    Test that the predicate at line 39 evaluates to False when parent is already in _subaccounts.\n    '
    var_1 = module_0.COA()
    var_2 = var_1.accounts
    var_3 = next(var_2)
    var_4 = var_3.code
    var_5 = '1.1'
    var_6 = 'First Sub'
    var_7 = bool(var_3 in var_1._subaccounts)
    assert var_7 is True
    var_8 = '1.2'
    var_9 = 'Second Sub'
    var_10 = var_1.subaccounts(var_3)
    var_11 = len(var_10)
    assert var_11 == 2



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.
# Failed to parse test_read_chart_of_accounts_call_empty.
# Failed to parse test_read_chart_of_accounts_call_returns_coa_type.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'General Ledger'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1001'
    var_6 = 'Cash'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_add_account_parent_already_in_subaccounts. Retrieved 11/51 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = '1'
    var_6 = 'Assets'
    var_7 = '1.1'
    var_8 = '1.2'
    var_9 = 'Current Assets'
    var_10 = 'Fixed Assets'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = '1000'
    var_2 = 'Cash'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/21 statements.
# Partially parsed test_subaccount_constructor_frozen. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'Chart1'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1100'
    var_6 = 'Cash'

def test_case_0():
    var_0 = 'asset'
    var_1 = 'Chart1'
    var_2 = '1000'
    var_3 = 'Assets'
    var_4 = '1100'
    var_5 = 'Cash'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info. Retrieved 9/37 statements.


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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info. Retrieved 9/34 statements.


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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_existing_account_with_matching_parent_name_code. Retrieved 9/41 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'equity'
    var_3 = 'revenue'
    var_4 = 'expense'
    var_5 = module_0.COA()
    var_6 = '1'
    var_7 = '1.1'
    var_8 = 'Current Assets'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info. Retrieved 13/20 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Code'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = module_0.COA()
    var_7 = 0
    var_8 = var_6.accounts
    var_9 = list(var_8)[var_7]
    var_10 = var_9.code
    var_11 = '100'
    var_12 = 'Test Account'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info. Retrieved 9/43 statements.


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
    var_8 = 'Test Sub Account'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/13 statements.
# Partially parsed test_add_returns_existing_account_with_matching_info. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_different_info. Retrieved 5/10 statements.
# Partially parsed test_add_updates_subaccounts_buffer. Retrieved 6/15 statements.
# Partially parsed test_add_nested_subaccounts. Retrieved 6/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1001'
    var_3 = 'Cash'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1001'
    var_3 = 'Cash'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1001'
    var_2 = 'Test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '9999'
    var_2 = '1001'
    var_3 = 'Test'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Parent account is not (yet) defined'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1001'
    var_3 = 'Cash'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'do not match existing chart of accounts member'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1001'
    var_3 = '1002'
    var_4 = 'Cash'
    var_5 = 'Bank'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1001'
    var_3 = 'Current Assets'
    var_4 = '100101'
    var_5 = 'Cash'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 9/28 statements.
# Partially parsed test_subaccount_constructor_frozen. Retrieved 8/27 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = '1000'
    var_3 = 'General Ledger'
    var_4 = module_0.COA(var_3)
    var_5 = '1'
    var_6 = 'Assets'
    var_7 = '1001'
    var_8 = 'Cash'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'asset'
    var_1 = '1'
    var_2 = 'Assets'
    var_3 = 'GL'
    var_4 = module_0.COA(var_3)
    var_5 = '1001'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1001'
    var_3 = 'Cash'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 2/13 statements.
# Partially parsed test_read_chart_of_accounts_call_returns_coa_type. Retrieved 1/10 statements.
# Partially parsed test_read_chart_of_accounts_call_multiple_invocations. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'Test that ReadChartOfAccounts protocol can be called and returns COA'
    var_1 = 'accounts'

def test_case_0():
    var_0 = 'Test that __call__ method returns correct COA type'

def test_case_0():
    var_0 = 'Test that __call__ can be invoked multiple times'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/29 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'General Ledger'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1100'
    var_6 = 'Cash'



# Parsed testcases at query #37
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
    var_1 = 'GAAP COA'
    var_2 = '2000'
    var_3 = 'Accounts Payable'

def test_case_0():
    var_0 = 'Asset'
    var_1 = 'COA'
    var_2 = '1000'
    var_3 = 'Cash'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/13 statements.
# Failed to parse test_read_chart_of_accounts_call_is_callable.
# Partially parsed test_read_chart_of_accounts_call_multiple_invocations. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'accounts'

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = var_1[0]
    assert var_2 == 3



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_add_account_with_inconsistent_information. Retrieved 10/37 statements.


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
    var_8 = 'Original Name'
    var_9 = 'Different Name'
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_add_account_with_inconsistent_information_raises_error. Retrieved 11/45 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = module_0.COA()
    var_6 = None
    var_7 = var_6.code
    var_8 = '100'
    var_9 = 'Original Name'
    var_10 = 'Different Name'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Account name, code and parent do not match'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/9 statements.
# Partially parsed test_add_returns_existing_account_with_same_properties. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_defined. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_different_properties. Retrieved 5/10 statements.
# Partially parsed test_add_populates_subaccounts_buffer. Retrieved 4/10 statements.
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
    var_2 = '1.1'
    var_3 = 'Test Sub-Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'can not be the parent of itself'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '1.1'
    var_3 = 'Test'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Parent account is not (yet) defined'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'do not match existing chart of accounts member'

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
    var_4 = 'First'
    var_5 = 'Second'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_add_account_with_mismatched_parent_raises_error. Retrieved 5/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Account'
    var_4 = '2'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Account name, code and parent do not match existing chart of accounts member.'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_account_with_inconsistent_name_raises_error. Retrieved 10/79 statements.


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
    var_8 = 'Original Name'
    var_9 = 'Different Name'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Account name, code and parent do not match'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_add_account_with_inconsistent_name_raises_error. Retrieved 11/20 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Code'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = module_0.COA()
    var_7 = '1'
    var_8 = '1.1'
    var_9 = 'Original Name'
    var_10 = 'Different Name'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Account name, code and parent do not match'



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.
# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Failed to parse test_read_chart_of_accounts_call_empty.




# Parsed testcases at query #46
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 'liability'
    var_2 = 'General Ledger'
    var_3 = '1000'
    var_4 = 'Assets'
    var_5 = '1001'
    var_6 = 'Cash'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_add_account_with_inconsistent_information. Retrieved 11/20 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'Code'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = module_0.COA()
    var_7 = '1'
    var_8 = '1.1'
    var_9 = 'Original Name'
    var_10 = 'Different Name'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Account name, code and parent do not match existing chart of accounts member.'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/32 statements.


def test_case_0():
    var_0 = 'SUB001'
    var_1 = 'Sub Account Name'
    var_2 = 'Asset'
    var_3 = 'Chart of Accounts'
    var_4 = 'PARENT001'
    var_5 = 'Parent Account'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_add_account_with_inconsistent_name_raises_error. Retrieved 10/69 statements.


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
    var_8 = 'Asset Account'
    var_9 = 'Different Name'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'do not match existing chart of accounts member'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/32 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = '1'
    var_4 = 'Assets'
    var_5 = 'Cash'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 4/27 statements.


def test_case_0():
    var_0 = 'Asset'
    var_1 = 'Standard COA'
    var_2 = '1000'
    var_3 = 'Cash'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'accounts'



