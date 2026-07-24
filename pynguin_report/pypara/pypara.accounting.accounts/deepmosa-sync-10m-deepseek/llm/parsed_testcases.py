####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 == var_0)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nodify_with_subaccounts. Retrieved 5/16 statements.
# Partially parsed test_nodify_without_subaccounts. Retrieved 3/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'
    var_3 = '2'
    var_4 = 'Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 5/26 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/22 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 9/17 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts
    var_5 = var_0.toplevel
    var_6 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Partial Asset'
    var_2 = '2'
    var_3 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = var_1.accounts
    var_6 = var_1.toplevel
    var_7 = None
    var_8 = '1'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_nodify_returns_node_with_account_and_subaccounts. Retrieved 5/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent'
    var_4 = 'Child'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/22 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '1'
    var_1 = 'Asset'
    var_2 = '2'
    var_3 = 'Liability'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_constructor_with_default_values. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.toplevel
    var_5 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Asset Account'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 6/15 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 5/27 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True
    var_5 = var_3.value
    var_6 = str(var_5)

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'
    var_4 = str(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_add_new_account_successfully. Retrieved 5/10 statements.
# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/14 statements.
# Partially parsed test_add_account_with_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_mismatched_details. Retrieved 6/16 statements.
# Partially parsed test_add_account_with_self_as_parent. Retrieved 3/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Child Account'
    var_5 = 'Different Name'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Account'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '001'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_add_account_with_nonexistent_parent. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = 'nonexistent'
    var_2 = 'child'
    var_3 = 'Child Account'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 6/8 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 5/27 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_0.toplevel
    var_5 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Balance Sheet'
    var_2 = '2'
    var_3 = 'Income Statement'
    var_4 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 5/15 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/29 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True
    var_5 = 1

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'
    var_4 = 1
    var_5 = var_0 + var_4
    var_6 = str(var_5)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 9/17 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 6/32 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/23 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 10/18 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts
    var_5 = var_0.accounts
    var_6 = enumerate(var_5)
    var_7 = 1
    var_8 = var_0.accounts

def test_case_0():
    var_0 = 'A'
    var_1 = 'Custom Asset'
    var_2 = 'L'
    var_3 = 'Custom Liability'
    var_4 = 2
    var_5 = 1

def test_case_0():
    var_0 = 'E'
    var_1 = 'Custom Expense'
    var_2 = 3
    var_3 = 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = var_1.accounts
    var_6 = var_1.accounts
    var_7 = enumerate(var_6)
    var_8 = 1
    var_9 = var_1.accounts



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'parent_code'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = 'sub_code'
    var_4 = 'Sub Account'



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_new_account_successfully. Retrieved 5/11 statements.
# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/14 statements.
# Partially parsed test_add_existing_account_with_mismatching_details. Retrieved 6/16 statements.
# Partially parsed test_add_account_with_parent_as_itself. Retrieved 4/10 statements.
# Partially parsed test_add_account_with_non_existent_parent. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'
    var_5 = 'Different Name'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = 'Self Parent Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'New Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_coa_constructor_with_rootspec. Retrieved 6/12 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0._accounts
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_0._subaccounts
    var_4 = len(var_3)
    assert var_4 == 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Liabilities'
    var_5 = (var_3, var_4)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1._accounts
    var_3 = len(var_2)
    assert var_3 == 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = (var_0, var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_add_account_parent_in_subaccounts_buffer. Retrieved 5/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Child Account'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Parent'
    var_2 = module_0.COA()
    var_3 = '1100'
    var_4 = 'Sub Account'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 4/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_method_creates_new_account. Retrieved 5/13 statements.
# Partially parsed test_add_method_returns_existing_account. Retrieved 5/15 statements.
# Partially parsed test_add_method_raises_error_for_self_parent. Retrieved 4/10 statements.
# Partially parsed test_add_method_raises_error_for_invalid_parent. Retrieved 4/8 statements.
# Partially parsed test_add_method_raises_error_for_inconsistent_account_info. Retrieved 7/18 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = 'Invalid Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '2'
    var_3 = 'New Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'
    var_5 = '2'
    var_6 = 'Different Name'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Account 1'
    var_3 = None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Parent'
    var_2 = module_0.COA()
    var_3 = '1001'
    var_4 = 'Sub Account'



# Parsed testcases at query #26
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 == var_0)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = var_1()
    var_4 = bool(var_2 != var_3)
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 8/22 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Account Name'
    var_4 = 'Parent Account'
    var_5 = '3'
    var_6 = 'Different Parent Account'
    var_7 = '3'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_account_with_inconsistent_information. Retrieved 6/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Account 2'
    var_4 = 'Parent Account'
    var_5 = 'Different Name'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_SubAccount_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'Test SubAccount'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 9/17 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 6/29 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/21 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts
    var_5 = var_0.accounts
    var_6 = enumerate(var_5)
    var_7 = 1
    var_8 = var_0.accounts

def test_case_0():
    var_0 = 'A'
    var_1 = 'Custom Asset'
    var_2 = 'L'
    var_3 = 'Custom Liability'
    var_4 = 2
    var_5 = 1

def test_case_0():
    var_0 = 'E'
    var_1 = 'Custom Equity'
    var_2 = 2
    var_3 = 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_COA_constructor_default. Retrieved 6/8 statements.
# Partially parsed test_COA_constructor_with_rootspec. Retrieved 5/25 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_0.toplevel
    var_5 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Balance Sheet'
    var_2 = '2'
    var_3 = 'Income Statement'
    var_4 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children_for_parent_account. Retrieved 4/11 statements.
# Partially parsed test_nodify_returns_node_with_correct_account_and_nested_children. Retrieved 7/19 statements.
# Partially parsed test_nodify_raises_no_error_for_account_with_multiple_children. Retrieved 6/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)
    var_4 = var_3.account
    var_5 = bool(var_3.account == var_2)
    assert var_5 is True
    var_6 = var_3.children
    var_7 = bool(var_3.children == [])
    assert var_7 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = 'Child Account'
    var_5 = 'Grandchild Account'
    var_6 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '3'
    var_4 = 'Child Account 1'
    var_5 = 'Child Account 2'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 5/8 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/22 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/18 statements.
# Partially parsed test_coa_constructor_accounts_frozen. Retrieved 8/13 statements.
# Partially parsed test_coa_constructor_toplevel_accounts. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = '1'
    var_4 = var_2[0].code
    var_5 = '2'
    var_6 = var_2[1].code
    var_7 = '3'
    var_8 = var_2[2].code
    var_9 = '4'
    var_10 = var_2[3].code
    var_11 = '5'
    var_12 = var_2[4].code

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/33 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 4/23 statements.


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
    var_9 = 'Expenses'

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'
    var_3 = 'Liabilities'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 13/46 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 8/41 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.accounts
    var_5 = var_0.toplevel
    var_6 = None

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
    var_10 = 0
    var_11 = 1
    var_12 = None

def test_case_0():
    var_0 = 'A'
    var_1 = 'Custom Assets'
    var_2 = 'E'
    var_3 = 'Custom Expenses'
    var_4 = '2'
    var_5 = '3'
    var_6 = '4'
    var_7 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 4/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_constructor_with_rootspec. Retrieved 6/19 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = var_0.toplevel
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = var_0.structure
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Liabilities'
    var_5 = (var_3, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_add_new_account_to_coa. Retrieved 5/13 statements.
# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/16 statements.
# Partially parsed test_add_account_with_invalid_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_self_as_parent. Retrieved 4/12 statements.
# Partially parsed test_add_account_with_mismatched_details. Retrieved 8/19 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'New Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = 'New Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'
    var_5 = '2'
    var_6 = 'Different Name'
    var_7 = var_0.add(var_1, var_2, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test___call___returns_COA_object. Retrieved 3/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 == var_0)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info. Retrieved 5/14 statements.
# Partially parsed test_add_existing_account_with_mismatched_info. Retrieved 6/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Sub Account'
    var_5 = 'Different Name'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Child Account'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_account_parent_already_in_subaccounts. Retrieved 5/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent'
    var_4 = 'Child'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_add_successfully_adds_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_conflicting_info. Retrieved 5/10 statements.
# Partially parsed test_add_returns_existing_account_when_info_matches. Retrieved 4/8 statements.


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
    var_2 = 'Invalid'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Invalid'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original'
    var_4 = 'Conflicting'
    var_5 = bool(False)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Sub Account'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '100'
    var_1 = 'Parent'
    var_2 = module_0.COA()
    var_3 = '101'
    var_4 = 'Sub'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_returns_existing_account_if_consistent. Retrieved 4/10 statements.
# Partially parsed test_add_raises_error_for_self_parent. Retrieved 3/7 statements.
# Partially parsed test_add_raises_error_for_undefined_parent. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_for_inconsistent_existing_account. Retrieved 6/12 statements.


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
    var_1 = '1'
    var_2 = 'Invalid'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Invalid'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Test Subaccount'
    var_4 = '1.1'
    var_5 = 'Different Name'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_add_account_with_inconsistent_info_raises_error. Retrieved 6/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent'
    var_4 = 'Child'
    var_5 = 'Different Name'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_add_successfully_adds_new_account. Retrieved 6/15 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 4/10 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_conflicting_details. Retrieved 7/18 statements.
# Partially parsed test_add_returns_existing_account_when_details_match. Retrieved 5/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Child Account'
    var_5 = []

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = 'Same Code'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Child Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Existing'
    var_5 = '1.1'
    var_6 = 'Different Name'
    var_7 = bool(False)
    assert var_7 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Existing'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_ReadChartOfAccounts___call__.




# Parsed testcases at query #27
#--------------------------

# Failed to parse test_ReadChartOfAccounts___call__.




# Parsed testcases at query #28
#--------------------------

# Failed to parse test_ReadChartOfAccounts___call__.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_method_parent_and_code_same. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = {}
    var_2 = var_0.__post_init__(var_1)
    var_3 = '1'
    var_4 = 'Test Account'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = '1001'
    var_4 = 'Sub Account'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_creates_new_subaccount_when_valid_input. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_returns_existing_account_when_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_existing_account_inconsistent. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'New Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Invalid'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Invalid'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original'
    var_4 = 'Different'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '001'
    var_3 = 'Parent Account'
    var_4 = 'COA1'
    var_5 = module_0.COA(var_4)
    var_6 = module_0.COA(var_4)



# Parsed testcases at query #34
#--------------------------

# Failed to parse test___call___returns_COA.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Sub Account'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_read_chart_of_accounts_call. Retrieved 2/5 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = []



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 5/12 statements.
# Partially parsed test_add_returns_existing_account_if_matches. Retrieved 5/15 statements.
# Partially parsed test_add_raises_error_for_self_parent. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_for_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_for_inconsistent_existing_account. Retrieved 9/22 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = 'Invalid'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1.1'
    var_3 = 'Child'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent1'
    var_4 = 'Parent2'
    var_5 = '1.1'
    var_6 = 'Old Name'
    var_7 = '1.1'
    var_8 = 'New Name'
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'SubAccountName'
    var_2 = '456'
    var_3 = 'ParentAccountName'
    var_4 = 'COA001'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #40
#--------------------------

# Failed to parse test___call___returns_COA.
# Failed to parse test___call___returns_different_COA_instances.




# Parsed testcases at query #41
#--------------------------

# Partially parsed test_sub_account_constructor. Retrieved 7/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'SubAccountName'
    var_2 = '456'
    var_3 = 'ParentAccountName'
    var_4 = 'COAName'
    var_5 = module_0.COA(var_4)
    var_6 = module_0.COA(var_4)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_add_account_with_inconsistent_information. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = 'Account 1'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()
    var_5 = module_0.COA()
    var_6 = 'Different Name'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_account_with_mismatched_information. Retrieved 6/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = '2'
    var_5 = 'Different Name'



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.




# Parsed testcases at query #45
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'SubAccountName'
    var_2 = '456'
    var_3 = 'ParentAccountName'
    var_4 = module_0.COA()



