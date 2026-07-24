####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 10/35 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/25 statements.


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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_coa_constructor_with_rootspec. Retrieved 6/13 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_1.toplevel
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_1.structure
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 5/26 statements.


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
    var_4 = '3'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_account_successfully. Retrieved 5/12 statements.
# Partially parsed test_add_account_with_existing_code_consistent_info. Retrieved 5/15 statements.
# Partially parsed test_add_account_with_existing_code_inconsistent_info. Retrieved 7/18 statements.
# Partially parsed test_add_account_with_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_self_as_parent. Retrieved 4/10 statements.


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
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Existing Child'
    var_5 = '1.1'
    var_6 = 'New Child Name'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
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
    var_3 = 'Self Parent Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = '1001'
    var_4 = 'Sub Account'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Parent'
    var_2 = module_0.COA()
    var_3 = '1001'
    var_4 = 'Sub Account'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 9/16 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 5/26 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/21 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 9/14 statements.
# Partially parsed test_coa_constructor_frozen. Retrieved 1/3 statements.


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
    var_4 = None

def test_case_0():
    var_0 = '1'
    var_1 = 'Partial Asset'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = var_1.accounts
    var_6 = enumerate(var_5)
    var_7 = 1
    var_8 = var_1.accounts

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_nodify_with_no_subaccounts. Retrieved 3/7 statements.
# Partially parsed test_nodify_with_subaccounts. Retrieved 5/16 statements.
# Partially parsed test_nodify_with_multiple_subaccounts. Retrieved 7/21 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Child Account 1'
    var_5 = '3'
    var_6 = 'Child Account 2'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_nodify_single_account. Retrieved 3/8 statements.
# Partially parsed test_nodify_account_with_subaccounts. Retrieved 5/16 statements.
# Partially parsed test_nodify_account_with_multiple_subaccounts. Retrieved 7/21 statements.
# Partially parsed test_nodify_account_with_nested_subaccounts. Retrieved 8/26 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'

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
    var_3 = '2'
    var_4 = 'Sub Account 1'
    var_5 = '3'
    var_6 = 'Sub Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'
    var_3 = '2'
    var_4 = 'Sub Account 1'
    var_5 = '3'
    var_6 = 'Sub Account 2'
    var_7 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_add_account_with_undefined_parent_raises_error. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_nodify_returns_correct_node. Retrieved 6/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = None
    var_4 = '2'
    var_5 = 'Child Account'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Test Account'
    var_4 = module_0.COA()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_empty_children_for_leaf_account. Retrieved 4/8 statements.
# Partially parsed test_nodify_returns_node_with_correct_account_and_children_for_parent_account. Retrieved 6/15 statements.
# Partially parsed test_nodify_returns_node_with_correct_account_and_nested_children. Retrieved 9/24 statements.
# Partially parsed test_nodify_returns_node_with_correct_account_for_root_account. Retrieved 4/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent Account'
    var_4 = '3'
    var_5 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent Account'
    var_4 = '3'
    var_5 = 'Child Account'
    var_6 = '4'
    var_7 = 'Grandchild Account'
    var_8 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)
    var_4 = var_3.account
    var_5 = bool(var_3.account == var_2)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 2/9 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'

def test_case_0():
    var_0 = '2'
    var_1 = 'Custom Liability'
    var_2 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_parent_instance_not_none. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Account 2'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 5/26 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/22 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 9/18 statements.


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
    var_4 = None

def test_case_0():
    var_0 = '100'
    var_1 = 'Custom Assets'
    var_2 = '2'
    var_3 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'
    var_6 = '2'
    var_7 = var_1.toplevel
    var_8 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_add_account_with_valid_parent_and_code. Retrieved 6/13 statements.
# Partially parsed test_add_account_with_nonexistent_parent. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_duplicate_code_consistent_info. Retrieved 5/11 statements.
# Partially parsed test_add_account_with_duplicate_code_inconsistent_info. Retrieved 6/13 statements.
# Partially parsed test_add_account_with_self_as_parent. Retrieved 3/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'
    var_5 = []

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account 1'
    var_4 = 'Child Account 2'
    var_5 = 'Parent Account'
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #21
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 is var_0)
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 12/39 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/37 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 6/11 statements.


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
    var_9 = 'Expenses'
    var_10 = 0
    var_11 = 1

def test_case_0():
    var_0 = 'A'
    var_1 = 'Custom Assets'
    var_2 = 'E'
    var_3 = 'Custom Expenses'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = str(var_0)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 10/35 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/26 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 6/12 statements.
# Partially parsed test_coa_constructor_none_rootspec. Retrieved 6/12 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_0)
    assert var_5 is True
    var_6 = str(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_0)
    assert var_5 is True
    var_6 = str(var_2)



# Parsed testcases at query #26
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 == var_0)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_ReadChartOfAccounts___call__. Retrieved 3/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_sub_account_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '100'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = '101'
    var_4 = 'Sub Account'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_COA_constructor_with_default_rootspec. Retrieved 4/5 statements.
# Partially parsed test_COA_constructor_with_custom_rootspec. Retrieved 2/9 statements.
# Partially parsed test_COA_constructor_with_partial_rootspec. Retrieved 2/13 statements.
# Partially parsed test_COA_constructor_with_empty_rootspec. Retrieved 6/9 statements.
# Partially parsed test_COA_constructor_with_none_rootspec. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 5/6 statements.
# Partially parsed test_nodify_returns_node_with_subaccounts_when_present. Retrieved 4/11 statements.
# Partially parsed test_nodify_raises_no_error_for_nonexistent_account. Retrieved 4/9 statements.
# Partially parsed test_nodify_maintains_tree_structure_correctly. Retrieved 8/30 statements.


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

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)
    var_4 = var_3.children
    var_5 = len(var_4)
    assert var_5 == 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = 'Nonexistent'
    var_3 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.2'
    var_4 = '1.1.1'
    var_5 = 'Child 1'
    var_6 = 'Child 2'
    var_7 = 'Grandchild'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_account_with_defined_parent. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Sub Account'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_coa_constructor_without_rootspec. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 10/33 statements.


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
    var_1 = 'Asset'
    var_2 = '2'
    var_3 = 'Liability'
    var_4 = '3'
    var_5 = 'Equity'
    var_6 = '4'
    var_7 = 'Revenue'
    var_8 = '5'
    var_9 = 'Expense'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '100'
    var_1 = 'Cash'
    var_2 = module_0.COA()
    var_3 = '1001'
    var_4 = 'Petty Cash'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_ReadChartOfAccounts_call.




# Parsed testcases at query #36
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 5/15 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 7/27 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 8/16 statements.


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
    var_0 = 'A1'
    var_1 = 'Custom Asset'
    var_2 = 'L1'
    var_3 = 'Custom Liability'
    var_4 = 1
    var_5 = var_0 + var_4
    var_6 = str(var_5)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_0)
    assert var_5 is True
    var_6 = 1
    var_7 = var_4 + var_6
    var_8 = str(var_7)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '100'
    var_1 = 'Parent Account'
    var_2 = 'Main COA'
    var_3 = '101'
    var_4 = 'Sub Account'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_add_account_with_defined_parent. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '2'
    var_4 = 'Child Account'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = 'assets'
    var_1 = 'liabilities'
    var_2 = 1000
    var_3 = 500
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = var_5()
    var_7 = bool(var_6 == var_4)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 3/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 5/26 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 4/22 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 9/18 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 8/17 statements.


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
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'
    var_6 = '2'
    var_7 = var_1.toplevel
    var_8 = None

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = '1'
    var_6 = '2'
    var_7 = var_1.toplevel



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 3/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Main COA'
    var_5 = module_0.COA(var_4)
    var_6 = module_0.COA(var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 6/21 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/15 statements.


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
    var_1 = 'Custom Asset'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Custom Liability'
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = '1'
    var_1 = 'Partial Asset'
    var_2 = (var_0, var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_add_account_with_nonexistent_parent. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 5/15 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 7/29 statements.
# Partially parsed test_coa_constructor_partial_custom_rootspec. Retrieved 3/24 statements.


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
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = '2000'
    var_3 = 'Custom Liability'
    var_4 = 1
    var_5 = var_0 + var_4
    var_6 = str(var_5)

def test_case_0():
    var_0 = '1000'
    var_1 = 'Custom Asset'
    var_2 = 1



# Parsed testcases at query #49
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 == var_0)
    assert var_3 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 6/29 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 4/23 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_1.toplevel
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_1.structure
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 0

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Liabilities'
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = (var_0, var_1)
    var_3 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_1.toplevel
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_1.structure
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 4/5 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/18 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/14 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 4/10 statements.


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
    var_0 = '1'
    var_1 = 'Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = '1'
    var_3 = '2'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_parent_account_not_defined_raises_error. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Test Account'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_nodify_returns_node_instance. Retrieved 3/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root Account'



# Parsed testcases at query #55
#--------------------------




def test_case_0():
    var_0 = 'assets'
    var_1 = 'liabilities'
    var_2 = 1000
    var_3 = 500
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = var_5()
    var_7 = bool(var_6 == var_4)
    assert var_7 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_read_chart_of_accounts_call.




# Parsed testcases at query #2
#--------------------------

# Partially parsed test_COA_constructor_default. Retrieved 5/10 statements.
# Partially parsed test_COA_constructor_custom_rootspec. Retrieved 7/29 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_0)
    assert var_5 is True
    var_6 = bool(var_2)
    assert var_6 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Asset Account'
    var_2 = '2'
    var_3 = 'Liability Account'
    var_4 = '1'
    var_5 = '2'
    var_6 = str(var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 10/44 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 7/37 statements.


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
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '20'
    var_3 = 'Liabilities'
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'parent'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = 'sub'
    var_4 = 'Sub Account'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 5/16 statements.
# Partially parsed test_nodify_returns_node_with_no_children_for_account_without_subaccounts. Retrieved 3/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root'
    var_3 = '2'
    var_4 = 'Sub'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Root'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/24 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/20 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.toplevel
    var_5 = list(var_4)
    var_6 = len(var_5)
    var_7 = bool(var_1)
    assert var_7 is True

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'

def test_case_0():
    var_0 = '100'
    var_1 = 'Partial Asset'
    var_2 = '3'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_add_successfully_adds_new_subaccount. Retrieved 5/12 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 4/10 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_different_properties. Retrieved 7/18 statements.
# Partially parsed test_add_returns_existing_account_when_properties_match. Retrieved 5/15 statements.


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
    var_3 = 'Same Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
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
    var_4 = 'Existing Child'
    var_5 = '1.1'
    var_6 = 'Different Name'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '1.1'
    var_4 = 'Existing Child'



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

# Partially parsed test_nodify_returns_node_with_correct_account_and_empty_children_for_leaf_account. Retrieved 4/8 statements.
# Partially parsed test_nodify_returns_node_with_correct_account_and_children_for_parent_account. Retrieved 8/20 statements.
# Partially parsed test_nodify_returns_node_with_correct_nested_structure. Retrieved 15/20 statements.


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
    var_2 = '1.1'
    var_3 = 'Parent Account'
    var_4 = '1.1.1'
    var_5 = 'Child Account 1'
    var_6 = '1.1.2'
    var_7 = 'Child Account 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Level 1 Account'
    var_6 = '1.1.1'
    var_7 = 'Level 2 Account'
    var_8 = var_0.nodify(var_2)
    var_9 = var_8.account
    var_10 = bool(var_8.account == var_2)
    assert var_10 is True
    var_11 = var_8.children
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_8.children[0].account
    var_14 = 0
    var_15 = var_8.children[var_14]
    var_16 = var_15.children
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = var_8.children[0].children[0].account



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 5/11 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 2/19 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = bool(var_1)
    assert var_4 is True
    var_5 = str(var_2)

def test_case_0():
    var_0 = 'A1'
    var_1 = 'Asset Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_0)
    assert var_5 is True
    var_6 = str(var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/9 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/24 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/20 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 8/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    var_4 = var_0.toplevel
    var_5 = list(var_4)
    var_6 = len(var_5)

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Asset'
    var_2 = '2'
    var_3 = 'Custom Liability'

def test_case_0():
    var_0 = '1'
    var_1 = 'Partial Asset'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = var_1.toplevel
    var_6 = list(var_5)
    var_7 = len(var_6)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 5/26 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 4/22 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 9/18 statements.


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
    var_0 = '1'
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
    var_5 = '1'
    var_6 = '2'
    var_7 = var_1.toplevel
    var_8 = None



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_ReadChartOfAccounts___call___returns_COA_object.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'ACC01'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = 'SUB01'
    var_4 = 'Sub Account'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'Test SubAccount'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_account_with_inconsistent_information. Retrieved 6/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Child Account'
    var_5 = 'Different Name'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_successfully_adds_new_account. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_returns_existing_account_when_details_match. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_existing_account_details_mismatch. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Invalid Account'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'Child Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Original Name'
    var_4 = 'Different Name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 == var_0)
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_add_method_adds_new_account. Retrieved 5/12 statements.
# Partially parsed test_add_method_throws_error_if_parent_equals_code. Retrieved 4/10 statements.
# Partially parsed test_add_method_throws_error_if_parent_not_defined. Retrieved 4/8 statements.
# Partially parsed test_add_method_throws_error_if_account_exists_with_conflicting_data. Retrieved 7/17 statements.
# Partially parsed test_add_method_returns_existing_account_if_data_matches. Retrieved 5/14 statements.


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
    var_3 = 'New Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'New Account'
    var_4 = bool(False)
    assert var_4 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'
    var_5 = '2'
    var_6 = 'New Account'
    var_7 = bool(False)
    assert var_7 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = '2'
    var_4 = 'Existing Account'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 3/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'SubAccount Name'
    var_2 = module_0.COA()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'SubAccount1'
    var_2 = '000'
    var_3 = 'Account1'
    var_4 = module_0.COA()



# Parsed testcases at query #22
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()
    var_3 = bool(var_2 == var_0)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



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

# Partially parsed test_add_successfully_adds_new_account. Retrieved 5/11 statements.
# Partially parsed test_add_raises_error_when_parent_and_code_are_same. Retrieved 4/10 statements.
# Partially parsed test_add_raises_error_when_parent_is_not_defined. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_inconsistent_info. Retrieved 6/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = 'Child Account'
    var_3 = module_0.COA()
    var_4 = 'Parent Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1'
    var_1 = module_0.COA()
    var_2 = 'Parent Account'
    var_3 = 'Same Code Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = 'Child Account'
    var_3 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = 'Child Account'
    var_3 = module_0.COA()
    var_4 = 'Parent Account'
    var_5 = 'Existing Account'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 3/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'SubAccountName'
    var_2 = module_0.COA()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_add_new_account. Retrieved 5/11 statements.
# Partially parsed test_add_existing_account. Retrieved 5/14 statements.
# Partially parsed test_add_account_with_inconsistent_details. Retrieved 6/16 statements.
# Partially parsed test_add_account_with_self_as_parent. Retrieved 3/6 statements.
# Partially parsed test_add_account_with_undefined_parent. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '2'
    var_4 = 'Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '2'
    var_4 = 'Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '2'
    var_4 = 'Child'
    var_5 = 'Inconsistent'
    var_6 = bool(False)
    assert var_6 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Account'
    var_3 = bool(False)
    assert var_3 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'SubAccountName'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_ReadChartOfAccounts___call___returns_COA. Retrieved 3/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0
    var_2 = var_1()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_sub_account_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Sub Account'
    var_2 = '456'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_new_subaccount_successfully. Retrieved 5/10 statements.
# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/14 statements.
# Partially parsed test_add_account_with_self_as_parent_raises_error. Retrieved 3/6 statements.
# Partially parsed test_add_account_with_nonexistent_parent_raises_error. Retrieved 4/8 statements.
# Partially parsed test_add_existing_account_with_conflicting_details_raises_error. Retrieved 6/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'
    var_5 = 'Different Name'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'parent'
    var_1 = 'Parent Account'
    var_2 = module_0.COA()
    var_3 = 'sub'
    var_4 = 'Sub Account'



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_read_chart_of_accounts_call_returns_coa.
# Failed to parse test_read_chart_of_accounts_call_returns_non_empty_coa.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_add_successfully_adds_new_account. Retrieved 5/10 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 4/9 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_inconsistent_info. Retrieved 6/14 statements.
# Partially parsed test_add_returns_existing_account_when_info_matches. Retrieved 5/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent Account'
    var_3 = 'Same Code Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent Account'
    var_4 = 'Existing Child'
    var_5 = 'Different Name'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_sub_account_constructor. Retrieved 7/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Sub Account 1'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Company A'
    var_5 = module_0.COA(var_4)
    var_6 = module_0.COA(var_4)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_ReadChartOfAccounts___call__.




# Parsed testcases at query #39
#--------------------------




import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = 'code'
    var_1 = 'name'
    var_2 = 'Account'
    var_3 = ()
    var_4 = 'type'
    var_5 = 'coa'
    var_6 = {var_4: var_4, var_5: var_5}
    var_7 = type(var_2, var_3, var_6)
    var_8 = var_7()
    var_9 = module_0.SubAccount(var_0, var_1, var_8)
    var_10 = var_9.code
    var_11 = bool(var_9.code == var_0)
    assert var_11 is True
    var_12 = var_9.name
    var_13 = bool(var_9.name == var_1)
    assert var_13 is True
    var_14 = var_9.parent
    var_15 = bool(var_9.parent == var_8)
    assert var_15 is True
    var_16 = var_9.type
    var_17 = bool(var_9.type == var_8.type)
    assert var_17 is True
    var_18 = var_9.coa
    var_19 = bool(var_9.coa == var_8.coa)
    assert var_19 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_add_account_with_mismatched_attributes. Retrieved 7/15 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = '002'
    var_2 = 'Parent Account'
    var_3 = module_0.COA()
    var_4 = module_0.COA()
    var_5 = 'Old Name'
    var_6 = 'New Name'



# Parsed testcases at query #41
#--------------------------

# Failed to parse test___call___returns_COA.
# Failed to parse test___call___returns_different_COA_instances.




