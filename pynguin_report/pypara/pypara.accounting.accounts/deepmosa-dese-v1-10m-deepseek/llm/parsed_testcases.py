####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/19 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_none_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_structure_property. Retrieved 4/9 statements.
# Partially parsed test_coa_constructor_find_method. Retrieved 4/10 statements.
# Partially parsed test_coa_constructor_subaccounts_method. Retrieved 1/5 statements.
# Partially parsed test_coa_constructor_nodify_method. Retrieved 1/5 statements.
# Partially parsed test_coa_constructor_iter_method. Retrieved 3/6 statements.
# Partially parsed test_coa_constructor_frozen_dataclass. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_0.accounts
    var_5 = {account.type for account in var_4}
    var_6 = str(var_2)

def test_case_0():
    var_0 = '1'
    var_1 = 'Custom Balance'
    var_2 = '2'
    var_3 = 'Custom Income'

def test_case_0():
    var_0 = 'A'
    var_1 = 'My Balance'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.structure
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '999'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/28 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/19 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_rootspec_none. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_accounts_are_frozen. Retrieved 4/7 statements.
# Partially parsed test_coa_constructor_structure_property. Retrieved 4/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = [code for (code, _) in var_0]
    var_5 = '1'
    var_6 = '2'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Balance'
    var_2 = '20'
    var_3 = 'Custom Income'

def test_case_0():
    var_0 = '99'
    var_1 = 'Partial Balance'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.structure
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 14/18 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/23 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/17 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 6/12 statements.
# Partially parsed test_coa_constructor_rootspec_none. Retrieved 6/12 statements.
# Partially parsed test_coa_constructor_frozen_dataclass. Retrieved 1/3 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = [code for (code, _) in var_0]
    var_5 = '1'
    var_6 = '2'
    var_7 = var_0.toplevel
    var_8 = list(var_7)
    var_9 = var_0.accounts
    var_10 = list(var_9)
    var_11 = var_0.structure
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 2

def test_case_0():
    var_0 = 'A'
    var_1 = 'Asset Account'
    var_2 = 'E'
    var_3 = 'Equity Account'

def test_case_0():
    var_0 = '10'
    var_1 = 'Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_read_chart_of_accounts_returns_coa. Retrieved 1/4 statements.
# Partially parsed test_read_chart_of_accounts_called_without_arguments. Retrieved 1/5 statements.
# Partially parsed test_read_chart_of_accounts_returns_different_coa_instances. Retrieved 2/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_returns_existing_account_when_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_account_exists_with_inconsistent_details. Retrieved 5/11 statements.


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
    var_3 = 'Existing Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing Account'
    var_4 = '2'



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/19 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/15 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_rootspec_none. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_0.accounts
    var_5 = {account.type for account in var_4}
    var_6 = str(var_2)

def test_case_0():
    var_0 = '1'
    var_1 = 'BalanceSheet'
    var_2 = '2'
    var_3 = 'IncomeStatement'

def test_case_0():
    var_0 = 'A'
    var_1 = 'Assets'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_add_account_parent_not_defined. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '2'
    var_2 = '21'
    var_3 = 'Sub Account'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_nodify_returns_node_with_children_for_account_with_subaccounts. Retrieved 9/11 statements.
# Partially parsed test_nodify_returns_node_with_nested_children. Retrieved 15/20 statements.
# Partially parsed test_nodify_raises_no_error_for_account_not_in_coa. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'
    var_6 = var_0.nodify(var_2)
    var_7 = var_6.children
    var_8 = len(var_7)
    assert var_8 == 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'
    var_6 = '1.1.1'
    var_7 = 'Sub Sub Account'
    var_8 = var_0.nodify(var_2)
    var_9 = var_8.children
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8.children[var_11]
    var_13 = var_12.children
    var_14 = len(var_13)
    assert var_14 == 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = '999'
    var_4 = 'Fake'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 6/31 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/21 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 7/17 statements.
# Partially parsed test_coa_constructor_none_rootspec. Retrieved 7/17 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = [code for (code, _) in var_0]
    var_5 = '1'
    var_6 = '2'

def test_case_0():
    var_0 = '10'
    var_1 = 'Custom Balance'
    var_2 = '20'
    var_3 = 'Custom Income'
    var_4 = '1'
    var_5 = '2'

def test_case_0():
    var_0 = '99'
    var_1 = 'Partial Balance'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = '1'
    var_6 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = '1'
    var_6 = '2'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_nodify_returns_node_with_children. Retrieved 3/11 statements.
# Partially parsed test_nodify_returns_node_with_nested_children. Retrieved 6/20 statements.
# Partially parsed test_nodify_for_account_without_children. Retrieved 4/6 statements.
# Partially parsed test_nodify_for_account_with_multiple_children. Retrieved 5/17 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Child Account'
    var_3 = '1.1.1'
    var_4 = 'Grandchild Account'
    var_5 = 0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Child 1'
    var_3 = '1.2'
    var_4 = 'Child 2'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account_and_children. Retrieved 7/25 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Test COA'
    var_2 = '1'
    var_3 = 'Root'
    var_4 = '1.1'
    var_5 = 'Sub Account'
    var_6 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/19 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/12 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 8/10 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 8/10 statements.
# Partially parsed test_coa_constructor_rootspec_default_name_capitalization. Retrieved 3/7 statements.
# Partially parsed test_coa_constructor_accounts_are_root_accounts. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_structure_property. Retrieved 4/9 statements.
# Partially parsed test_coa_constructor_find_existing_account. Retrieved 2/5 statements.
# Partially parsed test_coa_constructor_find_non_existing_account. Retrieved 2/4 statements.
# Partially parsed test_coa_constructor_subaccounts_empty_initially. Retrieved 2/5 statements.
# Partially parsed test_coa_constructor_nodify_root_account. Retrieved 2/5 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = [code for (code, _) in var_0]
    var_5 = '1'
    var_6 = '2'

def test_case_0():
    var_0 = '10'
    var_1 = 'MyBalance'
    var_2 = '20'
    var_3 = 'MyIncome'

def test_case_0():
    var_0 = '100'
    var_1 = 'CustomBalance'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = [code for (code, _) in var_1]
    var_6 = '1'
    var_7 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = [code for (code, _) in var_1]
    var_6 = '1'
    var_7 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.structure
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_nodify_returns_node_with_correct_account. Retrieved 5/6 statements.
# Partially parsed test_nodify_returns_node_with_subaccount_children. Retrieved 9/11 statements.
# Partially parsed test_nodify_returns_nested_structure_for_multiple_levels. Retrieved 15/20 statements.
# Partially parsed test_nodify_handles_account_with_multiple_subaccounts. Retrieved 12/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)
    var_4 = var_3.children

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'
    var_6 = var_0.nodify(var_2)
    var_7 = var_6.children
    var_8 = len(var_7)
    assert var_8 == 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account 1'
    var_6 = '1.1.1'
    var_7 = 'Sub Account 2'
    var_8 = var_0.nodify(var_2)
    var_9 = var_8.children
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8.children[var_11]
    var_13 = var_12.children
    var_14 = len(var_13)
    assert var_14 == 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account 1'
    var_6 = var_2.code
    var_7 = '1.2'
    var_8 = 'Sub Account 2'
    var_9 = var_0.nodify(var_2)
    var_10 = var_9.children
    var_11 = len(var_10)
    assert var_11 == 2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 1/4 statements.
# Partially parsed test___call___returns_coa_with_accounts. Retrieved 2/10 statements.
# Partially parsed test___call___returns_empty_coa. Retrieved 2/7 statements.
# Partially parsed test___call___invoked_without_arguments. Retrieved 1/5 statements.
# Partially parsed test___call___returns_different_coa_instances. Retrieved 2/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details_returns_account. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_coa_constructor_with_default_rootspec. Retrieved 6/9 statements.
# Partially parsed test_coa_constructor_with_custom_rootspec. Retrieved 4/19 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/15 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_with_none_rootspec. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_accounts_are_root_accounts. Retrieved 1/3 statements.
# Partially parsed test_coa_constructor_structure_property. Retrieved 4/11 statements.
# Partially parsed test_coa_constructor_frozen_dataclass. Retrieved 4/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_0.accounts
    var_5 = {account.type for account in var_4}

def test_case_0():
    var_0 = '1'
    var_1 = 'MyAsset'
    var_2 = '2'
    var_3 = 'MyLiability'

def test_case_0():
    var_0 = 'A'
    var_1 = 'CustomAsset'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = '1'
    var_6 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = '1'
    var_6 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.structure
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_subaccount_parent_not_in_subaccounts. Retrieved 4/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/22 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 4/21 statements.
# Partially parsed test_coa_constructor_rootspec_empty_dict. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_frozen_dataclass. Retrieved 1/5 statements.
# Partially parsed test_coa_constructor_accounts_iterable. Retrieved 7/9 statements.
# Partially parsed test_coa_constructor_structure_property. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = [code for (code, _) in var_0]
    var_5 = '1'
    var_6 = '2'

def test_case_0():
    var_0 = '1000'
    var_1 = 'CustomAsset'
    var_2 = '2000'
    var_3 = 'CustomLiability'

def test_case_0():
    var_0 = '300'
    var_1 = 'MyEquity'
    var_2 = '1'
    var_3 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = [acc for (_, acc) in var_0]
    var_4 = len(var_2)
    var_5 = len(var_3)
    var_6 = zip(var_2, var_3)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.structure
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test___call___returns_coa. Retrieved 1/4 statements.
# Partially parsed test___call___raises_exception_on_failure. Retrieved 1/5 statements.
# Partially parsed test___call___returns_different_coa_instances. Retrieved 2/8 statements.
# Partially parsed test___call___can_be_called_multiple_times. Retrieved 1/5 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

def test_case_0():
    var_0 = 'Read error'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info_returns_account. Retrieved 3/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Sub Account'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)
    var_6 = module_0.COA(var_4)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 2/4 statements.
# Partially parsed test___call___returns_different_coa_instances. Retrieved 3/10 statements.
# Partially parsed test___call___raises_exception. Retrieved 3/9 statements.
# Partially parsed test___call___returns_coa_with_expected_attributes. Retrieved 6/12 statements.
# Partially parsed test___call___can_be_called_multiple_times. Retrieved 1/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = lambda : var_0

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()
    var_2 = 0

def test_case_0():
    var_0 = ()
    var_1 = 'Test error'
    var_2 = ValueError(var_1)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Accounts Payable'
    var_4 = module_0.COA()
    var_5 = lambda : var_4

def test_case_0():
    var_0 = 0
    assert var_0 == 3



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '101'
    var_1 = 'Cash'
    var_2 = '100'
    var_3 = 'Assets'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info_returns_account. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_add_new_account_successfully. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account_with_matching_details. Retrieved 4/9 statements.
# Partially parsed test_add_existing_account_with_mismatched_parent. Retrieved 5/11 statements.
# Partially parsed test_add_existing_account_with_mismatched_name. Retrieved 5/10 statements.
# Partially parsed test_add_account_with_parent_as_self. Retrieved 3/6 statements.
# Partially parsed test_add_account_with_nonexistent_parent. Retrieved 4/8 statements.


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
    var_3 = 'Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '1.1'
    var_4 = 'Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'
    var_4 = 'Different Name'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Self Parent'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_add_new_subaccount_successfully. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account_with_matching_details. Retrieved 4/10 statements.
# Partially parsed test_add_raises_error_when_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_when_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_when_existing_account_mismatch. Retrieved 5/10 statements.


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
    var_3 = 'Existing Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Same Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing Sub Account'
    var_4 = 'Different Name'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 7/14 statements.
# Partially parsed test_add_existing_account_with_different_name. Retrieved 5/10 statements.
# Partially parsed test_add_existing_account_with_different_code. Retrieved 7/14 statements.
# Partially parsed test_add_existing_account_with_same_attributes. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child'
    var_4 = '3'
    var_5 = 'Different Parent'
    var_6 = 'Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child'
    var_4 = 'Different Name'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child'
    var_4 = '3'
    var_5 = 'Different Child'
    var_6 = 'Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child'



# Parsed testcases at query #37
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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent_raises_error. Retrieved 4/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'SubAccount'
    var_3 = '2'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 10/19 statements.
# Partially parsed test_add_existing_account_with_different_name. Retrieved 6/13 statements.
# Partially parsed test_add_existing_account_with_different_code. Retrieved 6/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'SubAccount1'
    var_4 = '1.2'
    var_5 = 'SubAccount2'
    var_6 = '1.2'
    var_7 = '1.1'
    var_8 = 'SubAccount1'
    var_9 = var_0.add(var_2, var_3, var_8)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'SubAccount1'
    var_4 = '1.1'
    var_5 = 'DifferentName'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'SubAccount1'
    var_4 = '1.2'
    var_5 = 'SubAccount1'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 7/14 statements.
# Partially parsed test_add_existing_account_with_different_name. Retrieved 5/10 statements.
# Partially parsed test_add_existing_account_with_different_code. Retrieved 6/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = '3'
    var_5 = 'Different Parent'
    var_6 = 'Child Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Different Name'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = '3'
    var_5 = 'Child Account'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 2/4 statements.
# Partially parsed test___call___returns_new_coa_each_call. Retrieved 2/9 statements.
# Partially parsed test___call___raises_exception_on_failure. Retrieved 4/9 statements.
# Partially parsed test___call___returns_coa_with_expected_attributes. Retrieved 4/7 statements.
# Partially parsed test___call___can_be_called_via_instance. Retrieved 3/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.ReadChartOfAccounts()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()
    var_1 = 0
    assert var_1 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()
    var_1 = ()
    var_2 = 'Read error'
    var_3 = ValueError(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = 'Asset'
    var_2 = 'Liability'
    var_3 = module_0.ReadChartOfAccounts()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.ReadChartOfAccounts()
    var_2 = var_1.__call__()



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent_raises_error. Retrieved 8/16 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Parent'
    var_4 = 'Child'
    var_5 = '3'
    var_6 = 'Different Parent'
    var_7 = 'Child'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 1/4 statements.
# Failed to parse test___call___returns_valid_coa.
# Partially parsed test___call___invoked_without_arguments. Retrieved 1/5 statements.
# Partially parsed test___call___returns_different_coa_instances. Retrieved 2/6 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_add_new_account_successfully. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account_with_matching_data. Retrieved 4/9 statements.
# Partially parsed test_add_account_with_parent_equal_to_code_raises_error. Retrieved 3/6 statements.
# Partially parsed test_add_account_with_nonexistent_parent_raises_error. Retrieved 4/8 statements.
# Partially parsed test_add_account_with_existing_code_but_mismatched_data_raises_error. Retrieved 5/10 statements.
# Partially parsed test_add_account_updates_subaccounts_buffer. Retrieved 4/8 statements.
# Partially parsed test_add_account_creates_subaccount_instance. Retrieved 4/8 statements.
# Partially parsed test_add_multiple_subaccounts_to_same_parent. Retrieved 6/16 statements.
# Partially parsed test_add_account_ensures_accounts_buffer_updated. Retrieved 4/7 statements.


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
    var_3 = 'Existing Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Account Name'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing Sub Account'
    var_4 = 'Different Name'

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
    var_3 = 'Sub Account'

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
    var_3 = 'Sub Account'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_with_default_rootspec. Retrieved 6/8 statements.
# Partially parsed test_constructor_with_custom_rootspec. Retrieved 4/27 statements.
# Partially parsed test_constructor_with_partial_rootspec. Retrieved 3/18 statements.
# Partially parsed test_constructor_accounts_are_frozen. Retrieved 3/6 statements.
# Partially parsed test_constructor_structure_property. Retrieved 4/7 statements.
# Partially parsed test_constructor_iter_method. Retrieved 3/5 statements.
# Partially parsed test_constructor_find_method. Retrieved 3/8 statements.
# Partially parsed test_constructor_subaccounts_method. Retrieved 5/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_0.accounts
    var_5 = {a.type for a in var_4}

def test_case_0():
    var_0 = '10'
    var_1 = 'CustomBalance'
    var_2 = '20'
    var_3 = 'CustomIncome'

def test_case_0():
    var_0 = '99'
    var_1 = 'OnlyBalance'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.structure
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '999'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '999'
    var_3 = 'Dummy'
    var_4 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_account_with_undefined_parent_raises_error. Retrieved 4/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '1'
    var_3 = 'Child Account'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_coa_constructor_default. Retrieved 7/11 statements.
# Partially parsed test_coa_constructor_with_rootspec. Retrieved 4/19 statements.
# Partially parsed test_coa_constructor_with_partial_rootspec. Retrieved 3/15 statements.
# Partially parsed test_coa_constructor_with_empty_rootspec. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_rootspec_none. Retrieved 5/7 statements.
# Partially parsed test_coa_constructor_structure_property. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = [code for (code, _) in var_0]
    var_5 = '1'
    var_6 = '2'

def test_case_0():
    var_0 = '10'
    var_1 = 'BalanceRoot'
    var_2 = '20'
    var_3 = 'IncomeRoot'

def test_case_0():
    var_0 = '100'
    var_1 = 'CustomBalance'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.structure
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_add_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_existing_account_inconsistent. Retrieved 5/10 statements.
# Partially parsed test_add_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_self_parent. Retrieved 3/6 statements.
# Partially parsed test_add_multiple_subaccounts. Retrieved 6/14 statements.
# Partially parsed test_add_nested_subaccount. Retrieved 6/12 statements.


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
    var_2 = '1.1'
    var_3 = 'Existing Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing Subaccount'
    var_4 = 'Different Name'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'New Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Self Parent'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Subaccount 1'
    var_4 = '1.2'
    var_5 = 'Subaccount 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = '1.1.1'
    var_4 = 'Child'
    var_5 = 'Grandchild'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_coa_constructor_default_rootspec. Retrieved 7/12 statements.
# Partially parsed test_coa_constructor_custom_rootspec. Retrieved 4/23 statements.
# Partially parsed test_coa_constructor_partial_rootspec. Retrieved 3/19 statements.
# Partially parsed test_coa_constructor_empty_rootspec. Retrieved 6/10 statements.
# Partially parsed test_coa_constructor_none_rootspec. Retrieved 6/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.accounts
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_0.accounts
    var_5 = {a.type for a in var_4}
    var_6 = str(var_2)

def test_case_0():
    var_0 = 'B'
    var_1 = 'Custom Balance'
    var_2 = 'I'
    var_3 = 'Custom Income'

def test_case_0():
    var_0 = '1'
    var_1 = 'My Balance'
    var_2 = '2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.COA(rootspec=var_0)
    var_2 = var_1.accounts
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = str(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_nodify_returns_node_with_children_for_account_with_subaccounts. Retrieved 9/11 statements.
# Partially parsed test_nodify_returns_node_with_nested_children. Retrieved 15/20 statements.
# Partially parsed test_nodify_returns_node_with_multiple_children. Retrieved 12/16 statements.
# Partially parsed test_nodify_raises_no_error_for_account_not_in_coa. Retrieved 5/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_0.nodify(var_2)

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'
    var_6 = var_0.nodify(var_2)
    var_7 = var_6.children
    var_8 = len(var_7)
    assert var_8 == 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account 1'
    var_6 = '1.1.1'
    var_7 = 'Sub Account 2'
    var_8 = var_0.nodify(var_2)
    var_9 = var_8.children
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8.children[var_11]
    var_13 = var_12.children
    var_14 = len(var_13)
    assert var_14 == 1

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account 1'
    var_6 = var_2.code
    var_7 = '1.2'
    var_8 = 'Sub Account 2'
    var_9 = var_0.nodify(var_2)
    var_10 = var_9.children
    var_11 = len(var_10)
    assert var_11 == 2

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = '999'
    var_4 = 'Fake'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_add_account_parent_not_in_subaccounts. Retrieved 6/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test___call___returns_coa. Retrieved 1/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 1/4 statements.
# Failed to parse test___call___returns_valid_coa.
# Failed to parse test___call___invoked_without_arguments.
# Partially parsed test___call___returns_different_coa_on_multiple_calls. Retrieved 2/8 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'



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

# Partially parsed test_add_existing_account_with_matching_info_returns_account. Retrieved 6/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = var_0.toplevel
    var_2 = next(var_1)
    var_3 = var_2.code
    var_4 = '1.1'
    var_5 = 'Sub Account'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details_returns_account. Retrieved 4/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 1/4 statements.
# Partially parsed test___call___returns_correct_coa_data. Retrieved 4/11 statements.
# Failed to parse test___call___is_callable.
# Partially parsed test___call___raises_exception_on_failure. Retrieved 1/5 statements.
# Partially parsed test___call___returns_empty_coa. Retrieved 2/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Revenue'

def test_case_0():
    var_0 = 'Read error'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.COA(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_returns_existing_account_if_matches. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_if_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_if_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_if_code_exists_with_different_parent. Retrieved 5/11 statements.
# Partially parsed test_add_raises_error_if_code_exists_with_different_name. Retrieved 5/10 statements.


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
    var_2 = '1.1'
    var_3 = 'Existing Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Same Code'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '1.1'
    var_4 = 'Same Name'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'First Name'
    var_4 = 'Different Name'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent. Retrieved 8/16 statements.
# Partially parsed test_add_existing_account_with_different_name. Retrieved 6/12 statements.
# Partially parsed test_add_existing_account_with_different_code. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '2'
    var_4 = 'Child'
    var_5 = '3'
    var_6 = 'Different Parent'
    var_7 = 'Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '2'
    var_4 = 'Child'
    var_5 = 'Different Child'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Parent'
    var_3 = '2'
    var_4 = 'Child'
    var_5 = '3'
    var_6 = 'Child'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info. Retrieved 4/9 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent_raises_error. Retrieved 7/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child'
    var_4 = '3'
    var_5 = 'Different Parent'
    var_6 = 'Child'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_existing_account_with_matching_info_returns_account. Retrieved 4/13 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Sub Account'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_account_with_existing_code_but_different_parent_raises_error. Retrieved 5/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = '1.1'
    var_4 = 'Child Account'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 1/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 7/12 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA(var_4)
    var_6 = module_0.COA(var_4)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_add_existing_account_with_different_parent_raises_error. Retrieved 4/14 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1.1'
    var_2 = 'Test Account'
    var_3 = '2'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_add_existing_account_with_matching_details. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '2'
    var_3 = 'Child Account'
    var_4 = 'Parent Account'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_read_chart_of_accounts_call_returns_coa. Retrieved 1/4 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_add_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_existing_account_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_inconsistent_existing_account. Retrieved 5/11 statements.


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
    var_3 = 'Existing Sub Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Same Code'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing Sub Account'
    var_4 = '2'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 5/10 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = module_0.COA()



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_returns_existing_account_if_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_if_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_if_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_if_account_exists_with_inconsistent_info. Retrieved 5/11 statements.


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
    var_2 = '1.1'
    var_3 = 'Existing Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Same Code'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '99'
    var_2 = '99.1'
    var_3 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing Subaccount'
    var_4 = '2'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test___call___returns_coa_instance. Retrieved 3/4 statements.
# Partially parsed test___call___returns_new_coa_each_call. Retrieved 4/5 statements.
# Partially parsed test___call___can_be_called_directly. Retrieved 2/4 statements.
# Partially parsed test___call___raises_exception_on_failure. Retrieved 5/9 statements.
# Partially parsed test___call___returns_coa_with_expected_attributes. Retrieved 5/7 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.ReadChartOfAccounts()
    var_2 = var_1.__call__()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()
    var_1 = module_0.COA()
    var_2 = var_0.__call__()
    var_3 = var_0.__call__()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.ReadChartOfAccounts()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.ReadChartOfAccounts()
    var_1 = ()
    var_2 = 'Read error'
    var_3 = ValueError(var_2)
    var_4 = var_0.__call__()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = 'Asset'
    var_2 = 'Liability'
    var_3 = module_0.ReadChartOfAccounts()
    var_4 = var_3.__call__()



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '001'
    var_1 = 'Test SubAccount'
    var_2 = '000'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #46
#--------------------------

# Partially parsed test___call___returns_coa. Retrieved 1/4 statements.
# Partially parsed test___call___returns_new_coa_instance_each_time. Retrieved 3/8 statements.
# Partially parsed test___call___raises_exception_on_failure. Retrieved 1/5 statements.
# Partially parsed test___call___returns_coa_with_expected_properties. Retrieved 5/12 statements.
# Failed to parse test___call___can_be_called_without_arguments.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = module_0.COA()
    var_2 = module_0.COA()

def test_case_0():
    var_0 = 'Read failed'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Revenue'
    var_4 = module_0.COA()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_subaccount_constructor. Retrieved 6/11 statements.


import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test SubAccount'
    var_2 = '100'
    var_3 = 'Parent Account'
    var_4 = 'Test COA'
    var_5 = module_0.COA()



# Parsed testcases at query #48
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



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_add_creates_new_subaccount. Retrieved 4/10 statements.
# Partially parsed test_add_returns_existing_account_if_consistent. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_if_parent_equals_code. Retrieved 3/6 statements.
# Partially parsed test_add_raises_error_if_parent_not_found. Retrieved 4/8 statements.
# Partially parsed test_add_raises_error_if_account_exists_with_inconsistent_info. Retrieved 5/11 statements.
# Partially parsed test_add_updates_subaccounts_buffer. Retrieved 4/8 statements.
# Partially parsed test_add_multiple_subaccounts. Retrieved 6/14 statements.
# Partially parsed test_add_subaccount_to_non_root_parent. Retrieved 6/12 statements.


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
    var_2 = '1.1'
    var_3 = 'Existing Subaccount'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = 'Same Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '999'
    var_2 = '999.1'
    var_3 = 'New Account'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Existing Subaccount'
    var_4 = '2'

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
    var_2 = '1.1'
    var_3 = 'Subaccount 1'
    var_4 = '1.2'
    var_5 = 'Subaccount 2'

import pypara.accounting.accounts as module_0

def test_case_0():
    var_0 = module_0.COA()
    var_1 = '1'
    var_2 = '1.1'
    var_3 = 'Intermediate'
    var_4 = '1.1.1'
    var_5 = 'Leaf Account'



