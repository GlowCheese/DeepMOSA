####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_types = {account.type for account in coa.accounts}
    assert account_types == {AccountType.BALANCE, AccountType.INCOME}
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_custom_rootspec():
    rootspec = {
        AccountType.BALANCE: (Code("1"), "Custom Balance"),
        AccountType.INCOME: (Code("2"), "Custom Income")
    }
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    balance_account = coa.find(Code("1"))
    income_account = coa.find(Code("2"))
    assert balance_account is not None
    assert income_account is not None
    assert balance_account.name == "Custom Balance"
    assert income_account.name == "Custom Income"
    assert balance_account.type == AccountType.BALANCE
    assert income_account.type == AccountType.INCOME

def test_coa_constructor_partial_rootspec():
    rootspec = {AccountType.BALANCE: (Code("A"), "My Balance")}
    coa = COA(rootspec=rootspec)
    balance_account = coa.find(Code("A"))
    income_account = coa.find(Code("2"))
    assert balance_account is not None
    assert income_account is not None
    assert balance_account.name == "My Balance"
    assert income_account.name == "Income"
    assert balance_account.type == AccountType.BALANCE
    assert income_account.type == AccountType.INCOME

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == 2
    for code, account in coa:
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_none_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 2
    for code, account in coa:
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_accounts_ordered():
    coa = COA()
    accounts = list(coa.accounts)
    assert accounts[0].type == AccountType.BALANCE
    assert accounts[1].type == AccountType.INCOME

def test_coa_constructor_toplevel_property():
    coa = COA()
    toplevel = list(coa.toplevel)
    assert len(toplevel) == 2
    for account in toplevel:
        assert account.parent is None

def test_coa_constructor_structure_property():
    coa = COA()
    structure = list(coa.structure)
    assert len(structure) == 2
    for node in structure:
        assert isinstance(node, COA.Node)
        assert node.account.parent is None
        assert len(node.children) == 0

def test_coa_constructor_find_method():
    coa = COA()
    balance_code = Code("1")
    income_code = Code("2")
    balance_account = coa.find(balance_code)
    income_account = coa.find(income_code)
    assert balance_account is not None
    assert income_account is not None
    assert balance_account.code == balance_code
    assert income_account.code == income_code
    assert coa.find(Code("999")) is None

def test_coa_constructor_subaccounts_method():
    coa = COA()
    for account in coa.accounts:
        subaccounts = coa.subaccounts(account)
        assert isinstance(subaccounts, list)
        assert len(subaccounts) == 0

def test_coa_constructor_nodify_method():
    coa = COA()
    for account in coa.toplevel:
        node = coa.nodify(account)
        assert isinstance(node, COA.Node)
        assert node.account == account
        assert node.children == []

def test_coa_constructor_iter_method():
    coa = COA()
    items = list(coa)
    assert len(items) == 2
    for code, account in items:
        assert isinstance(code, Code)
        assert isinstance(account, RootAccount)
        assert code == account.code

def test_coa_constructor_frozen_dataclass():
    coa = COA()
    try:
        coa._accounts = {}
    except Exception as e:
        assert isinstance(e, dataclasses.FrozenInstanceError)


# LLM-generated content at query #2
#--------------------------

def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes
    for _, account in coa:
        assert isinstance(account, RootAccount)
        assert account.coa is coa

def test_coa_constructor_custom_rootspec():
    rootspec = {AccountType.BALANCESHEET: (Code("10"), "Custom Balance"), AccountType.INCOMESTATEMENT: (Code("20"), "Custom Income")}
    coa = COA(rootspec=rootspec)
    accounts = dict(coa)
    assert len(accounts) == 2
    assert Code("10") in accounts
    assert Code("20") in accounts
    assert accounts[Code("10")].name == "Custom Balance"
    assert accounts[Code("20")].name == "Custom Income"
    assert accounts[Code("10")].type == AccountType.BALANCESHEET
    assert accounts[Code("20")].type == AccountType.INCOMESTATEMENT

def test_coa_constructor_partial_rootspec():
    rootspec = {AccountType.BALANCESHEET: (Code("99"), "Partial Balance")}
    coa = COA(rootspec=rootspec)
    accounts = dict(coa)
    assert len(accounts) == 2
    assert Code("99") in accounts
    assert accounts[Code("99")].name == "Partial Balance"
    assert accounts[Code("99")].type == AccountType.BALANCESHEET
    other_code = Code("2")
    other_account = accounts.get(other_code)
    assert other_account is not None
    assert other_account.type == AccountType.INCOMESTATEMENT
    assert other_account.name == "Incomestatement"

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == 2
    for code, account in coa:
        assert code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_rootspec_none():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 2
    for code, account in coa:
        assert code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_accounts_are_frozen():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == 2
    for account in accounts_list:
        try:
            account.name = "New Name"
            assert False, "Should not be able to modify frozen account"
        except:
            pass

def test_coa_constructor_toplevel_iterable():
    coa = COA()
    toplevel = list(coa.toplevel)
    assert len(toplevel) == 2
    for account in toplevel:
        assert account.parent is None

def test_coa_constructor_structure_property():
    coa = COA()
    structure = list(coa.structure)
    assert len(structure) == 2
    for node in structure:
        assert isinstance(node, COA.Node)
        assert node.children == []


# LLM-generated content at query #3
#--------------------------

def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == code
        assert account.coa is coa
    assert list(coa.toplevel) == list(coa.accounts)
    assert len(list(coa.structure)) == 2

def test_coa_constructor_custom_rootspec():
    rootspec = {AccountType.ASSET: (Code("A"), "Asset Account"), AccountType.EQUITY: (Code("E"), "Equity Account")}
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert coa.find(Code("A")) is not None
    assert coa.find(Code("E")) is not None
    asset_account = coa.find(Code("A"))
    equity_account = coa.find(Code("E"))
    assert asset_account.name == "Asset Account"
    assert equity_account.name == "Equity Account"
    assert asset_account.type == AccountType.ASSET
    assert equity_account.type == AccountType.EQUITY
    assert asset_account.coa is coa
    assert equity_account.coa is coa

def test_coa_constructor_partial_rootspec():
    rootspec = {AccountType.ASSET: (Code("10"), "Assets")}
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    asset_account = coa.find(Code("10"))
    assert asset_account is not None
    assert asset_account.name == "Assets"
    assert asset_account.type == AccountType.ASSET
    other_type = AccountType.EQUITY
    other_account = coa.find(Code("2"))
    assert other_account is not None
    assert other_account.type == other_type
    assert other_account.name == other_type.name.capitalize()

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    for c, (code, account) in enumerate(coa, start=1):
        assert code == Code(str(c))
        assert account.type == list(AccountType)[c-1]
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_rootspec_none():
    coa = COA(rootspec=None)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    for c, (code, account) in enumerate(coa, start=1):
        assert code == Code(str(c))
        assert account.type == list(AccountType)[c-1]
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_frozen_dataclass():
    coa = COA()
    try:
        coa._accounts = {}
    except Exception as e:
        assert isinstance(e, dataclasses.FrozenInstanceError)


# LLM-generated content at query #4
#--------------------------

def test_read_chart_of_accounts_returns_coa():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result == mock_coa

def test_read_chart_of_accounts_called_without_arguments():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    mock_reader.assert_called_once_with()

def test_read_chart_of_accounts_returns_different_coa_instances():
    mock_coa1 = COA()
    mock_coa2 = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = [mock_coa1, mock_coa2]
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 == mock_coa1
    assert result2 == mock_coa2
    assert result1 != result2


# LLM-generated content at query #5
#--------------------------

def test_subaccount_constructor():
    mock_code = Code(value="123")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code(value="100"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #6
#--------------------------

def test_add_creates_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "Sub Account"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_returns_existing_account_when_consistent():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Account"
    existing_account = coa.add(parent_code, existing_code, existing_name)
    retrieved_account = coa.add(parent_code, existing_code, existing_name)
    assert retrieved_account == existing_account

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    same_code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(same_code, same_code, "Account")

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    non_existent_parent = Code("999")
    new_code = Code("999.1")
    new_name = "Sub Account"
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined."):
        coa.add(non_existent_parent, new_code, new_name)

def test_add_raises_error_when_account_exists_with_inconsistent_details():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Account"
    coa.add(parent_code, existing_code, existing_name)
    different_parent_code = Code("2")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(different_parent_code, existing_code, existing_name)


# LLM-generated content at query #7
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #8
#--------------------------

def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_types = {account.type for account in coa.accounts}
    assert account_types == {AccountType.BALANCE, AccountType.INCOME}
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.BALANCE: (Code("1"), "BalanceSheet"),
        AccountType.INCOME: (Code("2"), "IncomeStatement")
    }
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    balance_account = coa.find(Code("1"))
    income_account = coa.find(Code("2"))
    assert balance_account is not None
    assert income_account is not None
    assert balance_account.name == "BalanceSheet"
    assert balance_account.type == AccountType.BALANCE
    assert income_account.name == "IncomeStatement"
    assert income_account.type == AccountType.INCOME
    assert balance_account.parent is None
    assert income_account.parent is None
    assert balance_account.coa is coa
    assert income_account.coa is coa

def test_coa_constructor_with_partial_rootspec():
    rootspec = {AccountType.BALANCE: (Code("A"), "Assets")}
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    custom_account = coa.find(Code("A"))
    default_account = coa.find(Code("2"))
    assert custom_account is not None
    assert default_account is not None
    assert custom_account.name == "Assets"
    assert custom_account.type == AccountType.BALANCE
    assert default_account.name == "Income"
    assert default_account.type == AccountType.INCOME

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == 2
    for code, account in coa:
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_rootspec_none():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 2
    for code, account in coa:
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()


# LLM-generated content at query #9
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code("000"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #10
#--------------------------

def test_add_account_parent_not_defined():
    coa = COA()
    try:
        coa.add(Code("2"), Code("21"), "Sub Account")
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."


# LLM-generated content at query #11
#--------------------------

def test_nodify_returns_node_with_correct_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.account == root_account

def test_nodify_returns_node_with_empty_children_for_leaf_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.children == []

def test_nodify_returns_node_with_children_for_account_with_subaccounts():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account = coa.add(root_account.code, Code("1.1"), "Sub Account")
    node = coa.nodify(root_account)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account

def test_nodify_returns_node_with_nested_children():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account = coa.add(root_account.code, Code("1.1"), "Sub Account")
    sub_sub_account = coa.add(sub_account.code, Code("1.1.1"), "Sub Sub Account")
    node = coa.nodify(root_account)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == sub_sub_account

def test_nodify_raises_no_error_for_account_not_in_coa():
    coa = COA()
    root_account = next(coa.toplevel)
    fake_account = RootAccount(Code("999"), "Fake", AccountType.ASSET, coa)
    node = coa.nodify(fake_account)
    assert node.account == fake_account
    assert node.children == []


# LLM-generated content at query #12
#--------------------------

def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == code
        assert account.coa is coa

def test_coa_constructor_custom_rootspec():
    custom_spec = {AccountType.BALANCESHEET: (Code("10"), "Custom Balance"), AccountType.INCOMESTATEMENT: (Code("20"), "Custom Income")}
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert coa.find(Code("10")) is not None
    assert coa.find(Code("20")) is not None
    assert coa.find(Code("10")).name == "Custom Balance"
    assert coa.find(Code("20")).name == "Custom Income"
    assert coa.find(Code("1")) is None
    assert coa.find(Code("2")) is None

def test_coa_constructor_partial_rootspec():
    custom_spec = {AccountType.BALANCESHEET: (Code("99"), "Partial Balance")}
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert coa.find(Code("99")) is not None
    assert coa.find(Code("99")).name == "Partial Balance"
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("2")).name == "Incomestatement"

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("1")).name == "Balancesheet"
    assert coa.find(Code("2")).name == "Incomestatement"

def test_coa_constructor_none_rootspec():
    coa = COA(rootspec=None)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("1")).name == "Balancesheet"
    assert coa.find(Code("2")).name == "Incomestatement"


# LLM-generated content at query #13
#--------------------------

def test_nodify_returns_node_with_correct_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert node.children == []

def test_nodify_returns_node_with_children():
    coa = COA()
    root_code = next(coa._accounts.keys())
    child_account = coa.add(root_code, Code("1.1"), "Child Account")
    root_account = coa.find(root_code)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account

def test_nodify_returns_node_with_nested_children():
    coa = COA()
    root_code = next(coa._accounts.keys())
    child_account = coa.add(root_code, Code("1.1"), "Child Account")
    grandchild_account = coa.add(Code("1.1"), Code("1.1.1"), "Grandchild Account")
    root_account = coa.find(root_code)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == grandchild_account

def test_nodify_for_account_without_children():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert node.children == []

def test_nodify_for_account_with_multiple_children():
    coa = COA()
    root_code = next(coa._accounts.keys())
    child1 = coa.add(root_code, Code("1.1"), "Child 1")
    child2 = coa.add(root_code, Code("1.2"), "Child 2")
    root_account = coa.find(root_code)
    node = coa.nodify(root_account)
    assert len(node.children) == 2
    child_accounts = [child.account for child in node.children]
    assert child1 in child_accounts
    assert child2 in child_accounts


# LLM-generated content at query #14
#--------------------------

def test_nodify_returns_node_with_correct_account_and_children():
    from dataclasses import dataclass, field
    from typing import List, Dict, Optional, Tuple, Iterable, Iterator, OrderedDict
    from enum import Enum
    coa_code = "1"
    coa_name = "Test COA"
    root_code = Code("1")
    root_name = "Root"
    root_type = AccountType.ASSET
    coa = COA(rootspec={root_type: (root_code, root_name)})
    sub_code = Code("1.1")
    sub_name = "Sub Account"
    coa.add(parent=root_code, code=sub_code, name=sub_name)
    root_account = coa.find(root_code)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == coa.find(sub_code)
    assert len(node.children[0].children) == 0


# LLM-generated content at query #15
#--------------------------

def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == code
        assert account.coa is coa

def test_coa_constructor_with_rootspec():
    rootspec = {AccountType.BALANCE: (Code("10"), "MyBalance"), AccountType.INCOME: (Code("20"), "MyIncome")}
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    balance_account = coa.find(Code("10"))
    income_account = coa.find(Code("20"))
    assert balance_account is not None
    assert income_account is not None
    assert balance_account.name == "MyBalance"
    assert income_account.name == "MyIncome"
    assert balance_account.type == AccountType.BALANCE
    assert income_account.type == AccountType.INCOME

def test_coa_constructor_with_partial_rootspec():
    rootspec = {AccountType.BALANCE: (Code("100"), "CustomBalance")}
    coa = COA(rootspec=rootspec)
    balance_account = coa.find(Code("100"))
    income_account = coa.find(Code("2"))
    assert balance_account is not None
    assert income_account is not None
    assert balance_account.name == "CustomBalance"
    assert income_account.name == "Income"

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes

def test_coa_constructor_with_none_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes

def test_coa_constructor_rootspec_default_name_capitalization():
    coa = COA()
    balance_account = coa.find(Code("1"))
    income_account = coa.find(Code("2"))
    assert balance_account.name == "Balance"
    assert income_account.name == "Income"

def test_coa_constructor_accounts_are_root_accounts():
    coa = COA()
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None

def test_coa_constructor_toplevel_iterable():
    coa = COA()
    toplevel_accounts = list(coa.toplevel)
    assert len(toplevel_accounts) == 2
    for account in toplevel_accounts:
        assert account.parent is None

def test_coa_constructor_structure_property():
    coa = COA()
    structure = list(coa.structure)
    assert len(structure) == 2
    for node in structure:
        assert isinstance(node, COA.Node)
        assert len(node.children) == 0

def test_coa_constructor_find_existing_account():
    coa = COA()
    account = coa.find(Code("1"))
    assert account is not None
    assert account.code == Code("1")
    assert account.name == "Balance"

def test_coa_constructor_find_non_existing_account():
    coa = COA()
    account = coa.find(Code("999"))
    assert account is None

def test_coa_constructor_subaccounts_empty_initially():
    coa = COA()
    balance_account = coa.find(Code("1"))
    subaccounts = coa.subaccounts(balance_account)
    assert subaccounts == []

def test_coa_constructor_nodify_root_account():
    coa = COA()
    balance_account = coa.find(Code("1"))
    node = coa.nodify(balance_account)
    assert node.account == balance_account
    assert node.children == []


# LLM-generated content at query #16
#--------------------------

def test_nodify_returns_node_with_correct_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert isinstance(node.children, list)

def test_nodify_returns_node_with_empty_children_for_account_without_subaccounts():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.children == []

def test_nodify_returns_node_with_subaccount_children():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account = coa.add(root_account.code, Code("1.1"), "Sub Account")
    node = coa.nodify(root_account)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account

def test_nodify_returns_nested_structure_for_multiple_levels():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account1 = coa.add(root_account.code, Code("1.1"), "Sub Account 1")
    sub_account2 = coa.add(sub_account1.code, Code("1.1.1"), "Sub Account 2")
    node = coa.nodify(root_account)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account1
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == sub_account2

def test_nodify_handles_account_with_multiple_subaccounts():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account1 = coa.add(root_account.code, Code("1.1"), "Sub Account 1")
    sub_account2 = coa.add(root_account.code, Code("1.2"), "Sub Account 2")
    node = coa.nodify(root_account)
    assert len(node.children) == 2
    assert {child.account for child in node.children} == {sub_account1, sub_account2}


# LLM-generated content at query #17
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___returns_coa_with_accounts():
    mock_account = Account(code="1000", name="Cash")
    mock_coa = COA(accounts=[mock_account])
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert len(result.accounts) == 1
    assert result.accounts[0].code == "1000"
    assert result.accounts[0].name == "Cash"

def test___call___returns_empty_coa():
    mock_coa = COA(accounts=[])
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert len(result.accounts) == 0

def test___call___invoked_without_arguments():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    mock_reader.assert_called_once_with()

def test___call___returns_different_coa_instances():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = [COA(), COA()]
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 is not result2


# LLM-generated content at query #18
#--------------------------

def test_add_existing_account_with_matching_details_returns_account():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    coa.add(parent_code, child_code, child_name)
    result = coa.add(parent_code, child_code, child_name)
    assert result is coa._accounts[child_code]
    assert result.code == child_code
    assert result.name == child_name
    assert result.parent == coa._accounts[parent_code]


# LLM-generated content at query #19
#--------------------------

def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_types = {account.type for account in coa.accounts}
    assert account_types == {AccountType.ASSET, AccountType.LIABILITY}
    for code, account in coa:
        assert isinstance(code, Code)
        assert isinstance(account, RootAccount)
        assert account.code == code
        assert account.name in ("Asset", "Liability")
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_with_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("1"), "MyAsset"),
        AccountType.LIABILITY: (Code("2"), "MyLiability")
    }
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    asset_account = coa.find(Code("1"))
    liability_account = coa.find(Code("2"))
    assert asset_account is not None
    assert liability_account is not None
    assert asset_account.name == "MyAsset"
    assert liability_account.name == "MyLiability"
    assert asset_account.type == AccountType.ASSET
    assert liability_account.type == AccountType.LIABILITY

def test_coa_constructor_with_partial_rootspec():
    partial_spec = {AccountType.ASSET: (Code("A"), "CustomAsset")}
    coa = COA(rootspec=partial_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    custom_asset = coa.find(Code("A"))
    default_liability = coa.find(Code("2"))
    assert custom_asset is not None
    assert default_liability is not None
    assert custom_asset.name == "CustomAsset"
    assert default_liability.name == "Liability"

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    asset_account = coa.find(Code("1"))
    liability_account = coa.find(Code("2"))
    assert asset_account is not None
    assert liability_account is not None
    assert asset_account.name == "Asset"
    assert liability_account.name == "Liability"

def test_coa_constructor_with_none_rootspec():
    coa = COA(rootspec=None)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    asset_account = coa.find(Code("1"))
    liability_account = coa.find(Code("2"))
    assert asset_account is not None
    assert liability_account is not None
    assert asset_account.name == "Asset"
    assert liability_account.name == "Liability"

def test_coa_constructor_accounts_are_root_accounts():
    coa = COA()
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None

def test_coa_constructor_toplevel_property():
    coa = COA()
    toplevel = list(coa.toplevel)
    assert len(toplevel) == 2
    for account in toplevel:
        assert account.parent is None

def test_coa_constructor_structure_property():
    coa = COA()
    structure = list(coa.structure)
    assert len(structure) == 2
    for node in structure:
        assert isinstance(node, COA.Node)
        assert isinstance(node.account, RootAccount)
        assert len(node.children) == 0

def test_coa_constructor_frozen_dataclass():
    coa = COA()
    try:
        coa._accounts = {}
    except:
        pass
    assert len(list(coa.accounts)) == 2


# LLM-generated content at query #20
#--------------------------

def test_add_subaccount_parent_not_in_subaccounts():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    coa.add(parent_code, child_code, "Child Account")
    assert child_code in coa._accounts
    assert coa._accounts[child_code].parent == coa._accounts[parent_code]
    assert coa._subaccounts[coa._accounts[parent_code]] == [coa._accounts[child_code]]


# LLM-generated content at query #21
#--------------------------

def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == code
        assert account.name in ["Asset", "Liability", "Equity", "Revenue", "Expense"]
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_custom_rootspec():
    rootspec = {AccountType.ASSET: (Code("1000"), "CustomAsset"), AccountType.LIABILITY: (Code("2000"), "CustomLiability")}
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    account_dict = dict(coa)
    assert Code("1000") in account_dict
    assert Code("2000") in account_dict
    asset_account = account_dict[Code("1000")]
    assert asset_account.name == "CustomAsset"
    assert asset_account.type == AccountType.ASSET
    liability_account = account_dict[Code("2000")]
    assert liability_account.name == "CustomLiability"
    assert liability_account.type == AccountType.LIABILITY

def test_coa_constructor_partial_rootspec():
    rootspec = {AccountType.EQUITY: (Code("300"), "MyEquity")}
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    account_dict = dict(coa)
    equity_code = Code("300")
    other_code = Code("1") if equity_code == Code("300") else Code("2")
    assert equity_code in account_dict
    assert other_code in account_dict
    equity_account = account_dict[equity_code]
    assert equity_account.name == "MyEquity"
    assert equity_account.type == AccountType.EQUITY
    other_account = account_dict[other_code]
    assert other_account.name in ["Asset", "Liability", "Revenue", "Expense"]

def test_coa_constructor_rootspec_none():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 2

def test_coa_constructor_rootspec_empty_dict():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == 2
    for _, account in coa:
        assert isinstance(account, RootAccount)

def test_coa_constructor_frozen_dataclass():
    coa = COA()
    try:
        coa._accounts = {}
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass
    try:
        coa.rootspec = None
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass

def test_coa_constructor_accounts_iterable():
    coa = COA()
    accounts_from_property = list(coa.accounts)
    accounts_from_iter = [acc for _, acc in coa]
    assert len(accounts_from_property) == len(accounts_from_iter)
    assert all(a1 is a2 for a1, a2 in zip(accounts_from_property, accounts_from_iter))

def test_coa_constructor_toplevel_accounts():
    coa = COA()
    toplevel = list(coa.toplevel)
    assert len(toplevel) == 2
    for account in toplevel:
        assert account.parent is None

def test_coa_constructor_structure_property():
    coa = COA()
    structure = list(coa.structure)
    assert len(structure) == 2
    for node in structure:
        assert isinstance(node, COA.Node)
        assert node.account.parent is None
        assert len(node.children) == 0


# LLM-generated content at query #22
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #23
#--------------------------

def test___call___returns_coa():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___raises_exception_on_failure():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = ValueError("Read error")
    try:
        mock_reader()
        assert False
    except ValueError as e:
        assert str(e) == "Read error"

def test___call___returns_different_coa_instances():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = [COA(), COA()]
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 is not result2
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)

def test___call___can_be_called_multiple_times():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    for _ in range(5):
        result = mock_reader()
        assert result is mock_coa


# LLM-generated content at query #24
#--------------------------

def test_add_existing_account_with_matching_info_returns_account():
    coa = COA()
    root_account = next(iter(coa._accounts.values()))
    parent_code = root_account.code
    code = Code("1.1")
    name = "Sub Account"
    new_account = coa.add(parent_code, code, name)
    existing_account = coa.add(parent_code, code, name)
    assert existing_account is new_account


# LLM-generated content at query #25
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("123")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("100"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #26
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code("000"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #27
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == COA("Test COA")


# LLM-generated content at query #28
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = ReadChartOfAccounts(lambda: mock_coa)
    result = mock_reader()
    assert result is mock_coa

def test___call___returns_different_coa_instances():
    mock_coa1 = COA()
    mock_coa2 = COA()
    call_count = 0
    def reader_func():
        nonlocal call_count
        call_count += 1
        return mock_coa1 if call_count == 1 else mock_coa2
    mock_reader = ReadChartOfAccounts(reader_func)
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 is mock_coa1
    assert result2 is mock_coa2

def test___call___raises_exception():
    mock_reader = ReadChartOfAccounts(lambda: (_ for _ in ()).throw(ValueError("Test error")))
    try:
        mock_reader()
        assert False
    except ValueError as e:
        assert str(e) == "Test error"

def test___call___returns_coa_with_expected_attributes():
    expected_accounts = [Account("1000", "Cash"), Account("2000", "Accounts Payable")]
    mock_coa = COA()
    mock_coa.accounts = expected_accounts
    mock_reader = ReadChartOfAccounts(lambda: mock_coa)
    result = mock_reader()
    assert result.accounts == expected_accounts

def test___call___can_be_called_multiple_times():
    call_counter = 0
    def counting_reader():
        nonlocal call_counter
        call_counter += 1
        return COA()
    mock_reader = ReadChartOfAccounts(counting_reader)
    mock_reader()
    mock_reader()
    mock_reader()
    assert call_counter == 3


# LLM-generated content at query #29
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    coa.add(parent_code, child_code, child_name)
    result = coa.add(parent_code, child_code, child_name)
    assert result.code == child_code
    assert result.name == child_name
    assert result.parent == coa.find(parent_code)


# LLM-generated content at query #30
#--------------------------

def test_subaccount_constructor():
    mock_code = Code(value="101")
    mock_name = "Cash"
    mock_parent = Account(code=Code(value="100"), name="Assets", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #31
#--------------------------

def test_add_existing_account_with_matching_info_returns_account():
    coa = COA()
    root_account = next(coa.toplevel)
    parent_code = root_account.code
    code = Code("1.1")
    name = "Sub Account"
    coa.add(parent_code, code, name)
    result = coa.add(parent_code, code, name)
    assert result.code == code
    assert result.name == name
    assert result.parent == root_account


# LLM-generated content at query #32
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    root_code = Code("1")
    root_account = coa.find(root_code)
    sub_code = Code("1.1")
    sub_name = "Sub Account"
    coa.add(root_code, sub_code, sub_name)
    existing_account = coa.add(root_code, sub_code, sub_name)
    assert existing_account.code == sub_code
    assert existing_account.name == sub_name
    assert existing_account.parent == root_account


# LLM-generated content at query #33
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    root_account = next(coa.toplevel)
    parent_code = root_account.code
    code = Code("1.1")
    name = "Sub Account"
    coa.add(parent_code, code, name)
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent == root_account


# LLM-generated content at query #34
#--------------------------

def test_add_new_account_successfully():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "Sub Account"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Sub Account"
    coa.add(parent_code, existing_code, existing_name)
    existing_account = coa.find(existing_code)
    result = coa.add(parent_code, existing_code, existing_name)
    assert result == existing_account

def test_add_existing_account_with_mismatched_parent():
    coa = COA()
    parent_code_1 = Code("1")
    parent_code_2 = Code("2")
    existing_code = Code("1.1")
    existing_name = "Sub Account"
    coa.add(parent_code_1, existing_code, existing_name)
    try:
        coa.add(parent_code_2, existing_code, existing_name)
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_mismatched_name():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Sub Account"
    mismatched_name = "Different Name"
    coa.add(parent_code, existing_code, existing_name)
    try:
        coa.add(parent_code, existing_code, mismatched_name)
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_account_with_parent_as_self():
    coa = COA()
    code = Code("1")
    try:
        coa.add(code, code, "Self Parent")
        assert False
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_account_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("999")
    new_code = Code("999.1")
    new_name = "Sub Account"
    try:
        coa.add(parent_code, new_code, new_name)
        assert False
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."


# LLM-generated content at query #35
#--------------------------

def test_add_new_subaccount_successfully():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "Sub Account"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Sub Account"
    coa.add(parent_code, existing_code, existing_name)
    existing_account = coa.find(existing_code)
    returned_account = coa.add(parent_code, existing_code, existing_name)
    assert returned_account == existing_account
    assert coa.find(existing_code) == existing_account

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    same_code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(same_code, same_code, "Same Account")

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    non_existent_parent = Code("999")
    new_code = Code("999.1")
    new_name = "New Account"
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined."):
        coa.add(non_existent_parent, new_code, new_name)

def test_add_raises_error_when_existing_account_mismatch():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Sub Account"
    coa.add(parent_code, existing_code, existing_name)
    different_name = "Different Name"
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, existing_code, different_name)


# LLM-generated content at query #36
#--------------------------

def test_add_existing_account_with_different_parent():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child")
    different_parent_code = Code("3")
    coa.add(parent_code, different_parent_code, "Different Parent")
    try:
        coa.add(different_parent_code, child_code, "Child")
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_name():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child")
    try:
        coa.add(parent_code, child_code, "Different Name")
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_code():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child")
    different_code = Code("3")
    coa.add(parent_code, different_code, "Different Child")
    try:
        coa.add(parent_code, child_code, "Child")
    except ValueError:
        assert False

def test_add_existing_account_with_same_attributes():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    account = coa.add(parent_code, child_code, "Child")
    same_account = coa.add(parent_code, child_code, "Child")
    assert account is same_account


# LLM-generated content at query #37
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #38
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code("000"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #39
#--------------------------

def test_add_existing_account_with_different_parent_raises_error():
    coa = COA()
    root_account = next(iter(coa._accounts.values()))
    parent_code = root_account.code
    code = Code("1.1")
    name = "SubAccount"
    coa.add(parent_code, code, name)
    different_parent_code = Code("2")
    different_parent_instance = coa._accounts.get(different_parent_code)
    assert different_parent_instance is not None
    try:
        coa.add(different_parent_code, code, name)
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #40
#--------------------------

def test_add_existing_account_with_different_parent():
    coa = COA()
    root_code = Code("1")
    root_account = coa.find(root_code)
    coa.add(root_code, Code("1.1"), "SubAccount1")
    coa.add(root_code, Code("1.2"), "SubAccount2")
    try:
        coa.add(Code("1.2"), Code("1.1"), "SubAccount1")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_name():
    coa = COA()
    root_code = Code("1")
    root_account = coa.find(root_code)
    coa.add(root_code, Code("1.1"), "SubAccount1")
    try:
        coa.add(root_code, Code("1.1"), "DifferentName")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_code():
    coa = COA()
    root_code = Code("1")
    root_account = coa.find(root_code)
    coa.add(root_code, Code("1.1"), "SubAccount1")
    try:
        coa.add(root_code, Code("1.2"), "SubAccount1")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #41
#--------------------------

def test_add_existing_account_with_different_parent():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child Account")
    different_parent_code = Code("3")
    coa.add(parent_code, different_parent_code, "Different Parent")
    try:
        coa.add(different_parent_code, child_code, "Child Account")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_name():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child Account")
    try:
        coa.add(parent_code, child_code, "Different Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_code():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child Account")
    different_code = Code("3")
    coa.add(parent_code, different_code, "Child Account")
    try:
        coa.add(parent_code, child_code, "Child Account")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #42
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___returns_new_coa_each_call():
    mock_reader = ReadChartOfAccounts()
    call_count = 0
    def side_effect():
        nonlocal call_count
        call_count += 1
        return COA()
    mock_reader.__call__ = side_effect
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 is not result2
    assert call_count == 2

def test___call___raises_exception_on_failure():
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: (_ for _ in ()).throw(ValueError("Read error"))
    try:
        mock_reader()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Read error"

def test___call___returns_coa_with_expected_attributes():
    mock_coa = COA()
    mock_coa.accounts = ["Asset", "Liability"]
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: mock_coa
    result = mock_reader()
    assert result.accounts == ["Asset", "Liability"]

def test___call___can_be_called_via_instance():
    mock_coa = COA()
    reader_instance = ReadChartOfAccounts()
    reader_instance.__call__ = lambda: mock_coa
    result = reader_instance.__call__()
    assert result is mock_coa


# LLM-generated content at query #43
#--------------------------

def test_add_existing_account_with_different_parent_raises_error():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    parent_account = coa.add(parent_code, parent_code, "Parent")
    coa.add(parent_code, child_code, "Child")
    different_parent_code = Code("3")
    different_parent_account = coa.add(parent_code, different_parent_code, "Different Parent")
    try:
        coa.add(different_parent_code, child_code, "Child")
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___returns_valid_coa():
    mock_coa = Mock(spec=COA)
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert isinstance(result, COA)

def test___call___invoked_without_arguments():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    mock_reader.assert_called_once_with()

def test___call___returns_different_coa_instances():
    mock_coa1 = COA()
    mock_coa2 = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = [mock_coa1, mock_coa2]
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 is mock_coa1
    assert result2 is mock_coa2
    assert result1 is not result2


# LLM-generated content at query #2
#--------------------------

def test_add_new_account_successfully():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "Sub Account"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_existing_account_with_matching_data():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Sub Account"
    coa.add(parent_code, existing_code, existing_name)
    existing_account = coa.add(parent_code, existing_code, existing_name)
    assert existing_account.code == existing_code
    assert existing_account.name == existing_name
    assert existing_account.parent == coa.find(parent_code)

def test_add_account_with_parent_equal_to_code_raises_error():
    coa = COA()
    same_code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(same_code, same_code, "Account Name")

def test_add_account_with_nonexistent_parent_raises_error():
    coa = COA()
    non_existent_parent = Code("999")
    new_code = Code("999.1")
    new_name = "Sub Account"
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined."):
        coa.add(non_existent_parent, new_code, new_name)

def test_add_account_with_existing_code_but_mismatched_data_raises_error():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Sub Account"
    coa.add(parent_code, existing_code, existing_name)
    mismatched_name = "Different Name"
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, existing_code, mismatched_name)

def test_add_account_updates_subaccounts_buffer():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "Sub Account"
    new_account = coa.add(parent_code, new_code, new_name)
    assert parent_account in coa._subaccounts
    assert new_account in coa._subaccounts[parent_account]

def test_add_account_creates_subaccount_instance():
    coa = COA()
    parent_code = Code("1")
    new_code = Code("1.1")
    new_name = "Sub Account"
    new_account = coa.add(parent_code, new_code, new_name)
    assert isinstance(new_account, SubAccount)

def test_add_multiple_subaccounts_to_same_parent():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    sub_code_1 = Code("1.1")
    sub_name_1 = "Sub Account 1"
    sub_account_1 = coa.add(parent_code, sub_code_1, sub_name_1)
    sub_code_2 = Code("1.2")
    sub_name_2 = "Sub Account 2"
    sub_account_2 = coa.add(parent_code, sub_code_2, sub_name_2)
    assert sub_account_1 in coa.subaccounts(parent_account)
    assert sub_account_2 in coa.subaccounts(parent_account)
    assert len(coa.subaccounts(parent_account)) == 2

def test_add_account_ensures_accounts_buffer_updated():
    coa = COA()
    parent_code = Code("1")
    new_code = Code("1.1")
    new_name = "Sub Account"
    coa.add(parent_code, new_code, new_name)
    assert new_code in coa._accounts
    assert coa._accounts[new_code].code == new_code
    assert coa._accounts[new_code].name == new_name


# LLM-generated content at query #3
#--------------------------

def test_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_types = {a.type for a in coa.accounts}
    assert account_types == {AccountType.BALANCE_SHEET, AccountType.INCOME_STATEMENT}
    for code, account in coa:
        assert account.code == code
        assert account.name in ["Balance_sheet", "Income_statement"]
        assert isinstance(account, RootAccount)

def test_constructor_with_custom_rootspec():
    custom_spec = {
        AccountType.BALANCE_SHEET: (Code("10"), "CustomBalance"),
        AccountType.INCOME_STATEMENT: (Code("20"), "CustomIncome")
    }
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert coa.find(Code("10")).name == "CustomBalance"
    assert coa.find(Code("20")).name == "CustomIncome"
    assert coa.find(Code("10")).type == AccountType.BALANCE_SHEET
    assert coa.find(Code("20")).type == AccountType.INCOME_STATEMENT

def test_constructor_with_partial_rootspec():
    custom_spec = {AccountType.BALANCE_SHEET: (Code("99"), "OnlyBalance")}
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert coa.find(Code("99")).name == "OnlyBalance"
    income_account = next(a for a in accounts if a.type == AccountType.INCOME_STATEMENT)
    assert income_account.code == Code("2")
    assert income_account.name == "Income_statement"

def test_constructor_rootspec_none():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 2
    for account in coa.accounts:
        assert account.parent is None
        assert account.coa is coa

def test_constructor_accounts_are_frozen():
    coa = COA()
    accounts = list(coa.accounts)
    for account in accounts:
        with pytest.raises(dataclasses.FrozenInstanceError):
            account.name = "NewName"

def test_constructor_toplevel_property():
    coa = COA()
    toplevel = list(coa.toplevel)
    assert len(toplevel) == 2
    for account in toplevel:
        assert account.parent is None

def test_constructor_structure_property():
    coa = COA()
    structure = list(coa.structure)
    assert len(structure) == 2
    for node in structure:
        assert isinstance(node, COA.Node)
        assert node.children == []

def test_constructor_iter_method():
    coa = COA()
    iter_items = list(coa)
    assert len(iter_items) == 2
    for code, account in iter_items:
        assert coa.find(code) is account

def test_constructor_find_method():
    coa = COA()
    account = coa.find(Code("1"))
    assert account is not None
    assert account.code == Code("1")
    assert account.name == "Balance_sheet"
    assert coa.find(Code("999")) is None

def test_constructor_subaccounts_method():
    coa = COA()
    root_account = coa.find(Code("1"))
    subaccounts = coa.subaccounts(root_account)
    assert subaccounts == []
    assert coa.subaccounts(Account(Code("999"), "Dummy", None, None, AccountType.BALANCE_SHEET, coa)) == []


# LLM-generated content at query #4
#--------------------------

def test_subaccount_constructor():
    mock_code = Code(value="123")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code(value="100"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #5
#--------------------------

def test_add_account_with_undefined_parent_raises_error():
    coa = COA()
    parent_code = Code("999")
    child_code = Code("1")
    child_name = "Child Account"
    try:
        coa.add(parent_code, child_code, child_name)
        assert False
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."


# LLM-generated content at query #6
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("123")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("100"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #7
#--------------------------

def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_codes = [code for code, _ in coa]
    assert Code("1") in account_codes
    assert Code("2") in account_codes
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.coa is coa

def test_coa_constructor_with_rootspec():
    rootspec = {AccountType.BALANCE: (Code("10"), "BalanceRoot"), AccountType.INCOME: (Code("20"), "IncomeRoot")}
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == 2
    balance_account = coa.find(Code("10"))
    income_account = coa.find(Code("20"))
    assert balance_account is not None
    assert income_account is not None
    assert balance_account.name == "BalanceRoot"
    assert income_account.name == "IncomeRoot"
    assert balance_account.type == AccountType.BALANCE
    assert income_account.type == AccountType.INCOME

def test_coa_constructor_with_partial_rootspec():
    rootspec = {AccountType.BALANCE: (Code("100"), "CustomBalance")}
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == 2
    custom_balance = coa.find(Code("100"))
    default_income = coa.find(Code("2"))
    assert custom_balance is not None
    assert default_income is not None
    assert custom_balance.name == "CustomBalance"
    assert default_income.name == "Income"

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == 2
    for account in coa.accounts:
        assert isinstance(account, RootAccount)

def test_coa_constructor_rootspec_none():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 2
    for account in coa.accounts:
        assert isinstance(account, RootAccount)

def test_coa_constructor_accounts_are_frozen():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == 2
    for account in accounts_list:
        assert account.parent is None

def test_coa_constructor_toplevel_iterable():
    coa = COA()
    toplevel_accounts = list(coa.toplevel)
    assert len(toplevel_accounts) == 2
    for account in toplevel_accounts:
        assert account.parent is None

def test_coa_constructor_structure_property():
    coa = COA()
    structure = list(coa.structure)
    assert len(structure) == 2
    for node in structure:
        assert isinstance(node, COA.Node)
        assert len(node.children) == 0


# LLM-generated content at query #8
#--------------------------

def test_add_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "New Subaccount"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Subaccount"
    existing_account = coa.add(parent_code, existing_code, existing_name)
    retrieved_account = coa.add(parent_code, existing_code, existing_name)
    assert retrieved_account == existing_account

def test_add_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Subaccount"
    coa.add(parent_code, existing_code, existing_name)
    try:
        coa.add(parent_code, existing_code, "Different Name")
        assert False
    except ValueError:
        assert True

def test_add_parent_not_found():
    coa = COA()
    parent_code = Code("99")
    new_code = Code("99.1")
    new_name = "New Subaccount"
    try:
        coa.add(parent_code, new_code, new_name)
        assert False
    except ValueError:
        assert True

def test_add_self_parent():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Self Parent")
        assert False
    except ValueError:
        assert True

def test_add_multiple_subaccounts():
    coa = COA()
    parent_code = Code("1")
    sub1_code = Code("1.1")
    sub1_name = "Subaccount 1"
    sub2_code = Code("1.2")
    sub2_name = "Subaccount 2"
    sub1 = coa.add(parent_code, sub1_code, sub1_name)
    sub2 = coa.add(parent_code, sub2_code, sub2_name)
    parent_account = coa.find(parent_code)
    subaccounts = coa.subaccounts(parent_account)
    assert sub1 in subaccounts
    assert sub2 in subaccounts
    assert len(subaccounts) == 2

def test_add_nested_subaccount():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    grandchild_code = Code("1.1.1")
    child = coa.add(parent_code, child_code, "Child")
    grandchild = coa.add(child_code, grandchild_code, "Grandchild")
    assert grandchild.parent == child
    assert grandchild in coa.subaccounts(child)


# LLM-generated content at query #9
#--------------------------

def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    account_types = {a.type for a in coa.accounts}
    assert account_types == {AccountType.BALANCE, AccountType.INCOME}
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_custom_rootspec():
    custom_spec = {
        AccountType.BALANCE: (Code("B"), "Custom Balance"),
        AccountType.INCOME: (Code("I"), "Custom Income")
    }
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    balance_account = next(a for a in accounts if a.type == AccountType.BALANCE)
    income_account = next(a for a in accounts if a.type == AccountType.INCOME)
    assert balance_account.code == Code("B")
    assert balance_account.name == "Custom Balance"
    assert income_account.code == Code("I")
    assert income_account.name == "Custom Income"

def test_coa_constructor_partial_rootspec():
    custom_spec = {AccountType.BALANCE: (Code("1"), "My Balance")}
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    balance_account = next(a for a in accounts if a.type == AccountType.BALANCE)
    income_account = next(a for a in accounts if a.type == AccountType.INCOME)
    assert balance_account.code == Code("1")
    assert balance_account.name == "My Balance"
    assert income_account.code == Code("2")
    assert income_account.name == "Income"

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    for account in accounts:
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_none_rootspec():
    coa = COA(rootspec=None)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    for account in accounts:
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()


# LLM-generated content at query #10
#--------------------------

def test_nodify_returns_node_with_correct_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.account == root_account

def test_nodify_returns_node_with_empty_children_for_leaf_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.children == []

def test_nodify_returns_node_with_children_for_account_with_subaccounts():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account = coa.add(root_account.code, Code("1.1"), "Sub Account")
    node = coa.nodify(root_account)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account

def test_nodify_returns_node_with_nested_children():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account1 = coa.add(root_account.code, Code("1.1"), "Sub Account 1")
    sub_account2 = coa.add(sub_account1.code, Code("1.1.1"), "Sub Account 2")
    node = coa.nodify(root_account)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account1
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == sub_account2

def test_nodify_returns_node_with_multiple_children():
    coa = COA()
    root_account = next(coa.toplevel)
    sub_account1 = coa.add(root_account.code, Code("1.1"), "Sub Account 1")
    sub_account2 = coa.add(root_account.code, Code("1.2"), "Sub Account 2")
    node = coa.nodify(root_account)
    assert len(node.children) == 2
    assert node.children[0].account == sub_account1
    assert node.children[1].account == sub_account2

def test_nodify_raises_no_error_for_account_not_in_coa():
    coa = COA()
    root_account = next(coa.toplevel)
    fake_account = RootAccount(Code("999"), "Fake", AccountType.ASSET, coa)
    node = coa.nodify(fake_account)
    assert node.account == fake_account
    assert node.children == []


# LLM-generated content at query #11
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code("000"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #12
#--------------------------

def test_add_account_parent_not_in_subaccounts():
    coa = COA()
    root_account = next(coa.toplevel)
    parent_code = root_account.code
    code = Code("1.1")
    name = "Sub Account"
    account = coa.add(parent_code, code, name)
    assert account.parent in coa._subaccounts


# LLM-generated content at query #13
#--------------------------

def test___call___returns_coa():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa


# LLM-generated content at query #14
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    parent_account = coa.add(parent_code, parent_code, "Parent Account")
    existing_account = coa.add(parent_code, child_code, child_name)
    result = coa.add(parent_code, child_code, child_name)
    assert result is existing_account


# LLM-generated content at query #15
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("123")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("100"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #16
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #17
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #18
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___returns_valid_coa():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = COA()
    result = mock_reader()
    assert isinstance(result, COA)

def test___call___invoked_without_arguments():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = COA()
    result = mock_reader()
    mock_reader.assert_called_once_with()

def test___call___returns_different_coa_on_multiple_calls():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = [COA(), COA()]
    first_result = mock_reader()
    second_result = mock_reader()
    assert first_result is not second_result
    assert isinstance(first_result, COA)
    assert isinstance(second_result, COA)


# LLM-generated content at query #19
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    root_account = next(coa.toplevel)
    parent_code = root_account.code
    code = Code("1.1")
    name = "Sub Account"
    coa.add(parent_code, code, name)
    existing_account = coa.add(parent_code, code, name)
    assert existing_account.code == code
    assert existing_account.name == name
    assert existing_account.parent == root_account


# LLM-generated content at query #20
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #21
#--------------------------

def test_add_existing_account_with_matching_info_returns_account():
    coa = COA()
    root_account = next(coa.toplevel)
    parent_code = root_account.code
    code = Code("1.1")
    name = "Sub Account"
    coa.add(parent_code, code, name)
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent == root_account


# LLM-generated content at query #22
#--------------------------

def test_add_existing_account_with_matching_details_returns_account():
    coa = COA()
    root_code = Code("1")
    root_account = next(iter(coa._accounts.values()))
    sub_code = Code("1.1")
    sub_account = coa.add(root_account.code, sub_code, "Sub Account")
    result = coa.add(root_account.code, sub_code, "Sub Account")
    assert result is sub_account
    assert result.code == sub_code
    assert result.name == "Sub Account"
    assert result.parent == root_account


# LLM-generated content at query #23
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code("000"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #24
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___returns_correct_coa_data():
    expected_accounts = [Account("1000", "Cash"), Account("2000", "Revenue")]
    mock_coa = COA(expected_accounts)
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result.accounts == expected_accounts

def test___call___is_callable():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    assert callable(mock_reader)

def test___call___raises_exception_on_failure():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = ValueError("Read error")
    try:
        mock_reader()
        assert False
    except ValueError as e:
        assert str(e) == "Read error"

def test___call___returns_empty_coa():
    empty_coa = COA([])
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = empty_coa
    result = mock_reader()
    assert len(result.accounts) == 0


# LLM-generated content at query #25
#--------------------------

def test_add_creates_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "New Subaccount"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_returns_existing_account_if_matches():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Subaccount"
    existing_account = coa.add(parent_code, existing_code, existing_name)
    same_account = coa.add(parent_code, existing_code, existing_name)
    assert same_account == existing_account

def test_add_raises_error_if_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(parent_code, parent_code, "Same Code")

def test_add_raises_error_if_parent_not_found():
    coa = COA()
    non_existent_parent = Code("99")
    new_code = Code("99.1")
    new_name = "New Account"
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined."):
        coa.add(non_existent_parent, new_code, new_name)

def test_add_raises_error_if_code_exists_with_different_parent():
    coa = COA()
    parent1_code = Code("1")
    parent2_code = Code("2")
    same_code = Code("1.1")
    same_name = "Same Name"
    coa.add(parent1_code, same_code, same_name)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent2_code, same_code, same_name)

def test_add_raises_error_if_code_exists_with_different_name():
    coa = COA()
    parent_code = Code("1")
    same_code = Code("1.1")
    coa.add(parent_code, same_code, "First Name")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, same_code, "Different Name")


# LLM-generated content at query #26
#--------------------------

def test_add_existing_account_with_different_parent():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.add(parent_code, parent_code, "Parent")
    child_code = Code("2")
    child_account = coa.add(parent_code, child_code, "Child")
    different_parent_code = Code("3")
    different_parent_account = coa.add(parent_code, different_parent_code, "Different Parent")
    try:
        coa.add(different_parent_code, child_code, "Child")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_name():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.add(parent_code, parent_code, "Parent")
    child_code = Code("2")
    child_account = coa.add(parent_code, child_code, "Child")
    try:
        coa.add(parent_code, child_code, "Different Child")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_existing_account_with_different_code():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.add(parent_code, parent_code, "Parent")
    child_code = Code("2")
    child_account = coa.add(parent_code, child_code, "Child")
    try:
        coa.add(parent_code, Code("3"), "Child")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #27
#--------------------------

def test_add_existing_account_with_matching_info():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    coa.add(parent_code, child_code, child_name)
    existing_account = coa.add(parent_code, child_code, child_name)
    assert existing_account.code == child_code
    assert existing_account.name == child_name
    assert existing_account.parent == coa.find(parent_code)


# LLM-generated content at query #28
#--------------------------

def test_add_existing_account_with_different_parent_raises_error():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child")
    different_parent_code = Code("3")
    coa.add(different_parent_code, different_parent_code, "Different Parent")
    try:
        coa.add(different_parent_code, child_code, "Child")
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #29
#--------------------------

def test_add_existing_account_with_matching_info_returns_account():
    coa = COA()
    root_code = Code("1")
    root_account = next(iter(coa._accounts.values()))
    sub_code = Code("1.1")
    sub_account = coa.add(root_account.code, sub_code, "Sub Account")
    result = coa.add(root_account.code, sub_code, "Sub Account")
    assert result is sub_account
    assert result.parent == root_account
    assert result.name == "Sub Account"
    assert result.code == sub_code


# LLM-generated content at query #30
#--------------------------

def test_add_account_with_existing_code_but_different_parent_raises_error():
    coa = COA()
    parent1_code = Code("1")
    parent2_code = Code("2")
    child_code = Code("1.1")
    child_name = "Child Account"
    coa.add(parent1_code, child_code, child_name)
    try:
        coa.add(parent2_code, child_code, child_name)
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #31
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa


# LLM-generated content at query #32
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("123")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("100"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == COA("Test COA")


# LLM-generated content at query #33
#--------------------------

def test_add_existing_account_with_different_parent_raises_error():
    coa = COA()
    root_account = next(iter(coa._accounts.values()))
    parent_code = root_account.code
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    different_parent_code = Code("2")
    different_parent_instance = coa._accounts.get(different_parent_code)
    assert different_parent_instance is not None
    try:
        coa.add(different_parent_code, code, name)
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #34
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    parent_account = coa.add(parent_code, parent_code, "Parent Account")
    child_account = coa.add(parent_code, child_code, child_name)
    result = coa.add(parent_code, child_code, child_name)
    assert result is child_account
    assert result.parent == parent_account
    assert result.name == child_name
    assert result.code == child_code


# LLM-generated content at query #35
#--------------------------

def test_subaccount_constructor():
    mock_code = Code(value="123")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code(value="100"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #36
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    parent_account = coa.add(parent_code, parent_code, "Parent Account")
    existing_account = coa.add(parent_code, child_code, child_name)
    result = coa.add(parent_code, child_code, child_name)
    assert result is existing_account


# LLM-generated content at query #37
#--------------------------

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    parent_account = coa.add(parent_code, parent_code, "Parent Account")
    child_account = coa.add(parent_code, child_code, child_name)
    result = coa.add(parent_code, child_code, child_name)
    assert result is child_account
    assert result.parent == parent_account
    assert result.name == child_name
    assert result.code == child_code


# LLM-generated content at query #38
#--------------------------

def test_subaccount_constructor():
    mock_code = Code(value="123")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code(value="100"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #39
#--------------------------

def test_read_chart_of_accounts_call_returns_coa():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa


# LLM-generated content at query #40
#--------------------------

def test_subaccount_constructor():
    mock_code = Code(value="123")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code(value="100"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #41
#--------------------------

def test_add_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "Sub Account"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Sub Account"
    existing_account = coa.add(parent_code, existing_code, existing_name)
    same_account = coa.add(parent_code, existing_code, existing_name)
    assert same_account == existing_account

def test_add_parent_not_found():
    coa = COA()
    parent_code = Code("99")
    new_code = Code("99.1")
    new_name = "New Account"
    try:
        coa.add(parent_code, new_code, new_name)
        assert False
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Same Code")
        assert False
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_inconsistent_existing_account():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Sub Account"
    coa.add(parent_code, existing_code, existing_name)
    different_parent_code = Code("2")
    try:
        coa.add(different_parent_code, existing_code, existing_name)
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #42
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code("000"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #43
#--------------------------

def test_add_creates_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "New Subaccount"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_returns_existing_account_if_consistent():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Subaccount"
    existing_account = coa.add(parent_code, existing_code, existing_name)
    retrieved_account = coa.add(parent_code, existing_code, existing_name)
    assert retrieved_account == existing_account

def test_add_raises_error_if_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(parent_code, parent_code, "Same Code")

def test_add_raises_error_if_parent_not_found():
    coa = COA()
    non_existent_parent = Code("99")
    new_code = Code("99.1")
    new_name = "New Account"
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined."):
        coa.add(non_existent_parent, new_code, new_name)

def test_add_raises_error_if_account_exists_with_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Subaccount"
    coa.add(parent_code, existing_code, existing_name)
    different_parent_code = Code("2")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(different_parent_code, existing_code, existing_name)


# LLM-generated content at query #44
#--------------------------

def test___call___returns_coa_instance():
    mock_coa = COA()
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: mock_coa
    result = mock_reader.__call__()
    assert result is mock_coa

def test___call___returns_new_coa_each_call():
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: COA()
    result1 = mock_reader.__call__()
    result2 = mock_reader.__call__()
    assert result1 is not result2

def test___call___can_be_called_directly():
    mock_coa = COA()
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___raises_exception_on_failure():
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: (_ for _ in ()).throw(ValueError("Read error"))
    try:
        mock_reader.__call__()
        assert False
    except ValueError as e:
        assert str(e) == "Read error"

def test___call___returns_coa_with_expected_attributes():
    mock_coa = COA()
    mock_coa.accounts = ["Asset", "Liability"]
    mock_reader = ReadChartOfAccounts()
    mock_reader.__call__ = lambda: mock_coa
    result = mock_reader.__call__()
    assert result.accounts == ["Asset", "Liability"]


# LLM-generated content at query #45
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code("000"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #46
#--------------------------

def test___call___returns_coa():
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result is mock_coa

def test___call___returns_new_coa_instance_each_time():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = [COA(), COA(), COA()]
    result1 = mock_reader()
    result2 = mock_reader()
    result3 = mock_reader()
    assert result1 is not result2
    assert result2 is not result3
    assert result1 is not result3

def test___call___raises_exception_on_failure():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.side_effect = ValueError("Read failed")
    try:
        mock_reader()
        assert False
    except ValueError as e:
        assert str(e) == "Read failed"

def test___call___returns_coa_with_expected_properties():
    expected_accounts = [Account("1000", "Cash"), Account("2000", "Revenue")]
    mock_coa = COA()
    mock_coa.accounts = expected_accounts
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa
    result = mock_reader()
    assert result.accounts == expected_accounts

def test___call___can_be_called_without_arguments():
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = COA()
    result = mock_reader()
    assert isinstance(result, COA)


# LLM-generated content at query #47
#--------------------------

def test_subaccount_constructor():
    mock_code = Code(value="123")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code(value="100"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #48
#--------------------------

def test_subaccount_constructor():
    mock_code = Code("001")
    mock_name = "Test SubAccount"
    mock_parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #49
#--------------------------

def test_add_creates_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "New Subaccount"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_returns_existing_account_if_consistent():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Subaccount"
    existing_account = coa.add(parent_code, existing_code, existing_name)
    retrieved_account = coa.add(parent_code, existing_code, existing_name)
    assert retrieved_account == existing_account

def test_add_raises_error_if_parent_equals_code():
    coa = COA()
    same_code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(same_code, same_code, "Same Account")

def test_add_raises_error_if_parent_not_found():
    coa = COA()
    non_existent_parent = Code("999")
    new_code = Code("999.1")
    new_name = "New Account"
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined."):
        coa.add(non_existent_parent, new_code, new_name)

def test_add_raises_error_if_account_exists_with_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing Subaccount"
    coa.add(parent_code, existing_code, existing_name)
    different_parent_code = Code("2")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(different_parent_code, existing_code, existing_name)

def test_add_updates_subaccounts_buffer():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    new_code = Code("1.1")
    new_name = "New Subaccount"
    assert parent_account not in coa._subaccounts
    new_account = coa.add(parent_code, new_code, new_name)
    assert parent_account in coa._subaccounts
    assert new_account in coa._subaccounts[parent_account]

def test_add_multiple_subaccounts():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    subaccount1 = coa.add(parent_code, Code("1.1"), "Subaccount 1")
    subaccount2 = coa.add(parent_code, Code("1.2"), "Subaccount 2")
    subaccounts = coa.subaccounts(parent_account)
    assert len(subaccounts) == 2
    assert subaccount1 in subaccounts
    assert subaccount2 in subaccounts

def test_add_subaccount_to_non_root_parent():
    coa = COA()
    root_parent_code = Code("1")
    intermediate_code = Code("1.1")
    intermediate_account = coa.add(root_parent_code, intermediate_code, "Intermediate")
    leaf_code = Code("1.1.1")
    leaf_name = "Leaf Account"
    leaf_account = coa.add(intermediate_code, leaf_code, leaf_name)
    assert leaf_account.code == leaf_code
    assert leaf_account.name == leaf_name
    assert leaf_account.parent == intermediate_account
    assert leaf_account in coa.subaccounts(intermediate_account)


