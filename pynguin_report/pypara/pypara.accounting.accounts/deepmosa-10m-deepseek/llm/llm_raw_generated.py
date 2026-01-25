####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.REVENUE: (Code("4"), "Revenue"),
        AccountType.EXPENSE: (Code("5"), "Expenses"),
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa is coa
        expected_code, expected_name = rootspec[account.type]
        assert account.code == expected_code
        assert account.name == expected_name

def test_coa_constructor_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa is coa
        if account.type in rootspec:
            expected_code, expected_name = rootspec[account.type]
            assert account.code == expected_code
            assert account.name == expected_name
        else:
            assert account.code == Code(str(account.type.value))
            assert account.name == account.type.name.capitalize()


# LLM-generated content at query #2
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == 0
    assert len(list(coa.toplevel)) == 0
    assert len(list(coa.structure)) == 0

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: ("1", "Assets"),
        AccountType.LIABILITY: ("2", "Liabilities")
    }
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == 2
    assert accounts[0][0] == "1"
    assert accounts[0][1].name == "Assets"
    assert accounts[0][1].type == AccountType.ASSET
    assert accounts[1][0] == "2"
    assert accounts[1][1].name == "Liabilities"
    assert accounts[1][1].type == AccountType.LIABILITY

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == 0
    assert len(list(coa.toplevel)) == 0
    assert len(list(coa.structure)) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Custom Liability"
    assert coa.find(Code("3")).name == "Equity"  # default for unspecified types


# LLM-generated content at query #4
#--------------------------

```
def test_add_account_successfully():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    new_account = coa.add(parent_code, Code("1.1"), "Child Account")
    assert new_account.code == Code("1.1")
    assert new_account.name == "Child Account"
    assert new_account.parent == parent_account
    assert new_account in coa._subaccounts[parent_account]

def test_add_account_with_existing_code_consistent_info():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("1.1"), "Child Account", parent_account)
    coa._accounts[Code("1.1")] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    result = coa.add(parent_code, Code("1.1"), "Child Account")
    assert result == existing_account

def test_add_account_with_existing_code_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("1.1"), "Existing Child", parent_account)
    coa._accounts[Code("1.1")] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    try:
        coa.add(parent_code, Code("1.1"), "New Child Name")
        assert False
    except ValueError:
        assert True

def test_add_account_with_nonexistent_parent():
    coa = COA()
    try:
        coa.add(Code("999"), Code("1.1"), "Child Account")
        assert False
    except ValueError:
        assert True

def test_add_account_with_self_as_parent():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    try:
        coa.add(parent_code, parent_code, "Self Parent Account")
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #5
#--------------------------

```
def test_subaccount_constructor():
    parent_account = Account(code=Code("1000"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("1001"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("1001")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #6
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("1000"), name="Parent", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("1001"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("1001")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #7
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.code == Code(str(i+1)) for i, account in enumerate(coa.accounts))
    assert all(account.name == account.type.name.capitalize() for account in coa.accounts)

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("A"), "Custom Asset"),
        AccountType.LIABILITY: (Code("L"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("A")).name == "Custom Asset"
    assert coa.find(Code("L")).name == "Custom Liability"
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("1"), "Partial Asset"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Partial Asset"
    assert coa.find(Code("2")).name == "Liability"  # Default naming for unspecified types
    assert all(isinstance(account, RootAccount) for account in coa.accounts)

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(account.code == Code(str(i+1)) for i, account in enumerate(coa.accounts))
    assert all(account.name == account.type.name.capitalize() for account in coa.accounts)

def test_coa_constructor_frozen():
    coa = COA()
    with pytest.raises(dataclasses.FrozenInstanceError):
        coa._accounts = {}


# LLM-generated content at query #8
#--------------------------

```python
def test_nodify_with_no_subaccounts():
    coa = COA()
    account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_with_subaccounts():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Parent Account", AccountType.ASSET, coa)
    child_account = SubAccount(Code("2"), "Child Account", parent_account)
    coa._accounts[Code("1")] = parent_account
    coa._accounts[Code("2")] = child_account
    coa._subaccounts[parent_account] = [child_account]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
    assert node.children[0].children == []

def test_nodify_with_multiple_subaccounts():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Parent Account", AccountType.ASSET, coa)
    child_account1 = SubAccount(Code("2"), "Child Account 1", parent_account)
    child_account2 = SubAccount(Code("3"), "Child Account 2", parent_account)
    coa._accounts[Code("1")] = parent_account
    coa._accounts[Code("2")] = child_account1
    coa._accounts[Code("3")] = child_account2
    coa._subaccounts[parent_account] = [child_account1, child_account2]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 2
    assert node.children[0].account == child_account1
    assert node.children[0].children == []
    assert node.children[1].account == child_account2
    assert node.children[1].children == []


# LLM-generated content at query #9
#--------------------------

```python
def test_nodify_single_account():
    coa = COA()
    account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    coa._accounts[Code("1")] = account
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_account_with_subaccounts():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    sub_account = SubAccount(Code("2"), "Sub Account", parent_account)
    coa._accounts[Code("1")] = parent_account
    coa._accounts[Code("2")] = sub_account
    coa._subaccounts[parent_account] = [sub_account]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == sub_account
    assert node.children[0].children == []

def test_nodify_account_with_multiple_subaccounts():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    sub_account1 = SubAccount(Code("2"), "Sub Account 1", parent_account)
    sub_account2 = SubAccount(Code("3"), "Sub Account 2", parent_account)
    coa._accounts[Code("1")] = parent_account
    coa._accounts[Code("2")] = sub_account1
    coa._accounts[Code("3")] = sub_account2
    coa._subaccounts[parent_account] = [sub_account1, sub_account2]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 2
    assert node.children[0].account == sub_account1
    assert node.children[0].children == []
    assert node.children[1].account == sub_account2
    assert node.children[1].children == []

def test_nodify_account_with_nested_subaccounts():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    sub_account1 = SubAccount(Code("2"), "Sub Account 1", parent_account)
    sub_account2 = SubAccount(Code("3"), "Sub Account 2", sub_account1)
    coa._accounts[Code("1")] = parent_account
    coa._accounts[Code("2")] = sub_account1
    coa._accounts[Code("3")] = sub_account2
    coa._subaccounts[parent_account] = [sub_account1]
    coa._subaccounts[sub_account1] = [sub_account2]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == sub_account1
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == sub_account2
    assert node.children[0].children[0].children == []


# LLM-generated content at query #10
#--------------------------

```python
def test_add_account_with_undefined_parent_raises_error():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    try:
        coa.add(parent_code, child_code, "Child Account")
        assert False, "Expected ValueError but no exception was raised"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."


# LLM-generated content at query #11
#--------------------------

```python
def test_subaccount_constructor():
    code = "123"
    name = "Test Account"
    parent = Account(code="100", name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent
    assert subaccount.type == parent.type
    assert subaccount.coa == parent.coa


# LLM-generated content at query #12
#--------------------------

```python
def test_nodify_returns_correct_node():
    coa = COA()
    parent_code = Code("1")
    parent_account = SubAccount(parent_code, "Parent Account", None)
    coa._accounts[parent_code] = parent_account
    child_code = Code("2")
    child_account = SubAccount(child_code, "Child Account", parent_account)
    coa._accounts[child_code] = child_account
    coa._subaccounts[parent_account] = [child_account]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account


# LLM-generated content at query #13
#--------------------------

```python
def test_subaccount_constructor():
    code = "123"
    name = "Test SubAccount"
    parent = Account(code="100", name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    sub_account = SubAccount(code=code, name=name, parent=parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent
    assert sub_account.type == parent.type
    assert sub_account.coa == parent.coa


# LLM-generated content at query #14
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(Code("100"), "Test Account", AccountType.ASSET, COA())
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent
    assert sub_account.type == parent.type
    assert sub_account.coa == parent.coa


# LLM-generated content at query #15
#--------------------------

```
def test_nodify_returns_node_with_correct_account_and_empty_children_for_leaf_account():
    coa = COA()
    account = coa.add(Code("1"), Code("2"), "Test Account")
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_returns_node_with_correct_account_and_children_for_parent_account():
    coa = COA()
    parent = coa.add(Code("1"), Code("2"), "Parent Account")
    child = coa.add(Code("2"), Code("3"), "Child Account")
    node = coa.nodify(parent)
    assert node.account == parent
    assert len(node.children) == 1
    assert node.children[0].account == child

def test_nodify_returns_node_with_correct_account_and_nested_children():
    coa = COA()
    parent = coa.add(Code("1"), Code("2"), "Parent Account")
    child = coa.add(Code("2"), Code("3"), "Child Account")
    grandchild = coa.add(Code("3"), Code("4"), "Grandchild Account")
    node = coa.nodify(parent)
    assert node.account == parent
    assert len(node.children) == 1
    assert node.children[0].account == child
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == grandchild

def test_nodify_returns_node_with_correct_account_for_root_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert isinstance(node, COA.Node)


# LLM-generated content at query #16
#--------------------------

```python
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)

def test_coa_constructor_custom_rootspec():
    rootspec = {AccountType.ASSET: (Code("1"), "Custom Asset")}
    coa = COA(rootspec=rootspec)
    account = coa.find(Code("1"))
    assert account is not None
    assert account.name == "Custom Asset"
    assert account.type == AccountType.ASSET

def test_coa_constructor_partial_rootspec():
    rootspec = {AccountType.LIABILITY: (Code("2"), "Custom Liability")}
    coa = COA(rootspec=rootspec)
    liability_account = coa.find(Code("2"))
    assert liability_account is not None
    assert liability_account.name == "Custom Liability"
    assert liability_account.type == AccountType.LIABILITY
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Asset"

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    for c, a in coa:
        assert a.name == a.type.name.capitalize()


# LLM-generated content at query #17
#--------------------------

```python
def test_parent_instance_not_none():
    coa = COA()
    parent_code = Code("1")
    coa.add(parent_code, Code("2"), "Account 2")
    parent_instance = coa.find(parent_code)
    assert parent_instance is not None


# LLM-generated content at query #18
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("100"), "Custom Assets"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("100")).name == "Custom Assets"
    assert coa.find(Code("2")).name == "Liability"  # Default name for second account type
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Asset"  # Default name for first account type
    assert coa.find(Code("2")).name == "Liability"  # Default name for second account type
    assert all(account.parent is None for account in coa.toplevel)


# LLM-generated content at query #19
#--------------------------

```
def test_add_account_with_valid_parent_and_code():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    account = coa.add(parent_code, child_code, child_name)
    assert account.code == child_code
    assert account.name == child_name
    assert account.parent.code == parent_code
    assert child_code in coa._accounts
    assert account in coa._subaccounts.get(coa._accounts[parent_code], [])

def test_add_account_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    try:
        coa.add(parent_code, child_code, child_name)
        assert False
    except ValueError:
        assert True

def test_add_account_with_duplicate_code_consistent_info():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    account1 = coa.add(parent_code, child_code, child_name)
    account2 = coa.add(parent_code, child_code, child_name)
    assert account1 == account2

def test_add_account_with_duplicate_code_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name1 = "Child Account 1"
    child_name2 = "Child Account 2"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa.add(parent_code, child_code, child_name1)
    try:
        coa.add(parent_code, child_code, child_name2)
        assert False
    except ValueError:
        assert True

def test_add_account_with_self_as_parent():
    coa = COA()
    code = Code("1")
    name = "Account"
    coa._accounts[code] = RootAccount(code, name, AccountType.ASSET, coa)
    try:
        coa.add(code, code, name)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #20
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("001")
    name = "Test SubAccount"
    parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #21
#--------------------------

```
def test___call___returns_COA_instance():
    mock_coa = COA()
    mock_reader = lambda: mock_coa
    result = mock_reader()
    assert result is mock_coa


# LLM-generated content at query #22
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    reader = MockReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #23
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.REVENUE: (Code("4"), "Revenue"),
        AccountType.EXPENSE: (Code("5"), "Expenses"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(account.code == custom_rootspec[account.type][0] for account in coa.accounts)
    assert all(account.name == custom_rootspec[account.type][1] for account in coa.accounts)

def test_coa_constructor_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("A"), "Custom Assets"),
        AccountType.EXPENSE: (Code("E"), "Custom Expenses"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    
    asset_account = next(acc for acc in coa.accounts if acc.type == AccountType.ASSET)
    assert asset_account.code == Code("A")
    assert asset_account.name == "Custom Assets"
    
    expense_account = next(acc for acc in coa.accounts if acc.type == AccountType.EXPENSE)
    assert expense_account.code == Code("E")
    assert expense_account.name == "Custom Expenses"
    
    other_types = set(AccountType) - {AccountType.ASSET, AccountType.EXPENSE}
    for acc in coa.accounts:
        if acc.type in other_types:
            assert acc.code == Code(str(acc.type.value))
            assert acc.name == acc.type.name.capitalize()

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    for acc in coa.accounts:
        assert acc.code == Code(str(acc.type.value))
        assert acc.name == acc.type.name.capitalize()


# LLM-generated content at query #24
#--------------------------

```python
def test_subaccount_constructor():
    code = "123"
    name = "Test SubAccount"
    parent = Account(code="100", name="Parent Account", type=AccountType.ASSET, coa=COA(name="Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent
    assert subaccount.type == parent.type
    assert subaccount.coa == parent.coa


# LLM-generated content at query #25
#--------------------------

```python
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa == coa

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.REVENUE: (Code("4"), "Revenue"),
        AccountType.EXPENSE: (Code("5"), "Expenses"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa == coa
        expected_code, expected_name = custom_rootspec[account.type]
        assert account.code == expected_code
        assert account.name == expected_name

def test_coa_constructor_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    
    # Check accounts specified in rootspec
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.type == AccountType.ASSET
    assert asset_account.name == "Assets"
    
    liability_account = coa.find(Code("2"))
    assert liability_account is not None
    assert liability_account.type == AccountType.LIABILITY
    assert liability_account.name == "Liabilities"
    
    # Check other accounts have default values
    for account in coa.accounts:
        if account.type not in partial_rootspec:
            assert account.code == Code(str(account.type.value))
            assert account.name == account.type.name.capitalize()

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa == coa
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_none_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa == coa
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()


# LLM-generated content at query #26
#--------------------------

```python
def test_read_chart_of_accounts_call():
    mock_coa = COA()
    mock_func = lambda: mock_coa
    assert mock_func() == mock_coa


# LLM-generated content at query #27
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    mock_coa = COA()
    mock_read_func = lambda: mock_coa
    assert isinstance(mock_read_func(), COA)


# LLM-generated content at query #28
#--------------------------

```python
def test_sub_account_constructor():
    parent_account = Account(code=Code("100"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("101"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("101")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account


# LLM-generated content at query #29
#--------------------------

```python
def test_COA_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)

def test_COA_constructor_with_custom_rootspec():
    custom_rootspec = {AccountType.ASSET: (Code("1"), "Custom Asset")}
    coa = COA(rootspec=custom_rootspec)
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Custom Asset"

def test_COA_constructor_with_partial_rootspec():
    custom_rootspec = {AccountType.ASSET: (Code("1"), "Custom Asset")}
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Custom Asset"

def test_COA_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Asset"

def test_COA_constructor_with_none_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == len(AccountType)
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Asset"


# LLM-generated content at query #30
#--------------------------

```
def test_nodify_returns_node_with_correct_account_and_children():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert isinstance(node.children, list)

def test_nodify_returns_empty_children_list_for_account_without_subaccounts():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert len(node.children) == 0

def test_nodify_returns_node_with_subaccounts_when_present():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    coa.add(parent_code, child_code, "Child Account")
    parent_account = coa.find(parent_code)
    node = coa.nodify(parent_account)
    assert len(node.children) == 1
    assert node.children[0].account.code == child_code

def test_nodify_raises_no_error_for_nonexistent_account():
    coa = COA()
    nonexistent_account = Account(Code("999"), "Nonexistent", None, coa)
    node = coa.nodify(nonexistent_account)
    assert node.account == nonexistent_account
    assert len(node.children) == 0

def test_nodify_maintains_tree_structure_correctly():
    coa = COA()
    parent_code = Code("1")
    child1_code = Code("1.1")
    child2_code = Code("1.2")
    grandchild_code = Code("1.1.1")
    coa.add(parent_code, child1_code, "Child 1")
    coa.add(parent_code, child2_code, "Child 2")
    coa.add(child1_code, grandchild_code, "Grandchild")
    parent_account = coa.find(parent_code)
    node = coa.nodify(parent_account)
    assert len(node.children) == 2
    assert any(child.account.code == child1_code for child in node.children)
    assert any(child.account.code == child2_code for child in node.children)
    child1_node = next(child for child in node.children if child.account.code == child1_code)
    assert len(child1_node.children) == 1
    assert child1_node.children[0].account.code == grandchild_code


# LLM-generated content at query #31
#--------------------------

```
def test_add_account_with_defined_parent():
    coa = COA()
    root_code = Code("1")
    root_account = coa._accounts[root_code]
    sub_code = Code("2")
    sub_name = "Sub Account"
    coa.add(root_code, sub_code, sub_name)
    assert coa.find(sub_code) is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    for account in accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Asset"),
        AccountType.LIABILITY: (Code("2"), "Liability"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.REVENUE: (Code("4"), "Revenue"),
        AccountType.EXPENSE: (Code("5"), "Expense"),
    }
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    for account in accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.code == rootspec[account.type][0]
        assert account.name == rootspec[account.type][1]


# LLM-generated content at query #33
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test Account"
    parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA())
    subaccount = SubAccount(code, name, parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #34
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("100"), name="Cash", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("1001"), name="Petty Cash", parent=parent_account)
    assert sub_account.code == Code("1001")
    assert sub_account.name == "Petty Cash"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #35
#--------------------------

```python
def test_ReadChartOfAccounts_call():
    class MockCOA:
        pass

    def mock_read_coa() -> MockCOA:
        return MockCOA()

    read_chart_of_accounts = ReadChartOfAccounts()
    read_chart_of_accounts.__call__ = mock_read_coa
    result = read_chart_of_accounts()
    assert isinstance(result, MockCOA)


# LLM-generated content at query #36
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(list(AccountType).index(account.type) + 1))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("A1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("L1"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        if account.type in custom_rootspec:
            assert account.code == custom_rootspec[account.type][0]
            assert account.name == custom_rootspec[account.type][1]
        else:
            assert account.code == Code(str(list(AccountType).index(account.type) + 1))
            assert account.name == account.type.name.capitalize()

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(list(AccountType).index(account.type) + 1))
        assert account.name == account.type.name.capitalize()


# LLM-generated content at query #37
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("100"), name="Parent Account", type=AccountType.ASSET, coa=COA(name="Main COA"))
    subaccount = SubAccount(code=Code("101"), name="Sub Account", parent=parent_account)
    assert subaccount.code == Code("101")
    assert subaccount.name == "Sub Account"
    assert subaccount.parent == parent_account


# LLM-generated content at query #38
#--------------------------

```
def test_add_account_with_defined_parent():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    coa.add(parent_code, Code("2"), "Child Account")


# LLM-generated content at query #39
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test Account"
    parent = Account(Code("456"), "Parent Account", AccountType.ASSET, COA())
    subaccount = SubAccount(code, name, parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #40
#--------------------------

```python
def test_read_chart_of_accounts_call():
    mock_coa = {"assets": 1000, "liabilities": 500}
    mock_reader = lambda: mock_coa
    result = mock_reader()
    assert result == mock_coa


# LLM-generated content at query #41
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    root_account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #42
#--------------------------

```
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Custom Liability"
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_partial_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Liability"
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Asset"
    assert coa.find(Code("2")).name == "Liability"
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_none_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Asset"
    assert coa.find(Code("2")).name == "Liability"
    assert all(account.parent is None for account in coa.toplevel)


# LLM-generated content at query #43
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    root_account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #44
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test Account"
    parent = Account(Code("100"), "Parent Account", AccountType.ASSET, COA("Main COA"))
    subaccount = SubAccount(code, name, parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent
    assert subaccount.type == AccountType.ASSET
    assert subaccount.coa == COA("Main COA")


# LLM-generated content at query #45
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: ("1", "Custom Asset"),
        AccountType.LIABILITY: ("2", "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Custom Asset"
    assert asset_account.type == AccountType.ASSET
    
    liability_account = coa.find(Code("2"))
    assert liability_account is not None
    assert liability_account.name == "Custom Liability"
    assert liability_account.type == AccountType.LIABILITY
    
    for account in coa.accounts:
        if account.code not in (Code("1"), Code("2")):
            assert account.name == account.type.name.capitalize()

def test_coa_constructor_with_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: ("1", "Partial Asset"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Partial Asset"
    assert asset_account.type == AccountType.ASSET
    
    for account in coa.accounts:
        if account.code != Code("1"):
            assert account.name == account.type.name.capitalize()


# LLM-generated content at query #46
#--------------------------

```python
def test_add_account_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."


# LLM-generated content at query #47
#--------------------------

```python
def test_subaccount_constructor():
    mock_code = "123"
    mock_name = "Test Account"
    mock_parent = Account(code="000", name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #48
#--------------------------

```python
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(list(AccountType).index(account.type) + 1))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1000"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2000"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for code, account in coa:
        assert isinstance(account, RootAccount)
        if account.type in custom_rootspec:
            expected_code, expected_name = custom_rootspec[account.type]
            assert account.code == expected_code
            assert account.name == expected_name
        else:
            assert account.code == Code(str(list(AccountType).index(account.type) + 1))
            assert account.name == account.type.name.capitalize()

def test_coa_constructor_partial_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1000"), "Custom Asset"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for code, account in coa:
        assert isinstance(account, RootAccount)
        if account.type in custom_rootspec:
            expected_code, expected_name = custom_rootspec[account.type]
            assert account.code == expected_code
            assert account.name == expected_name
        else:
            assert account.code == Code(str(list(AccountType).index(account.type) + 1))
            assert account.name == account.type.name.capitalize()


# LLM-generated content at query #49
#--------------------------

```python
def test_ReadChartOfAccounts___call___returns_COA():
    mock_coa = COA()
    mock_read_func = lambda: mock_coa
    assert mock_read_func() == mock_coa


# LLM-generated content at query #50
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == 0
    assert len(list(coa.toplevel)) == 0
    assert len(list(coa.structure)) == 0

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: ("1", "Assets"),
        AccountType.LIABILITY: ("2", "Liabilities")
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == 2
    assert len(list(coa.toplevel)) == 2
    assert len(list(coa.structure)) == 2
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: ("1", "Assets")
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == 1
    assert len(list(coa.toplevel)) == 1
    assert len(list(coa.structure)) == 1
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")) is None

def test_coa_constructor_with_empty_rootspec():
    rootspec = {}
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == 0
    assert len(list(coa.toplevel)) == 0
    assert len(list(coa.structure)) == 0


# LLM-generated content at query #51
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)

def test_coa_constructor_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"

def test_coa_constructor_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liability"

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert coa.find(Code("1")).name == "Asset"
    assert coa.find(Code("2")).name == "Liability"


# LLM-generated content at query #52
#--------------------------

```python
def test_parent_account_not_defined_raises_error():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."


# LLM-generated content at query #53
#--------------------------

```python
def test_subaccount_constructor():
    code = "123"
    name = "Test Account"
    parent = Account(code="100", name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent
    assert subaccount.type == parent.type
    assert subaccount.coa == parent.coa


# LLM-generated content at query #54
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    root_account = RootAccount(Code("1"), "Root Account", AccountType.ASSET, coa)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #55
#--------------------------

```python
def test_read_chart_of_accounts_call():
    mock_coa = {"assets": 1000, "liabilities": 500}
    mock_caller = lambda: mock_coa
    result = mock_caller()
    assert result == mock_coa


# LLM-generated content at query #56
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test Account"
    parent_account = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    subaccount = SubAccount(code, name, parent_account)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent_account


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_read_chart_of_accounts_call():
    class MockCOA:
        pass

    def mock_read_coa() -> MockCOA:
        return MockCOA()

    reader: ReadChartOfAccounts = mock_read_coa
    result = reader()
    assert isinstance(result, MockCOA)


# LLM-generated content at query #2
#--------------------------

```python
def test_COA_constructor_default():
    coa = COA(None)
    assert len(list(coa.accounts)) == len(AccountType)
    for code, account in coa:
        assert isinstance(code, Code)
        assert isinstance(account, RootAccount)
        assert account.code == code
        assert account.name == account.type.name.capitalize()
        assert account.parent is None
        assert account.coa is coa

def test_COA_constructor_custom_rootspec():
    rootspec = {AccountType.ASSET: (Code("1"), "Asset Account"), AccountType.LIABILITY: (Code("2"), "Liability Account")}
    coa = COA(rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for code, account in coa:
        assert isinstance(code, Code)
        assert isinstance(account, RootAccount)
        if account.type == AccountType.ASSET:
            assert account.code == Code("1")
            assert account.name == "Asset Account"
        elif account.type == AccountType.LIABILITY:
            assert account.code == Code("2")
            assert account.name == "Liability Account"
        else:
            assert account.code == Code(str(account.type.value))
            assert account.name == account.type.name.capitalize()
        assert account.parent is None
        assert account.coa is coa


# LLM-generated content at query #3
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.REVENUE: (Code("4"), "Revenue"),
        AccountType.EXPENSE: (Code("5"), "Expense"),
    }
    coa = COA(rootspec=custom_rootspec)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Revenue"
    assert coa.find(Code("5")).name == "Expense"

def test_coa_constructor_with_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("10"), "Assets"),
        AccountType.LIABILITY: (Code("20"), "Liabilities"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("10")).name == "Assets"
    assert coa.find(Code("20")).name == "Liabilities"
    assert coa.find(Code("1")).name == AccountType.EQUITY.name.capitalize()
    assert coa.find(Code("2")).name == AccountType.REVENUE.name.capitalize()
    assert coa.find(Code("3")).name == AccountType.EXPENSE.name.capitalize()


# LLM-generated content at query #4
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("parent"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("sub"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("sub")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #5
#--------------------------

```python
def test_nodify_returns_node_with_correct_account_and_children():
    coa = COA()
    root_account = RootAccount(Code("1"), "Root", AccountType.ASSET, coa)
    sub_account = SubAccount(Code("2"), "Sub", root_account)
    coa._accounts[Code("1")] = root_account
    coa._accounts[Code("2")] = sub_account
    coa._subaccounts[root_account] = [sub_account]
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == sub_account

def test_nodify_returns_node_with_no_children_for_account_without_subaccounts():
    coa = COA()
    root_account = RootAccount(Code("1"), "Root", AccountType.ASSET, coa)
    coa._accounts[Code("1")] = root_account
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 0


# LLM-generated content at query #6
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert len(list(coa.toplevel)) == len(AccountType)
    for account in coa.toplevel:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        assert account.coa is coa

def test_coa_constructor_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_spec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert len(list(coa.toplevel)) == len(AccountType)
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Custom Asset"
    assert asset_account.type == AccountType.ASSET
    liability_account = coa.find(Code("2"))
    assert liability_account is not None
    assert liability_account.name == "Custom Liability"
    assert liability_account.type == AccountType.LIABILITY

def test_coa_constructor_partial_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Partial Asset"),
    }
    coa = COA(rootspec=custom_spec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert len(list(coa.toplevel)) == len(AccountType)
    asset_account = coa.find(Code("100"))
    assert asset_account is not None
    assert asset_account.name == "Partial Asset"
    assert asset_account.type == AccountType.ASSET
    default_equity_account = coa.find(Code("3"))
    assert default_equity_account is not None
    assert default_equity_account.name == "Equity"
    assert default_equity_account.type == AccountType.EQUITY


# LLM-generated content at query #7
#--------------------------

```
def test_add_successfully_adds_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    new_account = coa.add(parent_code, Code("1.1"), "Child Account")
    assert new_account.code == Code("1.1")
    assert new_account.name == "Child Account"
    assert new_account.parent == parent_account
    assert new_account in coa._subaccounts[parent_account]

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    try:
        coa.add(parent_code, parent_code, "Same Account")
        assert False
    except ValueError:
        assert True

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    try:
        coa.add(Code("999"), Code("1.1"), "Child Account")
        assert False
    except ValueError:
        assert True

def test_add_raises_error_when_account_exists_with_different_properties():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("1.1"), "Existing Child", parent_account)
    coa._accounts[existing_account.code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    try:
        coa.add(parent_code, Code("1.1"), "Different Name")
        assert False
    except ValueError:
        assert True

def test_add_returns_existing_account_when_properties_match():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("1.1"), "Existing Child", parent_account)
    coa._accounts[existing_account.code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    result = coa.add(parent_code, Code("1.1"), "Existing Child")
    assert result == existing_account


# LLM-generated content at query #8
#--------------------------

```
def test_nodify_returns_node_instance():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #9
#--------------------------

```python
def test_nodify_returns_node_with_correct_account_and_empty_children_for_leaf_account():
    coa = COA()
    account = coa.add(Code("1"), Code("1.1"), "Sub Account")
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_returns_node_with_correct_account_and_children_for_parent_account():
    coa = COA()
    parent = coa.add(Code("1"), Code("1.1"), "Parent Account")
    child1 = coa.add(Code("1.1"), Code("1.1.1"), "Child Account 1")
    child2 = coa.add(Code("1.1"), Code("1.1.2"), "Child Account 2")
    node = coa.nodify(parent)
    assert node.account == parent
    assert len(node.children) == 2
    assert node.children[0].account == child1
    assert node.children[1].account == child2

def test_nodify_returns_node_with_correct_nested_structure():
    coa = COA()
    root = next(coa.toplevel)
    level1 = coa.add(root.code, Code("1.1"), "Level 1 Account")
    level2 = coa.add(Code("1.1"), Code("1.1.1"), "Level 2 Account")
    node = coa.nodify(root)
    assert node.account == root
    assert len(node.children) == 1
    assert node.children[0].account == level1
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == level2


# LLM-generated content at query #10
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {AccountType.ASSET: (Code("A1"), "Asset Account")}
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    asset_account = coa.find(Code("A1"))
    assert asset_account is not None
    assert asset_account.name == "Asset Account"
    for account in coa.accounts:
        if account.type != AccountType.ASSET:
            assert account.code == Code(str(account.type.value))
            assert account.name == account.type.name.capitalize()

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(account.type.value))
        assert account.name == account.type.name.capitalize()


# LLM-generated content at query #11
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert len(list(coa.toplevel)) == len(AccountType)

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    
    asset_account = coa.find(Code("1"))
    liability_account = coa.find(Code("2"))
    
    assert asset_account is not None
    assert asset_account.name == "Custom Asset"
    assert asset_account.type == AccountType.ASSET
    
    assert liability_account is not None
    assert liability_account.name == "Custom Liability"
    assert liability_account.type == AccountType.LIABILITY
    
    assert len(list(coa.accounts)) == len(AccountType)
    assert len(list(coa.toplevel)) == len(AccountType)

def test_coa_constructor_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("1"), "Partial Asset"),
    }
    coa = COA(rootspec=partial_rootspec)
    
    asset_account = coa.find(Code("1"))
    liability_account = coa.find(Code("2"))  # Should be default
    
    assert asset_account is not None
    assert asset_account.name == "Partial Asset"
    assert asset_account.type == AccountType.ASSET
    
    assert liability_account is not None
    assert liability_account.name == "Liability"  # Default name
    assert liability_account.type == AccountType.LIABILITY
    
    assert len(list(coa.accounts)) == len(AccountType)
    assert len(list(coa.toplevel)) == len(AccountType)

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    assert len(list(coa.toplevel)) == len(AccountType)
    
    for i, account in enumerate(coa.toplevel, start=1):
        assert account.code == Code(str(i))
        assert account.name == account.type.name.capitalize()


# LLM-generated content at query #12
#--------------------------

```
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Custom Liability"
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("1"), "Partial Asset"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Partial Asset"
    assert coa.find(Code("2")).name == "Liability"  # default name
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Asset"  # default name
    assert coa.find(Code("2")).name == "Liability"  # default name
    assert all(account.parent is None for account in coa.toplevel)


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadChartOfAccounts___call___returns_COA_object():
    class MockCOA:
        pass

    def mock_read_chart_of_accounts() -> MockCOA:
        return MockCOA()

    reader = ReadChartOfAccounts()
    reader.__call__ = mock_read_chart_of_accounts
    result = reader()
    assert isinstance(result, MockCOA)


# LLM-generated content at query #14
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("ACC01"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("SUB01"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("SUB01")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #15
#--------------------------

```python
def test_subaccount_constructor():
    code = Code()
    name = "Test SubAccount"
    parent = Account()
    sub_account = SubAccount(code=code, name=name, parent=parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #16
#--------------------------

```python
def test_add_account_with_inconsistent_information():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    code = Code("2")
    name = "Child Account"
    account = SubAccount(code, name, parent_account)
    coa._accounts[code] = account
    try:
        coa.add(parent_code, code, "Different Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #17
#--------------------------

```
def test_add_successfully_adds_new_account():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    parent_account = coa.find(parent_code)
    child_account = coa.add(parent_code, child_code, child_name)
    assert child_account.code == child_code
    assert child_account.name == child_name
    assert child_account.parent == parent_account
    assert child_code in coa._accounts
    assert child_account in coa._subaccounts[parent_account]

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Invalid Account")
        assert False
    except ValueError:
        assert True

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    parent_code = Code("99")
    child_code = Code("99.1")
    try:
        coa.add(parent_code, child_code, "Child Account")
        assert False
    except ValueError:
        assert True

def test_add_returns_existing_account_when_details_match():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    coa.add(parent_code, child_code, child_name)
    existing_account = coa.add(parent_code, child_code, child_name)
    assert existing_account.code == child_code
    assert existing_account.name == child_name
    assert existing_account.parent == coa.find(parent_code)

def test_add_raises_error_when_existing_account_details_mismatch():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    coa.add(parent_code, child_code, "Original Name")
    try:
        coa.add(parent_code, child_code, "Different Name")
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #18
#--------------------------

```python
def test___call__():
    coa_mock = COA()
    read_coa_mock = lambda: coa_mock
    assert read_coa_mock() == coa_mock


# LLM-generated content at query #19
#--------------------------

```python
def test_add_method_adds_new_account():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    new_account = coa.add(parent_code, Code("2"), "New Account")
    assert new_account.code == Code("2")
    assert new_account.name == "New Account"
    assert new_account.parent == parent_account
    assert new_account in coa._subaccounts[parent_account]

def test_add_method_throws_error_if_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    try:
        coa.add(parent_code, parent_code, "New Account")
        assert False
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_method_throws_error_if_parent_not_defined():
    coa = COA()
    try:
        coa.add(Code("1"), Code("2"), "New Account")
        assert False
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_method_throws_error_if_account_exists_with_conflicting_data():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("2"), "Existing Account", parent_account)
    coa._accounts[Code("2")] = existing_account
    try:
        coa.add(parent_code, Code("2"), "New Account")
        assert False
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."

def test_add_method_returns_existing_account_if_data_matches():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("2"), "Existing Account", parent_account)
    coa._accounts[Code("2")] = existing_account
    returned_account = coa.add(parent_code, Code("2"), "Existing Account")
    assert returned_account == existing_account


# LLM-generated content at query #20
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "SubAccount Name"
    parent = Account(type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=code, name=name, parent=parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #21
#--------------------------

```
def test_subaccount_constructor():
    code = Code("001")
    name = "SubAccount1"
    parent = Account(Code("000"), "Account1", AccountType.ASSET, COA())
    subaccount = SubAccount(code, name, parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #22
#--------------------------

```
def test___call___returns_COA():
    mock_coa = COA()
    mock_reader = lambda: mock_coa
    result = mock_reader()
    assert result == mock_coa


# LLM-generated content at query #23
#--------------------------

```python
def test_subaccount_constructor():
    code = "123"
    name = "Test SubAccount"
    parent = Account(code="100", name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=code, name=name, parent=parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #24
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(Code("456"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    subaccount = SubAccount(code, name, parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #25
#--------------------------

```python
def test_add_successfully_adds_new_account():
    code_parent = Code("1")
    code_child = Code("2")
    name_child = "Child Account"
    coa = COA()
    parent_account = RootAccount(code_parent, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[code_parent] = parent_account
    added_account = coa.add(code_parent, code_child, name_child)
    assert added_account.code == code_child
    assert added_account.name == name_child
    assert added_account.parent == parent_account
    assert code_child in coa._accounts
    assert parent_account in coa._subaccounts
    assert added_account in coa._subaccounts[parent_account]

def test_add_raises_error_when_parent_and_code_are_same():
    code = Code("1")
    coa = COA()
    parent_account = RootAccount(code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[code] = parent_account
    try:
        coa.add(code, code, "Same Code Account")
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_raises_error_when_parent_is_not_defined():
    code_parent = Code("1")
    code_child = Code("2")
    name_child = "Child Account"
    coa = COA()
    try:
        coa.add(code_parent, code_child, name_child)
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_raises_error_when_account_exists_with_inconsistent_info():
    code_parent = Code("1")
    code_child = Code("2")
    name_child = "Child Account"
    coa = COA()
    parent_account = RootAccount(code_parent, "Parent Account", AccountType.ASSET, coa)
    existing_account = SubAccount(code_child, "Existing Account", parent_account)
    coa._accounts[code_parent] = parent_account
    coa._accounts[code_child] = existing_account
    try:
        coa.add(code_parent, code_child, name_child)
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #26
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "SubAccountName"
    parent_account = Account(type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=code, name=name, parent=parent_account)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent_account


# LLM-generated content at query #27
#--------------------------

```python
def test_add_new_account():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    code = Code("2")
    name = "Child"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent == parent_account
    assert code in coa._accounts
    assert account in coa._subaccounts[parent_account]

def test_add_existing_account():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    code = Code("2")
    name = "Child"
    account = SubAccount(code, name, parent_account)
    coa._accounts[code] = account
    coa._subaccounts[parent_account] = [account]
    result = coa.add(parent_code, code, name)
    assert result == account

def test_add_account_with_inconsistent_details():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    code = Code("2")
    name = "Child"
    inconsistent_account = SubAccount(code, "Inconsistent", parent_account)
    coa._accounts[code] = inconsistent_account
    coa._subaccounts[parent_account] = [inconsistent_account]
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_self_as_parent():
    coa = COA()
    code = Code("1")
    name = "Account"
    try:
        coa.add(code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_undefined_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Child"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "SubAccountName"
    parent_account = Account()
    sub_account = SubAccount(code=code, name=name, parent=parent_account)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent_account


# LLM-generated content at query #29
#--------------------------

```python
def test_ReadChartOfAccounts___call___returns_COA():
    mock_coa = COA()
    mock_reader = lambda: mock_coa
    assert isinstance(mock_reader(), COA)


# LLM-generated content at query #30
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("001")
    name = "Test SubAccount"
    parent = Account(Code("000"), "Parent Account", AccountType.ASSET, COA("Test COA"))
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #31
#--------------------------

```python
def test_sub_account_constructor():
    code = Code("123")
    name = "Test Sub Account"
    parent = Account(Code("456"), "Parent Account", AccountType.ASSET, COA())
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #32
#--------------------------

```python
def test_add_new_subaccount_successfully():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    account = coa.add(parent_code, child_code, child_name)
    assert account.code == child_code
    assert account.name == child_name
    assert account.parent.code == parent_code
    assert child_code in coa._accounts
    assert account in coa._subaccounts[coa._accounts[parent_code]]

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    existing_account = SubAccount(child_code, child_name, coa._accounts[parent_code])
    coa._accounts[child_code] = existing_account
    coa._subaccounts[coa._accounts[parent_code]] = [existing_account]
    account = coa.add(parent_code, child_code, child_name)
    assert account is existing_account

def test_add_account_with_self_as_parent_raises_error():
    coa = COA()
    code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(code, code, "Account")

def test_add_account_with_nonexistent_parent_raises_error():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    with pytest.raises(ValueError, match="Parent account is not (yet) defined."):
        coa.add(parent_code, child_code, "Child Account")

def test_add_existing_account_with_conflicting_details_raises_error():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    existing_account = SubAccount(child_code, "Different Name", coa._accounts[parent_code])
    coa._accounts[child_code] = existing_account
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, child_code, child_name)


# LLM-generated content at query #33
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("parent"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("sub"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("sub")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #34
#--------------------------

```
def test_read_chart_of_accounts_call_returns_coa():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA(accounts=[], relationships=[])

    reader = MockReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)
    assert result.accounts == []
    assert result.relationships == []

def test_read_chart_of_accounts_call_returns_non_empty_coa():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            account = Account(id="1", name="Cash", type=AccountType.ASSET)
            relationship = Relationship(source="1", target="2", type=RelationshipType.HIERARCHY)
            return COA(accounts=[account], relationships=[relationship])

    reader = MockReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)
    assert len(result.accounts) == 1
    assert len(result.relationships) == 1
    assert result.accounts[0].id == "1"
    assert result.relationships[0].source == "1"


# LLM-generated content at query #35
#--------------------------

```
def test_add_successfully_adds_new_account():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    account = coa.add(parent_code, child_code, child_name)
    assert account.code == child_code
    assert account.name == child_name
    assert account.parent.code == parent_code
    assert child_code in coa._accounts
    assert account in coa._subaccounts[account.parent]

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    try:
        coa.add(parent_code, parent_code, "Same Code Account")
        assert False
    except ValueError:
        assert True

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    try:
        coa.add(parent_code, child_code, "Child Account")
        assert False
    except ValueError:
        assert True

def test_add_raises_error_when_account_exists_with_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[child_code] = SubAccount(child_code, "Existing Child", coa._accounts[parent_code])
    try:
        coa.add(parent_code, child_code, "Different Name")
        assert False
    except ValueError:
        assert True

def test_add_returns_existing_account_when_info_matches():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    child_name = "Child Account"
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    existing_account = SubAccount(child_code, child_name, coa._accounts[parent_code])
    coa._accounts[child_code] = existing_account
    coa._subaccounts[existing_account.parent] = [existing_account]
    account = coa.add(parent_code, child_code, child_name)
    assert account is existing_account


# LLM-generated content at query #36
#--------------------------

```python
def test_sub_account_constructor():
    code = Code("001")
    name = "Sub Account 1"
    parent_account = Account(
        Code("000"),
        "Parent Account",
        AccountType.ASSET,
        COA("Company A")
    )
    sub_account = SubAccount(code, name, parent_account)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == COA("Company A")


# LLM-generated content at query #37
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(Code("100"), "Parent Account", AccountType.ASSET, COA())
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent
    assert sub_account.type == parent.type
    assert sub_account.coa == parent.coa


# LLM-generated content at query #38
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    mock_instance = MockReadChartOfAccounts()
    result = mock_instance()
    assert isinstance(result, COA)


# LLM-generated content at query #39
#--------------------------

```python
def test_subaccount_constructor():
    mock_code = "code"
    mock_name = "name"
    mock_parent = type("Account", (), {"type": "type", "coa": "coa"})()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_parent.type
    assert sub_account.coa == mock_parent.coa


# LLM-generated content at query #40
#--------------------------

```python
def test_add_account_with_mismatched_attributes():
    code = Code("001")
    parent_code = Code("002")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, COA())
    coa = COA()
    coa._accounts[parent_code] = parent_account
    coa._accounts[code] = SubAccount(code, "Old Name", parent_account)
    try:
        coa.add(parent_code, code, "New Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #41
#--------------------------

```
def test___call___returns_COA():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    mock_reader = MockReadChartOfAccounts()
    result = mock_reader()
    assert isinstance(result, COA)

def test___call___returns_different_COA_instances():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    mock_reader = MockReadChartOfAccounts()
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 is not result2


