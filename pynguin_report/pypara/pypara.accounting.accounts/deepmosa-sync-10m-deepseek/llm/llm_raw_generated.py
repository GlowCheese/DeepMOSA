####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test___call___returns_COA():
    mock_coa = COA()
    mock_reader = lambda: mock_coa
    result = mock_reader()
    assert result == mock_coa


# LLM-generated content at query #2
#--------------------------

```python
def test_nodify_with_subaccounts():
    coa = COA()
    root_code = Code("1")
    root_account = RootAccount(root_code, "Root Account", AccountType.ASSET, coa)
    sub_code = Code("2")
    sub_account = SubAccount(sub_code, "Sub Account", root_account)
    coa._accounts[root_code] = root_account
    coa._accounts[sub_code] = sub_account
    coa._subaccounts[root_account] = [sub_account]
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == sub_account

def test_nodify_without_subaccounts():
    coa = COA()
    root_code = Code("1")
    root_account = RootAccount(root_code, "Root Account", AccountType.ASSET, coa)
    coa._accounts[root_code] = root_account
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 0


# LLM-generated content at query #3
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_spec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Custom Liability"
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_partial_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Partial Asset"),
    }
    coa = COA(rootspec=custom_spec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("100")).name == "Partial Asset"
    assert coa.find(Code("2")).name == "Liability"  # default name for second account type
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)
    assert coa.find(Code("1")).name == "Asset"  # default name for first account type


# LLM-generated content at query #4
#--------------------------

```python
def test_nodify_returns_node_with_account_and_subaccounts():
    # Setup test data
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    child_account = SubAccount(child_code, "Child", parent_account)
    
    # Mock accounts and subaccounts
    coa._accounts = {parent_code: parent_account, child_code: child_account}
    coa._subaccounts = {parent_account: [child_account]}
    
    # Call nodify
    node = coa.nodify(parent_account)
    
    # Assertions
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account


# LLM-generated content at query #5
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Asset"),
        AccountType.LIABILITY: (Code("2"), "Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Asset"
    assert coa.find(Code("2")).name == "Liability"


# LLM-generated content at query #6
#--------------------------

```python
def test_coa_constructor_with_default_values():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(a.parent is None for a in coa.toplevel)

def test_coa_constructor_with_custom_rootspec():
    rootspec = {AccountType.ASSET: (Code("1"), "Asset Account")}
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Asset Account"


# LLM-generated content at query #7
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(AccountType(account.type).value))
        assert account.name == AccountType(account.type).name.capitalize()

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        if account.type in custom_rootspec:
            assert account.code == custom_rootspec[account.type][0]
            assert account.name == custom_rootspec[account.type][1]
        else:
            assert account.code == Code(str(AccountType(account.type).value))
            assert account.name == AccountType(account.type).name.capitalize()


# LLM-generated content at query #8
#--------------------------

```python
def test_add_new_account_successfully():
    coa = COA()
    parent_code = Code("1")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    code = Code("1.1")
    name = "Child Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert code in coa._accounts
    assert account in coa._subaccounts[coa._accounts[parent_code]]

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    code = Code("1.1")
    name = "Child Account"
    existing_account = SubAccount(code, name, coa._accounts[parent_code])
    coa._accounts[code] = existing_account
    coa._subaccounts[coa._accounts[parent_code]] = [existing_account]
    account = coa.add(parent_code, code, name)
    assert account is existing_account

def test_add_account_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Child Account"
    try:
        coa.add(parent_code, code, name)
        assert False
    except ValueError:
        assert True

def test_add_account_with_mismatched_details():
    coa = COA()
    parent_code = Code("1")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    code = Code("1.1")
    name = "Child Account"
    existing_account = SubAccount(code, "Different Name", coa._accounts[parent_code])
    coa._accounts[code] = existing_account
    coa._subaccounts[coa._accounts[parent_code]] = [existing_account]
    try:
        coa.add(parent_code, code, name)
        assert False
    except ValueError:
        assert True

def test_add_account_with_self_as_parent():
    coa = COA()
    code = Code("1")
    try:
        coa.add(code, code, "Account")
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #9
#--------------------------

```python
def test_subaccount_constructor():
    mock_code = "123"
    mock_name = "Test SubAccount"
    mock_parent = Account(code="001", name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #10
#--------------------------

```
def test_add_account_with_nonexistent_parent():
    coa = COA()
    try:
        coa.add(Code("nonexistent"), Code("child"), "Child Account")
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."


# LLM-generated content at query #11
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_rootspec():
    rootspec = {AccountType.BALANCE_SHEET: (Code("1"), "Balance Sheet"), AccountType.INCOME_STATEMENT: (Code("2"), "Income Statement")}
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == 2
    assert all(account.parent is None for account in coa.toplevel)
    assert any(account.code == Code("1") and account.name == "Balance Sheet" for account in coa.toplevel)
    assert any(account.code == Code("2") and account.name == "Income Statement" for account in coa.toplevel)


# LLM-generated content at query #12
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.code == Code(str(list(AccountType).index(account.type) + 1))
        assert account.name == account.type.name.capitalize()

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for code, account in coa:
        assert isinstance(account, RootAccount)
        if account.type in rootspec:
            expected_code, expected_name = rootspec[account.type]
            assert account.code == expected_code
            assert account.name == expected_name
        else:
            assert account.code == Code(str(list(AccountType).index(account.type) + 1))
            assert account.name == account.type.name.capitalize()


# LLM-generated content at query #13
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in coa.accounts)
    assert all(a.code == Code(str(i+1)) for i, a in enumerate(coa.accounts))
    assert all(a.name == t.name.capitalize() for t, a in zip(AccountType, coa.accounts))

def test_coa_constructor_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("A"), "Custom Asset"),
        AccountType.LIABILITY: (Code("L"), "Custom Liability")
    }
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    assert accounts[0].code == Code("A")
    assert accounts[0].name == "Custom Asset"
    assert accounts[1].code == Code("L")
    assert accounts[1].name == "Custom Liability"
    assert all(isinstance(a, RootAccount) for a in accounts[2:])
    assert all(a.code == Code(str(i+1)) for i, a in enumerate(accounts[2:], start=2))
    assert all(a.name == t.name.capitalize() for t, a in zip(list(AccountType)[2:], accounts[2:]))

def test_coa_constructor_partial_rootspec():
    custom_spec = {
        AccountType.EXPENSE: (Code("E"), "Custom Expense")
    }
    coa = COA(rootspec=custom_spec)
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    assert accounts[3].code == Code("E")
    assert accounts[3].name == "Custom Expense"
    assert all(isinstance(a, RootAccount) for a in accounts)
    assert all(a.code == Code(str(i+1)) for i, a in enumerate(accounts) if i != 3)
    assert all(a.name == t.name.capitalize() for t, a in zip(AccountType, accounts) if t != AccountType.EXPENSE)

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in coa.accounts)
    assert all(a.code == Code(str(i+1)) for i, a in enumerate(coa.accounts))
    assert all(a.name == t.name.capitalize() for t, a in zip(AccountType, coa.accounts))


# LLM-generated content at query #14
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("parent_code"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("sub_code"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("sub_code")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_add_new_account_successfully():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    new_account_code = Code("2")
    new_account = coa.add(parent_code, new_account_code, "New Account")
    assert new_account.code == new_account_code
    assert new_account.name == "New Account"
    assert new_account.parent == parent_account
    assert new_account_code in coa._accounts
    assert new_account in coa._subaccounts[parent_account]

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account_code = Code("2")
    existing_account = SubAccount(existing_account_code, "Existing Account", parent_account)
    coa._accounts[existing_account_code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    retrieved_account = coa.add(parent_code, existing_account_code, "Existing Account")
    assert retrieved_account == existing_account

def test_add_existing_account_with_mismatching_details():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account_code = Code("2")
    existing_account = SubAccount(existing_account_code, "Existing Account", parent_account)
    coa._accounts[existing_account_code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    try:
        coa.add(parent_code, existing_account_code, "Different Name")
        assert False  # Should raise ValueError
    except ValueError:
        assert True

def test_add_account_with_parent_as_itself():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    try:
        coa.add(parent_code, parent_code, "Self Parent Account")
        assert False  # Should raise ValueError
    except ValueError:
        assert True

def test_add_account_with_non_existent_parent():
    coa = COA()
    parent_code = Code("1")
    new_account_code = Code("2")
    try:
        coa.add(parent_code, new_account_code, "New Account")
        assert False  # Should raise ValueError
    except ValueError:
        assert True


# LLM-generated content at query #17
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(coa._accounts) == 0
    assert len(coa._subaccounts) == 0

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: ("1", "Assets"),
        AccountType.LIABILITY: ("2", "Liabilities")
    }
    coa = COA(rootspec=rootspec)
    assert len(coa._accounts) == 2
    assert coa._accounts["1"].name == "Assets"
    assert coa._accounts["2"].name == "Liabilities"

def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec={})
    assert len(coa._accounts) == 0

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: ("1", "Assets")
    }
    coa = COA(rootspec=rootspec)
    assert len(coa._accounts) == 1
    assert coa._accounts["1"].name == "Assets"


# LLM-generated content at query #18
#--------------------------

```python
def test_add_account_parent_in_subaccounts_buffer():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    coa._subaccounts[parent_account] = []
    child_code = Code("2")
    child_account = SubAccount(child_code, "Child Account", parent_account)
    coa._accounts[child_code] = child_account
    coa._subaccounts[parent_account].append(child_account)
    coa.add(parent_code, child_code, "Child Account")


# LLM-generated content at query #19
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("1000"), name="Parent", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("1100"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("1100")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #20
#--------------------------

```
def test_nodify_returns_node_instance():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #21
#--------------------------

```python
def test_add_method_creates_new_account():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    new_account = coa.add(parent_code, Code("2"), "New Account")
    assert new_account.code == Code("2")
    assert new_account.name == "New Account"
    assert new_account.parent == parent_account
    assert new_account in coa.subaccounts(parent_account)

def test_add_method_returns_existing_account():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("2"), "Existing Account", parent_account)
    coa._accounts[Code("2")] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    retrieved_account = coa.add(parent_code, Code("2"), "Existing Account")
    assert retrieved_account == existing_account

def test_add_method_raises_error_for_self_parent():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    try:
        coa.add(parent_code, parent_code, "Invalid Account")
        assert False
    except ValueError:
        assert True

def test_add_method_raises_error_for_invalid_parent():
    coa = COA()
    invalid_parent_code = Code("999")
    try:
        coa.add(invalid_parent_code, Code("2"), "New Account")
        assert False
    except ValueError:
        assert True

def test_add_method_raises_error_for_inconsistent_account_info():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("2"), "Existing Account", parent_account)
    coa._accounts[Code("2")] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    try:
        coa.add(parent_code, Code("2"), "Different Name")
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = Account(Code("1"), "Account 1", None)
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #24
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


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test___call___returns_COA():
    mock_coa = COA()
    mock_reader = lambda: mock_coa
    result = mock_reader()
    assert result == mock_coa

def test___call___returns_new_COA_instance_each_time():
    mock_reader = lambda: COA()
    result1 = mock_reader()
    result2 = mock_reader()
    assert result1 != result2


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_add_existing_account_with_different_parent():
    coa = COA()
    parent_code = Code("1")
    account_code = Code("2")
    account_name = "Account Name"
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(account_code, account_name, parent_account)
    coa._accounts[account_code] = existing_account
    different_parent_account = RootAccount(Code("3"), "Different Parent Account", AccountType.LIABILITY, coa)
    coa._accounts[Code("3")] = different_parent_account
    try:
        coa.add(Code("3"), account_code, account_name)
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #29
#--------------------------

```python
def test_add_account_with_inconsistent_information():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Account 2"
    parent = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent
    coa._accounts[code] = SubAccount(code, "Different Name", parent)
    try:
        coa.add(parent_code, code, name)
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #30
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_SubAccount_constructor():
    mock_code = Code()
    mock_name = "Test SubAccount"
    mock_parent = Account()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #2
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in coa.accounts)
    assert all(a.code == Code(str(i+1)) for i, a in enumerate(coa.accounts))
    assert all(a.name == t.name.capitalize() for a, t in zip(coa.accounts, AccountType))

def test_coa_constructor_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("A"), "Custom Asset"),
        AccountType.LIABILITY: (Code("L"), "Custom Liability"),
    }
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    assert accounts[0].code == Code("A")
    assert accounts[0].name == "Custom Asset"
    assert accounts[1].code == Code("L") 
    assert accounts[1].name == "Custom Liability"
    assert all(a.code == Code(str(i+1)) for i, a in enumerate(accounts[2:], start=2))
    assert all(a.name == t.name.capitalize() for a, t in zip(accounts[2:], list(AccountType)[2:]))

def test_coa_constructor_partial_rootspec():
    rootspec = {
        AccountType.EQUITY: (Code("E"), "Custom Equity"),
    }
    coa = COA(rootspec=rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    assert accounts[2].code == Code("E")
    assert accounts[2].name == "Custom Equity"
    assert all(a.code == Code(str(i+1)) for i, a in enumerate(accounts) if i != 2)
    assert all(a.name == t.name.capitalize() for a, t in zip(accounts, AccountType) if t != AccountType.EQUITY)


# LLM-generated content at query #3
#--------------------------

```python
def test_COA_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == 2
    assert all(account.parent is None for account in coa.toplevel)

def test_COA_constructor_with_rootspec():
    rootspec = {
        AccountType.BALANCE_SHEET: (Code("1"), "Balance Sheet"),
        AccountType.INCOME_STATEMENT: (Code("2"), "Income Statement"),
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == 2
    assert all(account.parent is None for account in coa.toplevel)
    assert coa.find(Code("1")).name == "Balance Sheet"
    assert coa.find(Code("2")).name == "Income Statement"


# LLM-generated content at query #4
#--------------------------

```
def test_nodify_returns_node_with_correct_account_and_empty_children_for_leaf_account():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert node.children == []

def test_nodify_returns_node_with_correct_account_and_children_for_parent_account():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    coa.add(parent_code, child_code, "Child Account")
    parent_account = coa.find(parent_code)
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account.code == child_code

def test_nodify_returns_node_with_correct_account_and_nested_children():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    grandchild_code = Code("3")
    coa.add(parent_code, child_code, "Child Account")
    coa.add(child_code, grandchild_code, "Grandchild Account")
    parent_account = coa.find(parent_code)
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account.code == child_code
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account.code == grandchild_code

def test_nodify_raises_no_error_for_account_with_multiple_children():
    coa = COA()
    parent_code = Code("1")
    child1_code = Code("2")
    child2_code = Code("3")
    coa.add(parent_code, child1_code, "Child Account 1")
    coa.add(parent_code, child2_code, "Child Account 2")
    parent_account = coa.find(parent_code)
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 2
    assert {child.account.code for child in node.children} == {child1_code, child2_code}


# LLM-generated content at query #5
#--------------------------

```
def test_coa_constructor_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"

def test_coa_constructor_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liability"  # default name for LIABILITY

def test_coa_constructor_accounts_frozen():
    coa = COA()
    accounts = list(coa.accounts)
    assert accounts[0].code == Code("1")
    assert accounts[1].code == Code("2")
    assert accounts[2].code == Code("3")
    assert accounts[3].code == Code("4")
    assert accounts[4].code == Code("5")

def test_coa_constructor_toplevel_accounts():
    coa = COA()
    toplevel = list(coa.toplevel)
    assert len(toplevel) == len(AccountType)
    assert all(account.parent is None for account in toplevel)


# LLM-generated content at query #6
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    for account in accounts:
        assert isinstance(account, RootAccount)

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.REVENUE: (Code("4"), "Revenue"),
        AccountType.EXPENSE: (Code("5"), "Expenses")
    }
    coa = COA(rootspec=custom_rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    for account in accounts:
        assert isinstance(account, RootAccount)
        assert account.code == custom_rootspec[account.type][0]
        assert account.name == custom_rootspec[account.type][1]

def test_coa_constructor_with_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities")
    }
    coa = COA(rootspec=partial_rootspec)
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    for account in accounts:
        assert isinstance(account, RootAccount)
        if account.type in partial_rootspec:
            assert account.code == partial_rootspec[account.type][0]
            assert account.name == partial_rootspec[account.type][1]
        else:
            assert account.code == Code(str(account.type.value))
            assert account.name == account.type.name.capitalize()


# LLM-generated content at query #7
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
        AccountType.EXPENSE: (Code("5"), "Expense"),
    }
    coa = COA(rootspec=custom_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(account.code == custom_rootspec[account.type][0] for account in coa.accounts)
    assert all(account.name == custom_rootspec[account.type][1] for account in coa.accounts)
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_partial_rootspec():
    partial_rootspec = {
        AccountType.ASSET: (Code("A"), "Custom Assets"),
        AccountType.EXPENSE: (Code("E"), "Custom Expenses"),
    }
    coa = COA(rootspec=partial_rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("A")).name == "Custom Assets"
    assert coa.find(Code("E")).name == "Custom Expenses"
    assert coa.find(Code("2")).name == "Liability"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Revenue"
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)


# LLM-generated content at query #8
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #9
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
    assert sub_account.type == parent.type
    assert sub_account.coa == parent.coa


# LLM-generated content at query #10
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
    assert len(list(coa.toplevel)) == 2
    assert len(list(coa.structure)) == 2


# LLM-generated content at query #11
#--------------------------

```python
def test_subaccount_constructor():
    mock_code = Code(value="001")
    mock_name = "Test SubAccount"
    mock_parent = Account(code=Code(value="000"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent


# LLM-generated content at query #12
#--------------------------

```python
def test_add_new_account_to_coa():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_account.code] = parent_account
    new_account = coa.add(parent_account.code, Code("2"), "New Account")
    assert new_account.code == Code("2")
    assert new_account.name == "New Account"
    assert new_account.parent == parent_account
    assert parent_account.code in coa._subaccounts
    assert new_account in coa._subaccounts[parent_account]

def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_account.code] = parent_account
    existing_account = SubAccount(Code("2"), "Existing Account", parent_account)
    coa._accounts[existing_account.code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    retrieved_account = coa.add(parent_account.code, Code("2"), "Existing Account")
    assert retrieved_account == existing_account

def test_add_account_with_invalid_parent():
    coa = COA()
    try:
        coa.add(Code("1"), Code("2"), "New Account")
        assert False
    except ValueError:
        assert True

def test_add_account_with_self_as_parent():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_account.code] = parent_account
    try:
        coa.add(parent_account.code, parent_account.code, "New Account")
        assert False
    except ValueError:
        assert True

def test_add_account_with_mismatched_details():
    coa = COA()
    parent_account = RootAccount(Code("1"), "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_account.code] = parent_account
    existing_account = SubAccount(Code("2"), "Existing Account", parent_account)
    coa._accounts[existing_account.code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    try:
        coa.add(parent_account.code, Code("2"), "Different Name")
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #13
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(Code("456"), "Parent Account", AccountType.ASSET, COA())
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #14
#--------------------------

```python
def test___call___returns_COA_object():
    mock_coa = COA()
    mock_reader = lambda: mock_coa
    result = mock_reader()
    assert result == mock_coa
    assert isinstance(result, COA)


# LLM-generated content at query #15
#--------------------------

```python
def test_add_existing_account_with_matching_info():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    code = Code("2")
    name = "Sub Account"
    account = SubAccount(code, name, parent_account)
    coa._accounts[code] = account
    coa._subaccounts[parent_account] = [account]
    result = coa.add(parent_code, code, name)
    assert result == account

def test_add_existing_account_with_mismatched_info():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    code = Code("2")
    name = "Sub Account"
    account = SubAccount(code, name, parent_account)
    coa._accounts[code] = account
    coa._subaccounts[parent_account] = [account]
    try:
        coa.add(parent_code, code, "Different Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #16
#--------------------------

```python
def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    code = Code("2")
    name = "Child Account"
    account = SubAccount(code, name, parent_account)
    coa._accounts[code] = account
    coa._subaccounts[parent_account] = [account]
    result = coa.add(parent_code, code, name)
    assert result == account


# LLM-generated content at query #17
#--------------------------

```python
def test_add_account_parent_already_in_subaccounts():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    coa._subaccounts[parent_account] = []
    coa.add(parent_code, child_code, "Child")
    assert parent_account in coa._subaccounts


# LLM-generated content at query #18
#--------------------------

```
def test_add_successfully_adds_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    
    new_code = Code("1.1")
    new_name = "Test Subaccount"
    subaccount = coa.add(parent_code, new_code, new_name)
    
    assert subaccount.code == new_code
    assert subaccount.name == new_name
    assert subaccount.parent == parent_account
    assert subaccount in coa.subaccounts(parent_account)
    assert coa.find(new_code) == subaccount

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    try:
        coa.add(Code("999"), Code("1.1"), "Invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_when_account_exists_with_conflicting_info():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    coa.add(parent_code, existing_code, "Original")
    
    try:
        coa.add(parent_code, existing_code, "Conflicting")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_returns_existing_account_when_info_matches():
    coa = COA()
    parent_code = Code("1")
    existing_code = Code("1.1")
    existing_name = "Existing"
    original_account = coa.add(parent_code, existing_code, existing_name)
    
    duplicate_account = coa.add(parent_code, existing_code, existing_name)
    assert duplicate_account == original_account


# LLM-generated content at query #19
#--------------------------

```python
def test_add_existing_account_with_matching_details():
    coa = COA()
    parent_code = Code("1")
    root_account = RootAccount(parent_code, "Parent Account", AccountType.ASSET, coa)
    coa._accounts[parent_code] = root_account
    code = Code("2")
    name = "Sub Account"
    existing_account = SubAccount(code, name, root_account)
    coa._accounts[code] = existing_account
    coa._subaccounts[root_account] = [existing_account]
    result = coa.add(parent_code, code, name)
    assert result == existing_account


# LLM-generated content at query #20
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("100"), name="Parent", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("101"), name="Sub", parent=parent_account)
    assert sub_account.code == Code("101")
    assert sub_account.name == "Sub"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #21
#--------------------------

```
def test_add_creates_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa._accounts[parent_code]
    new_account = coa.add(parent_code, Code("1.1"), "Test Subaccount")
    assert isinstance(new_account, SubAccount)
    assert new_account.code == Code("1.1")
    assert new_account.name == "Test Subaccount"
    assert new_account.parent == parent_account
    assert new_account in coa._subaccounts[parent_account]

def test_add_returns_existing_account_if_consistent():
    coa = COA()
    parent_code = Code("1")
    coa.add(parent_code, Code("1.1"), "Test Subaccount")
    existing_account = coa.add(parent_code, Code("1.1"), "Test Subaccount")
    assert existing_account.code == Code("1.1")
    assert existing_account.name == "Test Subaccount"
    assert existing_account.parent == coa._accounts[parent_code]

def test_add_raises_error_for_self_parent():
    coa = COA()
    try:
        coa.add(Code("1"), Code("1"), "Invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_for_undefined_parent():
    coa = COA()
    try:
        coa.add(Code("99"), Code("99.1"), "Invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_for_inconsistent_existing_account():
    coa = COA()
    parent_code = Code("1")
    coa.add(parent_code, Code("1.1"), "Test Subaccount")
    try:
        coa.add(parent_code, Code("1.1"), "Different Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_add_account_with_inconsistent_info_raises_error():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("2")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    child_account = SubAccount(child_code, "Child", parent_account)
    coa._accounts[child_code] = child_account
    try:
        coa.add(parent_code, child_code, "Different Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```
def test_add_successfully_adds_new_account():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    new_account = coa.add(parent_code, Code("1.1"), "Child Account")
    assert new_account.code == Code("1.1")
    assert new_account.name == "Child Account"
    assert new_account.parent == parent_account
    assert new_account in coa._accounts.values()
    assert new_account in coa._subaccounts.get(parent_account, [])

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    try:
        coa.add(parent_code, parent_code, "Same Code")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    try:
        coa.add(Code("999"), Code("1.1"), "Child Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_when_account_exists_with_conflicting_details():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("1.1"), "Existing", parent_account)
    coa._accounts[existing_account.code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    try:
        coa.add(parent_code, Code("1.1"), "Different Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_returns_existing_account_when_details_match():
    coa = COA()
    parent_code = Code("1")
    parent_account = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    coa._accounts[parent_code] = parent_account
    existing_account = SubAccount(Code("1.1"), "Existing", parent_account)
    coa._accounts[existing_account.code] = existing_account
    coa._subaccounts[parent_account] = [existing_account]
    result = coa.add(parent_code, Code("1.1"), "Existing")
    assert result == existing_account


# LLM-generated content at query #25
#--------------------------

```python
def test_subaccount_constructor():
    code = "123"
    name = "Test Account"
    parent = Account(code="100", name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=code, name=name, parent=parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #26
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    class MockCOA:
        pass

    def mock_read_coa() -> MockCOA:
        return MockCOA()

    read_chart_of_accounts = ReadChartOfAccounts()
    read_chart_of_accounts.__call__ = mock_read_coa
    result = read_chart_of_accounts()
    assert isinstance(result, MockCOA)


# LLM-generated content at query #27
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    class MockCOA:
        pass

    def mock_read() -> MockCOA:
        return MockCOA()

    reader = ReadChartOfAccounts()
    reader.__call__ = mock_read
    result = reader()
    assert isinstance(result, MockCOA)


# LLM-generated content at query #28
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    class MockCOA:
        pass

    def mock_read_coa() -> MockCOA:
        return MockCOA()

    read_chart_of_accounts: ReadChartOfAccounts = mock_read_coa
    result = read_chart_of_accounts()
    assert isinstance(result, MockCOA)


# LLM-generated content at query #29
#--------------------------

```python
def test_add_method_parent_and_code_same():
    coa = COA()
    coa.__post_init__({})
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Account")
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."


# LLM-generated content at query #30
#--------------------------

```python
def test_subaccount_constructor():
    code = "123"
    name = "Test SubAccount"
    parent = Account(code="100", name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent
    assert subaccount.type == parent.type
    assert subaccount.coa == parent.coa


# LLM-generated content at query #31
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("1000"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=Code("1001"), name="Sub Account", parent=parent_account)
    assert sub_account.code == Code("1001")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa


# LLM-generated content at query #32
#--------------------------

```python
def test_add_creates_new_subaccount_when_valid_input():
    coa = COA()
    parent_code = Code("1")
    parent_account = coa._accounts[parent_code]
    new_code = Code("1.1")
    new_name = "New Subaccount"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent == parent_account
    assert new_account in coa._subaccounts[parent_account]
    assert new_code in coa._accounts

def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_when_parent_not_found():
    coa = COA()
    try:
        coa.add(Code("999"), Code("1.1"), "Invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_returns_existing_account_when_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Subaccount"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_raises_error_when_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    coa.add(parent_code, code, "Original")
    try:
        coa.add(parent_code, code, "Different")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #33
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent_account = Account(Code("001"), "Parent Account", AccountType.ASSET, COA("COA1"))
    sub_account = SubAccount(code, name, parent_account)
    
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == COA("COA1")


# LLM-generated content at query #34
#--------------------------

```python
def test___call___returns_COA():
    class MockCOA:
        pass

    def mock_read_chart_of_accounts() -> MockCOA:
        return MockCOA()

    read_chart_of_accounts: ReadChartOfAccounts = mock_read_chart_of_accounts
    result = read_chart_of_accounts()
    assert isinstance(result, MockCOA)


# LLM-generated content at query #35
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Sub Account"
    parent = Account(code=Code("456"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    sub_account = SubAccount(code=code, name=name, parent=parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #36
#--------------------------

```python
def test_read_chart_of_accounts_call():
    mock_coa = COA()
    mock_read_func = lambda: mock_coa
    read_chart_of_accounts = ReadChartOfAccounts()
    read_chart_of_accounts.__call__ = mock_read_func
    result = read_chart_of_accounts()
    assert result == mock_coa


# LLM-generated content at query #37
#--------------------------

```
def test_add_creates_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    new_account = coa.add(parent_code, Code("1.1"), "Child Account")
    assert isinstance(new_account, SubAccount)
    assert new_account.code == Code("1.1")
    assert new_account.name == "Child Account"
    assert new_account.parent.code == parent_code
    assert new_account in coa._subaccounts[coa._accounts[parent_code]]

def test_add_returns_existing_account_if_matches():
    coa = COA()
    parent_code = Code("1")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    existing_account = SubAccount(Code("1.1"), "Child", coa._accounts[parent_code])
    coa._accounts[Code("1.1")] = existing_account
    coa._subaccounts[coa._accounts[parent_code]] = [existing_account]
    result = coa.add(parent_code, Code("1.1"), "Child")
    assert result is existing_account

def test_add_raises_error_for_self_parent():
    coa = COA()
    parent_code = Code("1")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent", AccountType.ASSET, coa)
    try:
        coa.add(parent_code, parent_code, "Invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_for_nonexistent_parent():
    coa = COA()
    try:
        coa.add(Code("999"), Code("1.1"), "Child")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_raises_error_for_inconsistent_existing_account():
    coa = COA()
    parent_code = Code("1")
    other_parent_code = Code("2")
    coa._accounts[parent_code] = RootAccount(parent_code, "Parent1", AccountType.ASSET, coa)
    coa._accounts[other_parent_code] = RootAccount(other_parent_code, "Parent2", AccountType.LIABILITY, coa)
    existing_account = SubAccount(Code("1.1"), "Old Name", coa._accounts[parent_code])
    coa._accounts[Code("1.1")] = existing_account
    try:
        coa.add(other_parent_code, Code("1.1"), "New Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #38
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "SubAccountName"
    parent = Account(Code("456"), "ParentAccountName", AccountType.ASSET, COA("COA001"))
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


# LLM-generated content at query #39
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


# LLM-generated content at query #40
#--------------------------

```
def test___call___returns_COA():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    mock = MockReadChartOfAccounts()
    result = mock()
    assert isinstance(result, COA)

def test___call___returns_different_COA_instances():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    mock = MockReadChartOfAccounts()
    result1 = mock()
    result2 = mock()
    assert result1 is not result2


# LLM-generated content at query #41
#--------------------------

```python
def test_sub_account_constructor():
    code = Code("123")
    name = "SubAccountName"
    parent = Account(Code("456"), "ParentAccountName", AccountType.ASSET, COA("COAName"))
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == COA("COAName")


# LLM-generated content at query #42
#--------------------------

```python
def test_add_account_with_inconsistent_information():
    code = Code("1")
    parent_code = Code("2")
    name = "Account 1"
    parent_instance = RootAccount(parent_code, "Parent Account", AccountType.ASSET, COA())
    coa = COA()
    coa._accounts[code] = SubAccount(code, "Different Name", parent_instance)
    try:
        coa.add(parent_code, code, name)
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #43
#--------------------------

```python
def test_add_account_with_mismatched_information():
    coa = COA()
    parent_code = Code("1")
    coa.add(parent_code, Code("2"), "Child Account")
    try:
        coa.add(parent_code, Code("2"), "Different Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #44
#--------------------------

```python
def test_read_chart_of_accounts_call():
    class MockReadChartOfAccounts:
        def __call__(self) -> dict:
            return {"account1": 1000, "account2": 2000}

    reader = MockReadChartOfAccounts()
    result = reader()
    assert result == {"account1": 1000, "account2": 2000}


# LLM-generated content at query #45
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "SubAccountName"
    parent = Account(Code("456"), "ParentAccountName", AccountType.ASSET, COA())
    sub_account = SubAccount(code, name, parent)
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent


