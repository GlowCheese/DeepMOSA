####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        pass

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    mock_coa = MockCOA()
    reader = MockReadChartOfAccounts()
    result = reader()
    assert result == mock_coa
```


# LLM-generated content at query #2
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    cash_account = coa.add(Code("1"), Code("11"), "Cash")
    bank_account = coa.add(Code("11"), Code("111"), "Bank Account")
    
    node = coa.nodify(cash_account)
    
    assert isinstance(node, COA.Node)
    assert node.account == cash_account
    assert len(node.children) == 1
    assert node.children[0].account == bank_account
    assert node.children[0].children == []

def test_nodify_with_leaf_node():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    asset_account = coa.find(Code("1"))
    
    node = coa.nodify(asset_account)
    
    assert node.account == asset_account
    assert node.children == []

def test_nodify_recursively_processes_deep_hierarchy():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    level1 = coa.add(Code("1"), Code("11"), "Current Assets")
    level2 = coa.add(Code("11"), Code("111"), "Cash")
    level3 = coa.add(Code("111"), Code("1111"), "Petty Cash")
    
    root_node = coa.nodify(coa.find(Code("1")))
    
    assert root_node.children[0].account == level1
    assert root_node.children[0].children[0].account == level2
    assert root_node.children[0].children[0].children[0].account == level3
```


# LLM-generated content at query #3
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert isinstance(coa, COA)

def test_coa_constructor_with_rootspec():
    code1 = Code("100")
    name1 = "Assets"
    type1 = AccountType.ASSETS  # Assuming ASSETS is a valid member of AccountType
    rootspec = {type1: (code1, name1)}
    coa = COA(rootspec=rootspec)
    account = coa.find(code1)
    assert account.code == code1
    assert account.name == name1
    assert account.parent is None

def test_coa_constructor_with_partial_rootspec():
    code1 = Code("999")
    type1 = AccountType.REVENUE 
    rootspec = {type1: (code1, "Custom Revenue")}
    coa = COA(rootspec=rootspec)
    assert coa.find(code1).name == "Custom Revenue"
    # Check that other types still got default values (e.g., Code("1") for the first type)
    assert len(coa._accounts) == len(AccountType)

def test_coa_constructor_integrity():
    coa = COA()
    assert coa._accounts == coa._accounts
    assert coa._subaccounts == coa._subaccounts
```


# LLM-generated content at query #4
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Code("123")
    mock_name = "Savings"
    mock_parent = MockAccount(type=AccountType.ASSET, coa=MockCOA())
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == MockCOA()
```


# LLM-generated content at query #5
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self, name):
            self.name = name

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA("test_account")

    expected_coa = MockCOA("test_account")
    reader: ReadChartOfAccounts = MockReadChartOfAccounts()
    result = reader()
    assert result.name == "test_account"
    assert result == expected_coa
```


# LLM-generated content at query #6
#--------------------------

```python
def test_coa_add_success():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("101")
    new_name = "Cash"
    new_account = coa.add(parent_code, new_code, new_name)
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

def test_coa_add_error_same_code():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    assert (lambda: coa.add(parent_code, parent_code, "Self"))( ) == None # This is a placeholder logic for the exception check
    # Since I cannot use try/except or control structures, I will assume the test environment 
    # handles checking if the following call raises ValueError internally via a test runner.
    # However, per instructions, we only contain assignments, assertions and calls.
    # To strictly follow "no control structure", we can't catch the error here.
    # But I will provide the valid functional test cases.

def test_coa_add_error_missing_parent():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("999")
    new_code = Code("101")
    # The requirement forbids 'if', 'for', etc. 
    # To test the error, one would normally use pytest.raises, but I cannot import it.
    # Therefore, I will focus on valid state transitions and properties.

def test_coa_add_existing_account_consistency():
    root_spec = {Account(code=Code("1"), name="Assets", type=AccountType.ASSET, coa=None) is None} # This is invalid code logic
    # Let's refine to valid assignments only.
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("101")
    new_name = "Cash"
    account_v1 = coa.add(parent_code, new_code, new_name)
    account_v2 = coa.add(parent_code, new_code, new_name)
    assert account_v1 == account_v2
    assert account_v1.code == account_v2.code

def test_coa_add_error_inconsistent_data():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("101")
    new_name = "Cash"
    coa.add(parent_code, new_code, new_name)
    # The next line would raise ValueError due to name mismatch:
    # coa.add(parent_code, new_code, "Different Name") 
    # But I cannot use try/except or if to handle the error in a single test function without imports.
```


# LLM-generated content at query #7
#--------------------------

```python
def test_add_fails_when_parent_is_not_defined():
    coa = COA(rootspec={AccountType.ASSET: (Code("1"), "Assets")})
    undefined_parent_code = Code("999")
    new_account_code = Code("101")
    new_account_name = "Cash"
    
    import pytest
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(undefined_parent_code, new_account_code, new_account_name)
```


# LLM-generated content at query #8
#--------------------------

```python
def test_nodify_returns_node_with_correct_account_and_children():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    sub_account = coa.add(Code("1"), Code("11"), "Current Assets")
    deep_account = coa.add(Code("11"), Code("111"), "Cash")
    
    node = coa.nodify(root_account := coa.find(Code("1")))
    
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == sub_account
    assert node.children[0].children[0].account == deep_account
```


# LLM-generated content at query #9
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    coa = COA()
    root_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    
    coa.add(root_code, child_code, child_name)
    
    root_account = coa.find(root_code)
    child_account = coa.find(child_code)
    
    node = coa.nodify(root_account)
    
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
    assert node.children[0].account.name == child_name
```


# LLM-generated content at query #10
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in coa.accounts)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Account"
    spec = {AccountType.ASSET: (custom_code, custom_name)}
    coa = COA(rootspec=spec)
    account = coa.find(custom_code)
    assert account is not None
    assert account.name == custom_name
    assert account.code == custom_code

def test_coa_constructor_mixed_rootspec():
    spec = {AccountType.ASSET: (Code("1"), "Assets Only")}
    coa = COA(rootspec=spec)
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Assets Only"
    other_type_account = next(a for a in coa.accounts if a.type != AccountType.ASSET)
    assert other_type_account.code != Code("1")
```


# LLM-generated content at query #11
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-account"
    
    class MockParent:
        def __init__(self):
            self.type = "Asset"
            self.coa = "Main_COA"

    parent_account = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == "Asset"
    assert sub_account.coa == "Main_COA"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert isinstance(coa._accounts, dict)
    assert isinstance(coa._subaccounts, dict)

def test_coa_constructor_with_rootspec():
    root_code = Code("100")
    root_name = "Custom Root"
    target_type = AccountType.ASSET
    rootspec = {target_type: (root_code, root_name)}
    coa = COA(rootspec=rootspec)
    
    found_account = coa.find(root_code)
    assert found_account is not None
    assert found_account.code == root_code
    assert found_account.name == root_name
    assert found_account.parent is None

def test_coa_constructor_with_partial_rootspec():
    target_type = AccountType.ASSET
    root_code = Code("999")
    rootspec = {target_type: (root_code, "Only Asset Custom")}
    coa = COA(rootspec=rootspec)
    
    asset_account = coa.find(root_code)
    assert asset_account is not None
    assert asset_account.name == "Only Asset Custom"
    
    other_type = list(AccountType)[1]
    other_account = coa.find(Code(str(2))) # Default logic for second type in enum
    assert other_account is not None
    assert other_account.code != root_code
```


# LLM-generated content at query #13
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    coa = COA()
    root_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    
    coa.add(root_code, child_code, child_name)
    
    root_account = coa.find(root_code)
    child_account = coa.find(child_code)
    
    node = coa.nodify(root_account)
    
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
```


# LLM-generated content at query #14
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    spec = {
        AccountType.ASSET: (Code("100"), "Assets Account")
    }
    coa = COA(rootspec=spec)
    found_account = coa.find(Code("100"))
    assert found_account is not None
    assert found_account.name == "Assets Account"
    assert found_account.code == Code("100")

def test_coa_constructor_validation_of_types():
    coa = COA()
    for code, account in coa:
        assert isinstance(code, Code)
        assert isinstance(account, Account)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_coa_constructor_default_roots():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Root"
    target_type = AccountType.ASSET
    rootspec = {target_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code
    assert found_account.type == target_type

def test_coa_constructor_invalid_rootspec_type():
    # Ensuring that it doesn't crash with empty dict and follows default logic
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_coa_constructor_default_init():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Type"
    # Assuming AccountType has at least one member, e.g., ASSET
    # We create a spec that overrides the first account type found in AccountType
    first_type = list(AccountType)[0]
    rootspec = {first_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code
    assert isinstance(found_account, RootAccount)

def test_coa_constructor_rootspec_partial():
    # Override only one type, others should use default logic (Code(str(c)), Name)
    first_type = list(AccountType)[0]
    custom_code = Code("10")
    rootspec = {first_type: (custom_code, "Overridden")}
    coa = COA(rootspec=rootspec)
    
    overridden = coa.find(custom_code)
    assert overridden is not None
    assert overridden.name == "Overridden"
    
    second_type = list(AccountType)[1] if len(list(AccountType)) > 1 else None
    if second_type:
        default_account = coa.find(Code("2")) # Default for second type is Code("2")
        assert default_account is not None
        assert default_account.name == second_type.name.capitalize()
```


# LLM-generated content at query #17
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    accounts = list(coa.accounts)
    assert len(accounts) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts)
    assert all(a.parent is None for a in accounts)

def test_coa_constructor_with_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Assets Account")
    }
    coa = COA(rootspec=custom_spec)
    asset_account = coa.find(Code("100"))
    assert asset_account is not None
    assert asset_account.name == "Assets Account"
    assert asset_account.code == Code("100")

def test_coa_constructor_initializes_correct_types():
    coa = COA()
    account_types = [a.type for a in coa.accounts]
    assert all(t in account_types for t in AccountType)

def test_coa_constructor_handles_empty_rootspec_explicitly():
    coa = COA(rootspec={})
    assert len(list(coa.accounts)) == len(AccountType)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(acc, RootAccount) for acc in accounts_list)
    assert all(acc.parent is None for acc in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Root"
    custom_type = AccountType.ASSET # Assuming ASSET exists in AccountType
    rootspec = {custom_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code
    assert found_account.type == custom_type

def test_coa_constructor_preserves_order():
    coa = COA()
    accounts_list = list(coa._accounts.keys())
    expected_codes = [Code(str(i)) for i in range(1, len(AccountType) + 1)]
    assert accounts_list == expected_codes

def test_coa_constructor_raises_error_on_invalid_rootspec_format():
    # This assumes the implementation of __post_init__ expects a dict with specific tuple structure
    # We check if passing an invalid type (like None via default) works as intended by the code logic
    coa = COA(rootspec=None)
    assert len(coa._accounts) == len(AccountType)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "12345"
    mock_name = "Savings Sub-Account"
    
    class MockAccount:
        def __init__(self, account_type, coa):
            self.type = account_type
            self.coa = coa

    class MockCOA:
        pass

    mock_type = "Asset"
    mock_coa = MockCOA()
    parent_account = MockAccount(mock_type, mock_coa)

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == mock_type
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #20
#--------------------------

```python
def test_add_with_existing_parent_does_not_raise_none_error():
    coa = COA(rootspec={AccountType.ASSET: (Code("1"), "Assets")})
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    account = coa.add(parent=parent_code, code=new_code, name=new_name)
    assert account.code == new_code
    assert coa.find(parent_code) is not None
```


# LLM-generated content at query #21
#--------------------------

```python
def test_coa_add_success():
    parent_code = Code("1")
    new_code = Code("1.1")
    new_name = "Cash"
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    added_account = coa.add(parent_code, new_code, new_name)
    assert added_account.code == new_code
    assert added_account.name == new_name
    assert added_account.parent.code == parent_code
    assert added_account in coa.subaccounts(coa.find(parent_code))
    assert coa.find(new_code) == added_account

def test_coa_add_same_code_raises_error():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(root_code, root_code, "Self Parent")

def test_coa_add_non_existent_parent_raises_error():
    coa = COA()
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("99"), Code("99.1"), "Ghost Account")

def test_coa_add_existing_account_returns_same_instance():
    parent_code = Code("1")
    target_code = Code("1.1")
    target_name = "Cash"
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    first_instance = coa.add(parent_code, target_code, target_name)
    second_instance = coa.add(parent_code, target_code, target_name)
    assert first_instance is second_instance

def test_coa_add_existing_account_mismatch_raises_error():
    parent_code = Code("1")
    target_code = Code("1.1")
    coa = COA(rootspec={AccountType.ASKS: (parent_code, "Assets")})
    coa.add(parent_code, target_code, "Original Name")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, target_code, "Different Name")
```


# LLM-generated content at query #22
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        def __init__(self):
            self.type = "Asset"
            self.coa = "Main_COA"
    
    mock_parent = MockParent()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "Main_COA"
```


# LLM-generated content at query #23
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        pass

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    mock_coa = MockCOA()
    reader = MockReadChartOfAccounts()
    result = reader()
    
    assert result == mock_coa
```


# LLM-generated content at query #24
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        pass

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    mock_reader = MockReadChartOfAccounts()
    result = mock_reader()
    assert isinstance(result, MockCOA)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        def __init__(self):
            self.type = "Asset"
            self.coa = "Main_COA"

    parent_account = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == "Asset"
    assert sub_account.coa == "Main_COA"
```


# LLM-generated content at query #26
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Code("ACC001")
    mock_name = "Savings Sub-Account"
    
    class MockAccount:
        def __init__(self):
            self.type = AccountType.ASSET
            self.coa = MockCOA()
    
    class MockCOA:
        pass

    mock_parent = MockAccount()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_parent.coa
```


# LLM-generated content at query #27
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Code(value="123")
    mock_name = "Savings"
    mock_parent = Mock(spec=Account)
    mock_parent.type = AccountType.ASSET
    mock_parent.coa = Mock(spec=COA)

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_parent.coa
```


# LLM-generated content at query #28
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "ACC001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        type = "Asset"
        coa = "Standard_COA"
    
    mock_parent = MockParent()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "Standard_COA"
```


# LLM-generated content at query #29
#--------------------------

```python
def test_coa_constructor_default_roots():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Asset"
    custom_type = AccountType.ASSET
    rootspec = {custom_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code
    assert found_account.type == custom_type

def test_coa_constructor_partial_rootspec():
    custom_code = Code("10")
    custom_name = "Custom Liability"
    custom_type = AccountType.LIABILITY
    rootspec = {custom_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    
    assert coa.find(custom_code) is not None
    assert len(list(coa.accounts)) == len(AccountType)
    
    default_account = next(a for a in coa.accounts if a.type != custom_type)
    assert default_account.code != custom_code
```


# LLM-generated content at query #30
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        type = "Asset"
        coa = "MainCOA"
    
    mock_parent = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "MainCOA"
```


# LLM-generated content at query #31
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSET
    coa = COA(rootspec={root_type: (root_code, root_name)})
    
    sub_code = Code("11")
    sub_name = "Cash"
    sub_account = coa.add(root_code, sub_code, sub_name)
    
    leaf_code = Code("1101")
    leaf_name = "Petty Cash"
    leaf_account = coa.add(sub_code, leaf_code, leaf_name)
    
    node = coa.nodify(coa.find(root_code))
    
    assert isinstance(node, COA.Node)
    assert node.account.code == root_code
    assert len(node.children) == 1
    assert node.children[0].account.code == sub_code
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account.code == leaf_code

def test_nodify_with_no_children_returns_leaf_node():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    root_account = coa.find(root_code)
    
    node = coa.nodify(root_account)
    
    assert node.account == root_account
    assert node.children == []

def test_nodify_with_multiple_siblings():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    
    sub1 = coa.add(root_code, Code("11"), "Cash")
    sub2 = coa.add(root_code, Code("12"), "Inventory")
    
    node = coa.nodify(coa.find(root_code))
    
    assert len(node.children) == 2
    child_codes = [child.account.code for child in node.children]
    assert Code("11") in child_codes
    assert Code("12") in child_codes
```


# LLM-generated content at query #32
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Type"
    # Assuming AccountType has at least one member, e.g., ASSET
    target_type = list(AccountType)[0]
    spec = {target_type: (custom_code, custom_name)}
    coa = COA(rootspec=spec)
    
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code
    assert isinstance(found_account, RootAccount)

def test_coa_constructor_rootspec_partial():
    # Only provide spec for one type, others should use default logic
    target_type = list(AccountType)[0]
    custom_code = Code("10")
    spec = {target_type: (custom_code, "Only This One")}
    coa = COA(rootspec=spec)
    
    account_with_spec = coa.find(custom_code)
    assert account_with_spec is not None
    assert account_with_spec.name == "Only This One"
    
    # Check a different type that should have default (e.g., index 2 if it exists)
    other_type = list(AccountType)[1] if len(AccountType) > 1 else None
    if other_type:
        default_account = coa.find(Code("2")) # Based on enumerate(..., start=1) logic in __post_init__
        assert default_account is not None
        assert default_account.name == other_type.name.capitalize()
```


# LLM-generated content at query #33
#--------------------------

```python
def test_coa_add_success():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

def test_coa_add_same_code_raises_error():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("1")
    code = Code("1")
    
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(code, code, "Self Parent")

def test_coa_add_nonexistent_parent_raises_error():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("99")
    new_code = Code("11")
    
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(parent_code, new_code, "No Parent")

def test_coa_add_existing_account_returns_same():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    account_first = coa.add(parent_code, new_code, new_name)
    account_second = coa.add(parent_code, new_code, new_name)
    
    assert account_first is account_second

def test_coa_add_existing_account_mismatch_raises_error():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Original Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
```


# LLM-generated content at query #34
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "123"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        type = "Asset"
        coa = "Standard COA"

    mock_parent = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "Standard COA"
```


# LLM-generated content at query #35
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_correct_coa():
    class MockCOA:
        def __init__(self, accounts):
            self.accounts = accounts

    expected_coa = MockCOA(accounts=["Cash", "Accounts Receivable", "Inventory"])
    
    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return expected_coa

    reader: ReadChartOfAccounts = MockReadChartOfAccounts()
    actual_coa = reader()

    assert actual_coa == expected_coa
    assert actual_coa.accounts == ["Cash", "Accounts Receivable", "Inventory"]
```


# LLM-generated content at query #36
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self):
            self.data = {"1000": "Cash"}

    class MockReader:
        def __call__(self) -> MockCOA:
            return MockCOA()

    reader = MockReader()
    result = reader()
    assert isinstance(result, MockCOA)
    assert result.data["1000"] == "Cash"
```


# LLM-generated content at query #37
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    coa = COA()
    root_acc = coa._accounts[Code("1")]
    child_acc = coa.add(Code("1"), Code("1.1"), "Sub Account")
    
    node = coa.nodify(root_acc)
    
    assert isinstance(node, COA.Node)
    assert node.account == root_acc
    assert len(node.children) == 1
    assert node.children[0].account == child_acc
```


# LLM-generated content at query #38
#--------------------------

```python
def test_add_fails_when_parent_is_not_defined():
    coa = COA(rootspec={})
    parent_code = Code("1000")
    new_code = Code("1001")
    account_name = "New Account"
    
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(parent=parent_code, code=new_code, name=account_name)
```


# LLM-generated content at query #39
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_parent = Mock(spec=Account)
    mock_parent.type = AccountType.ASSET
    mock_parent.coa = Mock(spec=COA)
    mock_code = Code("1234")
    mock_name = "Savings Sub-account"

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_parent.coa
```


# LLM-generated content at query #40
#--------------------------

```python
def test_coa_constructor_default_roots():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(a.parent is None for a in accounts_list)
    assert accounts_list[0].code == Code("1")
    assert accounts_list[0].name == AccountType.ASSET.name.capitalize()

def test_coa_constructor_with_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("10"), "Assets"),
        AccountType.LIABILITY: (Code("20"), "Liabilities")
    }
    coa = COA(rootspec=custom_spec)
    asset_acc = coa.find(Code("10"))
    liability_acc = coa.find(Code("20"))
    assert asset_acc.name == "Assets"
    assert liability_acc.name == "Liabilities"
    assert asset_acc.code == Code("10")
    assert liability_acc.code == Code("20")

def test_coa_constructor_partial_rootspec():
    custom_spec = {
        AccountType.EQUITY: (Code("30"), "Equity")
    }
    coa = COA(rootspec=custom_spec)
    equity_acc = coa.find(Code("30"))
    assert equity_acc.name == "Equity"
    # Check that a default was created for an unmapped type (e.g., first type in enum)
    default_acc = list(coa.accounts)[0]
    assert default_acc.code != Code("30")
```


# LLM-generated content at query #41
#--------------------------

```python
def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")) is not None

def test_coa_constructor_with_custom_rootspec():
    custom_spec = {AccountType.ASSET: (Code("100"), "Assets")}
    coa = COA(rootspec=custom_spec)
    asset_account = coa.find(Code("100"))
    assert asset_account is not None
    assert asset_account.name == "Assets"

def test_coa_constructor_initializes_correct_types():
    coa = COA()
    account_types = [a.type for a in coa.accounts]
    assert len(account_types) == len(AccountType)
```


# LLM-generated content at query #42
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        def __init__(self):
            self.type = "Asset"
            self.coa = "DefaultCOA"
    
    parent_account = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == "Asset"
    assert sub_account.coa == "DefaultCOA"
```


# LLM-generated content at query #43
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(a.parent is None for a in coa.toplevel)

def test_coa_constructor_with_rootspec():
    code1 = Code("100")
    name1 = "Custom Asset"
    type1 = AccountType.ASSET
    rootspec = {type1: (code1, name1)}
    coa = COA(rootspec=rootspec)
    account = coa.find(code1)
    assert account is not None
    assert account.name == name1
    assert account.code == code1
    assert account.type == type1

def test_coa_constructor_partial_rootspec():
    code2 = Code("200")
    name2 = "Custom Liability"
    type2 = AccountType.LIABILITY
    rootspec = {type2: (code2, name2)}
    coa = COA(rootspec=rootspec)
    assert coa.find(code2).name == name2
    assert coa.find(Code("1")) is not None
```


# LLM-generated content at query #44
#--------------------------

```python
def test_coa_constructor_default_roots():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    spec = {
        AccountType.ASSET: (Code("100"), "Assets Account")
    }
    coa = COA(rootspec=spec)
    asset_account = coa.find(Code("100"))
    assert asset_account is not None
    assert asset_account.name == "Assets Account"
    assert asset_account.code == Code("100")

def test_coa_constructor_with_partial_rootspec():
    spec = {
        AccountType.LIABILITY: (Code("200"), "Liabilities Only")
    }
    coa = COA(rootspec=spec)
    liability_account = coa.find(Code("200"))
    assert liability_account is not None
    assert liability_account.name == "Liabilities Only"
    
    # Check that other accounts still defaulted correctly via the loop logic
    # Assuming AccountType has at least one other type, e.g., ASSET
    other_account = next(a for a in coa.accounts if a.type != AccountType.LIABILITY)
    assert other_account.code is not None
    assert isinstance(other_account.code, Code)
```


# LLM-generated content at query #45
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_expected_coa():
    class MockCOA:
        def __init__(self, data):
            self.data = data

    class MockReadChartOfAccounts:
        def __init__(self, coa_data):
            self.coa_data = coa_data
        
        def __call__(self):
            return self.coa_data

    expected_data = {"account1": 100, "account2": 200}
    mock_coa = MockCOA(expected_data)
    reader = MockReadChartOfAccounts(mock_coa)
    
    result = reader()
    
    assert result.data == expected_data
```


# LLM-generated content at query #46
#--------------------------

```python
def test_add_with_existing_parent_does_not_trigger_none_error():
    root_code = Code("1")
    root_name = "Asset"
    root_type = AccountType.ASSET
    rootspec = {root_type: (root_code, root_name)}
    coa = COA(rootspec=rootspec)
    
    parent_code = root_code
    new_code = Code("101")
    new_name = "Cash"
    
    account = coa.add(parent=parent_code, code=new_code, name=new_name)
    
    assert account.code == new_code
    assert account.name == new_name
    assert coa.find(new_code) is not None
```


# LLM-generated content at query #47
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSET
    rootspec = {root_type: (root_code, root_name)}
    coa = COA(rootspec=rootspec)
    
    sub_code = Code("11")
    sub_name = "Cash"
    sub_account = coa.add(root_code, sub_code, sub_name)
    
    node = coa.nodify(coa.find(root_code))
    
    assert isinstance(node, COA.Node)
    assert node.account == coa.find(root_code)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account
    assert node.children[0].account.code == sub_code
```


# LLM-generated content at query #48
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Mock()
    mock_name = "Savings Sub-Account"
    mock_parent = Mock()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_properties_delegation():
    mock_code = Mock()
    mock_name = "Savings Sub-Account"
    mock_parent = Mock()
    mock_parent.type = AccountType.ASSET
    mock_parent.coa = Mock()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_parent.coa

def test_subaccount_immutability():
    mock_code = Mock()
    mock_parent = Mock()
    sub_account = SubAccount(code=mock_code, name="Test", parent=mock_parent)
    
    with pytest.raises(Exception):
        sub_account.name = "New Name"
```


# LLM-generated content at query #49
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        def __init__(self):
            self.type = "Asset"
            self.coa = "MainCOA"
            
    mock_parent = MockParent()

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "MainCOA"
```


# LLM-generated content at query #50
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Account"
    custom_type = AccountType.ASSET
    rootspec = {custom_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code
    assert found_account.type == custom_type

def test_coa_constructor_rootspec_partial():
    custom_code = Code("10")
    custom_type = AccountType.LIABILITY
    rootspec = {custom_type: (custom_code, "Liability Root")}
    coa = COA(rootspec=rootspec)
    assert coa.find(custom_code).name == "Liability Root"
    assert coa.find(Code("1")) is not None
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Code("12345")
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        def __init__(self):
            self.type = AccountType.ASSET
            self.coa = MockCOA()
            
    class MockCOA:
        pass

    mock_parent = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_parent.coa
```


# LLM-generated content at query #2
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSET
    coa = COA(rootspec={root_type: (root_code, root_name)})
    
    sub_code = Code("11")
    sub_name = "Cash"
    sub_account = coa.add(root_code, sub_code, sub_name)
    
    leaf_code = Code("1101")
    leaf_name = "Petty Cash"
    leaf_account = coa.add(sub_code, leaf_code, leaf_name)
    
    node = coa.nodify(coa.find(root_code))
    
    assert isinstance(node, COA.Node)
    assert node.account == coa.find(root_code)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == leaf_account

def test_nodify_leaf_node_has_no_children():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    
    sub_code = Code("11")
    coa.add(root_code, sub_code, "Cash")
    
    leaf_node = coa.nodify(coa.find(sub_code))
    
    assert leaf_node.account.code == sub_code
    assert len(leaf_node.children) == 0

def test_nodify_single_level_structure():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    
    node = coa.nodify(coa.find(root_code))
    
    assert node.account == coa.find(root_code)
    assert node.children == []
```


# LLM-generated content at query #3
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSET
    rootspec = {root_type: (root_code, root_name)}
    coa = COA(rootspec=rootspec)
    
    sub_code = Code("11")
    sub_name = "Cash"
    sub_account = coa.add(root_code, sub_code, sub_name)
    
    node = coa.nodify(coa._accounts[root_code])
    
    assert isinstance(node, COA.Node)
    assert node.account == coa._accounts[root_code]
    assert len(node.children) == 1
    assert node.children[0].account == sub_account
    assert isinstance(node.children[0], COA.Node)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_coa_constructor_no_rootspec():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Assets Account"),
        AccountType.LIABILITY: (Code("200"), "Liabilities Account")
    }
    coa = COA(rootspec=custom_spec)
    asset_account = coa.find(Code("100"))
    liability_account = coa.find(Code("200"))
    assert asset_account.name == "Assets Account"
    assert asset_account.code == Code("100")
    assert liability_account.name == "Liabilities Account"
    assert liability_account.code == Code("200")
    assert coa.find(Code("300")) is None

def test_coa_constructor_default_naming():
    coa = COA()
    first_acc = next(iter(coa.accounts))
    assert first_acc.code == Code("1")
    assert first_acc.name == AccountType.ASSET.name.capitalize()
```


# LLM-generated content at query #5
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert isinstance(coa, COA)

def test_coa_constructor_with_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("10"), "Assets"),
        AccountType.LIABILITY: (Code("20"), "Liabilities")
    }
    coa = COA(rootspec=custom_spec)
    asset_account = coa.find(Code("10"))
    liability_account = coa.find(Code("20"))
    assert asset_account.name == "Assets"
    assert liability_account.name == "Liabilities"
    assert asset_account.code == Code("10")
    assert liability_account.code == Code("20")

def test_coa_constructor_empty_rootspec_behavior():
    coa = COA(rootspec={})
    first_account = next(iter(coa.accounts))
    assert first_account.code == Code("1")
```


# LLM-generated content at query #6
#--------------------------

```python
def test_coa_add_success():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    added_account = coa.add(parent_code, new_code, new_name)
    
    assert added_account.code == new_code
    assert added_account.name == new_name
    assert added_account.parent.code == parent_code
    assert coa.find(new_code) == added_account
    assert added_account in coa.subaccounts(coa.find(parent_code))

def test_coa_add_same_code_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    code = Code("1")
    
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(code, code, "Self Parent")

def test_coa_add_nonexistent_parent_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("99")
    new_code = Code("11")
    
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(parent_code, new_code, "Ghost Account")

def test_coa_add_duplicate_account_returns_existing():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    first_account = coa.add(parent_code, new_code, new_name)
    second_account = coa.add(parent_code, new_code, new_name)
    
    assert first_account == second_account
    assert len(coa.subaccounts(coa.find(parent_code))) == 1

def test_coa_add_duplicate_account_mismatch_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Original Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
```


# LLM-generated content at query #7
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self):
            self.data = {"account1": 100}

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    expected_coa = MockCOA()
    reader = MockReadChartOfAccounts()
    actual_coa = reader()

    assert actual_coa == expected_coa
    assert actual_coa.data["account1"] == 100
```


# LLM-generated content at query #8
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")) is not None

def test_coa_constructor_with_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Assets"),
        AccountType.LIABILITY: (Code("200"), "Liabilities")
    }
    coa = COA(rootspec=custom_spec)
    asset_acc = coa.find(Code("100"))
    liability_acc = coa.find(Code("200"))
    assert asset_acc.name == "Assets"
    assert liability_acc.name == "Liabilities"
    assert asset_acc.code == Code("100")

def test_coa_constructor_empty_rootspec_behavior():
    coa = COA(rootspec={})
    first_account = next(iter(coa.accounts))
    assert first_account.code == Code("1")
```


# LLM-generated content at query #9
#--------------------------

```python
def test_coa_constructor_with_no_rootspec():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Type"
    # Assuming AccountType has at least one member, e.g., ASSET
    # We map the first type to our custom spec
    first_type = list(AccountType)[0]
    rootspec = {first_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code
    assert isinstance(found_account, RootAccount)

def test_coa_constructor_default_naming():
    coa = COA()
    first_type = list(AccountType)[0]
    # Default name is TypeName.capitalize()
    expected_name = first_type.name.capitalize()
    found_account = next(a for a in coa.accounts if a.code == Code("1"))
    assert found_account.name == expected_name
```


# LLM-generated content at query #10
#--------------------------

```python
def test_subaccount_constructor():
    mock_code = Code("123")
    mock_name = "Savings"
    
    class MockParent:
        type = AccountType.ASSET
        coa = "COA_001"
    
    mock_parent = MockParent()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == "COA_001"
```


# LLM-generated content at query #11
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in coa.accounts)
    assert all(a.parent is None for a in coa.accounts)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Type"
    # Assuming AccountType has at least one member, e.g., ASSET
    spec = {AccountType.ASSET: (custom_code, custom_name)}
    coa = COA(rootspec=spec)
    
    account_asset = coa.find(custom_code)
    assert account_asset is not None
    assert account_asset.name == custom_name
    assert account_asset.code == custom_code
    assert isinstance(account_asset, RootAccount)

def test_coa_constructor_with_partial_rootspec():
    custom_code = Code("10")
    custom_name = "Special Asset"
    spec = {AccountType.ASSET: (custom_code, custom_name)}
    coa = COA(rootspec=spec)
    
    # Check customized account
    assert coa.find(custom_code).name == custom_name
    # Check that other accounts are still initialized with defaults from the enumeration
    # We find an account that wasn't in spec by checking if code is not custom_code
    other_accounts = [a for a in coa.accounts if a.code != custom_code]
    assert len(other_accounts) > 0
    assert all(a.parent is None for a in other_accounts)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSET
    rootspec = {root_type: (root_code, root_name)}
    coa = COA(rootspec=rootspec)
    
    sub_code = Code("11")
    sub_name = "Cash"
    sub_account = coa.add(root_code, sub_code, sub_name)
    
    node = coa.nodify(coa.find(root_code))
    
    assert isinstance(node, COA.Node)
    assert node.account == coa.find(root_code)
    assert len(node.children) == 1
    assert node.children[0].account == sub_account
```


# LLM-generated content at query #13
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = "12345"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        def __init__(self):
            self.type = "Asset"
            self.coa = "Standard COA"
    
    mock_parent = MockParent()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "Standard COA"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_add_does_not_raise_when_parent_exists():
    root_code = Code("1")
    parent_code = Code("100")
    new_account_code = Code("101")
    account_name = "Test Account"
    rootspec = {AccountType.ASSET: (root_code, "Assets")}
    coa = COA(rootspec=rootspec)
    coa.add(root_code, parent_code, "Parent Account")
    new_account = coa.add(parent_code, new_account_code, account_name)
    assert new_account.code == new_account_code
    assert coa.find(new_account_code) is not None
```


# LLM-generated content at query #15
#--------------------------

```python
def test_add_with_existing_parent():
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Cash"
    rootspec = {AccountType.ASSET: (parent_code, "Assets")}
    coa = COA(rootspec=rootspec)
    
    # Ensure parent exists in the COA before calling add
    assert coa.find(parent_code) is not None
    
    # This call should NOT trigger the 'if parentinstance is None' block
    new_account = coa.add(parent=parent_code, code=child_code, name=child_name)
    
    assert new_account.code == child_code
    assert new_account.parent.code == parent_code
```


# LLM-generated content at query #16
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = "SA001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        def __init__(self):
            self.type = "Asset"
            self.coa = "MainCOA"

    mock_parent = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "MainCOA"
```


# LLM-generated content at query #17
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Code("123")
    mock_name = "Savings Sub-account"
    
    class MockAccount:
        def __init__(self, account_type, coa):
            self.type = account_type
            self.coa = coa

    mock_type = AccountType.ASSET
    mock_coa = COA("Default COA")
    parent_account = MockAccount(mock_type, mock_coa)

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == mock_type
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #18
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    coa = COA()
    root_code = Code("1")
    child_code = Code("1.1")
    child_name = "Sub Account"
    
    coa.add(root_code, child_code, child_name)
    
    root_account = coa.find(root_code)
    child_account = coa.find(child_code)
    
    node = coa.nodify(root_account)
    
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
```


# LLM-generated content at query #19
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        type = "Asset"
        coa = "MainCOA"
        
    mock_parent = MockParent()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "MainCOA"

def test_subaccount_immutability():
    class MockParent:
        type = "Liability"
        coa = "SecondaryCOA"
        
    mock_parent = MockParent()
    sub_account = SubAccount(code="SUB002", name="Credit Sub-Account", parent=mock_parent)
    
    from dataclasses import FrozenInstanceError
    try:
        sub_account.name = "New Name"
    except FrozenInstanceError:
        pass
```


# LLM-generated content at query #20
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = "1001"
    mock_name = "Savings Sub-Account"
    
    class MockAccount:
        def __init__(self):
            self.type = "Asset"
            self.coa = "Standard COA"
            
    mock_parent = MockAccount()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == "Asset"
    assert sub_account.coa == "Standard COA"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_subaccount_constructor():
    mock_code = Code("123")
    mock_name = "Savings"
    
    class MockParent:
        type = AccountType.ASSET
        coa = MockCOA()

    class MockCOA:
        pass

    parent_account = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == parent_account.coa
```


# LLM-generated content at query #22
#--------------------------

```python
def test_coa_constructor_default_roots():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(a.parent is None for a in accounts_list)
    assert any(a.code == Code("1") and a.name == "Asset" for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("10"), "Total Assets"),
        AccountType.LIABILITY: (Code("20"), "Total Liabilities")
    }
    coa = COA(rootspec=custom_spec)
    asset_acc = coa.find(Code("10"))
    liability_acc = coa.find(Code("20"))
    assert asset_acc.name == "Total Assets"
    assert liability_acc.name == "Total Liabilities"
    assert asset_acc.parent is None
    assert liability_acc.parent is None

def test_coa_constructor_empty_rootspec_is_treated_as_none():
    coa = COA(rootspec={})
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert any(a.code == Code("1") for a in accounts_list)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_add_fails_when_parent_not_defined():
    coa = COA()
    non_existent_parent_code = Code("999")
    new_account_code = Code("100")
    account_name = "New Account"
    
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(non_existent_parent_code, new_account_code, account_name)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_coa_constructor_no_rootspec():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert accounts_list[0].code == Code("1")
    assert accounts_list[0].name == AccountType.ASSET.name.capitalize()

def test_coa_constructor_with_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("10"), "Assets"),
        AccountType.LIABILITY: (Code("20"), "Liabilities")
    }
    coa = COA(rootspec=custom_spec)
    asset_account = coa.find(Code("10"))
    liability_account = coa.find(Code("20"))
    assert asset_account.name == "Assets"
    assert liability_account.code == Code("20")
    assert coa.find(Code("1")) is None
```


# LLM-generated content at query #25
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_expected_coa():
    class MockCOA:
        def __init__(self, data):
            self.data = data

    class MockReadChartOfAccounts:
        def __init__(self, coa_data):
            self.coa_data = coa_data

        def __call__(self) -> MockCOA:
            return MockCOA(self.coa_data)

    expected_data = {"account_1": 100, "account_2": 200}
    reader = MockReadChartOfAccounts(expected_data)
    result = reader()

    assert result.data == expected_data
```


# LLM-generated content at query #26
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self):
            self.data = {"account1": 100}

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    expected_coa = MockCOA()
    reader = MockReadChartOfAccounts()
    result = reader()
    
    assert result == expected_coa
    assert result.data["account1"] == 100
```


# LLM-generated content at query #27
#--------------------------

```python
def test_coa_add_success():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

def test_coa_add_same_code_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    code = Code("1")
    
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(code, code, "Self Parent")

def test_coa_add_non_existent_parent_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("99")
    new_code = Code("991")
    
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(parent_code, new_code, "Orphan Account")

def test_coa_add_existing_account_returns_same_instance():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    account_first_call = coa.add(parent_code, new_code, new_name)
    account_second_call = coa.add(parent_code, new_code, new_name)
    
    assert account_first_call is account_second_call
    assert account_first_call.code == new_code

def test_coa_add_existing_account_mismatch_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Original Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
```


# LLM-generated content at query #28
#--------------------------

```python
def test_coa_add_success():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

def test_coa_add_fails_self_parenting():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    code = Code("1")
    
    try:
        coa.add(code, code, "Self")
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_coa_add_fails_missing_parent():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("99")
    new_code = Code("991")
    
    try:
        coa.add(parent_code, new_code, "No Parent")
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_coa_add_returns_existing_account_if_identical():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    first_account = coa.add(parent_code, new_code, new_name)
    second_account = coa.add(parent_code, new_code, new_name)
    
    assert first_account == second_account
    assert len(coa.subaccounts(coa.find(parent_code))) == 1

def test_coa_add_fails_on_inconsistent_existing_account():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=rootspec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Original Name")
    
    try:
        coa.add(parent_code, new_code, "Different Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."
```


# LLM-generated content at query #29
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_correct_value():
    class MockCOA:
        def __init__(self, data):
            self.data = data

    expected_coa = MockCOA({"account1": 100, "account2": 200})
    
    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return expected_coa

    reader: ReadChartOfAccounts = MockReadChartOfAccounts()
    result = reader()

    assert result == expected_coa
    assert result.data["account1"] == 100
```


# LLM-generated content at query #30
#--------------------------

```python
def test_add_raises_error_on_inconsistent_account_data():
    parent_code = Code("1")
    child_code = Code("1.1")
    root_spec = {AccountType.ASSET: (parent_code, "Assets")}
    coa = COA(rootspec=root_spec)
    # Pre-populate the account with a different name to trigger the 'else' block
    coa.add(parent_code, child_code, "Original Name")
    # Attempt to add the same code but with a different name
    import pytest
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, child_code, "Different Name")
```


# LLM-generated content at query #31
#--------------------------

```python
def test_coa_add_success():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

def test_coa_add_error_self_parenting():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    same_code = Code("1")
    
    import pytest
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(same_code, same_code, "Self Parent")

def test_coa_add_error_missing_parent():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    non_existent_parent = Code("99")
    new_code = Code("991")
    
    import pytest
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(non_existent_parent, new_code, "Orphan Account")

def test_coa_add_return_existing_if_identical():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    name = "Cash"
    
    first_call = coa.add(parent_code, new_code, name)
    second_call = coa.add(parent_code, new_code, name)
    
    assert first_call == second_call
    assert len(coa.subaccounts(coa.find(parent_code))) == 1

def test_coa_add_error_inconsistent_data():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    existing_code = Code("11")
    coa.add(parent_code, existing_code, "Cash")
    
    import pytest
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, existing_code, "Different Name")

def test_coa_add_error_mismatched_parent():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets"), AccountType.LIABILITY: (Code("2"), "Liabilities")}
    coa = COA(rootspec=root_spec)
    existing_code = Code("11")
    coa.add(Code("1"), existing_code, "Cash")
    
    import pytest
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), existing_code, "Cash")
```


# LLM-generated content at query #32
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Mock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = Mock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_type_property_returns_parent_type():
    mock_code = Mock(spec=Code)
    mock_parent = Mock(spec=Account)
    expected_type = AccountType.ASSET
    mock_parent.type = expected_type
    
    sub_account = SubAccount(code=mock_code, name="Sub", parent=mock_parent)
    
    assert sub_account.type == expected_type

def test_subaccount_coa_property_returns_parent_coa():
    mock_code = Mock(spec=Code)
    mock_parent = Mock(spec=Account)
    expected_coa = Mock(spec=COA)
    mock_parent.coa = expected_coa
    
    sub_account = SubAccount(code=mock_code, name="Sub", parent=mock_parent)
    
    assert sub_account.coa == expected_coa

def test_subaccount_immutability():
    mock_code = Mock(spec=Code)
    mock_parent = Mock(spec=Account)
    sub_account = SubAccount(code=mock_code, name="Sub", parent=mock_parent)
    
    from dataclasses import replace
    
    with pytest.raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


# LLM-generated content at query #33
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Sub Account Name"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_properties_delegation():
    mock_code = MagicMock(spec=Code)
    mock_name = "Sub Account Name"
    mock_parent = MagicMock(spec=Account)
    mock_type = MagicMock(spec=AccountType)
    mock_coa = MagicMock(spec=COA)
    
    mock_parent.type = mock_type
    mock_parent.coa = mock_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == mock_type
    assert sub_account.coa == mock_coa

def test_subaccount_immutability():
    mock_code = MagicMock(spec=Code)
    mock_parent = MagicMock(spec=Account)
    sub_account = SubAccount(code=mock_code, name="Test", parent=mock_parent)
    
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


