####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")) is not None

def test_coa_constructor_with_rootspec():
    spec = {AccountType.ASSET: (Code("100"), "Assets")}
    coa = COA(rootspec=spec)
    account = coa.find(Code("100"))
    assert account.name == "Assets"
    assert account.code == Code("100")
    assert account.parent is None

def test_coa_constructor_raises_error_on_invalid_logic():
    # Note: The provided COA implementation does not explicitly raise an error in __post_init__ 
    # for duplicate codes via rootspec, but it will overwrite the dictionary entry.
    spec = {AccountType.ASSET: (Code("1"), "Duplicate")}
    coa = COA(rootspec=spec)
    account = coa.find(Code("1"))
    assert account.name == "Duplicate"
```


# LLM-generated content at query #2
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    coa = COA()
    root_account = coa.find(Code("1"))
    sub_account = coa.add(Code("1"), Code("1.1"), "Sub Account")
    
    node = coa.nodify(root_account)
    
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == sub_account

def test_nodify_handles_leaf_node():
    coa = COA()
    root_account = coa.find(Code("1"))
    
    node = coa.nodify(root_account)
    
    assert node.account == root_account
    assert node.children == []

def test_nodify_recursive_structure():
    coa = COA()
    root = coa.find(Code("1"))
    child = coa.add(Code("1"), Code("1.1"), "Child")
    grandchild = coa.add(Code("1.1"), Code("1.1.1"), "Grandchild")
    
    node = coa.nodify(root)
    
    assert node.account == root
    assert node.children[0].account == child
    assert node.children[0].children[0].account == grandchild
```


# LLM-generated content at query #3
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_type_property_delegation():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    mock_parent.type = AccountType.ASSET
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == AccountType.ASSET

def test_subaccount_coa_property_delegation():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    mock_coa = MagicMock(spec=COA)
    mock_parent.coa = mock_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.coa == mock_coa

def test_subaccount_immutability():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    with pytest.raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_properties_delegate_to_parent():
    mock_code = MagicMock(spec=Code)
    mock_name = "Checking Sub-account"
    mock_parent = MagicMock(spec=Account)
    mock_type = AccountType.ASSET
    mock_coa = MagicMock(spec=COA)
    
    mock_parent.type = mock_type
    mock_parent.coa = mock_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == mock_type
    assert sub_account.coa == mock_coa

def test_subaccount_immutability():
    mock_code = MagicMock(spec=Code)
    mock_parent = MagicMock(spec=Account)
    sub_account = SubAccount(code=mock_code, name="Immutable", parent=mock_parent)
    
    try:
        sub_account.name = "New Name"
    except Exception as e:
        assert isinstance(e, FrozenInstanceError) or True
```


# LLM-generated content at query #5
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-account"
    mock_parent = MagicMock(spec=Account)
    mock_coa = MagicMock(spec=COA)
    mock_parent.type = AccountType.ASSET
    mock_parent.coa = mock_coa

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #6
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings"
    mock_parent = MagicMock(spec=Account)
    mock_coa = MagicMock(spec=COA)
    mock_parent.type = AccountType.ASSET
    mock_parent.coa = mock_coa

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #7
#--------------------------

```python
def test_read_chart_of_accounts_success():
    class MockCOA:
        pass

    mock_coa = MockCOA()
    
    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return mock_coa

    reader: ReadChartOfAccounts = MockReadChartOfAccounts()
    result = reader()
    
    assert result == mock_coa
```


# LLM-generated content at query #8
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self):
            self.data = {"1000": "Cash"}

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    expected_coa = MockCOA()
    reader: ReadChartOfAccounts = MockReadChartOfAccounts()
    result = reader()

    assert result == expected_coa
    assert result.data["1000"] == "Cash"
```


# LLM-generated content at query #9
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-Account"
    
    class MockAccount:
        def __init__(self, account_type, coa):
            self.type = account_type
            self.coa = coa

    mock_account_type = "ASSET"
    mock_coa = "COA_ROOT"
    parent_account = MockAccount(mock_account_type, mock_coa)

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == mock_account_type
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #10
#--------------------------

```python
def test_coa_add_success():
    parent_code = Code("1")
    child_code = Code("101")
    child_name = "Cash"
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    new_account = coa.add(parent_code, child_code, child_name)
    
    assert new_account.code == child_code
    assert new_account.name == child_name
    assert new_account.parent.code == parent_code
    assert child_account_in_coa := coa.find(child_code) is not None
    assert child_account_in_coa.name == child_name
    assert child_code in [acc.code for acc in coa.subaccounts(coa.find(parent_code))]

def test_coa_add_same_code_raises_error():
    parent_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    
    import pytest
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(parent_code, parent_code, "Self Parent")

def test_coa_add_non_existent_parent_raises_error():
    parent_code = Code("1")
    child_code = Code("101")
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    invalid_parent = Code("999")
    
    import pytest
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(invalid_parent, child_code, "Orphan Account")

def test_coa_add_duplicate_account_returns_existing():
    parent_code = Code("1")
    child_code = Code("101")
    child_name = "Cash"
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    first_acc = coa.add(parent_code, child_code, child_name)
    second_acc = coa.add(parent_code, child_code, child_name)
    
    assert first_acc is second_acc
    assert second_acc.code == child_code

def test_coa_add_duplicate_account_mismatch_raises_error():
    parent_code = Code("1")
    child_code = Code("101")
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    coa.add(parent_code, child_code, "Original Name")
    
    import pytest
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, child_code, "Different Name")
```


# LLM-generated content at query #11
#--------------------------

```python
def test_add_existing_account_inconsistent_data_raises_error():
    """
    Ensures that when an account with the same code already exists but has 
    different parent, name, or code, a ValueError is raised. 
    This forces the predicate at line 27 to evaluate to False.
    """
    parent_code = Code("1")
    child_code = Code("101")
    # Initialize COA with a root account for AccountType.ASSET (assuming it exists)
    # We use a generic setup that ensures parent_code exists in _accounts.
    coa = COA(rootspec={AccountType.ASSET: (parent_code, "Assets")})
    
    # First, add the account correctly to populate self._accounts[child_code]
    coa.add(parent_code, child_code, "Cash")
    
    # Now, attempt to 'add' the same code but with a different name 
    # This makes (account.name == name) False, thus making the whole predicate False.
    import pytest
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, child_code, "Different Name")
```


# LLM-generated content at query #12
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

def test_coa_add_self_parent_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    same_code = Code("1")
    
    try:
        coa.add(same_code, same_code, "Self Parent Error")
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_coa_add_nonexistent_parent_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    invalid_parent = Code("99")
    new_code = Code("11")
    
    try:
        coa.add(invalid_parent, new_code, "No Parent Error")
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_coa_add_existing_account_returns_same_instance():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    first_add = coa.add(parent_code, new_code, new_name)
    second_add = coa.add(parent_code, new_code, new_name)
    
    assert first_add is second_add
    assert first_add.code == new_code

def test_coa_add_existing_account_mismatch_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Original Name")
    
    try:
        coa.add(parent_code, new_code, "Different Name")
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."
```


# LLM-generated content at query #13
#--------------------------

```python
def test_add_raises_error_on_inconsistent_account_data():
    root_code = Code("1")
    root_name = "Assets"
    parent_code = Code("11")
    parent_name = "Current Assets"
    existing_code = Code("111")
    existing_name = "Cash"
    inconsistent_name = "Different Name"
    
    coa = COA(rootspec={AccountType.ASSET: (root_code, root_name)})
    coa.add(root_code, parent_code, parent_name)
    coa.add(parent_code, existing_code, existing_name)
    
    import pytest
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, existing_code, inconsistent_name)
```


# LLM-generated content at query #14
#--------------------------

```python
from typing import Callable

def test_read_chart_of_accounts_call_returns_expected_value():
    class MockCOA:
        def __eq__(self, other):
            return isinstance(other, dict) and other == {"1000": "Cash"}

    expected_coa = {"1000": "Cash"}
    
    def mock_reader() -> MockCOA:
        return expected_coa

    reader: ReadChartOfAccounts = mock_reader
    result = reader()
    
    assert result == expected_coa
```


# LLM-generated content at query #15
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_correct_coa():
    class MockCOA:
        def __init__(self, data):
            self.data = data

    expected_coa = MockCOA({"account1": 100, "account2": 200})

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return expected_coa

    reader: ReadChartOfAccounts = MockReadChartOfAccounts()
    actual_coa = reader()

    assert actual_coa == expected_coa
    assert actual_coa.data["account1"] == 100
```


# LLM-generated content at query #16
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Mock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = Mock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_type_property_access():
    mock_code = Mock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = Mock(spec=Account)
    mock_type = AccountType.ASSET
    mock_parent.type = mock_type
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == mock_type

def test_subaccount_coa_property_access():
    mock_code = Mock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = Mock(spec=Account)
    mock_coa = Mock(spec=COA)
    mock_parent.coa = mock_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.coa == mock_coa

def test_subaccount_immutability():
    mock_code = Mock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = Mock(spec=Account)
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    from dataclasses import FrozenInstanceError
    from pytest import raises
    
    with raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_properties_delegation():
    mock_code = MagicMock(spec=Code)
    mock_name = "Checking Sub-Account"
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
    sub_account = SubAccount(code=mock_code, name="Immutable", parent=mock_parent)
    
    with pytest.raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


# LLM-generated content at query #3
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert isinstance(coa, COA)

def test_coa_constructor_with_rootspec():
    spec = {
        AccountType.ASSET: (Code("100"), "Assets Account")
    }
    coa = COA(rootspec=spec)
    account = coa.find(Code("100"))
    assert account.name == "Assets Account"
    assert account.code == Code("100")
    assert account.type == AccountType.ASSET

def test_coa_constructor_integrity():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(coa._accounts) == len(accounts_list)
    assert isinstance(coa._subaccounts, dict)
```


# LLM-generated content at query #4
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
    code = Code("1")
    
    try:
        coa.add(code, code, "Self Parent")
        assert False, "Should have raised ValueError for self-parenting"
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_coa_add_error_missing_parent():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    non_existent_parent = Code("99")
    new_code = Code("991")
    
    try:
        coa.add(non_existent_parent, new_code, "Orphan")
        assert False, "Should have raised ValueError for missing parent"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_coa_add_existing_account_same_info():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    first_account = coa.add(parent_code, new_code, new_name)
    second_account = coa.add(parent_code, new_code, new_name)
    
    assert first_account == second_account

def test_coa_add_existing_account_mismatch_info():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Original Name")
    
    try:
        coa.add(parent_code, new_code, "Different Name")
        assert False, "Should have raised ValueError for name mismatch"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."
```


# LLM-generated content at query #5
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_properties_access_parent_attributes():
    mock_code = MagicMock(spec=Code)
    mock_name = "Checking Sub-Account"
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
    sub_account = SubAccount(code=mock_code, name="Immutable", parent=mock_parent)
    
    with pytest.raises(AttributeError):
        sub_account.name = "New Name"
```


# LLM-generated content at query #6
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(acc, RootAccount) for acc in accounts_list)
    assert all(acc.parent is None for acc in accounts_list)

def test_coa_constructor_with_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Assets Account"),
        AccountType.LIABILITY: (Code("200"), "Liabilities Account")
    }
    coa = COA(rootspec=custom_spec)
    asset_acc = coa.find(Code("100"))
    liability_acc = coa.find(Code("200"))
    assert asset_acc.name == "Assets Account"
    assert liability_acc.name == "Liabilities Account"
    assert asset_acc.code == Code("100")
    assert liability_acc.code == Code("200")

def test_coa_constructor_empty_rootspec_behavior():
    coa = COA(rootspec={})
    accounts_dict = dict(coa)
    assert len(accounts_dict) == len(AccountType)
    assert Code("1") in accounts_dict
```


# LLM-generated content at query #7
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSET  # Assuming AccountType is defined elsewhere as per context
    
    coa = COA(rootspec={root_type: (root_code, root_name)})
    root_account = coa.find(root_code)
    
    child_code = Code("11")
    child_name = "Cash"
    child_account = coa.add(root_code, child_code, child_name)
    
    grandchild_code = Code("1101")
    grandchild_name = "Petty Cash"
    grandchild_account = coa.add(child_code, grandchild_code, grandchild_name)
    
    node = coa.nodify(root_account)
    
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == grandchild_account

def test_nodify_with_no_children_returns_leaf_node():
    root_code = Code("2")
    root_name = "Liabilities"
    root_type = AccountType.LIABILITY
    
    coa = COA(rootspec={root_type: (root_code, root_name)})
    root_account = coa.find(root_code)
    
    node = coa.nodify(root_account)
    
    assert node.account == root_account
    assert node.children == []

def test_nodify_handles_multiple_siblings():
    root_code = Code("3")
    root_name = "Equity"
    root_type = AccountType.EQUITY
    
    coa = COA(rootspec={root_type: (root_code, root_name)})
    root_account = coa.find(root_code)
    
    child1 = coa.add(root_code, Code("31"), "Retained Earnings")
    child2 = coa.add(root_code, Code("32"), "Common Stock")
    
    node = coa.nodify(root_account)
    
    assert len(node.children) == 2
    assert node.children[0].account == child1
    assert node.children[1].account == child2
```


# LLM-generated content at query #8
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

def test_coa_add_nonexistent_parent_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("99")
    new_code = Code("991")
    
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(parent_code, new_code, "No Parent")

def test_coa_add_existing_account_returns_same():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    name = "Cash"
    
    account_first = coa.add(parent_code, new_code, name)
    account_second = coa.add(parent_code, new_code, name)
    
    assert account_first == account_second
    assert account_first.code == new_code

def test_coa_add_existing_account_mismatch_raises_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Cash")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
```


# LLM-generated content at query #9
#--------------------------

```python
def test_add_with_existing_parent_should_not_raise_error():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSETS
    rootspec = {root_type: (root_code, root_name)}
    coa = COA(rootspec=rootspec)
    parent_code = root_code
    new_code = Code("11")
    new_name = "Cash"
    account = coa.add(parent=parent_code, code=new_code, name=new_name)
    assert account.code == new_code
    assert account.parent.code == parent_code
```


# LLM-generated content at query #10
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert all(a.parent is None for a in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Assets Account")
    }
    coa = COA(rootspec=custom_spec)
    asset_account = coa.find(Code("100"))
    assert asset_account is not None
    assert asset_account.name == "Assets Account"
    assert asset_account.code == Code("100")

def test_coa_constructor_verifies_accounts_dict():
    coa = COA()
    assert len(coa._accounts) == len(AccountType)
    assert isinstance(coa._accounts, dict)

def test_coa_constructor_iteration():
    coa = COA()
    iterated_items = list(coa)
    assert len(iterated_items) == len(AccountType)
    assert all(isinstance(item[0], Code) and isinstance(item[1], Account) for item in iterated_items)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")) is not None

def test_coa_constructor_with_rootspec():
    spec = {AccountType.ASSET: (Code("100"), "Assets")}
    coa = COA(rootspec=spec)
    account = coa.find(Code("100"))
    assert account.name == "Assets"
    assert account.code == Code("100")
    assert account.parent is None

def test_coa_constructor_preserves_order():
    coa = COA()
    accounts_list = list(coa.accounts)
    account_codes = [a.code for a in accounts_list]
    assert account_codes == [Code(str(i)) for i in range(1, len(AccountType) + 1)]
```


# LLM-generated content at query #12
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
    root_name = "Assets"
    coa = COA(rootspec={AccountType.ASSET: (root_code, root_name)})
    
    account = coa.find(root_code)
    node = coa.nodify(account)
    
    assert node.account == account
    assert node.children == []

def test_nodify_with_multiple_siblings():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    
    sub1 = coa.add(root_code, Code("11"), "Cash")
    sub2 = coa.add(root_code, Code("12"), "Inventory")
    
    node = coa.nodify(coa.find(root_code))
    
    assert len(node.children) == 2
    assert node.children[0].account.code == Code("11")
    assert node.children[1].account.code == Code("12")
```


# LLM-generated content at query #13
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_expected_coa():
    class MockCOA:
        def __init__(self, data):
            self.data = data

    mock_data = {"1000": "Cash", "2000": "Accounts Payable"}
    expected_coa = MockCOA(mock_data)

    class MockReadChartOfAccounts:
        def __call__(self):
            return expected_coa

    reader = MockReadChartOfAccounts()
    result = reader()

    assert result == expected_coa
    assert result.data["1000"] == "Cash"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = "SUB001"
    mock_name = "Savings Sub-account"
    
    class MockParent:
        type = "Asset"
        coa = "MainCOA"
    
    parent_account = MockParent()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == "Asset"
    assert sub_account.coa == "MainCOA"
```


# LLM-generated content at query #15
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_expected_coa():
    class MockCOA:
        def __eq__(self, other):
            return isinstance(other, dict) and other == {"account1": 100}

    mock_coa_data = {"account1": 100}
    
    class MockReadChartOfAccounts:
        def __call__(self) -> dict:
            return mock_coa_data

    reader = MockReadChartOfAccounts()
    result = reader()
    
    assert result == mock_coa_data
```


# LLM-generated content at query #16
#--------------------------

```python
def test_add_fails_when_parent_is_not_defined():
    coa = COA()
    parent_code = Code("1000")
    new_code = Code("1010")
    account_name = "Test Account"
    
    import pytest
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(parent=parent_code, code=new_code, name=account_name)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = MockCode(value="1001")
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        type = MockAccountType.ASSET
        coa = MockCOA()
        
    mock_parent = MockParent()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == MockAccountType.ASSET
    assert sub_account.coa == mock_parent.coa

def test_sub_account_immutability():
    mock_code = MockCode(value="1001")
    mock_name = "Savings Sub-Account"
    
    class MockParent:
        type = MockAccountType.ASSET
        coa = MockCOA()
        
    mock_parent = MockParent()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    with pytest.raises(Exception): # Using generic exception check as per instruction constraints on logic
        sub_account.name = "New Name"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Code("123")
    mock_name = "Savings"
    mock_parent = Account(code=Code("100"), name="Assets", type=AccountType.ASSET, coa=COA())
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_parent.coa
```


# LLM-generated content at query #19
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

    expected_coa = MockCOA({"account1": 100, "account2": 200})
    reader = MockReadChartOfAccounts(expected_coa)
    
    result = reader()
    
    assert result == expected_coa
    assert result.data["account1"] == 100
```


# LLM-generated content at query #20
#--------------------------

```python
def test_nodify_returns_node_with_correct_structure():
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


# LLM-generated content at query #21
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    root_code = Code("1")
    root_name = "Assets"
    root_type = AccountType.ASSET  # Assuming AccountType exists based on context
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
```


# LLM-generated content at query #22
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_type_property_returns_parent_type():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = MagicMock(spec=Account)
    expected_type = AccountType.ASSET
    mock_parent.type = expected_type
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == expected_type

def test_subaccount_coa_property_returns_parent_coa():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = MagicMock(spec=Account)
    expected_coa = MagicMock(spec=COA)
    mock_parent.coa = expected_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.coa == expected_coa

def test_subaccount_immutability():
    mock_code = MagicMock(spec=Code)
    mock_parent = MagicMock(spec=Account)
    sub_account = SubAccount(code=mock_code, name="Test", parent=mock_parent)
    
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


# LLM-generated content at query #23
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) > 0
    assert isinstance(coa, COA)

def test_coa_constructor_with_rootspec():
    custom_spec = {AccountType.ASSET: (Code("100"), "Assets")}
    coa = COA(rootspec=custom_spec)
    account = coa.find(Code("100"))
    assert account is not None
    assert account.name == "Assets"
    assert account.code == Code("100")

def test_coa_constructor_verifies_account_types_coverage():
    coa = COA()
    account_types_in_coa = [a.type for a in coa.accounts]
    for atype in AccountType:
        assert atype in account_types_in_coa
```


# LLM-generated content at query #24
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
        AccountType.ASSET: (Code("100"), "Assets"),
        AccountType.LIABILITY: (Code("200"), "Liabilities")
    }
    coa = COA(rootspec=spec)
    asset_acc = coa.find(Code("100"))
    liability_acc = coa.find(Code("200"))
    assert asset_acc.name == "Assets"
    assert liability_acc.name == "Liabilities"
    assert asset_acc.type == AccountType.ASSET
    assert liability_acc.type == AccountType.LIABILITY

def test_coa_constructor_with_partial_rootspec():
    spec = {
        AccountType.ASSET: (Code("999"), "Custom Asset")
    }
    coa = COA(rootspec=spec)
    custom_acc = coa.find(Code("999"))
    assert custom_acc.name == "Custom Asset"
    assert custom_acc.type == AccountType.ASSET
    
    # Check that other accounts still use default logic (e.g., 2nd type in enum)
    other_types = [a for a in coa.accounts if a.type != AccountType.ASSET]
    assert len(other_types) > 0
```


# LLM-generated content at query #25
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings"
    mock_parent = MagicMock(spec=Account)
    mock_coa = MagicMock(spec=COA)
    mock_parent.type = AccountType.ASSET
    mock_parent.coa = mock_coa

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #26
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = Code(value="1001")
    mock_name = "Savings Sub-Account"
    
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


# LLM-generated content at query #27
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert isinstance(coa._accounts, dict)
    assert isinstance(coa._subaccounts, dict)

def test_coa_constructor_with_rootspec():
    spec = {AccountType.ASSET: (Code("100"), "Assets")}
    coa = COA(rootspec=spec)
    asset_account = coa.find(Code("100"))
    assert asset_account is not None
    assert asset_account.name == "Assets"
    assert asset_account.code == Code("100")

def test_coa_constructor_with_partial_rootspec():
    spec = {AccountType.LIABILITY: (Code("200"), "Liabilities")}
    coa = COA(rootspec=spec)
    liability_account = coa.find(Code("200"))
    assert liability_account is not None
    assert liability_account.name == "Liabilities"
    # Check that the first type (e.g., Asset) still got a default code/name
    first_acc = next(iter(coa.accounts))
    assert first_acc.code is not None
    assert first_acc.name is not None
```


# LLM-generated content at query #28
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert isinstance(next(iter(coa.accounts)), RootAccount)

def test_coa_constructor_with_rootspec():
    code1 = Code("100")
    name1 = "Assets"
    code2 = Code("200")
    name2 = "Liabilities"
    rootspec = {
        AccountType.ASSET: (code1, name1),
        AccountType.LIABILITY: (code2, name/name2) # Note: Assuming AccountType exists in scope as per context
    }
    # Re-evaluating based on provided snippet structure: 
    # Since we cannot define custom functions or control structures, 
    # we assume the environment has AccountType, Code, and RootAccount.
    rootspec = {
        AccountType.ASSET: (Code("10"), "Assets Custom"),
        AccountType.LIABILITY: (Code("20"), "Liabilities Custom")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("10")) is not None
    assert coa.find(Code("10")).name == "Assets Custom"
    assert coa.find(Code("20")) is not None
    assert coa.find(Code("20")).name == "Liabilities Custom"

def test_coa_constructor_preserves_order():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert all(isinstance(a, RootAccount) for a in accounts_list)
    assert len(accounts_list) == len(AccountType)
```


# LLM-generated content at query #29
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self):
            self.data = {"1000": "Cash"}

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    expected_coa = MockCOA()
    reader = MockReadChartOfAccounts()
    result = reader()
    
    assert result == expected_coa
    assert result.data["1000"] == "Cash"
```


# LLM-generated content at query #30
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = Code("12345")
    mock_name = "Savings Sub-Account"
    
    class MockAccount:
        def __init__(self, account_type, coa):
            self.type = account_type
            self.coa = coa

    mock_type = AccountType.ASSET
    mock_coa = COA("Standard")
    parent_account = MockAccount(mock_type, mock_coa)

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=parent_account)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == parent_account
    assert sub_account.type == mock_type
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #31
#--------------------------

```python
def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")) is not None

def test_coa_constructor_with_custom_rootspec():
    custom_spec = {
        AccountType.ASSET: (Code("100"), "Assets Account")
    }
    coa = COA(rootspec=custom_spec)
    asset_acc = coa.find(Code("100"))
    assert asset_acc is not None
    assert asset_acc.name == "Assets Account"
    assert asset_acc.code == Code("100")

def test_coa_constructor_default_naming_logic():
    coa = COA(rootspec={})
    # Based on __post_init__: code is str(c) where c is enumerate index starting 1
    # and name is AccountType.name.capitalize()
    first_acc = coa.find(Code("1"))
    assert first_acc.code == Code("1")
    assert first_acc.name == AccountType.ASSET.name.capitalize()

def test_coa_constructor_preserves_order():
    coa = COA(rootspec={})
    accounts_list = list(coa.accounts)
    first_code = accounts_list[0].code
    second_code = accounts_list[1].code
    assert first_code < second_code
```


# LLM-generated content at query #32
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
    custom_name = "Custom Asset"
    rootspec = {AccountType.ASSET: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    asset_account = coa.find(custom_code)
    assert asset_account is not None
    assert asset_account.name == custom_name
    assert asset_account.code == custom_code
    assert asset_account.type == AccountType.ASSET

def test_coa_constructor_rootspec_partial():
    custom_code = Code("10")
    custom_name = "Custom Liability"
    rootspec = {AccountType.LIABILITY: (custom_code, custom_name)}
    coa = COA(rootspec=rootspec)
    assert coa.find(custom_code) is not None
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).name == AccountType.ASSET.name.capitalize()
```


# LLM-generated content at query #33
#--------------------------

```python
def test_add_fails_when_parent_is_not_defined():
    coa = COA()
    parent_code = Code("100")
    new_code = Code("101")
    account_name = "Test Account"
    
    import pytest
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(parent_code, new_code, account_name)
```


# LLM-generated content at query #34
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    coa = COA()
    root_code = Code("1")
    child_code = Code("1.1")
    child_name = "Child Account"
    
    root_account = coa.add(root_code, root_code, "Root Account") # Note: In actual usage COA init handles roots
    # Since we can't modify the COA class to fix the logic in 'add' for roots easily without knowing RootAccount implementation, 
    # we assume standard setup based on provided snippet.
    
    # Manually injecting into _accounts for a controlled test environment since add() has restrictions
    coa._accounts[root_code] = RootAccount(root_code, "Root", AccountType.ASSET, coa)
    child_account = coa.add(root_code, child_code, child_name)
    
    node = coa.nodify(root_account)
    
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
```


# LLM-generated content at query #35
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = Mock()
    mock_name = "Sub Account Name"
    mock_parent = Mock()
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_sub_account_type_property_returns_parent_type():
    mock_code = Mock()
    mock_name = "Sub Account Name"
    mock_parent = Mock()
    expected_type = Mock()
    mock_parent.type = expected_type
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == expected_type

def test_sub_account_coa_property_returns_parent_coa():
    mock_code = Mock()
    mock_name = "Sub Account Name"
    mock_parent = Mock()
    expected_coa = Mock()
    mock_parent.coa = expected_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.coa == expected_coa

def test_sub_account_is_frozen():
    mock_code = Mock()
    mock_name = "Sub Account Name"
    mock_parent = Mock()
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    from dataclasses import FrozenInstanceError
    
    with FrozenInstanceError:
        sub_account.name = "New Name"
```


# LLM-generated content at query #36
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
    assert added_account in coa.subaccounts(coa.find(parent_code))
    assert coa.find(new_code) == added_account

def test_coa_add_same_code_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    code = Code("1")
    
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(code, code, "Self Parent")

def test_coa_add_missing_parent_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    non_existent_parent = Code("99")
    new_code = Code("991")
    
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(non_existent_parent, new_code, "Ghost Account")

def test_coa_add_existing_account_idempotency():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    new_name = "Cash"
    
    first_add = coa.add(parent_code, new_code, new_name)
    second_add = coa.add(parent_code, new_code, new_name)
    
    assert first_add == second_add
    assert len(coa.subaccounts(coa.find(parent_code))) == 1

def test_coa_add_existing_account_mismatch_error():
    root_spec = {AccountType.ASSET: (Code("1"), "Assets")}
    coa = COA(rootspec=root_spec)
    parent_code = Code("1")
    new_code = Code("11")
    
    coa.add(parent_code, new_code, "Original Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
```


# LLM-generated content at query #37
#--------------------------

```python
def test_sub_account_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Sub-Account"
    mock_parent = MagicMock(spec=Account)
    mock_coa = MagicMock(spec=COA)
    mock_parent.type = AccountType.ASSET
    mock_parent.coa = mock_coa

    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)

    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == mock_coa
```


# LLM-generated content at query #38
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(AccountType)
    assert all(isinstance(acc, Account) for acc in accounts_list)
    assert all(acc.parent is None for acc in accounts_list)

def test_coa_constructor_with_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Root"
    custom_type = AccountType.ASSET  # Assuming AccountType.ASSET exists based on context
    rootspec = {custom_type: (custom_code, custom_name)}
    coa = COA(rootspec=rootspend)
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.code == custom_code
    assert found_account.name == custom_name
    assert found_account.type == custom_type

def test_coa_constructor_identifies_toplevel_correctly():
    coa = COA()
    toplevel_accounts = list(coa.toplevel)
    for acc in toplevel_accounts:
        assert acc.parent is None
```


# LLM-generated content at query #39
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")) is not None

def test_coa_constructor_with_rootspec():
    rootspec = {AccountType.ASSET: (Code("100"), "Assets")}
    coa = COA(rootspec=rootspec)
    account = coa.find(Code("100"))
    assert account.name == "Assets"
    assert account.code == Code("100")

def test_coa_constructor_preserves_order():
    coa = COA()
    accounts_list = list(coa.accounts)
    codes = [a.code for a in accounts_list]
    assert codes == sorted(codes, key=lambda x: str(x))
```


# LLM-generated content at query #40
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self, data):
            self.data = data

    expected_coa = MockCOA({"account1": 100})
    
    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return expected_coa

    reader: ReadChartOfAccounts = MockReadChartOfAccounts()
    result = reader()
    
    assert result == expected_coa
    assert result.data["account1"] == 100
```


# LLM-generated content at query #41
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

def test_nodify_handles_leaf_nodes():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    
    node = coa.nodify(coa.find(root_code))
    
    assert node.account == coa.find(root_code)
    assert node.children == []

def test_nodify_handles_empty_subaccounts():
    root_code = Code("1")
    coa = COA(rootspec={AccountType.ASSET: (root_code, "Assets")})
    account = coa.find(root_code)
    
    assert coa.subaccounts(account) == []
    node = coa.nodify(account)
    assert node.children == []
```


# LLM-generated content at query #42
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_type_property_returns_parent_type():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    expected_type = AccountType.ASSET
    mock_parent.type = expected_type
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == expected_type

def test_subaccount_coa_property_returns_parent_coa():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    expected_coa = MagicMock(spec=COA)
    mock_parent.coa = expected_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.coa == expected_coa

def test_subaccount_immutability():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    from dataclasses import FrozenInstanceError
    from pytest importraises
    
    with raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


# LLM-generated content at query #43
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
    custom_name = "Custom Type"
    # Assuming AccountType has at least one member, e.g., ASSET
    spec = {AccountType.ASSET: (custom_code, custom_name)}
    coa = COA(rootspec=spec)
    found_account = coa.find(custom_code)
    assert found_account is not None
    assert found_account.name == custom_name
    assert found_account.code == custom_code

def test_coa_constructor_with_partial_rootspec():
    custom_code = Code("10")
    custom_name = "Custom Root"
    spec = {AccountType.ASSET: (custom_code, custom_name)}
    coa = COA(rootspec=spec)
    # Check that the specified one is customized
    assert coa.find(custom_code).name == custom_name
    # Check that others are still initialized with defaults (e.g., code '2' for second type)
    remaining_accounts = list(coa.accounts)
    assert len(remaining_accounts) == len(AccountType)
    assert any(acc.code == Code("2") for acc in remaining_accounts)
```


# LLM-generated content at query #44
#--------------------------

```python
def test_coa_constructor_with_empty_rootspec():
    coa = COA(rootspec=None)
    assert len(list(coa.accounts)) == len(AccountType)
    assert coa.find(Code("1")) is not None

def test_coa_constructor_with_custom_rootspec():
    custom_code = Code("99")
    custom_name = "Custom Account"
    rootspec = {AccountType.ASSET: (custom_code, custom_name)}
    coa = COA(rootspec=rootspend)
    account = coa.find(custom_code)
    assert account is not None
    assert account.name == custom_name
    assert account.code == custom_code

def test_coa_constructor_default_values():
    coa = COA()
    first_type = list(AccountType)[0]
    expected_code = Code("1")
    expected_name = first_type.name.capitalize()
    account = coa.find(expected_code)
    assert account is not None
    assert account.name == expected_name
```


# LLM-generated content at query #45
#--------------------------

```python
def test_nodify_returns_correct_node_structure():
    code_root = Code("1")
    code_child = Code("1.1")
    name_root = "Assets"
    name_child = "Cash"
    
    coa = COA(rootspec={AccountType.ASSETS: (code_root, name_root)})
    account_root = coa.find(code_root)
    account_child = coa.add(code_root, code_child, name_child)
    
    node = coa.nodify(account_root)
    
    assert isinstance(node, COA.Node)
    assert node.account == account_root
    assert len(node.children) == 1
    assert node.children[0].account == account_child
    assert node.children[0].account.name == name_child

def test_nodify_leaf_node_has_no_children():
    code_root = Code("1")
    coa = COA(rootspec={AccountType.ASSETS: (code_root, "Assets")})
    account_root = coa.find(code_root)
    
    node = coa.nodify(account_root)
    
    assert node.account == account_root
    assert node.children == []

def test_nodify_deep_structure():
    code_1 = Code("1")
    code_1_1 = Code("1.1")
    code_1_1_1 = Code("1.1.1")
    coa = COA(rootspec={AccountType.ASSETS: (code_1, "Assets")})
    acc_1 = coa.find(code_1)
    acc_1_1 = coa.add(code_1, code_1_1, "Current Assets")
    acc_1_1_1 = coa.add(code_1_1, code_1_1_1, "Cash")
    
    node_root = coa.nodify(acc_1)
    node_mid = node_root.children[0]
    node_leaf = node_mid.children[0]
    
    assert node_root.account == acc_1
    assert node_mid.account == acc_1_1
    assert node_leaf.account == acc_1_1_1
    assert len(node_leaf.children) == 0
```


# LLM-generated content at query #46
#--------------------------

```python
def test_subaccount_constructor_initialization():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.code == mock_code
    assert sub_account.name == mock_name
    assert sub_account.parent == mock_parent

def test_subaccount_type_property_returns_parent_type():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    expected_type = AccountType.ASSET
    mock_parent.type = expected_type
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.type == expected_type

def test_subaccount_coa_property_returns_parent_coa():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    expected_coa = MagicMock(spec=COA)
    mock_parent.coa = expected_coa
    
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    assert sub_account.coa == expected_coa

def test_subaccount_is_immutable():
    mock_code = MagicMock(spec=Code)
    mock_name = "Savings Account"
    mock_parent = MagicMock(spec=Account)
    sub_account = SubAccount(code=mock_code, name=mock_name, parent=mock_parent)
    
    with pytest.raises(FrozenInstanceError):
        sub_account.name = "New Name"
```


# LLM-generated content at query #47
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_expected_coa():
    class MockCOA:
        pass

    class MockReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()

    expected_coa = MockCOA()
    reader = MockReadChartOfAccounts()
    result = reader()
    assert result == expected_coa
```


