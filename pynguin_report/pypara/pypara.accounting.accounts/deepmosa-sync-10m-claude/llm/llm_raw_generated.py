####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_read_chart_of_accounts_call():
    from typing import Protocol
    
    class COA:
        def __init__(self, accounts=None):
            self.accounts = accounts or []
    
    class ReadChartOfAccounts(Protocol):
        def __call__(self) -> COA:
            ...
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA(accounts=["Account1", "Account2"])
    
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    assert isinstance(result, COA)
    assert result.accounts == ["Account1", "Account2"]


# LLM-generated content at query #2
#--------------------------

```python
def test_nodify_creates_node_with_account():
    from collections import OrderedDict
    
    coa = COA()
    root_account = coa.find(Code("1"))
    
    node = coa.nodify(root_account)
    
    assert node.account == root_account
    assert node.children == []


def test_nodify_creates_node_with_children():
    from collections import OrderedDict
    
    coa = COA()
    root_account = coa.find(Code("1"))
    child_account = coa.add(Code("1"), Code("1.1"), "Test Child")
    
    node = coa.nodify(root_account)
    
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account


def test_nodify_creates_nested_tree_structure():
    from collections import OrderedDict
    
    coa = COA()
    root_account = coa.find(Code("1"))
    child_account = coa.add(Code("1"), Code("1.1"), "Test Child")
    grandchild_account = coa.add(Code("1.1"), Code("1.1.1"), "Test Grandchild")
    
    node = coa.nodify(root_account)
    
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == grandchild_account


def test_nodify_returns_node_instance():
    from collections import OrderedDict
    
    coa = COA()
    root_account = coa.find(Code("1"))
    
    node = coa.nodify(root_account)
    
    assert isinstance(node, COA.Node)


def test_nodify_with_multiple_children():
    from collections import OrderedDict
    
    coa = COA()
    root_account = coa.find(Code("1"))
    child1 = coa.add(Code("1"), Code("1.1"), "Child 1")
    child2 = coa.add(Code("1"), Code("1.2"), "Child 2")
    child3 = coa.add(Code("1"), Code("1.3"), "Child 3")
    
    node = coa.nodify(root_account)
    
    assert len(node.children) == 3
    assert node.children[0].account == child1
    assert node.children[1].account == child2
    assert node.children[2].account == child3


# LLM-generated content at query #3
#--------------------------

```python
def test_subaccount_constructor():
    from dataclasses import dataclass
    from enum import Enum
    
    class AccountType(Enum):
        ASSET = "asset"
        LIABILITY = "liability"
    
    class Code(str):
        pass
    
    @dataclass(frozen=True)
    class COA:
        name: str
    
    @dataclass(frozen=True)
    class Account:
        code: Code
        name: str
        type: AccountType
        coa: COA
    
    coa_instance = COA(name="Chart of Accounts")
    parent_account = Account(code=Code("1000"), name="Assets", type=AccountType.ASSET, coa=coa_instance)
    
    code = Code("1100")
    name = "Cash"
    
    sub_account = SubAccount(code=code, name=name, parent=parent_account)
    
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == coa_instance


# LLM-generated content at query #4
#--------------------------

```python
def test_nodify_returns_coa_node_instance():
    from collections import OrderedDict
    from dataclasses import dataclass
    
    # Create a mock Account class
    @dataclass
    class Account:
        code: str
        name: str
        parent: 'Account' = None
    
    # Create a mock RootAccount class
    @dataclass
    class RootAccount(Account):
        type: str = None
        coa: 'COA' = None
    
    # Create a mock SubAccount class
    @dataclass
    class SubAccount(Account):
        pass
    
    # Create instances
    coa = COA()
    root_account = Account(code="1", name="Root")
    
    # Call nodify
    node = coa.nodify(root_account)
    
    # Verify the result is a COA.Node instance
    assert isinstance(node, COA.Node)
    assert node.account == root_account
    assert isinstance(node.children, list)


# LLM-generated content at query #5
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert coa._accounts is not None
    assert coa._subaccounts is not None
    assert len(coa._accounts) == 5
    for account in coa._accounts.values():
        assert isinstance(account, RootAccount)

def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1000"), "Assets"),
        AccountType.LIABILITY: (Code("2000"), "Liabilities"),
    }
    coa = COA(rootspec=custom_rootspec)
    asset_account = coa.find(Code("1000"))
    assert asset_account is not None
    assert asset_account.code == Code("1000")
    assert asset_account.name == "Assets"
    assert asset_account.type == AccountType.ASSET

def test_coa_constructor_partial_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("5000"), "Custom Assets"),
    }
    coa = COA(rootspec=custom_rootspec)
    custom_asset = coa.find(Code("5000"))
    assert custom_asset is not None
    assert custom_asset.name == "Custom Assets"
    default_liability = coa.find(Code("2"))
    assert default_liability is not None
    assert default_liability.name == "Liability"

def test_coa_constructor_initializes_empty_subaccounts():
    coa = COA()
    assert isinstance(coa._subaccounts, dict)
    assert len(coa._subaccounts) == 0

def test_coa_constructor_creates_root_accounts_in_order():
    coa = COA()
    accounts_list = list(coa._accounts.values())
    assert len(accounts_list) == 5
    assert all(acc.parent is None for acc in accounts_list)

def test_coa_constructor_with_none_rootspec():
    coa = COA(rootspec=None)
    assert len(coa._accounts) == 5
    for account in coa._accounts.values():
        assert isinstance(account, RootAccount)

def test_coa_constructor_accounts_are_frozen():
    coa = COA()
    with_error = False
    try:
        coa._accounts = {}
    except Exception:
        with_error = True
    assert with_error


# LLM-generated content at query #6
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    accounts = list(coa.accounts)
    assert len(accounts) == len(list(AccountType))
    for account in accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None


def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
    }
    coa = COA(rootspec=custom_rootspec)
    asset_account = coa.find(Code("1"))
    assert asset_account is not None
    assert asset_account.name == "Assets"
    assert asset_account.type == AccountType.ASSET
    
    liability_account = coa.find(Code("2"))
    assert liability_account is not None
    assert liability_account.name == "Liabilities"
    assert liability_account.type == AccountType.LIABILITY


def test_coa_constructor_initializes_all_account_types():
    coa = COA()
    account_types = list(AccountType)
    accounts = list(coa.accounts)
    assert len(accounts) == len(account_types)
    for i, account_type in enumerate(account_types, start=1):
        found_account = None
        for account in accounts:
            if account.type == account_type:
                found_account = account
                break
        assert found_account is not None


def test_coa_constructor_with_partial_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("100"), "Total Assets"),
    }
    coa = COA(rootspec=custom_rootspec)
    asset_account = coa.find(Code("100"))
    assert asset_account is not None
    assert asset_account.name == "Total Assets"
    
    accounts = list(coa.accounts)
    assert len(accounts) == len(list(AccountType))


def test_coa_constructor_frozen():
    coa = COA()
    try:
        coa._accounts = {}
        assert False, "Expected frozen dataclass to prevent attribute assignment"
    except (AttributeError, TypeError):
        pass


def test_coa_constructor_creates_root_accounts():
    coa = COA()
    toplevel_accounts = list(coa.toplevel)
    assert len(toplevel_accounts) == len(list(AccountType))
    for account in toplevel_accounts:
        assert account.parent is None
        assert isinstance(account, RootAccount)


# LLM-generated content at query #7
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    
    assert coa._accounts is not None
    assert coa._subaccounts is not None
    assert len(coa._accounts) > 0
    
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(list(AccountType))
    
    for account in accounts_list:
        assert isinstance(account, RootAccount)
        assert account.parent is None


def test_coa_constructor_with_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSETS: (Code("1"), "Assets"),
        AccountType.LIABILITIES: (Code("2"), "Liabilities"),
    }
    
    coa = COA(rootspec=custom_rootspec)
    
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("2")).name == "Liabilities"


def test_coa_constructor_accounts_are_root_accounts():
    coa = COA()
    
    for code, account in coa:
        assert isinstance(account, RootAccount)
        assert account.parent is None


def test_coa_constructor_creates_all_account_types():
    coa = COA()
    
    account_types = list(AccountType)
    accounts_count = len(list(coa.accounts))
    
    assert accounts_count == len(account_types)


def test_coa_constructor_empty_rootspec_uses_defaults():
    coa = COA(rootspec={})
    
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(list(AccountType))


def test_coa_constructor_none_rootspec_uses_defaults():
    coa = COA(rootspec=None)
    
    accounts_list = list(coa.accounts)
    assert len(accounts_list) == len(list(AccountType))


# LLM-generated content at query #8
#--------------------------

```python
def test_add_creates_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Test Sub-Account"
    
    result = coa.add(parent_code, child_code, child_name)
    
    assert result.code == child_code
    assert result.name == child_name
    assert result.parent.code == parent_code
    assert coa.find(child_code) == result


def test_add_raises_error_when_parent_equals_code():
    coa = COA()
    code = Code("1")
    
    try:
        coa.add(code, code, "Test")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "An account can not be the parent of itself" in str(e)


def test_add_raises_error_when_parent_not_defined():
    coa = COA()
    parent_code = Code("99")
    child_code = Code("99.1")
    
    try:
        coa.add(parent_code, child_code, "Test")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Parent account is not (yet) defined" in str(e)


def test_add_returns_existing_account_if_consistent():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Test Sub-Account"
    
    account1 = coa.add(parent_code, child_code, child_name)
    account2 = coa.add(parent_code, child_code, child_name)
    
    assert account1 == account2
    assert account1 is account2


def test_add_raises_error_when_account_exists_with_different_info():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    
    coa.add(parent_code, child_code, "Original Name")
    
    try:
        coa.add(parent_code, child_code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Account name, code and parent do not match" in str(e)


def test_add_updates_subaccounts_buffer():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    
    parent_account = coa.find(parent_code)
    coa.add(parent_code, child_code, "Test Sub-Account")
    
    subaccounts = coa.subaccounts(parent_account)
    
    assert len(subaccounts) == 1
    assert subaccounts[0].code == child_code


def test_add_multiple_subaccounts_to_same_parent():
    coa = COA()
    parent_code = Code("1")
    child_code1 = Code("1.1")
    child_code2 = Code("1.2")
    
    account1 = coa.add(parent_code, child_code1, "First Child")
    account2 = coa.add(parent_code, child_code2, "Second Child")
    
    parent_account = coa.find(parent_code)
    subaccounts = coa.subaccounts(parent_account)
    
    assert len(subaccounts) == 2
    assert account1 in subaccounts
    assert account2 in subaccounts


# LLM-generated content at query #9
#--------------------------

```python
def test_add_existing_account_returns_existing():
    from collections import OrderedDict
    from enum import Enum
    
    class AccountType(Enum):
        ASSET = 1
        LIABILITY = 2
        EQUITY = 3
        REVENUE = 4
        EXPENSE = 5
    
    class Code(str):
        pass
    
    class Account:
        def __init__(self, code, name, account_type, coa):
            self.code = code
            self.name = name
            self.type = account_type
            self.coa = coa
            self.parent = None
    
    class RootAccount(Account):
        pass
    
    # Create COA instance
    coa = COA()
    
    # Get a root account to use as parent
    parent_code = Code("1")
    parent_account = coa.find(parent_code)
    
    # Add a new account
    child_code = Code("1.1")
    child_name = "Test Account"
    added_account = coa.add(parent_code, child_code, child_name)
    
    # Verify the account is in _accounts (predicate at line 22 should be True)
    assert child_code in coa._accounts
    
    # Add the same account again with same parameters
    result_account = coa.add(parent_code, child_code, child_name)
    
    # Verify it returns the existing account
    assert result_account is added_account
    assert result_account.code == child_code
    assert result_account.name == child_name


# LLM-generated content at query #10
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    class MockCOA:
        def __init__(self):
            self.accounts = []
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()
    
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    assert result is not None
    assert isinstance(result, MockCOA)
    assert hasattr(result, 'accounts')


def test_read_chart_of_accounts_call_is_callable():
    class MockCOA:
        pass
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()
    
    reader = ConcreteReadChartOfAccounts()
    
    assert callable(reader)


def test_read_chart_of_accounts_call_returns_consistent_type():
    class MockCOA:
        def __init__(self, name="default"):
            self.name = name
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA("test_coa")
    
    reader = ConcreteReadChartOfAccounts()
    result1 = reader()
    result2 = reader()
    
    assert type(result1) == type(result2)
    assert isinstance(result1, MockCOA)
    assert isinstance(result2, MockCOA)


# LLM-generated content at query #11
#--------------------------

```python
def test_subaccount_constructor():
    from dataclasses import dataclass
    
    # Create a mock parent account
    @dataclass(frozen=True)
    class AccountType:
        name: str
    
    @dataclass(frozen=True)
    class COA:
        name: str
    
    @dataclass(frozen=True)
    class Account:
        type: AccountType
        coa: COA
    
    @dataclass(frozen=True)
    class Code:
        value: str
    
    @dataclass(frozen=True)
    class SubAccount:
        code: Code
        name: str
        parent: Account
        
        @property
        def type(self):
            return self.parent.type
        
        @property
        def coa(self):
            return self.parent.coa
    
    # Create test data
    code = Code(value="SUB001")
    account_type = AccountType(name="Asset")
    coa = COA(name="Standard COA")
    parent_account = Account(type=account_type, coa=coa)
    
    # Test constructor
    sub_account = SubAccount(code=code, name="Sub Account 1", parent=parent_account)
    
    # Assertions
    assert sub_account.code == code
    assert sub_account.name == "Sub Account 1"
    assert sub_account.parent == parent_account
    assert sub_account.type == account_type
    assert sub_account.coa == coa


# LLM-generated content at query #12
#--------------------------

```python
def test_read_chart_of_accounts_call():
    class MockCOA:
        def __init__(self):
            self.accounts = []
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()
    
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    assert isinstance(result, MockCOA)
    assert hasattr(result, 'accounts')
    assert result.accounts == []


def test_read_chart_of_accounts_call_returns_coa_instance():
    class SimpleCOA:
        def __init__(self, name="Default COA"):
            self.name = name
    
    class SimpleReadChartOfAccounts:
        def __call__(self) -> SimpleCOA:
            return SimpleCOA("Test COA")
    
    reader = SimpleReadChartOfAccounts()
    coa = reader()
    
    assert isinstance(coa, SimpleCOA)
    assert coa.name == "Test COA"


def test_read_chart_of_accounts_call_multiple_invocations():
    class CounterCOA:
        def __init__(self, count):
            self.count = count
    
    call_count = 0
    
    class CountingReadChartOfAccounts:
        def __call__(self) -> CounterCOA:
            nonlocal call_count
            call_count += 1
            return CounterCOA(call_count)
    
    reader = CountingReadChartOfAccounts()
    result1 = reader()
    result2 = reader()
    result3 = reader()
    
    assert result1.count == 1
    assert result2.count == 2
    assert result3.count == 3
    assert call_count == 3


# LLM-generated content at query #13
#--------------------------

```python
def test_subaccount_constructor():
    from dataclasses import dataclass
    
    # Create a mock parent Account
    @dataclass(frozen=True)
    class AccountType:
        name: str
    
    @dataclass(frozen=True)
    class COA:
        name: str
    
    @dataclass(frozen=True)
    class Code:
        value: str
    
    @dataclass(frozen=True)
    class Account:
        code: Code
        name: str
        type: AccountType
        coa: COA
    
    @dataclass(frozen=True)
    class SubAccount:
        code: Code
        name: str
        parent: Account
        
        @property
        def type(self) -> AccountType:
            return self.parent.type
        
        @property
        def coa(self) -> COA:
            return self.parent.coa
    
    # Create test data
    code = Code(value="ACC001")
    parent_code = Code(value="PARENT001")
    account_type = AccountType(name="Asset")
    coa = COA(name="Standard COA")
    parent = Account(code=parent_code, name="Parent Account", type=account_type, coa=coa)
    
    # Test constructor
    sub_account = SubAccount(code=code, name="Sub Account", parent=parent)
    
    assert sub_account.code == code
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent
    assert sub_account.type == account_type
    assert sub_account.coa == coa


def test_subaccount_constructor_frozen():
    from dataclasses import dataclass
    
    @dataclass(frozen=True)
    class AccountType:
        name: str
    
    @dataclass(frozen=True)
    class COA:
        name: str
    
    @dataclass(frozen=True)
    class Code:
        value: str
    
    @dataclass(frozen=True)
    class Account:
        code: Code
        name: str
        type: AccountType
        coa: COA
    
    @dataclass(frozen=True)
    class SubAccount:
        code: Code
        name: str
        parent: Account
        
        @property
        def type(self) -> AccountType:
            return self.parent.type
        
        @property
        def coa(self) -> COA:
            return self.parent.coa
    
    code = Code(value="ACC001")
    parent_code = Code(value="PARENT001")
    account_type = AccountType(name="Asset")
    coa = COA(name="Standard COA")
    parent = Account(code=parent_code, name="Parent Account", type=account_type, coa=coa)
    
    sub_account = SubAccount(code=code, name="Sub Account", parent=parent)
    
    try:
        sub_account.code = Code(value="NEW_CODE")
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, TypeError):
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_subaccount_constructor():
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccountType:
        name: str
    
    @dataclass
    class MockCOA:
        name: str
    
    @dataclass
    class MockAccount:
        type: MockAccountType
        coa: MockCOA
    
    # Create instances
    mock_type = MockAccountType(name="Asset")
    mock_coa = MockCOA(name="Standard COA")
    mock_parent = MockAccount(type=mock_type, coa=mock_coa)
    
    code = "1000"
    name = "Cash"
    
    # Create SubAccount instance
    sub_account = SubAccount(code=code, name=name, parent=mock_parent)
    
    # Assertions
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_type
    assert sub_account.coa == mock_coa


def test_subaccount_constructor_with_different_values():
    from dataclasses import dataclass
    
    @dataclass
    class MockAccountType:
        name: str
    
    @dataclass
    class MockCOA:
        name: str
    
    @dataclass
    class MockAccount:
        type: MockAccountType
        coa: MockCOA
    
    mock_type = MockAccountType(name="Liability")
    mock_coa = MockCOA(name="IFRS COA")
    mock_parent = MockAccount(type=mock_type, coa=mock_coa)
    
    code = "2000"
    name = "Accounts Payable"
    
    sub_account = SubAccount(code=code, name=name, parent=mock_parent)
    
    assert sub_account.code == code
    assert sub_account.name == name
    assert sub_account.parent == mock_parent
    assert sub_account.type == mock_type
    assert sub_account.coa == mock_coa


def test_subaccount_is_frozen():
    from dataclasses import dataclass
    
    @dataclass
    class MockAccountType:
        name: str
    
    @dataclass
    class MockCOA:
        name: str
    
    @dataclass
    class MockAccount:
        type: MockAccountType
        coa: MockCOA
    
    mock_type = MockAccountType(name="Equity")
    mock_coa = MockCOA(name="Test COA")
    mock_parent = MockAccount(type=mock_type, coa=mock_coa)
    
    sub_account = SubAccount(code="3000", name="Capital", parent=mock_parent)
    
    try:
        sub_account.code = "3001"
        assert False, "Expected FrozenInstanceError"
    except Exception as e:
        assert "frozen" in str(type(e).__name__).lower()


# LLM-generated content at query #15
#--------------------------

```python
def test_add_basic_subaccount():
    from collections import OrderedDict
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Test SubAccount"
    
    result = coa.add(parent_code, child_code, child_name)
    
    assert result.code == child_code
    assert result.name == child_name
    assert result.parent.code == parent_code
    assert child_code in coa._accounts


def test_add_same_parent_and_code_raises_error():
    coa = COA()
    code = Code("1")
    
    try:
        coa.add(code, code, "Test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "An account can not be the parent of itself" in str(e)


def test_add_nonexistent_parent_raises_error():
    coa = COA()
    parent_code = Code("999")
    child_code = Code("999.1")
    
    try:
        coa.add(parent_code, child_code, "Test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Parent account is not (yet) defined" in str(e)


def test_add_existing_account_with_matching_info_returns_existing():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    child_name = "Test SubAccount"
    
    first_result = coa.add(parent_code, child_code, child_name)
    second_result = coa.add(parent_code, child_code, child_name)
    
    assert first_result is second_result
    assert len([a for a in coa.accounts]) == 2


def test_add_existing_account_with_mismatched_info_raises_error():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    
    coa.add(parent_code, child_code, "Original Name")
    
    try:
        coa.add(parent_code, child_code, "Different Name")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Account name, code and parent do not match" in str(e)


def test_add_multiple_subaccounts_to_same_parent():
    coa = COA()
    parent_code = Code("1")
    
    child1 = coa.add(parent_code, Code("1.1"), "First Child")
    child2 = coa.add(parent_code, Code("1.2"), "Second Child")
    
    subaccounts = coa.subaccounts(coa.find(parent_code))
    
    assert len(subaccounts) == 2
    assert child1 in subaccounts
    assert child2 in subaccounts


def test_add_nested_subaccounts():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    grandchild_code = Code("1.1.1")
    
    child = coa.add(parent_code, child_code, "Child")
    grandchild = coa.add(child_code, grandchild_code, "Grandchild")
    
    assert grandchild.parent == child
    assert grandchild.code == grandchild_code
    assert coa.find(grandchild_code) == grandchild


def test_add_account_updates_subaccounts_buffer():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    
    parent_account = coa.find(parent_code)
    coa.add(parent_code, child_code, "Test Child")
    
    assert parent_account in coa._subaccounts
    assert len(coa._subaccounts[parent_account]) == 1
    assert coa._subaccounts[parent_account][0].code == child_code


# LLM-generated content at query #16
#--------------------------

```python
def test_subaccount_constructor():
    from dataclasses import dataclass
    from enum import Enum
    
    class AccountType(Enum):
        ASSET = "asset"
        LIABILITY = "liability"
    
    @dataclass(frozen=True)
    class COA:
        name: str
    
    @dataclass(frozen=True)
    class Account:
        code: str
        name: str
        type: AccountType
        coa: COA
    
    @dataclass(frozen=True)
    class Code:
        value: str
    
    @dataclass(frozen=True)
    class SubAccount:
        code: Code
        name: str
        parent: Account
        
        @property
        def type(self) -> AccountType:
            return self.parent.type
        
        @property
        def coa(self) -> COA:
            return self.parent.coa
    
    coa = COA(name="Chart of Accounts")
    parent_account = Account(code="1000", name="Assets", type=AccountType.ASSET, coa=coa)
    code = Code(value="1001")
    
    sub_account = SubAccount(code=code, name="Cash", parent=parent_account)
    
    assert sub_account.code == code
    assert sub_account.name == "Cash"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == coa


# LLM-generated content at query #17
#--------------------------

```python
def test_read_chart_of_accounts_call():
    class MockCOA:
        def __init__(self):
            self.accounts = []
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> MockCOA:
            return MockCOA()
    
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    assert result is not None
    assert isinstance(result, MockCOA)
    assert hasattr(result, 'accounts')
    assert result.accounts == []


def test_read_chart_of_accounts_call_returns_coa():
    class SimpleCOA:
        def __init__(self, data=None):
            self.data = data or {}
    
    class TestableReadChartOfAccounts:
        def __call__(self) -> SimpleCOA:
            return SimpleCOA({'account1': 'value1'})
    
    reader = TestableReadChartOfAccounts()
    coa = reader()
    
    assert isinstance(coa, SimpleCOA)
    assert coa.data == {'account1': 'value1'}


def test_read_chart_of_accounts_call_multiple_times():
    class CounterCOA:
        def __init__(self, count):
            self.count = count
    
    call_count = 0
    
    class MultiCallReadChartOfAccounts:
        def __call__(self) -> CounterCOA:
            nonlocal call_count
            call_count += 1
            return CounterCOA(call_count)
    
    reader = MultiCallReadChartOfAccounts()
    result1 = reader()
    result2 = reader()
    
    assert result1.count == 1
    assert result2.count == 2
    assert call_count == 2


# LLM-generated content at query #18
#--------------------------

```python
def test_subaccount_constructor():
    from dataclasses import dataclass
    from typing import TYPE_CHECKING
    
    # Create mock Account and COA objects
    @dataclass(frozen=True)
    class MockCOA:
        name: str
    
    @dataclass(frozen=True)
    class MockAccount:
        code: str
        name: str
        type: str
        coa: MockCOA
    
    # Create test data
    test_code = "1000"
    test_name = "Cash Account"
    test_coa = MockCOA(name="Standard COA")
    test_parent = MockAccount(code="1", name="Assets", type="asset", coa=test_coa)
    
    # Create SubAccount instance
    sub_account = SubAccount(code=test_code, name=test_name, parent=test_parent)
    
    # Assert constructor sets attributes correctly
    assert sub_account.code == test_code
    assert sub_account.name == test_name
    assert sub_account.parent == test_parent
    assert sub_account.type == "asset"
    assert sub_account.coa == test_coa
    
    # Assert the instance is frozen (immutable)
    try:
        sub_account.code = "2000"
        assert False, "SubAccount should be frozen"
    except AttributeError:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_add_account_with_inconsistent_information_raises_error():
    from enum import Enum
    from typing import NewType
    
    Code = NewType('Code', str)
    
    class AccountType(Enum):
        ASSET = 1
        LIABILITY = 2
        EQUITY = 3
        REVENUE = 4
        EXPENSE = 5
    
    class Account:
        def __init__(self, code, name, account_type, coa):
            self.code = code
            self.name = name
            self.type = account_type
            self.coa = coa
            self.parent = None
    
    class RootAccount(Account):
        pass
    
    coa = COA(rootspec={AccountType.ASSET: (Code("1"), "Assets")})
    
    parent_code = Code("1")
    account_code = Code("100")
    
    coa.add(parent_code, account_code, "Test Account")
    
    try:
        coa.add(parent_code, account_code, "Different Name")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Account name, code and parent do not match" in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_add_basic_subaccount():
    coa = COA()
    root_account = coa.find(Code("1"))
    new_account = coa.add(Code("1"), Code("1.1"), "Test Account")
    assert new_account.code == Code("1.1")
    assert new_account.name == "Test Account"
    assert new_account.parent == root_account


def test_add_nested_subaccount():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "First Level")
    new_account = coa.add(Code("1.1"), Code("1.1.1"), "Second Level")
    assert new_account.code == Code("1.1.1")
    assert new_account.name == "Second Level"
    assert new_account.parent.code == Code("1.1")


def test_add_duplicate_account_same_details():
    coa = COA()
    account1 = coa.add(Code("1"), Code("1.1"), "Test Account")
    account2 = coa.add(Code("1"), Code("1.1"), "Test Account")
    assert account1 == account2


def test_add_parent_not_defined():
    coa = COA()
    try:
        coa.add(Code("99"), Code("99.1"), "Test Account")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Parent account is not (yet) defined" in str(e)


def test_add_account_is_own_parent():
    coa = COA()
    try:
        coa.add(Code("1.1"), Code("1.1"), "Test Account")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "An account can not be the parent of itself" in str(e)


def test_add_duplicate_account_different_name():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Original Name")
    try:
        coa.add(Code("1"), Code("1.1"), "Different Name")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Account name, code and parent do not match existing chart of accounts member" in str(e)


def test_add_duplicate_account_different_parent():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    coa.add(Code("2"), Code("2.1"), "Another Account")
    try:
        coa.add(Code("2"), Code("1.1"), "Test Account")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Account name, code and parent do not match existing chart of accounts member" in str(e)


def test_add_multiple_accounts_same_parent():
    coa = COA()
    account1 = coa.add(Code("1"), Code("1.1"), "First Child")
    account2 = coa.add(Code("1"), Code("1.2"), "Second Child")
    subaccounts = coa.subaccounts(coa.find(Code("1")))
    assert len(subaccounts) == 2
    assert account1 in subaccounts
    assert account2 in subaccounts


def test_add_account_in_subaccounts_buffer():
    coa = COA()
    parent = coa.find(Code("1"))
    new_account = coa.add(Code("1"), Code("1.1"), "Test Account")
    subaccounts = coa.subaccounts(parent)
    assert new_account in subaccounts


# LLM-generated content at query #21
#--------------------------

```python
def test_add_account_with_inconsistent_name_raises_error():
    from enum import Enum
    from collections import OrderedDict
    
    class AccountType(Enum):
        ASSET = 1
        LIABILITY = 2
        EQUITY = 3
        REVENUE = 4
        EXPENSE = 5
    
    class Code:
        def __init__(self, value):
            self.value = value
        
        def __eq__(self, other):
            return self.value == other.value if isinstance(other, Code) else self.value == other
        
        def __hash__(self):
            return hash(self.value)
        
        def __repr__(self):
            return f"Code({self.value})"
    
    class Account:
        def __init__(self, code, name, account_type, coa, parent=None):
            self.code = code
            self.name = name
            self._type = account_type
            self._coa = coa
            self.parent = parent
        
        @property
        def type(self):
            return self._type
        
        @property
        def coa(self):
            return self._coa
    
    class RootAccount(Account):
        pass
    
    coa = COA(rootspec={AccountType.ASSET: (Code("1"), "Assets")})
    
    parent_code = Code("1")
    account_code = Code("1.1")
    
    coa.add(parent_code, account_code, "Current Assets")
    
    try:
        coa.add(parent_code, account_code, "Different Name")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "do not match existing chart of accounts member" in str(e)


