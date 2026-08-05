####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    converts accounts into a tree structure of COA.Node instances.
    """
    coa = COA()
    
    # Setup a hierarchy:
    # Assets (1) -> Liquidity (1000) -> Bank Account (1001)
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    
    # Test 1: Verify the structure of a leaf node (no children)
    # We find an account that has no sub-accounts in our setup (e.g., Liabilities '2')
    liabilities = coa.find(Code("2"))
    node_leaf = coa.nodify(liabilities)
    
    assert isinstance(node_leaf, COA.Node)
    assert node_leaf.account.code == Code("2")
    assert node_leaf.children == []

    # Test 2: Verify the structure of a mid-level node (has children)
    # Liquidity (1000) should have Bank Account (1001) as a child
    node_mid = coa.nodify(liquidity)
    
    assert isinstance(node_mid, COA.Node)
    assert node_mid.account.code == Code("1000")
    assert len(node_mid.children) == 1
    assert node_mid.children[0].account.code == Code("1001")
    assert node_mid.children[0].children == []

    # Test 3: Verify the structure of a root node with deep nesting
    # Assets (1) should contain Liquidity (1000), which contains Bank Account (1001)
    node_root = coa.nodify(coa.find(Code("1")))
    
    assert isinstance(node_root, COA.Node)
    assert node_root.account.code == Code("1")
    # Check depth 1
    assert len(node_root.children) == 1
    child_node = node_root.children[0]
    assert child_node.account.code == Code("1000")
    # Check depth 2
    assert len(child_node.children) == 1
    grandchild_node = child_node.children[0]
    assert grandchild_node.account.code == Code("1001")
    assert grandchild_node.children == []

    # Test 4: Verify node content integrity
    # Ensure the Node holds the actual account object properties correctly via its reference
    assert node_root.account.name == "Assets"
    assert child_node.account.name == "Liquidity"
    assert grandchild_node.account.name == "Bank Account"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable 
    object that satisfies the interface.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader: ReadChartOfAccounts = MockReader()

    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_COA___iter__():
    """
    Tests the __iter__ method of the COA class to ensure it correctly 
    yields the account code and account object for all accounts in the COA.
    """
    # Initialize a default COA (contains 5 core accounts)
    coa = COA()
    
    # Add some sub-accounts to test iteration with more than just roots
    liquidity_code = Code("1000")
    liquidity_name = "Liquidity"
    coa.add(Code("1"), liquidity_code, liquidity_name)
    
    bank_code = Code("1001")
    bank_name = "Bank Account"
    coa.add(liquidity_code, bank_code, bank_name)

    # Prepare expected data based on the logic of COA initialization and additions
    # Root accounts: 1:Assets, 2:Liabilities, 3:Equities, 4:Revenues, 5:Expenses
    expected_items = [
        (Code("1"), coa.find(Code("1"))),
        (Code("2"), coa.find(Code("2"))),
        (Code("3"), coa.find(Code("3"))),
        (Code("4"), coa.find(Code("4"))),
        (Code("5"), coa.find(Code("5"))),
        (liquidity_code, coa.find(liquidity_code)),
        (bank_code, coa.find(bank_code)),
    ]

    # Verify the number of items yielded matches our expected count
    iterated_items = list(coa)
    assert len(iterated_items) == len(expected_items)

    # Verify each (code, account) pair in the iteration matches expectation
    for i, (expected_code, expected_account) in enumerate(expected_items):
        actual_code, actual_account = iterated_items[i]
        assert actual_code == expected_code
        assert actual_account == expected_account
        # Also verify the content of the account retrieved via iteration
        assert actual_account.code == expected_code

    # Verify that iterating twice yields the same result (COA is not a one-time generator)
    second_iteration = list(coa)
    assert second_iteration == iterated_items
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_COA___iter__():
    """
    Tests the __iter__ method of the COA class to ensure it correctly yields 
    (code, account) pairs for both default and custom initialized accounts.
    """
    # Test Case 1: Default Initialization
    coa = COA()
    default_accounts = list(coa)
    
    assert len(default_accounts) == 5
    # Verify core accounts exist with correct types/codes as per docstring
    expected_defaults = [
        (Code("1"), AccountType.ASSETS),
        (Code("2"), AccountType.LIABILITIES),
        (Code("3"), AccountType.EQUITIES),
        (Code("4"), AccountType.REVENUES),
        (Code("5"), AccountType.EXPENSES),
    ]
    
    for i, (code, account) in enumerate(default_accounts):
        expected_code, expected_type = expected_defaults[i]
        assert code == expected_code
        assert account.type == expected_type

    # Test Case 2: Initialization with custom rootspec
    custom_spec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    coa_custom = COA(rootspec=custom_spec)
    custom_accounts = list(coa_custom)
    
    assert len(custom_accounts) == 5
    for code, account in custom_accounts:
        assert code in [Code("10"), Code("20"), Code("30"), Code("40"), Code("50")]
        assert account.name.startswith("Custom")

    # Test Case 3: Iteration after adding sub-accounts
    coa_extended = COA()
    parent_code = Code("1")
    child_code = Code("1000")
    child_name = "Liquidity"
    
    coa_extended.add(parent_code, child_code, child_name)
    
    # The iterator should now include the new sub-account
    all_items = list(coa_extended)
    assert len(all_items) == 6
    
    # Check if the newly added account is present in the iteration
    found_child = False
    for code, account in all_items:
        if code == child_code:
            assert account.name == child_name
            assert isinstance(account, SubAccount)
            assert account.parent.code == parent_code
            found_child = True
    
    assert found_child is True
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account to a root account
    parent_code = Code("1")  # Assets (default)
    new_code = Code("1000")
    new_name = "Liquidity"
    
    sub_acc = coa.add(parent_code, new_code, new_name)
    
    assert sub_acc.code == new_code
    assert sub_acc.name == new_name
    assert sub_acc.parent.code == parent_code
    assert sub_acc.type == AccountType.ASSETS
    assert coa.find(new_code) == sub_acc
    assert sub_acc in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of the newly created account)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_acc = coa.add(new_code, child_code, child_name)
    
    assert child_acc.parent.code == new_code
    assert child_acc in coa.subaccounts(sub_acc)

    # Test adding an existing account with identical parameters (idempotency)
    existing_acc = coa.add(new_code, child_code, child_name)
    assert existing_acc is child_acc

    # Test error: Account cannot be parent of itself
    with pytest.raises(ValueError, match="An account can not be the same as its parent"):
        # Note: The implementation uses `if parent == code`, so we trigger that
        coa.add(new_code, new_code, "Self")

    # Test error: Parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined"):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Mismatching existing account data (Name mismatch)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(parent_code, child_code, "Different Name")

    # Test error: Mismatching existing account data (Parent mismatch)
    # Create a different account with same code but different parent structure
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("2"), child_code, child_name)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a functional 
    implementation (a callable) that returns a COA instance.
    """
    # Arrange: Create a mock or a dummy function that conforms to the protocol
    mock_coa = COA()
    
    def mock_reader() -> COA:
        return mock_coa

    # Assert that the implementation is callable and returns a COA object
    assert callable(mock_reader)
    result = mock_reader()
    
    assert isinstance(result, COA)
    assert result is mock_coa
    
    # Verify functionality with an actual populated COA via the reader
    def real_reader() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("100"), "Cash")
        return coa

    result_real = real_reader()
    assert isinstance(result_real, COA)
    assert result_real.find(Code("100")).name == "Cash"

    # Test with a mock object that specifically mimics the protocol signature
    mock_protocol_impl = MagicMock(spec=ReadChartOfAccounts)
    mock_protocol_impl.return_value = mock_coa
    
    assert mock_protocol_impl() == mock_coa
    mock_protocol_impl.assert_called_once()
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that adheres to its signature.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the Protocol for testing
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()
    
    # Execute the call
    result = reader()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable 
    object (like a function or a mock) that conforms to the signature.
    """
    # Arrange: Create a dummy COA instance and a mock implementation of the protocol
    mock_coa = MagicMock(spec=COA)
    
    # A function is a valid implementation of the ReadChartOfAccounts protocol
    def mock_reader() -> COA:
        return mock_coa

    # Act: Call the implementation
    result = mock_reader()

    # Assert: Verify the result is the expected COA instance and type matches
    assert result == mock_coa
    assert isinstance(result, COA)

    # Additional test using a Mock object to ensure compatibility with Protocol requirements
    mock_callable = MagicMock(spec=ReadChartOfAccounts)
    mock_callable.return_value = mock_coa
    
    result_from_mock = mock_callable()
    assert result_from_mock == mock_coa
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test an object that 
    implements the protocol signature.
    """
    # Create a mock function/object that follows the ReadChartOfAccounts protocol
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned by the callable
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_COA_nodify():
    # Initialize a basic COA
    coa = COA()
    
    # 1. Test nodify on a RootAccount (Top level)
    # By default, '1' is Assets
    assets_account = coa.find(Code("1"))
    node = coa.nodify(assets_account)
    
    assert isinstance(node, COA.Node)
    assert node.account.code == Code("1")
    assert node.account.name == "Assets"
    # Root accounts have no sub-accounts by default in this setup
    assert node.children == []

    # 2. Test nodify on a SubAccount (Nested level)
    # Add Liquidity under Assets
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    # Add Bank Account under Liquidity
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    
    # Get the node for the parent (Liquidity)
    liquidity_node = coa.nodify(liquidity)
    
    assert liquidity_node.account.code == Code("1000")
    assert len(liquidity_node.children) == 1
    
    # Check if the child node is correctly constructed (Bank Account)
    child_node = liquidity_node.children[0]
    assert child_node.account.code == Code("1001")
    assert child_node.account.name == "Bank Account"
    assert child_node.children == []

    # 3. Test full tree structure via nodify on the top level
    # The structure property uses nodify internally for all toplevel accounts
    structure = list(coa.structure)
    
    # We expect 5 top-level nodes (Assets, Liabilities, Equities, Revenues, Expenses)
    # plus any additional top-level accounts if we had added them directly to root
    assert len(structure) == 5
    
    # Find the Assets node in the structure and verify depth
    assets_node = next(n for n in structure if n.account.code == Code("1"))
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account.code == Code("1000")
    assert assets_node.children[0].children[0].account.code == Code("1001")

def test_COA_nodify_empty_subaccounts():
    coa = COA()
    # Verify that an account with no children results in an empty list of children in Node
    acc = coa.find(Code("2")) # Liabilities
    node = coa.nodify(acc)
    assert node.children == []
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test 1: Successful addition of a sub-account
    parent_code = Code("1")  # Assets is default root
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test 2: Successful addition of a nested sub-account (grandchild)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.code == grandchild_code
    assert grandchild_account.parent.code == new_code
    assert grandchild_account in coa.subaccounts(new_account)

    # Test 3: Adding an existing account with identical info (Idempotency)
    existing_account = coa.add(new_code, grandchild_code, grandchild_name)
    assert existing_account == grandchild_account

    # Test 4: Exception - Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test 5: Exception - Parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test 6: Exception - Inconsistent data (same code, different name/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(new_code, grandchild_code, "Different Name")

    # Test 7: Exception - Inconsistent data (same code, different parent)
    # Using a different root as parent for an existing child code
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), grandchild_code, "Bank Account")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable 
    object (like a function or a class with __call__) that follows the protocol.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the Protocol
    class MockReader:
        def __init__(self, coa_to_return):
            self.coa_to_return = coa_to_return
            
        def __call__(self) -> COA:
            return self.coa_to_return

    # Instantiate the reader with our mock COA
    reader = MockReader(mock_coa)

    # Execute the __call__ method
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a mock or 
    a concrete implementation that adheres to the protocol.
    """
    # Create a mock that follows the ReadChartOfAccounts protocol signature
    # The protocol defines __call__ returning a COA instance
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define what the call should return
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == expected_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___concrete_implementation():
    """
    Tests the __call__ method using a concrete implementation of the protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    result = reader()

    assert isinstance(result, COA)
    # Verify default COA contents as per docstring
    assert result.find(Code("1")).name == 'Assets'
    assert result.find(Code("5")).name == 'Expenses'
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a standard sub-account
    parent_code = Code("1")  # Assets (default root)
    new_code = Code("1000")
    new_name = "Liquidity"
    
    sub_acc = coa.add(parent_code, new_code, new_name)
    
    assert sub_acc.code == new_code
    assert sub_acc.name == new_name
    assert sub_acc.parent.code == parent_code
    assert sub_acc.type == AccountType.ASSETS
    assert coa.find(new_code) == sub_acc
    assert sub_acc in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested child (multi-level)
    child_code = Code("1001")
    child_name = "Bank Account"
    grandchild_acc = coa.add(new_code, child_code, child_name)
    
    assert grandchild_acc.parent.code == new_code
    assert grandchild_acc.code == child_code
    assert grandchild_acc in coa.subaccounts(sub_acc)

    # Test idempotency (adding the exact same account should return existing and not raise error)
    existing_acc = coa.add(new_code, child_code, child_name)
    assert existing_acc == grandchild_acc

    # Test Error: Parent is itself
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test Error: Parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test Error: Inconsistent data (same code, different name/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(new_code, child_code, "Wrong Name")
        
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Pointing to a different parent but using existing code
        coa.add(Code("2"), child_code, child_name)

    # Test Error: Inconsistent data (same code, same name, but mismatching parent via logic)
    # Note: The implementation checks `account.parent == parentinstance`. 
    # If we try to add an existing code with a different parent, it triggers the error.
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("3"), child_code, child_name)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    
    # Test successful addition of a sub-account
    sub_account = coa.add(parent_code, new_code, new_name)
    assert sub_account.code == new_code
    assert sub_account.name == new_name
    assert sub_account.parent.code == parent_code
    assert coa.find(new_code) == sub_account
    assert sub_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a child to the newly created sub-account (multi-level)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    assert child_account.parent.code == new_code
    assert child_account.code == child_code
    assert child_account in coa.subaccounts(sub_account)

    # Test adding an existing account with identical details (should return existing)
    existing = coa.add(new_code, child_code, child_name)
    assert existing is child_account

    # Test adding an existing account with different name/parent (should raise ValueError)
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(new_code, child_code, "Different Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        # Attempting to re-add child_code but pointing to a different parent
        coa.add(Code("2"), child_code, child_name)

    # Test adding an account where parent is the same as the new account code
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test adding an account with a non-existent parent
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a sub-account to a root account (Assets)
    liquidity_code = Code("1000")
    liquidity_name = "Liquidity"
    parent_code = Code("1")
    
    liquidity_acc = coa.add(parent_code, liquidity_code, liquidity_name)
    
    assert liquidity_acc.code == liquidity_code
    assert liquidity_acc.name == liquidity_name
    assert liquidity_acc.parent.code == parent_code
    assert liquidity_acc.type == AccountType.ASSETS
    assert coa.find(liquidity_code) == liquidity_acc

    # Test adding a sub-account to a previously added sub-account (Nested)
    bank_acc_code = Code("1001")
    bank_acc_name = "Bank Account"
    bank_acc = coa.add(liquidity_code, bank_acc_code, bank_acc_name)
    
    assert bank_acc.code == bank_acc_code
    assert bank_acc.parent.code == liquidity_code
    assert bank_acc.parent.parent.code == parent_code
    assert bank_acc in coa.subaccounts(liquidity_acc)

    # Test adding an existing account with same parameters (Idempotency)
    duplicate_acc = coa.add(parent_code, liquidity_code, liquidity_name)
    assert duplicate_acc == liquidity_acc

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(liquidity_code, liquidity_code, "Self")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Mismatching attributes for existing code (Name mismatch)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, liquidity_code, "Different Name")

    # Test error: Mismatching attributes for existing code (Parent mismatch)
    # First create a different child for Assets
    cash_code = Code("1002")
    coa.add(parent_code, cash_code, "Cash")
    # Try to re-add 1002 but point it to Liabilities instead of Assets
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), cash_code, "Cash")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a functional 
    implementation (a callable) that returns a COA instance.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the Protocol
    def mock_reader() -> COA:
        return mock_coa

    # Verify the callable matches the expected behavior (returning a COA)
    result = mock_reader()
    
    assert result == mock_coa
    assert isinstance(result, COA)

    # Test with an actual COA instance to ensure compatibility 
    # with the Protocol's intended usage
    real_coa = COA()
    def real_reader() -> COA:
        return real_coa
    
    assert real_reader() == real_coa
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a mock or a concrete implementation.
    """
    # Create a dummy COA to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol for testing
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()
    
    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)

def test_ReadChartOfAccounts___call___with_mock():
    """
    Tests the __call__ method using a MagicMock that adheres to the protocol.
    """
    # Create a mock object that behaves like a ReadChartOfAccounts callable
    reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define what the call should return
    expected_coa = COA()
    reader.return_value = expected_coa

    # Execute the call
    result = reader()

    # Assertions
    assert result == expected_coa
    reader.assert_called_once()
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account to a root account
    parent_code = Code("1")  # Assets (default)
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of the newly created account)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # Test idempotency: adding the same account with identical details should return existing instance
    existing_account = coa.add(parent_code, new_code, new_name)
    assert existing_account is new_account

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Adding an existing code with different details (name or parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Attempting to re-add 1001 but claiming its parent is '2' (Liabilities) instead of '1000'
        coa.add(Code("2"), child_code, child_name)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account to a root account
    parent_code = Code("1")  # Assets (default)
    new_code = Code("1000")
    new_name = "Liquidity"
    sub_account = coa.add(parent_code, new_code, new_name)
    
    assert sub_account.code == new_code
    assert sub_account.name == new_name
    assert sub_account.parent.code == parent_code
    assert sub_account.type == AccountType.ASSETS
    assert coa.find(new_code) == sub_account
    assert sub_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of child)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.parent.code == new_code
    assert grandchild_account.code == grandchild_code
    assert grandchild_account in coa.subaccounts(sub_account)

    # Test idempotency (adding the exact same account again should return existing)
    same_account = coa.add(new_code, Code("1000"), "Liquidity")
    assert same_account == sub_account

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Conflict - same code but different name/parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Change the parent for an existing code
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable 
    that returns a COA instance.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol for testing
    class MockReadCOA:
        def __call__(self) -> COA:
            return mock_coa

    # Instantiate the implementation
    reader = MockReadCOA()

    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it via a concrete 
    implementation (a callable) to verify it returns a COA instance.
    """
    # Create a mock or a simple function that adheres to the protocol
    def mock_reader() -> COA:
        return COA()

    # Verification of type/interface adherence
    # In Python, Protocols are structural; we verify the behavior 
    # of an object implementing the __call__ signature.
    reader: ReadChartOfAccounts = mock_reader
    
    result = reader()

    # Assertions
    assert isinstance(result, COA), "The callable must return an instance of COA"
    assert len(list(result)) == 5, "Default COA should have 5 root accounts"
    assert result.find(Code("1")).name == "Assets"

def test_ReadChartOfAccounts___call___with_custom_logic():
    """
    Tests a more complex implementation of the ReadChartOfAccounts protocol.
    """
    class CustomReader:
        def __call__(self) -> COA:
            # Implementation that adds custom accounts
            coa = COA()
            coa.add(Code("1"), Code("100"), "Cash")
            return coa

    reader: ReadChartOfAccounts = CustomReader()
    result = reader()

    assert isinstance(result, COA)
    assert result.find(Code("100")).name == "Cash"
    assert result.find(Code("100")).parent.code == Code("1")
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a top-level subaccount (child of RootAccount)
    parent_code = Code("1")  # Assets
    new_code = Code("1000")
    new_name = "Liquidity"
    liquidity_acc = coa.add(parent_code, new_code, new_name)
    
    assert liquidity_acc.code == new_code
    assert liquidity_acc.name == new_name
    assert liquidity_acc.parent.code == parent_code
    assert liquidity_acc.type == AccountType.ASSETS
    assert coa.find(new_code) == liquidity_acc
    
    # Test successful addition of a nested subaccount (child of SubAccount)
    sub_parent_code = Code("1000")
    sub_new_code = Code("1001")
    sub_new_name = "Bank Account"
    bank_acc = coa.add(sub_parent_code, sub_new_code, sub_new_name)
    
    assert bank_acc.code == sub_new_code
    assert bank_acc.parent.code == sub_parent_code
    assert bank_acc.parent.name == "Liquidity"
    assert bank_acc in coa.subaccounts(liquidity_acc)

    # Test adding an existing account with identical info (should return existing)
    existing_acc = coa.add(parent_code, new_code, new_name)
    assert existing_acc is liquidity_acc

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Inconsistent information (different name for same code/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    # Test error: Inconsistent information (different parent for same code)
    # First create a different branch
    coa.add(Code("2"), Code("2000"), "Liabilities Sub")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2000"), new_code, "Liquidity")
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    transforms accounts into a tree-like structure of Node instances.
    """
    coa = COA()
    
    # 1. Test Root Account Nodification (No children)
    assets_account = coa.find(Code("1"))
    root_node = coa.nodify(assets_account)
    
    assert isinstance(root_node, COA.Node)
    assert root_node.account.code == Code("1")
    assert root_node.account.name == "Assets"
    assert root_node.children == []

    # 2. Test Sub-Account Nodification (With children)
    # Add a level 2 account: Assets -> Liquidity
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    # Add a level 3 account: Assets -> Liquidity -> Bank Account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    
    # Nodify the top-level Assets account again to see the full tree
    complex_node = coa.nodify(assets_account)
    
    # Verify structure: Root -> Node(Liquidity) -> Node(Bank Account)
    assert len(complex_node.children) == 1
    
    liquidity_node = complex_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    
    bank_node = liquidity_node.children[0]
    assert bank_node.account.code == Code("1001")
    assert bank_node.account.name == "Bank Account"
    assert bank_node.children == []

    # 3. Test Nodify on a leaf SubAccount directly
    leaf_node = coa.nodify(bank_account)
    assert isinstance(leaf_node, COA.Node)
    assert leaf_node.account.code == Code("1001")
    assert leaf_node.children == []

    # 4. Test with custom rootspec to ensure nodify handles non-standard codes
    custom_coa = COA(rootspec={
        AccountType.ASSETS: (Code("99"), "Custom Assets")
    })
    custom_root = custom_coa.find(Code("99"))
    custom_node = custom_coa.nodify(custom_root)
    assert custom_node.account.code == Code("99")
    assert custom_node.account.name == "Custom Assets"
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a sub-account to an existing root account
    parent_code = Code("1")  # Assets is default code '1'
    new_code = Code("1000")
    new_name = "Liquidity"
    
    sub_acc = coa.add(parent_code, new_code, new_name)
    
    assert sub_acc.code == new_code
    assert sub_acc.name == new_name
    assert sub_acc.parent.code == parent_code
    assert sub_acc.type == AccountType.ASSETS
    assert coa.find(new_code) == sub_acc
    assert sub_acc in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of child)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_acc = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_acc.parent.code == new_code
    assert grandchild_acc.code == grandchild_code
    assert grandchild_acc in coa.subaccounts(sub_acc)

    # Test adding an existing account with same parameters (should return existing)
    existing = coa.add(new_code, grandchild_code, grandchild_name)
    assert existing is grandchild_acc

    # Test error: Parent and Code are the same
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(Code("1000"), Code("1000"), "Self Parent")

    # Test error: Parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Inconsistent data (same code, different name/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Change the parent of 1000 to something else via a mismatching add call
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a mock or 
    a concrete implementation that follows the signature.
    """
    # Create a mock that implements the ReadChartOfAccounts protocol
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned by the call
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_concrete_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    result = reader()

    assert isinstance(result, COA)
    # Verify the default root accounts exist as per COA implementation
    assert result.find(Code("1")).name == "Assets"
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that adheres to its structure.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a function that implements the ReadChartOfAccounts protocol
    def mock_reader() -> COA:
        return mock_coa

    # Verify the reader follows the protocol (is callable and returns COA)
    assert callable(mock_reader)
    result = mock_reader()
    
    assert isinstance(result, COA)
    assert result == mock_coa

    # Test with a class-based implementation of the protocol
    class ConcreteReader:
        def __init__(self, coa_to_return: COA):
            self.coa = coa_to_return
        
        def __call__(self) -> COA:
            return self.coa

    concrete_reader = ConcreteReader(mock_coa)
    assert callable(concrete_reader)
    assert concrete_reader() == mock_coa
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a mock or a 
    concrete implementation that follows the protocol.
    """
    # Create a mock object that implements the ReadChartOfAccounts protocol
    # The protocol requires a __call__ method that returns a COA instance.
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Setup the expected return value: a new Chart of Accounts instance
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the __call__ method
    actual_coa = mock_reader()

    # Assertions
    assert actual_coa == expected_coa
    assert isinstance(actual_coa, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___concrete_implementation():
    """
    Tests the __call__ method using a real concrete class following the protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    coa = reader()

    assert isinstance(coa, COA)
    assert len(list(coa)) == 5  # Default root accounts
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a functional 
    implementation or a Mock that adheres to the protocol.
    """
    # Create a mock COA instance to be returned
    mock_coa = MagicMock(spec=COA)
    
    # Define a function that matches the ReadChartOfAccounts signature:
    # def __call__(self) -> COA:
    def mock_reader() -> COA:
        return mock_coa

    # In Python, a protocol is a structural type. 
    # We verify that our implementation satisfies the interface and returns correctly.
    result = mock_reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a compatible callable.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = Mock(spec=COA)
    
    # Define a concrete implementation of the protocol
    def mock_reader() -> COA:
        return mock_coa

    # Verify that the object is an instance of the Protocol (runtime checkable)
    assert isinstance(mock_reader, ReadChartOfAccounts)

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test an object that 
    implements the expected signature.
    """
    # Create a mock that implements the ReadChartOfAccounts protocol
    # The protocol requires a __call__ method returning a COA instance
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Setup a dummy COA to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == expected_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    # Verify the default accounts are present in a new COA
    assert result.find(Code("1")).name == 'Assets'
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation 
    (or a Mock) to verify it behaves as expected when called.
    """
    # Arrange: Create a mock that implements the ReadChartOfAccounts protocol
    # A Mock object in Python can be configured to return a specific value when called.
    mock_reader = Mock(spec=ReadChartOfAccounts)
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Act: Call the mock as if it were a ReadChartOfAccounts instance
    result = mock_reader()

    # Assert: Verify the return value is the expected COA and the call was made
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_concrete_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    # Arrange: Define a simple concrete class implementing the Protocol
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    
    # Act
    result = reader()

    # Assert
    assert isinstance(result, COA)
    # Verify it contains the default 5 core accounts as per module docstring
    expected_codes = [Code("1"), Code("2"), Code("imitation"), Code("4"), Code("5")] # Note: Docstring shows '1'-'5'
    # Checking specific known code from COA initialization
    assert result.find(Code("1")).name == "Assets"
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it via a functional implementation.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define an implementation of the ReadChartOfAccounts protocol
    def coa_reader_impl() -> COA:
        return mock_coa

    # Verify the implementation matches the expected return type/value
    result = coa_reader_impl()
    
    assert result == mock_coa
    assert isinstance(result, COA)

    # Test with a real COA instance to ensure integration
    real_coa = COA()
    def real_coa_reader_impl() -> COA:
        return real_coa

    result_real = real_coa_reader_impl()
    assert result_real.find(Code("1")).name == "Assets"
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    compatible callable object (a Mock or a function).
    """
    # Arrange: Create a mock that follows the ReadChartOfAccounts protocol
    # The protocol specifies a __call__ method returning a COA instance.
    expected_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = expected_coa

    # Act: Call the object as a function
    result = mock_reader()

    # Assert: Verify the result is the expected COA and the call was made
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_concrete_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    # Arrange: A real function that matches the protocol signature
    def concrete_reader() -> COA:
        return COA(rootspec={
            AccountType.ASSETS: (Code("10"), "Custom Assets")
        })

    # Act
    result = concrete_reader()

    # Assert
    assert isinstance(result, COA)
    assert result.find(Code("10")).name == "Custom Assets"
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable object.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol for testing
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()
    
    # Execute the call
    result = reader()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a concrete 
    implementation (a callable) that follows the protocol.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the ReadChartOfAccounts protocol
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()

    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #37
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test an object 
    that satisfies this protocol.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol for testing
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()

    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #38
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it via a concrete 
    implementation or a mock that adheres to the protocol.
    """
    # Create a mock object that follows the ReadChartOfAccounts protocol
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define what the __call__ should return (a COA instance)
    expected_coa = COA()
    mock_reader.return_value = expected_coa
    
    # Execute the call
    result = mock_reader()
    
    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_concrete_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA(rootspec={
                AccountType.ASSETS: (Code("10"), "Custom Assets")
            })

    reader = SimpleCOAReader()
    coa = reader()
    
    assert isinstance(coa, COA)
    assert coa.find(Code("10")).name == "Custom Assets"
```


# LLM-generated content at query #39
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable 
    that returns a COA instance.
    """
    # Create a mock COA instance
    mock_coa = MagicMock(spec=COA)
    
    # Define a dummy implementation of the protocol
    def coa_reader_implementation() -> COA:
        return mock_coa

    # Verify the implementation adheres to the expected return type
    result = coa_reader_implementation()
    assert result == mock_coa
    assert isinstance(result, COA)

    # Testing with a real COA instance to ensure compatibility
    real_coa = COA()
    def real_coa_reader() -> COA:
        return real_coa

    result_real = real_coa_reader()
    assert result_real.find(Code("1")).name == "Assets"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that adheres to the protocol signature.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the Protocol
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()
    
    # Execute the call
    result = reader()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_COA___iter__():
    """
    Tests the __iter__ method of the COA class to ensure it correctly yields 
    the account code and account object for all accounts in the chart of accounts.
    """
    # Initialize a default COA (contains 5 core accounts)
    coa = COA()
    
    # Add additional accounts to test iteration over both root and sub-accounts
    liquidity_code = Code("1000")
    liquidity_name = "Liquidity"
    bank_account_code = Code("1001")
    bank_account_name = "Bank Account"
    
    parent_acc = coa.add(Code("1"), liquidity_code, liquidity_name)
    coa.add(liquidity_code, bank_account_code, bank_account_name)

    # Prepare expected data: list of (Code, AccountName)
    # The order should match the insertion order in _accounts (OrderedDict)
    expected_accounts = [
        (Code("1"), "Assets"),
        (Code("2"), "Liabilities"),
        (Code("3"), "Equities"),
        (Code("4"), "Revenues"),
        (Code("5"), "Expenses"),
        (liquidity_code, liquidity_name),
        (bank_account_code, bank_account_name)
    ]

    # Actual iteration results
    actual_accounts = []
    for code, account in coa:
        actual_accounts.append((code, account.name))

    # Assertions
    assert len(actual_accounts) == len(expected_accounts), "The number of iterated accounts does not match."
    assert actual_accounts == expected_accounts, "The iterated items (code and name) do not match the expected values or order."

    # Verify that we can also iterate through the structure directly via __iter__ 
    # to check if the objects returned are valid Account instances
    for code, account in coa:
        assert isinstance(code, Code)
        assert hasattr(account, "name")
        assert hasattr(account, "type")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a sub-account
    parent_code = Code("1")  # Assets (default root)
    new_code = Code("1000")
    new_name = "Liquidity"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of the child)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.parent.code == new_code
    assert grandchild_account in coa.subaccounts(new_account)

    # Test adding an existing account with identical info (should return existing)
    existing = coa.add(new_code, grandchild_code, grandchild_name)
    assert existing is grandchild_account

    # Test error: Account cannot be parent of itself
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Inconsistent data (name mismatch)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(new_code, grandchild_code, "Wrong Name")

    # Test error: Inconsistent data (parent mismatch)
    # Note: We must use a new code to avoid the 'existing' check logic for different parents
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), Code("1001"), "Bank Account")
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.
    """
    # Define a concrete implementation of the protocol
    class MockCOAReader:
        def __init__(self, coa_to_return: COA):
            self.coa_to_return = coa_to_return

        def __call__(self) -> COA:
            return self.coa_to_return

    # Create a sample COA to return
    expected_coa = COA()
    
    # Initialize the reader with our expected COA
    reader = MockCOAReader(expected_coa)

    # Execute the __call__ method
    result = reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    assert result.find(Code("1")).name == 'Assets'

    # Test with a MagicMock to ensure protocol compliance for any callable returning COA
    mock_reader: ReadChartOfAccounts = MagicMock(return_value=expected_coa)
    
    result_from_mock = mock_reader()
    
    assert result_from_mock == expected_coa
    mock_reader.assert_called_once()
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ protocol implementation of ReadChartOfAccounts.
    Since ReadChartOfAccounts is a Protocol, we test it by creating a 
    concrete implementation (a callable) and verifying it behaves as expected.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol
    def mock_reader() -> COA:
        return mock_coa

    # Verify that the callable matches the expected return type/behavior
    result = mock_reader()
    
    assert result == mock_coa
    assert isinstance(result, COA)

    # Test with a class-based implementation as well
    class ConcreteReader:
        def __call__(self) -> COA:
            return mock_coa

    concrete_reader = ConcreteReader()
    assert concrete_reader() == mock_coa
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable 
    that returns a COA instance.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    # Initialize the reader
    reader: ReadChartOfAccounts = MockReader()

    # Execute the __call__ method
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of an object implementing the 
    ReadChartOfAccounts protocol.
    """
    # Create a mock object that follows the ReadChartOfAccounts protocol
    # The protocol defines __call__ returning a COA instance
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned by the mock
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable object 
    that satisfies the interface.
    """
    # Create a mock that mimics a function/callable satisfying ReadChartOfAccounts
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Create a dummy COA instance to be returned by the reader
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleReader()
    result = reader()
    
    assert isinstance(result, COA)
    # Verify default roots in a fresh COA
    assert result.find(Code("1")).name == "Assets"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.
    """
    # Define a mock COA to be returned by our reader
    mock_coa = MagicMock(spec=COA)
    
    # Create a concrete implementation of the protocol
    class MockCOAReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockCOAReader()
    
    # Execute the call
    result = reader()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    transforms a flat account structure into a tree-like Node structure.
    """
    coa = COA()
    
    # Setup: Create a hierarchy
    # Root: Assets (1) -> Sub: Liquidity (1000) -> SubSub: Bank Account (1001)
    # Root: Liabilities (2)
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # 1. Test nodify on a leaf node (Bank Account)
    leaf_node = coa.nodify(bank_account)
    assert isinstance(leaf_node, COA.Node)
    assert leaf_node.account.code == Code("1001")
    assert leaf_node.account.name == "Bank Account"
    assert len(leaf_node.children) == 0

    # 2. Test nodify on a middle node (Liquidity)
    parent_node = coa.nodify(liquidity)
    assert isinstance(parent_node, COA.Node)
    assert parent_node.account.code == Code("1000")
    assert len(parent_node.children) == 1
    assert parent_node.children[0].account.code == Code("1001")

    # 3. Test nodify on a root node (Assets)
    root_node = coa.nodify(coa.find(Code("1")))
    assert isinstance(root_node, COA.Node)
    assert root_node.account.code == Code("1")
    assert len(root_node.children) == 1
    # Check deep nesting: Root -> Liquidity -> Bank Account
    assert root_node.children[0].account.code == Code("1000")
    assert root_node.children[0].children[0].account.code == Code("1001")

    # 4. Test nodify on a simple root node with no children (Liabilities)
    liabilities_node = coa.nodify(coa.find(Code("2")))
    assert liabilities_node.account.code == Code("2")
    assert len(liabilities_node.children) == 0

    # 5. Verify the structure integrity through the property 'structure'
    structure = list(coa.structure)
    # Should have 5 top-level nodes (Assets, Liabilities, Equities, Revenues, Expenses)
    assert len(structure) == 5
    
    # Find Assets node in structure to verify deep child exists
    assets_node = next(n for n in structure if n.account.code == Code("1"))
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account.code == Code("1000")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it via a concrete implementation.
    """
    # Setup: Create a mock COA to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Implementation of the protocol
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()

    # Execution
    result = reader()

    # Verification
    assert result == mock_coa
    assert isinstance(result, COA)

def test_ReadChartOfAccounts___call___with_logic():
    """
    Tests a more complex implementation of the ReadChartOfAccounts protocol
    to ensure it correctly returns a populated COA.
    """
    class RealReader:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Cash")
            return coa

    reader = RealReader()
    result = reader()

    # Verification
    assert isinstance(result, COA)
    assert result.find(Code("1000")).name == "Cash"
    assert result.find(Code("1")).name == "Assets"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that adheres to its signature.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol for testing
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()
    
    # Execute the call
    result = reader()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of an object implementing the 
    ReadChartOfAccounts protocol.
    """
    # Create a mock that follows the ReadChartOfAccounts protocol
    # Since it's a Protocol with __call__, we can use a MagicMock
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned by the mock
    expected_coa = COA()
    mock_reader.return_value = expected_coa
    
    # Execute the __call__ method
    result = mock_reader()
    
    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly
    converts account structures into a tree-like Node structure.
    """
    coa = COA()
    
    # Setup a hierarchy:
    # 1 (Assets) -> 1000 (Liquidity) -> 1001 (Bank Account)
    # 2 (Liabilities)
    
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Test top-level nodes conversion (the roots)
    nodes = list(coa.structure)
    
    # We expect 5 root nodes from the default COA initialization:
    # Assets, Liabilities, Equities, Revenues, Expenses
    assert len(nodes) == 5
    
    # Find the node for Assets (Code '1')
    assets_node = next(n for n in nodes if n.account.code == Code("1"))
    
    # Verify Assets node properties
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1
    
    # Check the child of Assets (Liquidity)
    liquidity_node = assets_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    
    # Check the grandchild (Bank Account)
    bank_account_node = liquidity_node.children[0]
    assert bank_account_node.account.code == Code("1001")
    assert bank_account_node.account.name == "Bank Account"
    assert len(bank_account_node.children) == 0
    
    # Verify a non-modified root node (Liabilities)
    liabilities_node = next(n for n in nodes if n.account.code == Code("2"))
    assert liabilities_node.account.name == "Liabilities"
    assert len(liabilities_node.children) == 0

    # Verify type of returned object
    assert isinstance(assets_node, COA.Node)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it via a mock or 
    a concrete implementation that follows the signature.
    """
    # Create a dummy COA instance to be returned by the callable
    expected_coa = COA()
    
    # Create a mock that adheres to the ReadChartOfAccounts protocol
    # The protocol defines __call__ with no arguments returning COA
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == expected_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_concrete_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleReader:
        def __call__(self) -> COA:
            return COA(rootspec={
                AccountType.ASSETS: (Code("10"), "Custom Assets")
            })

    reader = SimpleReader()
    coa = reader()

    assert isinstance(coa, COA)
    assert coa.find(Code("10")).name == "Custom Assets"
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # 1. Test successful addition of a root-level sub-account (child of an existing root)
    parent_code = Code("1")  # Assets
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account

    # 2. Test successful addition of a nested sub-account (child of the newly created account)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.parent.code == new_code
    assert grandchild_account.code == grandchild_code
    assert coa.find(grandchild_code).parent.code == new_code

    # 3. Test error: Account cannot be parent of itself
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # 4. Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Non-existent Parent")

    # 5. Test error: Inconsistent data (same code, different name/parent)
    # We try to add '1000' again but with a different name
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, Code("1000"), "Different Name")

    # 6. Test idempotency: Adding the exact same account again should return the existing instance without error
    existing_account = coa.add(parent_code, Code("1000"), "Liquidity")
    assert existing_account == new_account
    
    # 7. Verify sub-account retrieval via COA helper
    sub_list = coa.subaccounts(coa.find(parent_code))
    assert new_account in sub_list
    assert grandchild_account in coa.subaccounts(new_account)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    converts account flat structures into a tree-like Node structure.
    """
    coa = COA()
    
    # Setup hierarchy:
    # Assets (1) -> Liquidity (1000) -> Bank Account (1001)
    # Liabilities (2)
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Test finding specific nodes via nodify starting from top-level accounts
    # We look for the 'Assets' node (RootAccount)
    assets_node = None
    for node in coa.structure:
        if node.account.code == Code("1"):
            assets_node = node
            break
            
    assert assets_node is not None
    assert assets_node.account.name == "Assets"
    
    # Verify Level 1: Liquidity (Child of Assets)
    assert len(assets_node.children) == 1
    liquidity_node = assets_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    
    # Verify Level 2: Bank Account (Child of Liquidity)
    assert len(liquidity_node.children) == 1
    bank_node = liquidity_node.children[0]
    assert bank_node.account.code == Code("1001")
    assert bank_node.account.name == "Bank Account"
    assert len(bank_node.children) == 0  # Leaf node
    
    # Verify the 'Liabilities' node exists in structure as a separate root
    liabilities_node = None
    for node in coa.structure:
        if node.account.code == Code("2"):
            liabilities_node = node
            break
    assert liabilities_node is not None
    assert len(liabilities_node.children) == 0

    # Test nodify on a specific account directly (even if it's a sub-account)
    bank_account_node = coa.nodify(bank_account)
    assert bank_account_node.account.code == Code("1001")
    assert bank_account_node.children == []

    # Test nodify on an account with no children (leaf node behavior)
    expenses_node = None
    for node in coa.structure:
        if node.account.code == Code("5"):
            expenses_node = node
            break
    assert expenses_node is not None
    assert len(expenses_node.children) == 0
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account to a root account
    parent_code = Code("1")  # Assets
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of the newly created account)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.parent.code == new_code
    assert grandchild_account.code == grandchild_code
    assert grandchild_account in coa.subaccounts(new_account)

    # Test adding an existing account with identical details (should return existing)
    existing_account = coa.add(new_code, grandchild_code, grandchild_name)
    assert existing_account is grandchild_account

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the same as its parent"):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined"):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Inconsistent details for existing code (different name/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(new_code, grandchild_code, "Different Name")
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    converts accounts into a hierarchical tree structure of Nodes.
    """
    coa = COA()
    
    # Setup a hierarchy:
    # [1] Assets
    #    [1000] Liquidity
    #        [1001] Bank Account
    
    root_assets = coa.find(Code("1"))
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")

    # 1. Test nodify on a leaf node (Bank Account)
    leaf_node = coa.nodify(bank_account)
    assert isinstance(leaf_node, COA.Node)
    assert leaf_node.account == bank_account
    assert len(leaf_node.children) == 0

    # 2. Test nodify on an intermediate node (Liquidity)
    mid_node = coa.nodify(liquidity)
    assert isinstance(mid_node, COA.Node)
    assert mid_node.account == liquidity
    assert len(mid_node.children) == 1
    assert mid_node.children[0].account == bank_account

    # 3. Test nodify on a root node (Assets)
    root_node = coa.nodify(root_assets)
    assert isinstance(root_node, COA.Node)
    assert root_node.account == root_assets
    assert len(root_node.children) == 1
    assert root_node.children[0].account == liquidity
    assert root_node.children[0].children[0].account == bank_account

    # 4. Test nodify on a node with no sub-accounts (Liabilities)
    liabilities = coa.find(Code("2"))
    liab_node = coa.nodify(liabilities)
    assert liab_node.account == liabilities
    assert len(liab_node.children) == 0

    # 5. Verify the entire structure matches expected depth and content
    structure = list(coa.structure)
    # Find the Assets node in the structure list (it might not be the first depending on order, 
    # but since COA uses OrderedDict it should be)
    assets_node = next(n for n in structure if n.account.code == Code("1"))
    
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account.code == Code("1000")
    assert assets_node.children[0].children[0].account.code == Code("1001")
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test 1: Successfully adding a sub-account to a root account
    parent_code = Code("1")  # Assets
    new_code = Code("1000")
    new_name = "Liquidity"
    sub_acct = coa.add(parent_code, new_code, new_name)
    
    assert sub_acct.code == new_code
    assert sub_acct.name == new_name
    assert sub_acct.parent.code == parent_code
    assert sub_acct.type == AccountType.ASSETS
    assert coa.find(new_code) == sub_acct
    assert sub_acct in coa.subaccounts(coa.find(parent_code))

    # Test 2: Successfully adding a nested sub-account (grandchild)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_acct = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_code == sub_acct.parent.code # Error in logic? No, check: grandchild's parent is liquidity
    assert grandchild_acct.parent.code == new_code
    assert grandchild_acct.name == grandchild_name
    assert grandchild_acct in coa.subaccounts(sub_acct)

    # Test 3: Adding an existing account with same parameters should return the existing instance
    existing_acct = coa.add(new_code, grandchild_code, grandchild_name)
    assert existing_acct is grandchild_acct

    # Test 4: Error - Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test 5: Error - Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test 6: Error - Inconsistency (Trying to re-add same code but with different name or parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(new_code, grandchild_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Trying to point 1001 to a different parent (2) while keeping same code/name
        coa.add(Code("2"), grandchild_code, grandchild_name)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a top-level subaccount (child of root)
    liquidity_code = Code("1000")
    parent_code = Code("1")
    liquidity_name = "Liquidity"
    liquidity_acc = coa.add(parent_code, liquidity_code, liquidity_name)
    
    assert liquidity_acc.code == liquidity_code
    assert liquidity_acc.name == liquidity_name
    assert liquidity_acc.parent.code == parent_code
    assert coa.find(liquidity_code) == liquidity_acc
    assert liquidity_acc in coa.subaccounts(coa.find(parent_code))

    # Test successful addition of a nested subaccount (child of child)
    bank_code = Code("1001")
    bank_name = "Bank Account"
    bank_acc = coa.add(liquidity_code, bank_code, bank_name)
    
    assert bank_acc.code == bank_code
    assert bank_acc.parent.code == liquidity_code
    assert bank_acc in coa.subaccounts(liquidity_acc)

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the same as its parent"):
        coa.add(bank_code, bank_code, "Self Parent")

    # Test error: Parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined"):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Adding existing account with different name/parent (inconsistency)
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(liquidity_code, liquidity_code, "Different Name")
        
    # Test idempotency: Adding exact same account again should return existing instance
    existing_acc = coa.add(liquidity_code, liquidity_code, liquidity_name)
    assert existing_acc is liquidity_acc

    # Test error: Adding existing code but with different name/parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("1"), Code("1000"), "New Name For Liquidity")
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a sub-account to a root account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test successful addition of a nested sub-account (multi-level)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.parent.code == new_code
    assert grandchild_account.code == grandchild_code
    assert grandchild_account in coa.subaccounts(new_account)

    # Test adding an existing account with identical parameters returns the same instance
    existing_account = coa.add(new_code, grandchild_code, grandchild_name)
    assert existing_account is grandchild_account

    # Test error: An account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Adding an existing code with different name/parent (Inconsistency)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, grandchild_code, "Different Name")

    # Test error: Adding an existing code with a different parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), grandchild_code, grandchild_name)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a sub-account to a root account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    
    sub_acc = coa.add(parent_code, new_code, new_name)
    
    assert sub_acc.code == new_code
    assert sub_acc.name == new_name
    assert sub_acc.parent.code == parent_code
    assert sub_acc.type == AccountType.ASSETS
    assert coa.find(new_code) == sub_acc
    assert sub_acc in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of child)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_acc = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_acc.parent.code == new_code
    assert grandchild_acc.name == grandchild_name
    assert grandchild_acc in coa.subaccounts(sub_acc)

    # Test adding an existing account with identical info (idempotency)
    existing_acc = coa.add(new_code, grandchild_code, grandchild_name)
    assert existing_acc == grandchild_acc

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Inconsistent data for existing code (name mismatch)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(new_code, grandchild_code, "Different Name")

    # Test error: Inconsistent data for existing code (parent mismatch)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), grandchild_code, grandchild_name)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly transforms 
    the flat account structure into a hierarchical tree of COA.Node instances.
    """
    coa = COA()
    
    # Setup a hierarchy:
    # Assets (1) -> Liquidity (1000) -> Bank Account (1001)
    # Liabilities (2)
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Test Root Node (Assets)
    assets_account = coa.find(Code("1"))
    root_node = coa.nodify(assets_account)
    
    assert isinstance(root_node, COA.Node)
    assert root_node.account.code == Code("1")
    assert len(root_node.children) == 1
    
    # Test Middle Node (Liquidity)
    liquidity_node = root_node.children[0]
    assert isinstance(liquidity_node, COA.Node)
    assert liquidity_node.account.code == Code("1000")
    assert len(liquidity_node.children) == 1
    
    # Test Leaf Node (Bank Account)
    bank_node = liquidity_node.children[0]
    assert isinstance(bank_node, COA.Node)
    assert bank_node.account.code == Code("1001")
    assert len(bank_node.children) == 0
    
    # Test a leaf node without children (Liabilities)
    liabilities_account = coa.find(Code("2"))
    liabilities_node = coa.nodify(liabilities_account)
    assert liabilities_node.account.code == Code("2")
    assert len(liabilities_node.children) == 0

    # Test the full structure iteration (the top-level nodes)
    structure = list(coa.structure)
    # We expect 5 top level accounts: Assets, Liabilities, Equities, Revenues, Expenses
    assert len(structure) == 5
    
    # Verify deep nesting exists in the structure generator
    # Find the node for Assets in the structure
    assets_struct_node = next(n for n in structure if n.account.code == Code("1"))
    assert len(assets_struct_node.children) == 1
    assert assets_struct_node.children[0].account.code == Code("1000")
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly transforms 
    a flat list of accounts into a tree-like structure of COA.Node instances.
    """
    coa = COA()
    
    # Setup: Create a hierarchy
    # 1 (Assets) -> 1000 (Liquidity) -> 1001 (Bank Account)
    # 2 (Liabilities)
    parent_code = Code("1")
    child_code = Code("1000")
    grandchild_code = Code("1001")
    
    liquidity = coa.add(parent_code, child_code, "Liquidity")
    bank_account = coa.add(child_code, grandchild_code, "Bank Account")
    coa.add(Code("2"), Code("2000"), "Current Liabilities")

    # Execution: Convert the top-level structure to nodes
    nodes = list(coa.structure)
    
    # Find specific nodes in the generated tree for assertion
    # We expect 5 top level nodes initially (Assets, Liabilities, Equities, Revenues, Expenses)
    # Plus any we added as top-level if they were roots, but here we added subaccounts.
    # The structure property iterates over coa.toplevel.
    
    # Identify the 'Assets' node in the tree
    assets_node = next(n for n in nodes if n.account.code == Code("1"))
    liabilities_node = next(n for n in nodes if n.account.code == Code("2"))
    
    # 1. Verify Assets Node (Root)
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1  # Only Liquidity is a child of Assets
    
    # 2. Verify Liquidity Node (Child of Assets)
    liquidity_node = assets_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1  # Bank Account is a child of Liquidity
    
    # 3. Verify Bank Account Node (Grandchild of Assets)
    bank_account_node = liquidity_node.children[0]
    assert bank_account_node.account.code == Code("1001")
    assert bank_account_node.account.name == "Bank Account"
    assert len(bank_account_node.children) == 0  # Leaf node
    
    # 4. Verify Liabilities Node (Sibling of Assets)
    # Note: In the provided COA implementation, adding a subaccount doesn't remove the root,
    # but 'structure' iterates over 'toplevel'.
    assert liabilities_node.account.code == Code("2")
    # We added 2000 as a child of 2, so we need to find the node for account 2
    # If 2000 was added via coa.add(Code("2"), ...), it's a subaccount, not a top-level account.
    # Therefore, 'liabilities_node' (Account 2) should now have 1 child.
    assert len(liabilities_node.children) == 1
    assert liabilities_node.children[0].account.code == Code("2000")

    # 5. Verify leaf nodes for the default accounts (Equities, Revenues, Expenses)
    equities_node = next(n for n in nodes if n.account.code == Code("3"))
    assert len(equities_node.children) == 0
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a mock or a 
    concrete implementation that adheres to the protocol.
    """
    # Arrange: Create a dummy COA instance to be returned by our callable
    expected_coa = COA()
    
    # A function (or Mock) that implements the ReadChartOfAccounts protocol
    # The protocol requires: def __call__(self) -> COA:
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = expected_coa

    # Act: Execute the callable
    result = mock_reader()

    # Assert: Verify the result is the expected COA and the call was made
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    # Arrange: Define a concrete class that implements the Protocol
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    
    # Act
    result = reader()

    # Assert
    assert isinstance(result, COA)
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    
    sub_acc = coa.add(parent_code, new_code, new_name)
    
    assert sub_acc.code == new_code
    assert sub_acc.name == new_name
    assert sub_acc.parent.code == parent_code
    assert sub_acc.type == AccountType.ASSETS
    assert coa.find(new_code) == sub_acc
    assert sub_acc in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of child)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_acc = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_acc.parent.code == new_code
    assert grandchild_acc in coa.subaccounts(sub_acc)

    # Test adding an existing account with identical info (should return same instance)
    existing_acc = coa.add(new_code, new_code, new_name)
    assert existing_acc is sub_acc

    # Test error: Account cannot be parent of itself
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent does not exist
    bogus_parent = Code("9999")
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(bogus_parent, Code("99991"), "Ghost Account")

    # Test error: Inconsistency (same code, different name/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Try to re-add same code but with a different parent (e.g., 2 instead of 1)
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object (like a function or a class with __call__) 
    that satisfies the protocol.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the Protocol
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader = MockReader()

    # Execute the __call__ method
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that adheres to its signature.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol for testing
    class MockReadCOA:
        def __call__(self) -> COA:
            return mock_coa

    # Instantiate the implementation
    reader = MockReadCOA()

    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object (like a function or a mock) that conforms to the signature.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = Mock(spec=COA)
    
    # Define a dummy implementation of the protocol
    def mock_reader() -> COA:
        return mock_coa

    # Verify the type/protocol conformance via a simple check 
    # (In Python, Protocols are checked via isinstance with @runtime_checkable)
    assert isinstance(mock_reader, ReadChartOfAccounts)

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account to a root account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of child)
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.parent.code == new_code
    assert grandchild_account.code == grandchild_code
    assert grandchild_account in coa.subaccounts(new_account)

    # Test adding an account that already exists with identical info (idempotency)
    existing_account = coa.add(parent_code, new_code, new_name)
    assert existing_account == new_account

    # Test error: Parent is the same as child
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test error: Conflict - same code but different name or parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Using a different parent for an existing code
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ protocol implementation of ReadChartOfAccounts.
    Since ReadChartOfAccounts is a Protocol, we test it by creating 
    a callable object that adheres to the signature and verifying 
    it returns a COA instance.
    """
    # Create a mock function that follows the ReadChartOfAccounts signature
    mock_reader = MagicMock(return_value=COA())

    # Verify the mock is callable (simulating an implementation of the protocol)
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

    # Test with a concrete implementation to ensure Protocol compliance
    class ConcreteReader:
        def __call__(self) -> COA:
            return COA()

    concrete_reader = ConcreteReader()
    assert isinstance(concrete_reader(), COA)
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable object.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a class that implements the ReadChartOfAccounts protocol
    class MockCOAReader:
        def __call__(self) -> COA:
            return mock_coa

    # Instantiate the reader
    reader = MockCOAReader()
    
    # Verify the object is an instance of the Protocol (via runtime checkable if possible, 
    # but here we test the functionality of the call)
    result = reader()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
    assert not isinstance(reader, str)  # Ensure it's our callable object
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test an implementation 
    to verify it adheres to the expected behavior (returning a COA instance).
    """
    # Create a mock implementation that follows the ReadChartOfAccounts protocol
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == expected_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_concrete_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    # Verify core accounts exist in the returned COA
    assert result.find(Code("1")).name == "Assets"
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that adheres to its structure.
    """
    # Create a mock COA instance to be returned by our callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol for testing purposes
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    # Initialize the reader
    reader = MockReader()

    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    converts a flat account structure into a tree-like Node structure.
    """
    coa = COA()
    
    # Setup: Add a hierarchy
    # Root: Assets [1]
    # Child: Liquidity [1000] (Parent: 1)
    # Grandchild: Bank Account [1001] (Parent: 1000)
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    
    # Test 1: Verify the root node (Assets) contains the correct hierarchy
    root_assets = coa.find(Code("1"))
    root_node = coa.nodify(root_assets)
    
    assert isinstance(root_node, COA.Node)
    assert root_node.account.code == Code("1")
    assert len(root_node.children) == 1
    
    # Test 2: Verify the first level child (Liquidity)
    liquidity_node = root_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    
    # Test 3: Verify the leaf node (Bank Account)
    bank_node = liquidity_node.children[0]
    assert bank_node.account.code == Code("1001")
    assert bank_node.account.name == "Bank Account"
    assert len(bank_node.children) == 0
    
    # Test 4: Verify nodify on a leaf node directly (no children)
    leaf_only_node = coa.nodify(bank_account)
    assert leaf_only_node.account.code == Code("1001")
    assert leaf_only_node.children == []

    # Test 5: Verify nodify on a mid-level node (Liquidity) independently
    mid_node = coa.nodify(liquidity)
    assert mid_node.account.code == Code("1000")
    assert len(mid_node.children) == 1
    assert mid_node.children[0].account.code == Code("1001")

    # Test 6: Verify traversal of the entire structure via nodify on a root account
    # We check that all top-level accounts are represented in the structure
    all_nodes = list(coa.structure)
    assert len(all_nodes) == 5  # Assets, Liabilities, Equities, Revenues, Expenses
    
    # Find the node for 'Expenses' (the last root)
    expenses_node = next(n for n in all_nodes if n.account.code == Code("5"))
    assert expenses_node.account.name == "Expenses"
    assert len(expenses_node.children) == 0
```


