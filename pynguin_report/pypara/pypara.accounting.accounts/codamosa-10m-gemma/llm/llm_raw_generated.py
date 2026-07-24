####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # Test adding an existing account with identical details (should return existing)
    existing_account = coa.add(new_code, child_code, child_name)
    assert existing_account is child_account

    # Test adding an existing account with different name (should raise ValueError)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(new_code, child_code, "Different Name")

    # Test adding an account where parent is the same as the account code
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test adding an account with a non-existent parent
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a top-level sub-account
    parent_code = Code("1")  # Assets
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test successful addition of a nested sub-account (multi-level)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)
    assert coa.find(child_code).parent.name == "Liquidity"

    # Test adding an existing account with identical info (idempotency)
    existing_account = coa.add(new_code, child_code, child_name)
    assert existing_account is child_account

    # Test error: Parent is the same as the new account code
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Account code exists but with different name or parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, child_code, "Different Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), child_code, child_name) # Different parent (Liabilities)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of an object conforming to the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test it using a mock or a concrete implementation.
    """
    # Create a mock that implements the ReadChartOfAccounts protocol
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Define a return value that is a valid COA instance
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_concrete_implementation():
    """
    Tests the protocol behavior using a concrete implementation.
    """
    class ConcreteReader:
        def __call__(self) -> COA:
            return COA()

    reader = ConcreteReader()
    result = reader()

    assert isinstance(result, COA)
    assert len(list(result)) == 5  # Default core accounts
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that adheres to the protocol.
    """
    # Create a mock object that implements the __call__ signature
    # defined in the ReadChartOfAccounts protocol.
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Setup the return value to be a COA instance
    expected_coa = COA()
    mock_reader.return_value = expected_coa
    
    # Execute the call
    result = mock_reader()
    
    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation_compliance():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    
    # Verify the instance satisfies the protocol
    assert isinstance(reader, ReadChartOfAccounts)
    
    # Verify the call returns the expected type
    result = reader()
    assert isinstance(result, COA)
    assert len(list(result.toplevel)) == 5
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_COA___iter__():
    """
    Tests the __iter__ method of the COA class to ensure it correctly
    iterates over the default root accounts and custom added accounts.
    """
    # Test 1: Default initialization
    coa = COA()
    
    # Check the number of default accounts (5 core accounts)
    default_accounts = list(coa)
    assert len(default_accounts) == 5
    
    # Check specific default values from the docstring/logic
    # (1, Assets), (2, Liabilities), (3, Equities), (4, Revenues), (5, Expenses)
    assert default_accounts[0] == (Code("1"), coa.find(Code("1")))
    assert default_accounts[0][1].name == "Assets"
    assert default_accounts[4] == (Code("5"), coa.find(Code("5")))
    assert default_accounts[4][1].name == "Expenses"

    # Test 2: Iteration after adding sub-accounts
    liquidity_code = Code("1000")
    bank_code = Code("1001")
    coa.add(Code("1"), liquidity_code, "Liquidity")
    coa.add(liquidity_code, bank_code, "Bank Account")
    
    # Convert to list to check contents
    all_accounts_iterated = list(coa)
    
    # The length should now be 7 (5 roots + 2 new)
    assert len(all_accounts_iterated) == 7
    
    # Verify the presence of the new accounts in the iteration
    codes_in_iter = [code for code, acct in all_accounts_iterated]
    assert liquidity_code in codes_in_iter
    assert bank_code in codes_in_iter
    
    # Verify names are correct during iteration
    found_liquidity = next(acct for code, acct in coa if code == liquidity_code)
    assert found_liquidity.name == "Liquidity"
    
    found_bank = next(acct for code, acct in coa if code == bank_code)
    assert found_bank.name == "Bank Account"

def test_COA___iter___custom_roots():
    """
    Tests __iter__ with a custom rootspec provided during initialization.
    """
    custom_spec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities")
    }
    coa = COA(rootspec=custom_spec)
    
    # The number of accounts remains 5, but codes and names change for the specified types
    # The remaining types (Equities, Revenues, Expenses) should still follow default logic
    
    # Check custom asset
    asset_acc = coa.find(Code("A"))
    assert asset_acc is not None
    assert asset_acc.name == "Custom Assets"
    
    # Check default expense (the 5th enum member)
    expense_acc = coa.find(Code("5"))
    assert expense_acc is not None
    assert expense_acc.name == "Expenses"
    
    # Check iteration sequence contains the custom code
    codes = [code for code, _ in coa]
    assert Code("A") in codes
    assert Code("L") in codes
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a sub-account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test successful addition of a nested sub-account
    grandchild_code = Code("1001")
    grandchild_name = "Bank Account"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.code == grandchild_code
    assert grandchild_account.parent.code == new_code
    assert grandchild_account in coa.subaccounts(new_account)

    # Test adding an existing account with same details (idempotency)
    existing_account = coa.add(parent_code, new_code, new_name)
    assert existing_account == new_account

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Inconsistent details for existing account (different name)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    # Test error: Inconsistent details for existing account (different parent)
    # First, create a sibling
    coa.add(parent_code, Code("1002"), "Other")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("1002"), new_code, new_name)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a valid implementation.
    """
    # Create a mock implementation that follows the ReadChartOfAccounts protocol
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned by the reader
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_concrete_class():
    """
    Tests the __call__ method using a concrete class implementation.
    """
    class ConcreteReader:
        def __call__(self) -> COA:
            return COA(rootspec={
                AccountType.ASSETS: (Code("10"), "Custom Assets")
            })

    reader = ConcreteReader()
    result = reader()

    assert isinstance(result, COA)
    assert result.find(Code("10")).name == "Custom Assets"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    # Create a mock object that implements the ReadChartOfAccounts protocol
    # Since ReadChartOfAccounts is a Protocol with a __call__ method,
    # we can use a MagicMock or a simple lambda/function.
    
    mock_coa = MagicMock(spec=COA)
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == mock_coa
    mock_reader.assert_called_once()
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol (runtime_checkable), we test 
    a concrete implementation that follows the signature.
    """
    # Create a mock that implements the ReadChartOfAccounts protocol
    # The protocol defines a __call__ method returning a COA instance
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == expected_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_protocol_compliance():
    """
    Tests that a class implementing the protocol is recognized by runtime_checkable.
    """
    class ValidReader:
        def __call__(self) -> COA:
            return COA()

    class InvalidReader:
        def __call__(self, x: int) -> None:
            pass

    valid_reader = ValidReader()
    invalid_reader = InvalidReader()

    assert isinstance(valid_reader, ReadChartOfAccounts)
    assert not isinstance(invalid_reader, ReadChartOfAccounts)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test 1: Successfully add a sub-account to a root account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test 2: Successfully add a nested sub-account (child of a sub-account)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # Test 3: Adding an existing account with same details should return the same instance
    duplicate_account = coa.add(new_code, new_code, new_name)
    assert duplicate_account is new_account

    # Test 4: Error - Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Non-existent")

    # Test 5: Error - Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test 6: Error - Conflict (Adding same code but different name or parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Change parent of existing code 1000 to code 2
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test a functional implementation.
    """
    # Create a mock implementation of the ReadChartOfAccounts protocol
    # The protocol expects a callable that returns a COA instance
    mock_coa = COA()
    mock_reader = Mock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == mock_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_real_logic():
    """
    Tests a real functional implementation of the ReadChartOfAccounts protocol.
    """
    def real_reader_impl() -> COA:
        # A real implementation would typically parse a file or DB
        # Here we return a customized COA
        spec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets")
        }
        return COA(rootspec=spec)

    # Verify the implementation adheres to the protocol and returns expected COA
    result = real_reader_impl()
    
    assert isinstance(result, COA)
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("10")).type == AccountType.ASSETS
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly transforms
    the flat account structure into a tree of COA.Node objects.
    """
    coa = COA()
    
    # Setup a hierarchy:
    # 1: Assets (Root)
    #   1000: Liquidity (Sub of 1)
    #     1001: Bank Account (Sub of 1000)
    # 2: Liabilities (Root)
    
    parent_code = Code("1")
    sub_code = Code("1000")
    sub_sub_code = Code("1001")
    
    coa.add(parent_code, sub_code, "Liquidity")
    coa.add(sub_code, sub_sub_code, "Bank Account")
    
    # Retrieve the top-level nodes from the structure
    # structure is a map/iterator over coa.toplevel
    nodes = list(coa.structure)
    
    # Find the node for Assets (Code '1')
    assets_node = next((n for n in nodes if n.account.code == parent_code), None)
    assert assets_node is not None
    assert assets_node.account.name == "Assets"
    
    # Check level 1: Liquidity should be a child of Assets
    assert len(assets_node.children) == 1
    liquidity_node = assets_node.children[0]
    assert liquidity_node.account.code == sub_code
    assert liquidity_node.account.name == "Liquidity"
    
    # Check level 2: Bank Account should be a child of Liquidity
    assert len(liquidity_node.children) == 1
    bank_node = liquidity_node.children[0]
    assert bank_node.account.code == sub_sub_code
    assert bank_node.account.name == "Bank Account"
    assert len(bank_node.children) == 0  # Leaf node
    
    # Verify that other root nodes exist and are not part of the Assets tree
    liabilities_node = next((n for n in nodes if n.account.code == Code("2")), None)
    assert liabilities_node is not None
    assert len(liabilities_node.children) == 0
    
    # Verify total number of top-level nodes (Assets, Liabilities, Equities, Revenues, Expenses)
    assert len(nodes) == 5
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ protocol implementation for ReadChartOfAccounts.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation
    to verify the signature and behavior.
    """
    # Create a mock COA instance
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol
    class MockCOAReader:
        def __call__(self) -> COA:
            return mock_coa

    reader: ReadChartOfAccounts = MockCOAReader()
    
    # Execute the call
    result = reader()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    # Create a mock function that matches the ReadChartOfAccounts protocol
    # The protocol defines a callable that returns a COA instance
    mock_coa_instance = COA()
    mock_reader = Mock(return_value=mock_coa_instance)
    
    # Ensure the mock is treated as a ReadChartOfAccounts type
    # (In a real scenario, this would be a function/class implementing the protocol)
    reader: ReadChartOfAccounts = mock_reader

    # Execute the call
    result = reader()

    # Assertions
    assert result == mock_coa_instance
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.
    """
    # Arrange: Create a mock function/object that matches the ReadChartOfAccounts protocol
    # The protocol expects a callable that returns a COA instance.
    mock_coa = COA()
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    mock_reader.return_value = mock_coa

    # Act: Call the object
    result = mock_reader()

    # Assert: Verify the result is the expected COA instance and the call was recorded
    assert result == mock_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___concrete_implementation():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    # Arrange: Define a concrete implementation of the protocol
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    
    # Act
    result = reader()

    # Assert
    assert isinstance(result, COA)
    assert len(list(result.toplevel)) == 5
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    transforms the flat account structure into a hierarchical tree of Nodes.
    """
    coa = COA()
    
    # 1. Test nodify on a RootAccount (Top-level)
    # By default, COA initializes with 5 root accounts: 1:Assets, 2:Liabilities, etc.
    assets_acc = coa.find(Code("1"))
    assert assets_acc is not None
    assert assets_acc.name == "Assets"
    
    node_root = coa.nodify(assets_acc)
    
    assert isinstance(node_root, COA.Node)
    assert node_root.account.code == Code("1")
    assert node_root.children == []

    # 2. Test nodify with a SubAccount (Nested structure)
    # Add: 1000 (Liquidity) as child of 1 (Assets)
    # Add: 1001 (Bank Account) as child of 1000 (Liquidity)
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_acc = coa.add(Code("1000"), Code("1001"), "Bank Account")
    
    node_liquidity = coa.nodify(liquidity)
    
    # Check liquidity node structure
    assert node_liquidity.account.code == Code("1000")
    assert len(node_liquidity.children) == 1
    
    # Check the child node (Bank Account)
    child_node = node_liquidity.children[0]
    assert isinstance(child_node, COA.Node)
    assert child_node.account.code == Code("1001")
    assert child_node.account.name == "Bank Account"
    assert child_node.children == []

    # 3. Test nodify on the top-level Assets node after additions
    # The root Assets node should now contain the tree of its sub-accounts
    node_assets_updated = coa.nodify(assets_acc)
    
    assert len(node_assets_updated.children) == 1
    # The first child should be the Liquidity node we inspected above
    assert node_assets_updated.children[0].account.code == Code("1000")
    assert node_assets_updated.children[0].children[0].account.code == Code("1001")

    # 4. Test nodify on an account with no children (an existing root like Liabilities)
    liabilities_acc = coa.find(Code("2"))
    node_liabilities = coa.nodify(liabilities_acc)
    assert node_liabilities.account.code == Code("2")
    assert node_liabilities.children == []
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.
    """
    # Arrange: Create a mock or a concrete implementation of the Protocol
    # A simple lambda or a function satisfies the Protocol signature
    expected_coa = COA()
    
    # Implementation of the protocol
    def coa_reader() -> COA:
        return expected_coa

    # We can use a mock to verify the call behavior
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    mock_reader.return_value = expected_coa

    # Act
    result = mock_reader()

    # Assert
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

    # Test with a real functional implementation
    real_reader: ReadChartOfAccounts = coa_reader
    assert real_reader() == expected_coa
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a top-level sub-account
    parent_code = Code("1")  # Assets
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test successful addition of a nested sub-account
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account.type == AccountType.ASSETS
    assert child_account in coa.subaccounts(new_account)

    # Test adding an existing account with same details (idempotency)
    existing_account = coa.add(new_code, child_code, child_name)
    assert existing_account is child_account

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Inconsistent data for existing account (different name)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(new_code, child_code, "Different Name")

    # Test error: Inconsistent data for existing account (different parent)
    # Create a new branch to test parent mismatch
    coa.add(Code("2"), Code("2000"), "Liabilities Branch")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2000"), child_code, child_name)
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a Mock 
    that conforms to the protocol's signature.
    """
    # Create a mock object that implements the ReadChartOfAccounts protocol
    # The protocol defines __call__ as returning a COA instance.
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Create a dummy COA instance to be returned by the mock
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    mock object or a concrete implementation that follows the protocol.
    """
    # Create a mock that implements the protocol (has a __call__ method)
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Define a fake COA to be returned
    fake_coa = COA()
    mock_reader.return_value = fake_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == fake_coa
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
    # Verify the default COA structure as defined in the module
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("5")).name == "Expenses"
```


# LLM-generated content at query #21
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

    # Test adding a nested sub-account (child of a sub-account)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account.type == AccountType.ASSETS
    assert child_account in coa.subaccounts(new_account)

    # Test adding an existing account with identical info (should return existing)
    existing_account = coa.add(parent_code, new_code, new_name)
    assert existing_account == new_account

    # Test error: Account cannot be parent of itself
    with pytest.raises(ValueError, match="An account can not be the same parent of itself"):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined"):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Inconsistent information (different name for same code/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(parent_code, new_code, "Different Name")

    # Test error: Inconsistent information (different parent for same code)
    # We need to create a new parent first to avoid "Parent not defined" error
    coa.add(parent_code, Code("2000"), "Other Parent")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("2000"), new_code, new_name)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test 1: Successfully add a sub-account to a root account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test 2: Successfully add a nested sub-account (child of a child)
    child_code = Code("1001")
    child_name = "Bank Account"
    grandchild_account = coa.add(new_code, child_code, child_name)
    
    assert grandchild_account.code == child_code
    assert grandchild_account.parent.code == new_code
    assert grandchild_account in coa.subaccounts(new_account)

    # Test 3: Adding an existing account with identical info should return the same instance
    existing_account = coa.add(parent_code, new_code, new_name)
    assert existing_account is new_account

    # Test 4: Error - Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test 5: Error - Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test 6: Error - Adding an existing code with different name or parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # Test adding an existing account with same parameters (should return existing)
    existing_account = coa.add(parent_code, new_code, new_name)
    assert existing_account == new_account

    # Test error: account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: mismatching data for existing code
    # (Code exists, but name/parent provided is different)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
        
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), new_code, "Different Parent")
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test it by verifying that 
    a compatible callable returns a COA instance.
    """
    # Create a mock function that adheres to the ReadChartOfAccounts protocol
    # The protocol specifies: def __call__(self) -> COA: ...
    mock_reader = Mock(return_value=COA())

    # Execute the callable
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation_behavior():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class ConcreteReader:
        def __call__(self) -> COA:
            return COA()

    reader = ConcreteReader()
    result = reader()
    
    assert isinstance(result, COA)
    assert len(list(result.toplevel)) == 5
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that matches its signature.
    """
    # Create a mock object that implements the protocol signature
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Create a dummy COA instance to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation_behavior():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    # Define a concrete class that follows the protocol
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    
    # Verify it is compatible with the protocol via runtime check
    assert isinstance(reader, ReadChartOfAccounts)
    
    # Verify the call returns a COA
    result = reader()
    assert isinstance(result, COA)
    assert len(list(result.toplevel)) == 5
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object (like a function or a mock) that adheres to the protocol.
    """
    # Arrange: Create a mock COA instance
    mock_coa = MagicMock(spec=COA)
    
    # Create a callable that follows the ReadChartOfAccounts protocol
    # In Python, a function or a class with __call__ can satisfy this Protocol
    def mock_reader() -> COA:
        return mock_coa

    # Act: Call the implementation
    result = mock_reader()

    # Assert: The result should be the COA instance returned by the callable
    assert result == mock_coa
    assert isinstance(result, COA)

    # Additional test: Verify it works with a class implementing __call__
    class ConcreteReader:
        def __init__(self, coa_to_return: COA):
            self.coa = coa_to_return
        
        def __call__(self) -> COA:
            return self.coa

    concrete_reader = ConcreteReader(mock_coa)
    assert concrete_reader() == mock_coa
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    converts the flat account structure into a tree of COA.Node objects.
    """
    coa = COA()
    
    # Setup a hierarchy:
    # 1 (Assets) -> 1000 (Liquidity) -> 1001 (Bank Account)
    # 2 (Liabilities)
    
    # Add sub-accounts
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Test the structure generation (which uses nodify)
    nodes = list(coa.structure)
    
    # We expect 5 top-level nodes (the default roots)
    assert len(nodes) == 5
    
    # Find the Assets node (RootAccount with code '1')
    assets_node = next(n for n in nodes if n.account.code == Code("1"))
    
    # Check Assets node properties
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1
    
    # Check Liquidity node (Child of Assets)
    liquidity_node = assets_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    
    # Check Bank Account node (Child of Liquidity)
    bank_account_node = liquidity_node.children[0]
    assert bank_account_node.account.code == Code("1001")
    assert bank_account_node.account.name == "Bank Account"
    assert len(bank_account_node.children) == 0
    
    # Verify a leaf node (Liabilities)
    liabilities_node = next(n for n in nodes if n.account.code == Code("2"))
    assert liabilities_node.account.name == "Liabilities"
    assert len(liabilities_node.children) == 0

    # Test nodify directly on a specific account
    direct_node = coa.nodify(bank_account)
    assert direct_node.account.code == Code("1001")
    assert direct_node.children == []
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a valid implementation.
    """
    # Create a mock implementation that follows the ReadChartOfAccounts protocol
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Define a dummy COA to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_real_logic():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    result = reader()

    assert isinstance(result, COA)
    # Verify the default core accounts exist in the returned COA
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("5")).name == "Expenses"
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of an object implementing the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test it using a callable mock 
    that returns a COA instance.
    """
    # Create a mock COA instance
    mock_coa = MagicMock(spec=COA)
    
    # Create a mock function that implements the ReadChartOfAccounts protocol
    # In Python, a function/callable object implements the protocol if it has __call__
    read_coa_callable = MagicMock(return_value=mock_coa)
    
    # Execute the callable
    result = read_coa_callable()
    
    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
    read_coa_callable.assert_called_once()

def test_ReadChartOfAccounts_implementation_integration():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    # A concrete implementation of the protocol
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    
    # Ensure the result is a COA instance and has the expected default accounts
    result = reader()
    assert isinstance(result, COA)
    
    # Verify the default root accounts exist as per COA __post_init__
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("5")).name == "Expenses"
    assert result.find(Code("999")) is None
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    # Arrange
    # Create a mock object that implements the ReadChartOfAccounts protocol
    mock_reader: ReadChartOfAccounts = MagicMock()
    
    # Define a mock COA instance to be returned by the call
    mock_coa = MagicMock(spec=COA)
    mock_reader.return_value = mock_coa

    # Act
    # Execute the __call__ method
    result = mock_reader()

    # Assert
    # Verify that the result is the expected COA instance
    assert result == mock_coa
    # Verify that the __call__ method was actually invoked once
    mock_reader.assert_called_once()
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.parent.code == new_code
    assert child_account.code == child_code
    assert child_account in coa.subaccounts(new_account)

    # Test adding an existing account with same details (should return existing)
    existing_account = coa.add(new_code, child_code, child_name)
    assert existing_account is child_account

    # Test error: Account cannot be parent of itself
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Inconsistent details for existing account
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Attempt to redefine 1001 with a different name but same code/parent
        coa.add(new_code, child_code, "Different Name")

    # Test error: Inconsistent parent for existing account
    # First, create a different account with same code but different parent
    # (This requires a unique code not already in coa, but we'll use a new one)
    # To trigger the specific 'inconsistent' error, we need to target an existing code
    # but provide a different parent.
    # Since we can't easily bypass the 'code in self._accounts' check without a new code,
    # we verify the logic via a unique code that tries to claim a different parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # We use a code that exists (1000) but try to claim it belongs to '2' (Liabilities)
        coa.add(Code("2"), Code("1000"), "Liquidity")
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation
    (a callable) that adheres to the protocol.
    """
    # Create a mock that matches the signature of ReadChartOfAccounts
    # A ReadChartOfAccounts object must be a callable that returns a COA instance
    mock_reader: ReadChartOfAccounts = Mock(return_value=COA())

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_concrete_implementation():
    """
    Tests the behavior of a real function implementation of the protocol.
    """
    def concrete_reader() -> COA:
        return COA()

    # Verify the function adheres to the protocol type via type checking (runtime)
    # and returns the correct type.
    assert isinstance(concrete_reader, ReadChartOfAccounts)
    
    result = concrete_reader()
    assert isinstance(result, COA)
    assert len(list(result.toplevel)) == 5
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of an object implementing the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test a callable that returns a COA instance.
    """
    # Create a mock function that follows the ReadChartOfAccounts protocol
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Create a real COA instance to be returned by the mock
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == expected_coa
    mock_reader.assert_called_once()
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly 
    transforms accounts into a tree structure of COA.Node instances.
    """
    coa = COA()
    
    # Setup a tree structure:
    # 1: Assets
    #   1000: Liquidity
    #     1001: Bank Account
    # 2: Liabilities
    
    # Add sub-accounts
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Test the nodify method on the top-level 'Assets' account
    assets_account = coa.find(Code("1"))
    root_node = coa.nodify(assets_account)
    
    # Assertions for the root node (Assets)
    assert isinstance(root_node, COA.Node)
    assert root_node.account.code == Code("1")
    assert root_node.account.name == "Assets"
    assert len(root_node.children) == 1
    
    # Assertions for the first child (Liquidity)
    liquidity_node = root_node.children[0]
    assert isinstance(liquidity_node, COA.Node)
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    
    # Assertions for the grandchild (Bank Account)
    bank_node = liquidity_node.children[0]
    assert isinstance(bank_node, COA.Node)
    assert bank_node.account.code == Code("1001")
    assert bank_node.account.name == "Bank Account"
    assert len(bank_node.children) == 0
    
    # Test nodify on a leaf node (Bank Account)
    leaf_node = coa.nodify(bank_account)
    assert leaf_node.account.code == Code("1001")
    assert leaf_node.children == []

    # Test nodify on a node with no children (Liabilities)
    liabilities_account = coa.find(Code("2"))
    liabilities_node = coa.nodify(liabilities_account)
    assert liabilities_node.account.code == Code("2")
    assert liabilities_node.children == []
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test it by creating 
    a compatible callable object.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol
    class MockCOAReader:
        def __call__(self) -> COA:
            return mock_coa

    # Instantiate the reader
    reader = MockCOAReader()

    # Verify the __call__ functionality
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
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test an implementation 
    to verify the call signature and return type.
    """
    # Create a mock implementation that adheres to the ReadChartOfAccounts protocol
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define a fake COA to be returned by the reader
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == expected_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation_logic():
    """
    Tests a concrete implementation of ReadChartOfAccounts to ensure 
    it correctly returns a COA object.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    assert len(list(result.toplevel)) == 5
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ protocol of ReadChartOfAccounts.
    Since ReadChartOfAccounts is a Protocol, we test it by verifying 
    that a callable object conforming to the protocol behaves as expected.
    """
    # Arrange
    mock_coa = Mock(spec=COA)
    
    # Define a concrete implementation of the protocol
    def mock_reader() -> COA:
        return mock_coa

    # Verify the function signature/type matches the protocol expectation
    # (In a real scenario, this would be used for type checking)
    reader: ReadChartOfAccounts = mock_reader

    # Act
    result = reader()

    # Assert
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #10
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

    # Test adding a nested sub-account (child of the newly created sub-account)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # Test idempotency: adding the exact same account again should return the same instance
    same_account = coa.add(parent_code, new_code, new_name)
    assert same_account is new_account

    # Test Error: An account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test Error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test Error: Inconsistency (trying to re-add same code but with different name or parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Trying to change the parent of 1000 from '1' to '2'
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a functional implementation.
    """
    # Create a mock implementation of the ReadChartOfAccounts protocol
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Define a real COA instance to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts___call___with_custom_coa():
    """
    Tests that the __call__ method can return a COA with custom root specifications.
    """
    custom_spec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets")
    }
    
    def custom_reader_impl() -> COA:
        return COA(rootspec=custom_spec)

    # Execute the implementation
    result = custom_reader_impl()

    # Assertions
    assert isinstance(result, COA)
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("10")).type == AccountType.ASSETS
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test a valid implementation.
    """
    # Create a mock implementation of the ReadChartOfAccounts protocol
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Setup a dummy COA to be returned by the call
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation_logic():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            # Implementation that creates a COA with custom roots
            spec = {
                AccountType.ASSETS: (Code("10"), "Custom Assets")
            }
            return COA(rootspec=spec)

    reader = SimpleCOAReader()
    coa = reader()

    # Verify the returned object is a COA and contains the expected custom data
    assert isinstance(coa, COA)
    assert coa.find(Code("10")).name == "Custom Assets"
    assert coa.find(Code("10")).type == AccountType.ASSETS
```


# LLM-generated content at query #13
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
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account (child of a sub-account)
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # Test idempotency: adding the same account again should return the same instance
    duplicate_account = coa.add(parent_code, new_code, new_name)
    assert duplicate_account is new_account

    # Test Error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test Error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # Test Error: Inconsistent data (same code, different name/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
        
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), new_code, new_name)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test an implementation 
    of it to verify the callability and return type.
    """
    # Create a mock implementation of the ReadChartOfAccounts protocol
    mock_reader: ReadChartOfAccounts = MagicMock()
    
    # Setup a fake COA to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation_logic():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol.
    """
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    reader = SimpleCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    assert len(list(result.toplevel)) == 5
    assert result.find(Code("1")).name == "Assets"
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ protocol of ReadChartOfAccounts.
    Since ReadChartOfAccounts is a Protocol, we test it by verifying 
    that a callable object conforming to the protocol behaves as expected.
    """
    # Create a mock COA instance to be returned by the callable
    mock_coa = MagicMock(spec=COA)
    
    # Define a function that matches the ReadChartOfAccounts protocol signature
    def mock_reader() -> COA:
        return mock_coa

    # Verify the callable satisfies the protocol (implicitly via usage)
    # and returns the correct object
    result = mock_reader()
    
    assert result == mock_coa
    assert isinstance(result, COA)

    # Test with a lambda implementation
    lambda_reader: ReadChartOfAccounts = lambda: mock_coa
    assert lambda_reader() == mock_coa
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    # Create a mock object that follows the ReadChartOfAccounts protocol
    # Since ReadChartOfAccounts is a Protocol with a __call__ method,
    # we can use a MagicMock or a simple callable.
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Define a fake COA to be returned by the call
    fake_coa = COA()
    mock_reader.return_value = fake_coa

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == fake_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    callable object that follows the protocol.
    """
    # Arrange
    mock_coa = Mock(spec=COA)
    
    # Create a mock function that matches the ReadChartOfAccounts protocol
    # The protocol specifies: def __call__(self) -> COA:
    read_coa_func = Mock(return_value=mock_coa)
    
    # Act
    result = read_coa_func()
    
    # Assert
    assert result == mock_coa
    read_coa_func.assert_called_once()

def test_ReadChartOfAccounts_implementation_logic():
    """
    Tests a concrete implementation of the ReadChartOfAccounts protocol
    to ensure it returns a valid COA instance.
    """
    # Arrange
    def concrete_reader() -> COA:
        return COA()
    
    # Assert type compatibility (Protocol check)
    assert isinstance(concrete_reader, ReadChartOfAccounts)
    
    # Act
    result = concrete_reader()
    
    # Assert
    assert isinstance(result, COA)
    assert len(list(result)) == 5  # Default COA has 5 core accounts
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ protocol implementation of ReadChartOfAccounts.
    Since ReadChartOfAccounts is a Protocol, we test it using a mock 
    that adheres to the protocol signature.
    """
    # Arrange
    # Create a mock object that mimics a function/callable matching the protocol
    mock_reader = Mock(spec=ReadChartOfAccounts)
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Act
    result = mock_reader()

    # Assert
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    compatible callable object.
    """
    # Arrange
    mock_coa = Mock(spec=COA)
    
    # Define a concrete implementation of the Protocol
    class MockReader:
        def __call__(self) -> COA:
            return mock_coa

    reader: ReadChartOfAccounts = MockReader()

    # Act
    result = reader()

    # Assert
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_COA_nodify():
    coa = COA()
    
    # Setup a hierarchy: 
    # Assets [1]
    #   Liquidity [1000]
    #     Bank Account [1001]
    
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Retrieve the root account for Assets
    assets_acc = coa.find(Code("1"))
    
    # Test nodify on a leaf node (Bank Account)
    leaf_node = coa.nodify(bank_account)
    assert isinstance(leaf_node, COA.Node)
    assert leaf_node.account.code == Code("1001")
    assert leaf_node.account.name == "Bank Account"
    assert leaf_node.children == []

    # Test nodify on an intermediate node (Liquidity)
    mid_node = coa.nodify(liquidity)
    assert isinstance(mid_node, COA.Node)
    assert mid_node.account.code == Code("1000")
    assert len(mid_node.children) == 1
    assert mid_node.children[0].account.code == Code("1001")

    # Test nodify on a root node (Assets)
    root_node = coa.nodify(assets_acc)
    assert isinstance(root_node, COA.Node)
    assert root_node.account.code == Code("1")
    assert len(root_node.children) == 1
    assert root_node.children[0].account.code == Code("1000")
    assert root_node.children[0].children[0].account.code == Code("1001")

    # Test nodify on a root node with no children (Liabilities)
    liabilities_acc = coa.find(Code("2"))
    empty_root_node = coa.nodify(liabilities_acc)
    assert empty_root_node.account.code == Code("2")
    assert empty_root_node.children == []
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_COA_nodify():
    """
    Tests the nodify method of the COA class to ensure it correctly
    transforms a flat account structure into a hierarchical tree of COA.Node instances.
    """
    coa = COA()
    
    # Setup a hierarchy:
    # 1 (Assets) -> 1000 (Liquidity) -> 1001 (Bank Account)
    # 2 (Liabilities)
    
    # Add sub-accounts
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_account = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Test finding root nodes for nodify
    # The top level accounts should be the 5 core accounts defined in __post_init__
    toplevel_accounts = list(coa.toplevel)
    assert len(toplevel_accounts) == 5
    
    # Convert structure to list to inspect nodes
    nodes = list(coa.structure)
    
    # Verify we have 5 top-level nodes (Assets, Liabilities, Equities, Revenues, Expenses)
    assert len(nodes) == 5
    
    # Find the node corresponding to 'Assets' (Code '1')
    assets_node = next(n for n in nodes if n.account.code == Code("1"))
    
    # Verify Assets node properties
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1
    
    # Verify the child of Assets is 'Liquidity'
    liquidity_node = assets_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    
    # Verify the grandchild is 'Bank Account'
    bank_node = liquidity_node.children[0]
    assert bank_node.account.code == Code("1001")
    assert bank_node.account.name == "Bank Account"
    assert len(bank_node.children) == 0
    
    # Verify a leaf node like 'Liabilities' (Code '2') has no children
    liabilities_node = next(n for n in nodes if n.account.code == Code("2"))
    assert liabilities_node.account.name == "Liabilities"
    assert len(liabilities_node.children) == 0

    # Verify the type of objects returned are COA.Node
    assert isinstance(assets_node, COA.Node)
    assert isinstance(liquidity_node, COA.Node)
    assert isinstance(bank_node, COA.Node)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.
    """
    # Create a mock implementation of the ReadChartOfAccounts protocol
    mock_reader = MagicMock(spec=ReadChartOfAccounts)
    
    # Setup a dummy COA to be returned
    expected_coa = COA()
    mock_reader.return_value = expected_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert result == expected_coa
    assert isinstance(result, COA)
    mock_reader.assert_called_once()
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a functional implementation.
    """
    # Arrange
    mock_coa = MagicMock(spec=COA)
    
    # Define a concrete implementation of the protocol
    def coa_reader_implementation() -> COA:
        return mock_coa

    # Act
    reader: ReadChartOfAccounts = coa_reader_implementation()
    result = reader()

    # Assert
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test successful addition of a sub-account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account
    
    # Test adding a nested sub-account
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)
    
    # Test adding an existing account with identical details (idempotency)
    existing_account = coa.add(new_code, child_code, child_name)
    assert existing_account == child_account

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Account exists but details (name/parent) do not match
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), new_code, "Different Parent")
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts protocol implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation
    (a callable) that adheres to the protocol.
    """
    # Create a mock object that behaves like a ReadChartOfAccounts implementation
    # The protocol specifies it must be a callable returning a COA instance.
    mock_reader = Mock(spec=ReadChartOfAccounts)
    
    # Define a fake COA to be returned
    fake_coa = COA()
    mock_reader.return_value = fake_coa

    # Execute the call
    result = mock_reader()

    # Assertions
    assert isinstance(result, COA)
    assert result == fake_coa
    mock_reader.assert_called_once()

def test_ReadChartOfAccounts_implementation_logic():
    """
    Tests a concrete function implementation of the ReadChartOfAccounts protocol.
    """
    def concrete_reader() -> COA:
        # Custom initialization as described in the docstring
        spec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets")
        }
        return COA(rootspec=spec)

    # Verify implementation returns correct COA structure
    coa = concrete_reader()
    assert isinstance(coa, COA)
    assert coa.find(Code("10")).name == "Custom Assets"
    assert coa.find(Code("10")).type == AccountType.ASSETS
```


# LLM-generated content at query #26
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

    # Test adding a nested sub-account
    child_code = Code("1001")
    child_name = "Bank Account"
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # Test adding an existing account with identical details (should return existing)
    existing_account = coa.add(new_code, Code("1000"), "Liquidity")
    assert existing_account == new_account

    # Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")

    # Test error: Inconsistent details for existing code (different name/parent)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, new_code, "Different Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing of accounts member."):
        # Note: The original code's error message for parent mismatch is slightly 
        # truncated in the logic check if it fails the specific 'account.parent == parentinstance' check
        coa.add(Code("2"), new_code, "Liquidity")
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of the ReadChartOfAccounts protocol.
    Since ReadChartOfAccounts is a Protocol, we test it using a 
    compatible callable object.
    """
    # Create a mock object that implements the protocol
    mock_coa = MagicMock(spec=COA)
    
    # Define a callable that matches the ReadChartOfAccounts signature
    def mock_reader() -> COA:
        return mock_coa

    # Verify the callable is an instance of the Protocol
    # Note: Protocols are runtime_checkable, so isinstance works
    assert isinstance(mock_reader, ReadChartOfAccounts)

    # Execute the __call__ method
    result = mock_reader()

    # Assertions
    assert result == mock_coa
    assert isinstance(result, COA)
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_ReadChartOfAccounts___call__():
    """
    Tests the __call__ method of a ReadChartOfAccounts implementation.
    Since ReadChartOfAccounts is a Protocol, we test a concrete implementation.
    """
    # Create a mock COA instance to be returned
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

def test_ReadChartOfAccounts___call___with_real_logic():
    """
    Tests the __call__ method with a functional implementation 
    to ensure it correctly returns a COA instance.
    """
    class RealReader:
        def __call__(self) -> COA:
            # Returns a default COA
            return COA()

    reader = RealReader()
    result = reader()
    
    assert isinstance(result, COA)
    # Verify default root accounts exist as per docstring
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("5")).name == "Expenses"
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # 1. Test adding a valid sub-account to a root account
    parent_code = Code("1")  # Assets
    new_code = Code("1000")
    new_name = "Liquidity"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert new_account.type == AccountType.ASSETS
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # 2. Test adding a nested sub-account (child of child)
    child_code = Code("1001")
    child_name = "Bank Account"
    
    child_account = coa.add(new_code, child_code, child_name)
    
    assert child_account.code == child_code
    assert child_account.parent.code == new_code
    assert child_account in coa.subaccounts(new_account)

    # 3. Test adding an existing account with same parameters (should return existing)
    existing_account = coa.add(new_code, child_code, child_name)
    assert existing_account is child_account

    # 4. Test error: Account cannot be its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # 5. Test error: Parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99991"), "Ghost Account")

    # 6. Test error: Conflict - same code but different name or parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Attempt to re-add 1000 but with a different name
        coa.add(parent_code, new_code, "Different Name")

    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        # Attempt to re-add 1001 but under a different parent
        coa.add(Code("2"), child_code, child_name)
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_COA_nodify():
    coa = COA()
    
    # 1. Test nodify on a RootAccount (top level)
    # The default COA has '1' as Assets (RootAccount)
    root_assets = coa.find(Code("1"))
    node = coa.nodify(root_assets)
    
    assert isinstance(node, COA.Node)
    assert node.account.code == Code("1")
    assert node.account.name == "Assets"
    assert node.children == []

    # 2. Test nodify with a SubAccount (nested)
    # Add a sub-account to Assets
    sub_account = coa.add(Code("1"), Code("1000"), "Liquidity")
    
    # Add a sub-sub-account to Liquidity
    sub_sub_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    
    # Nodify the top level account again
    root_node = coa.nodify(root_assets)
    
    # Verify the tree structure
    # Level 0: Assets
    assert root_node.account.code == Code("1")
    assert len(root_node.children) == 1
    
    # Level 1: Liquidity
    liquidity_node = root_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    
    # Level 2: Bank Account
    bank_node = liquidity_node.children[0]
    assert bank_node.account.code == Code("1001")
    assert bank_node.account.name == "Bank Account"
    assert bank_node.children == []

    # 3. Test nodify on the SubAccount directly
    sub_node = coa.nodify(sub_account)
    assert sub_node.account.code == Code("1000")
    assert len(sub_node.children) == 1
    assert sub_node.children[0].account.code == Code("1001")
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_COA_add():
    coa = COA()
    
    # Test adding a valid sub-account
    parent_code = Code("1")
    new_code = Code("1000")
    new_name = "Liquidity"
    
    new_account = coa.add(parent_code, new_code, new_name)
    
    assert new_account.code == new_code
    assert new_account.name == new_name
    assert new_account.parent.code == parent_code
    assert coa.find(new_code) == new_account
    assert new_account in coa.subaccounts(coa.find(parent_code))

    # Test adding a nested sub-account
    grandchild_code = Code("10001")
    grandchild_name = "Cash"
    grandchild_account = coa.add(new_code, grandchild_code, grandchild_name)
    
    assert grandchild_account.parent.code == new_code
    assert grandchild_account.type == AccountType.ASSETS
    assert grandchild_account in coa.subaccounts(new_account)

    # Test adding an existing account with same parameters (should return existing)
    existing_account = coa.add(parent_code, new_code, new_name)
    assert existing_account is new_account

    # Test adding an existing account with different name (should raise ValueError)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(parent_code, new_code, "Different Name")

    # Test adding an existing account with different parent (should raise ValueError)
    # Note: We need a different parent that exists, e.g., Liabilities (Code "2")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("2"), new_code, new_name)

    # Test adding account where parent is same as code (should raise ValueError)
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(new_code, new_code, "Self Parent")

    # Test adding account where parent does not exist (should raise ValueError)
    with pytest.raises(ValueError, match="Parent account is not \(yet\) defined."):
        coa.add(Code("9999"), Code("99999"), "Ghost Account")
```


