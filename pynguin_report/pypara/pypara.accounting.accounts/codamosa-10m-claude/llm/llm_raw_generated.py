####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    
    # Setup: Create a COA instance
    coa = COA()
    
    # Test 1: Successfully add a sub-account to a root account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.type == AccountType.ASSETS
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Successfully add a nested sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.type == AccountType.ASSETS
    assert bank_account.parent == liquidity
    assert coa.find(Code("1001")) == bank_account
    
    # Test 3: Add to different account type (Liabilities)
    current_liabilities = coa.add(Code("2"), Code("2000"), "Current Liabilities")
    assert current_liabilities.code == Code("2000")
    assert current_liabilities.type == AccountType.LIABILITIES
    assert current_liabilities.parent.code == Code("2")
    
    # Test 4: Return existing account if already added with same properties
    existing = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert existing == liquidity
    assert existing.code == Code("1000")
    assert existing.name == "Liquidity"
    
    # Test 5: Raise error when parent and code are the same
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1000"), Code("1000"), "Same Code")
    
    # Test 6: Raise error when parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined"):
        coa.add(Code("9999"), Code("9998"), "Non-existent Parent")
    
    # Test 7: Raise error when account exists with different properties
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    # Test 8: Raise error when account exists with different parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("2"), Code("1000"), "Liquidity")
    
    # Test 9: Verify subaccounts are properly tracked
    subaccounts_of_1 = coa.subaccounts(coa.find(Code("1")))
    assert liquidity in subaccounts_of_1
    
    subaccounts_of_1000 = coa.subaccounts(coa.find(Code("1000")))
    assert bank_account in subaccounts_of_1000
    
    # Test 10: Add multiple sub-accounts to same parent
    savings_account = coa.add(Code("1000"), Code("1002"), "Savings Account")
    assert savings_account.code == Code("1002")
    assert savings_account.parent == liquidity
    
    subaccounts_of_1000 = coa.subaccounts(liquidity)
    assert len(subaccounts_of_1000) == 2
    assert bank_account in subaccounts_of_1000
    assert savings_account in subaccounts_of_1000


# LLM-generated content at query #2
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    # Setup
    coa = COA()
    
    # Test 1: Add a sub-account to a root account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity.type == AccountType.ASSETS
    assert liquidity.coa == coa
    
    # Test 2: Verify account is in COA
    found = coa.find(Code("1000"))
    assert found is not None
    assert found.code == Code("1000")
    assert found.name == "Liquidity"
    
    # Test 3: Add a sub-account to a sub-account
    bankaccount = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccount.code == Code("1001")
    assert bankaccount.name == "Bank Account"
    assert bankaccount.parent.code == Code("1000")
    assert bankaccount.type == AccountType.ASSETS
    
    # Test 4: Verify sub-accounts are tracked
    subaccounts_of_1 = coa.subaccounts(coa.find(Code("1")))
    assert len(subaccounts_of_1) == 1
    assert subaccounts_of_1[0].code == Code("1000")
    
    subaccounts_of_1000 = coa.subaccounts(coa.find(Code("1000")))
    assert len(subaccounts_of_1000) == 1
    assert subaccounts_of_1000[0].code == Code("1001")
    
    # Test 5: Add account to different account type
    liability = coa.add(Code("2"), Code("2000"), "Payables")
    assert liability.type == AccountType.LIABILITIES
    
    # Test 6: Error when parent and code are same
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1"), Code("1"), "Invalid")
    
    # Test 7: Error when parent doesn't exist
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined"):
        coa.add(Code("9999"), Code("9999"), "Invalid")
    
    # Test 8: Adding same account twice returns existing account if consistent
    liquidity_again = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity_again == liquidity
    assert liquidity_again.code == Code("1000")
    
    # Test 9: Error when adding same code with different name
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    # Test 10: Error when adding same code with different parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("2"), Code("1000"), "Liquidity")
    
    # Test 11: Multiple sub-accounts under same parent
    cash = coa.add(Code("1"), Code("1002"), "Cash")
    receivables = coa.add(Code("1"), Code("1003"), "Receivables")
    
    subaccounts_of_1_all = coa.subaccounts(coa.find(Code("1")))
    assert len(subaccounts_of_1_all) == 3
    codes = [acc.code for acc in subaccounts_of_1_all]
    assert Code("1000") in codes
    assert Code("1002") in codes
    assert Code("1003") in codes


# LLM-generated content at query #3
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify it's callable and returns a COA instance
    assert callable(read_coa)
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call__with_custom_spec():
    """Test ReadChartOfAccounts with custom root specification."""
    custom_spec = {
        AccountType.ASSETS: (Code("100"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("300"), "Custom Equities"),
        AccountType.REVENUES: (Code("400"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("100")) is not None
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    def read_coa() -> COA:
        return COA()
    
    result1 = read_coa()
    result2 = read_coa()
    
    # Both should be COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    
    # They should be different instances
    assert result1 is not result2
    
    # But have the same structure
    assert len(list(result1.accounts)) == len(list(result2.accounts))


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function through the protocol
    result = read_coa()
    
    # Verify the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can find accounts
    assets = result.find(Code("1"))
    assert assets is not None
    assert assets.name == "Assets"
    assert assets.type == AccountType.ASSETS


# LLM-generated content at query #5
#--------------------------

```python
def test_COA___iter__():
    """Test the __iter__ method of COA class."""
    # Create a COA instance with default root accounts
    coa = COA()
    
    # Collect all items from iteration
    items = list(coa)
    
    # Should have 5 default root accounts
    assert len(items) == 5
    
    # Each item should be a tuple of (Code, Account)
    for code, account in items:
        assert isinstance(code, str)
        assert hasattr(account, 'code')
        assert hasattr(account, 'name')
        assert hasattr(account, 'type')
        assert code == account.code
    
    # Verify the default accounts are present in order
    codes = [code for code, _ in items]
    names = [account.name for _, account in items]
    
    assert codes == ['1', '2', '3', '4', '5']
    assert names == ['Assets', 'Liabilities', 'Equities', 'Revenues', 'Expenses']
    
    # Verify account types
    types = [account.type for _, account in items]
    assert types == [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES
    ]


def test_COA___iter__with_subaccounts():
    """Test __iter__ includes both root and sub-accounts."""
    coa = COA()
    
    # Add some sub-accounts
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    # Collect all items from iteration
    items = list(coa)
    
    # Should have 7 accounts total (5 root + 2 sub)
    assert len(items) == 7
    
    # Verify all codes are present
    codes = [code for code, _ in items]
    assert Code("1") in codes
    assert Code("1000") in codes
    assert Code("1001") in codes
    
    # Verify iteration maintains order
    assert codes[:5] == ['1', '2', '3', '4', '5']
    assert codes[5:] == ['1000', '1001']


def test_COA___iter__empty_iteration():
    """Test that iteration can be called multiple times."""
    coa = COA()
    
    # First iteration
    items1 = list(coa)
    
    # Second iteration
    items2 = list(coa)
    
    # Both iterations should produce the same results
    assert len(items1) == len(items2)
    assert items1 == items2


def test_COA___iter__returns_iterator():
    """Test that __iter__ returns an iterator."""
    coa = COA()
    
    # Get the iterator
    iterator = iter(coa)
    
    # Should be able to call next on it
    code1, account1 = next(iterator)
    assert code1 == Code("1")
    assert account1.name == "Assets"
    
    code2, account2 = next(iterator)
    assert code2 == Code("2")
    assert account2.name == "Liabilities"


# LLM-generated content at query #6
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test ReadChartOfAccounts protocol __call__ method."""
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify that the implementation is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    
    # Verify the protocol can be used as a type hint
    def process_coa_reader(reader: ReadChartOfAccounts) -> COA:
        return reader()
    
    coa_result = process_coa_reader(read_coa)
    assert isinstance(coa_result, COA)
    assert coa_result.find(Code("1000")) is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify it's callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all default account types are present
    account_types = {acct.type for acct in accounts}
    assert AccountType.ASSETS in account_types
    assert AccountType.LIABILITIES in account_types
    assert AccountType.EQUITIES in account_types
    assert AccountType.REVENUES in account_types
    assert AccountType.EXPENSES in account_types


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with custom implementation."""
    def custom_read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    result = custom_read_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    def read_coa() -> COA:
        return COA()
    
    result1 = read_coa()
    result2 = read_coa()
    
    # Both should be COA instances but different objects
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert result1 is not result2
    
    # Both should have the same default structure
    assert len(list(result1.accounts)) == len(list(result2.accounts))


# LLM-generated content at query #8
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the reader
    reader = SimpleCOAReader()
    
    # Call the reader and verify it returns a COA instance
    result = reader()
    
    assert isinstance(result, COA)
    assert result is not None


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts.__call__ with custom root specification."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Assets Custom"),
        AccountType.LIABILITIES: (Code("20"), "Liabilities Custom"),
        AccountType.EQUITIES: (Code("30"), "Equities Custom"),
        AccountType.REVENUES: (Code("40"), "Revenues Custom"),
        AccountType.EXPENSES: (Code("50"), "Expenses Custom"),
    }
    
    class CustomCOAReader:
        def __call__(self) -> COA:
            return COA(rootspec=custom_rootspec)
    
    reader = CustomCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("10")).name == "Assets Custom"
    assert result.find(Code("20")).name == "Liabilities Custom"
    assert result.find(Code("30")).name == "Equities Custom"
    assert result.find(Code("40")).name == "Revenues Custom"
    assert result.find(Code("50")).name == "Expenses Custom"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts.__call__ can be invoked multiple times."""
    
    class MultiCallCOAReader:
        def __call__(self) -> COA:
            return COA()
    
    reader = MultiCallCOAReader()
    
    # Call multiple times
    result1 = reader()
    result2 = reader()
    
    # Both should be valid COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    
    # They should be different instances
    assert result1 is not result2
    
    # But have the same structure
    assert list(result1.accounts) != list(result2.accounts)  # Different instances
    assert len(list(result1.accounts)) == len(list(result2.accounts))


# LLM-generated content at query #9
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call___with_custom_spec():
    """Test ReadChartOfAccounts with custom root specification."""
    custom_spec = {
        AccountType.ASSETS: (Code("100"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("300"), "Custom Equities"),
        AccountType.REVENUES: (Code("400"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Verify both have the same structure
    accounts1 = [acc.code for acc in coa1.accounts]
    accounts2 = [acc.code for acc in coa2.accounts]
    assert accounts1 == accounts2


# LLM-generated content at query #10
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        """Concrete implementation that returns a COA instance."""
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).type == AccountType.EXPENSES
    
    # Test that multiple calls work correctly
    result2 = read_coa()
    assert isinstance(result2, COA)
    assert result2.find(Code("1")) is not None
    
    # Verify that different instances are created
    assert result is not result2


# LLM-generated content at query #11
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"
    
    # Verify the callable conforms to ReadChartOfAccounts protocol
    assert callable(read_coa)
    
    # Test multiple calls return independent COA instances
    result2 = read_coa()
    assert isinstance(result2, COA)
    assert result is not result2


# LLM-generated content at query #12
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("1")).type == AccountType.ASSETS
    
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("4")).type == AccountType.REVENUES
    
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__custom_implementation():
    """Test ReadChartOfAccounts protocol with custom implementation."""
    
    # Define a custom implementation that returns a COA with custom root accounts
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "My Assets"),
        AccountType.LIABILITIES: (Code("20"), "My Liabilities"),
        AccountType.EQUITIES: (Code("30"), "My Equities"),
        AccountType.REVENUES: (Code("40"), "My Revenues"),
        AccountType.EXPENSES: (Code("50"), "My Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    # Call the function and verify it returns the custom COA
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    assert result.find(Code("10")).name == "My Assets"
    assert result.find(Code("20")).name == "My Liabilities"
    assert result.find(Code("30")).name == "My Equities"
    assert result.find(Code("40")).name == "My Revenues"
    assert result.find(Code("50")).name == "My Expenses"


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the accounts have the correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts_protocol_compliance():
    """Test that ReadChartOfAccounts protocol is properly defined and callable."""
    
    # Define multiple implementations to test protocol flexibility
    def read_coa_simple() -> COA:
        return COA()
    
    def read_coa_custom() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    # Both should be valid implementations
    assert callable(read_coa_simple)
    assert callable(read_coa_custom)
    
    # Both should return COA instances
    coa1 = read_coa_simple()
    coa2 = read_coa_custom()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # coa2 should have additional accounts
    assert coa2.find(Code("1000")) is not None
    assert coa1.find(Code("1000")) is None


def test_ReadChartOfAccounts_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1100"), "Checking Account")
        return coa
    
    # Call multiple times
    coa_instance1 = read_coa()
    coa_instance2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(coa_instance1, COA)
    assert isinstance(coa_instance2, COA)
    
    # They should be different instances
    assert coa_instance1 is not coa_instance2
    
    # Both should have the added account
    assert coa_instance1.find(Code("1100")) is not None
    assert coa_instance2.find(Code("1100")) is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result is not None
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all root account types are present
    account_types = {acc.type for acc in accounts}
    assert AccountType.ASSETS in account_types
    assert AccountType.LIABILITIES in account_types
    assert AccountType.EQUITIES in account_types
    assert AccountType.REVENUES in account_types
    assert AccountType.EXPENSES in account_types


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both calls succeeded
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts with custom root specification."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    
    # Verify custom codes are used
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("30")).name == "Custom Equities"
    assert result.find(Code("40")).name == "Custom Revenues"
    assert result.find(Code("50")).name == "Custom Expenses"


# LLM-generated content at query #15
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("1")).type == AccountType.ASSETS
    
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("4")).type == AccountType.REVENUES
    
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that modifies the COA."""
    
    def read_coa_custom() -> COA:
        coa = COA()
        # Add custom accounts
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa_custom, ReadChartOfAccounts)
    
    # Call and verify custom accounts are present
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    liquidity = result.find(Code("1000"))
    assert liquidity is not None
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    
    bank_account = result.find(Code("1001"))
    assert bank_account is not None
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.code == Code("1000")


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    call_count = 0
    
    def read_coa_tracked() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    assert isinstance(read_coa_tracked, ReadChartOfAccounts)
    
    # Call multiple times
    result1 = read_coa_tracked()
    result2 = read_coa_tracked()
    result3 = read_coa_tracked()
    
    # Verify each call returns a new COA instance
    assert call_count == 3
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #16
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify it implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the default root accounts
    accounts = list(result)
    assert len(accounts) == 5
    assert accounts[0][0] == Code("1")
    assert accounts[1][0] == Code("2")
    assert accounts[2][0] == Code("3")
    assert accounts[3][0] == Code("4")
    assert accounts[4][0] == Code("5")


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test that ReadChartOfAccounts can return a customized COA."""
    
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    bank_account = result.find(Code("1001"))
    assert bank_account is not None
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.name == "Liquidity"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    assert call_count == 3
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    # Each call should return a new instance
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #17
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected default accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    
    # Define a concrete implementation with custom root spec
    def read_custom_coa() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("100"), "Assets Custom"),
            AccountType.LIABILITIES: (Code("200"), "Liabilities Custom"),
            AccountType.EQUITIES: (Code("300"), "Equities Custom"),
            AccountType.REVENUES: (Code("400"), "Revenues Custom"),
            AccountType.EXPENSES: (Code("500"), "Expenses Custom"),
        }
        return COA(rootspec=rootspec)
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("100")).name == "Assets Custom"
    assert result.find(Code("200")).name == "Liabilities Custom"
    assert result.find(Code("300")).name == "Equities Custom"
    assert result.find(Code("400")).name == "Revenues Custom"
    assert result.find(Code("500")).name == "Expenses Custom"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    
    # Add account to first COA
    liquidity = coa1.add(Code("1"), Code("1000"), "Liquidity")
    
    # Verify second COA is unaffected
    assert coa2.find(Code("1000")) is None
    assert coa1.find(Code("1000")) is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected content
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___basic_coa():
    """Test ReadChartOfAccounts with a simple COA returning function."""
    
    def simple_reader() -> COA:
        return COA()
    
    coa = simple_reader()
    
    # Verify basic root accounts exist
    assert coa.find(Code("1")) is not None  # Assets
    assert coa.find(Code("2")) is not None  # Liabilities
    assert coa.find(Code("3")) is not None  # Equities
    assert coa.find(Code("4")) is not None  # Revenues
    assert coa.find(Code("5")) is not None  # Expenses


def test_ReadChartOfAccounts___call___complex_coa():
    """Test ReadChartOfAccounts with a complex COA structure."""
    
    def complex_reader() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        coa.add(Code("2"), Code("2000"), "Short-term Debt")
        return coa
    
    coa = complex_reader()
    
    # Verify structure
    assert coa.find(Code("1000")).name == "Liquidity"
    assert coa.find(Code("1001")).name == "Bank Account"
    assert coa.find(Code("1001")).parent.name == "Liquidity"
    assert coa.find(Code("2000")).name == "Short-term Debt"


def test_ReadChartOfAccounts___call___protocol_compliance():
    """Test that any callable returning COA satisfies ReadChartOfAccounts protocol."""
    
    class CoaFactory:
        def __call__(self) -> COA:
            return COA()
    
    factory = CoaFactory()
    
    # Should be callable and return COA
    result = factory()
    assert isinstance(result, COA)


# LLM-generated content at query #19
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify the returned COA is functional
    assets = result.find(Code("1"))
    assert assets is not None
    assert assets.name == "Assets"
    assert assets.type == AccountType.ASSETS


# LLM-generated content at query #20
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the expected default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify it's a valid ReadChartOfAccounts implementation
    assert callable(read_coa)
    
    # Test multiple calls return valid COA instances
    result2 = read_coa()
    assert isinstance(result2, COA)
    assert len(list(result2.accounts)) == 5


# LLM-generated content at query #21
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    def read_coa() -> COA:
        """Sample implementation of ReadChartOfAccounts."""
        return COA()
    
    # Verify the callable conforms to the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify root account types and names
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).type == AccountType.EXPENSES
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times with consistent results."""
    
    def read_coa() -> COA:
        """Sample implementation of ReadChartOfAccounts."""
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Both should have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name
    assert len(list(coa1.toplevel)) == len(list(coa2.toplevel))


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    
    def read_coa_custom() -> COA:
        """Custom implementation of ReadChartOfAccounts with modified root accounts."""
        custom_rootspec = {
            AccountType.ASSETS: (Code("A"), "My Assets"),
            AccountType.LIABILITIES: (Code("B"), "My Liabilities"),
            AccountType.EQUITIES: (Code("C"), "My Equities"),
            AccountType.REVENUES: (Code("D"), "My Revenues"),
            AccountType.EXPENSES: (Code("E"), "My Expenses"),
        }
        return COA(rootspec=custom_rootspec)
    
    # Call and verify custom implementation
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify custom codes and names
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("B")).name == "My Liabilities"
    assert result.find(Code("C")).name == "My Equities"
    assert result.find(Code("D")).name == "My Revenues"
    assert result.find(Code("E")).name == "My Expenses"


# LLM-generated content at query #22
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test the __call__ method of ReadChartOfAccounts protocol."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify that read_coa is callable and conforms to ReadChartOfAccounts protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    
    # Test another implementation
    def read_empty_coa() -> COA:
        return COA()
    
    assert callable(read_empty_coa)
    result2 = read_empty_coa()
    assert isinstance(result2, COA)
    
    # Verify empty COA has default root accounts
    assert result2.find(Code("1")) is not None
    assert result2.find(Code("1")).type == AccountType.ASSETS
    assert result2.find(Code("2")) is not None
    assert result2.find(Code("2")).type == AccountType.LIABILITIES
    assert result2.find(Code("3")) is not None
    assert result2.find(Code("3")).type == AccountType.EQUITIES
    assert result2.find(Code("4")) is not None
    assert result2.find(Code("4")).type == AccountType.REVENUES
    assert result2.find(Code("5")) is not None
    assert result2.find(Code("5")).type == AccountType.EXPENSES


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify that the function implements the ReadChartOfAccounts protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the returned COA has the correct account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts protocol with a custom COA configuration."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("E"), "Custom Equities"),
        AccountType.REVENUES: (Code("R"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")).name == "Custom Liabilities"
    assert result.find(Code("E")).name == "Custom Equities"
    assert result.find(Code("R")).name == "Custom Revenues"
    assert result.find(Code("X")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_tracking() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_tracking()
    result2 = read_coa_with_tracking()
    result3 = read_coa_with_tracking()
    
    # Verify all calls returned COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    
    # Verify call count
    assert call_count == 3
    
    # Verify each instance is independent
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #24
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it conforms to the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify root accounts are of correct types
    account_types = {acc.type for acc in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    
    # Create a more complex implementation
    def read_custom_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    bank_account = result.find(Code("1001"))
    assert bank_account is not None
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.name == "Liquidity"


def test_ReadChartOfAccounts___call___callable():
    """Test that ReadChartOfAccounts is callable."""
    
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Verify calling it returns COA
    coa = read_coa()
    assert isinstance(coa, COA)


# LLM-generated content at query #25
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify account types are correct
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_implementation():
    """
    Test ReadChartOfAccounts with a custom implementation that returns a customized COA.
    """
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Bank")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Bank"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    call_count = 0
    
    def read_coa_with_count() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_count()
    result2 = read_coa_with_count()
    
    # Verify both calls succeeded and returned COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert call_count == 2


# LLM-generated content at query #26
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts __call__ can be invoked multiple times."""
    
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    # Each call should return a different instance
    assert coa1 is not coa2


def test_ReadChartOfAccounts___call___returns_coa_with_defaults():
    """Test that ReadChartOfAccounts __call__ returns COA with default root accounts."""
    
    def read_coa() -> COA:
        return COA()
    
    coa = read_coa()
    
    # Verify all default root accounts are present
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("1")).type == AccountType.ASSETS
    
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("2")).type == AccountType.LIABILITIES
    
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("3")).type == AccountType.EQUITIES
    
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("4")).type == AccountType.REVENUES
    
    assert coa.find(Code("5")) is not None
    assert coa.find(Code("5")).name == "Expenses"
    assert coa.find(Code("5")).type == AccountType.EXPENSES


# LLM-generated content at query #27
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable and conforms to the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    root_accounts = list(result.toplevel)
    assert len(root_accounts) == 5
    
    # Verify each root account has the correct type
    account_types = [account.type for account in root_accounts]
    assert AccountType.ASSETS in account_types
    assert AccountType.LIABILITIES in account_types
    assert AccountType.EQUITIES in account_types
    assert AccountType.REVENUES in account_types
    assert AccountType.EXPENSES in account_types


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts protocol with a custom COA configuration."""
    
    # Define a concrete implementation with custom root spec
    def read_custom_coa() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "Assets Custom"),
            AccountType.LIABILITIES: (Code("20"), "Liabilities Custom"),
            AccountType.EQUITIES: (Code("30"), "Equities Custom"),
            AccountType.REVENUES: (Code("40"), "Revenues Custom"),
            AccountType.EXPENSES: (Code("50"), "Expenses Custom"),
        }
        return COA(rootspec=rootspec)
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Assets Custom"
    assert result.find(Code("20")).name == "Liabilities Custom"
    assert result.find(Code("30")).name == "Equities Custom"
    assert result.find(Code("40")).name == "Revenues Custom"
    assert result.find(Code("50")).name == "Expenses Custom"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    call_count = 0
    
    def read_coa_with_tracking() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_tracking()
    coa2 = read_coa_with_tracking()
    
    # Verify both calls succeeded and returned distinct instances
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2


# LLM-generated content at query #28
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation conforms to the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    assert all(isinstance(a, Account) for a in accounts)


def test_ReadChartOfAccounts___call___with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA initialization."""
    
    def read_custom_coa() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Custom Equities"),
            AccountType.REVENUES: (Code("40"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
        }
        return COA(rootspec=rootspec)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("30")).name == "Custom Equities"
    assert result.find(Code("40")).name == "Custom Revenues"
    assert result.find(Code("50")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = [0]
    
    def read_coa_with_tracking() -> COA:
        call_count[0] += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_tracking()
    result2 = read_coa_with_tracking()
    result3 = read_coa_with_tracking()
    
    assert call_count[0] == 3
    assert all(isinstance(r, COA) for r in [result1, result2, result3])
    # Each call should produce independent COA instances
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #29
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance and call it
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    # Verify the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify root accounts have correct types and names
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).type == AccountType.EXPENSES
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class ConcreteReadChartOfAccounts:
        def __init__(self):
            self.call_count = 0
        
        def __call__(self) -> COA:
            self.call_count += 1
            return COA()
    
    reader = ConcreteReadChartOfAccounts()
    
    # Call multiple times
    result1 = reader()
    result2 = reader()
    
    assert reader.call_count == 2
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    # Each call should return a different COA instance
    assert result1 is not result2


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts returning COA with custom rootspec."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "My Assets"),
        AccountType.LIABILITIES: (Code("20"), "My Liabilities"),
        AccountType.EQUITIES: (Code("30"), "My Equities"),
        AccountType.REVENUES: (Code("40"), "My Revenues"),
        AccountType.EXPENSES: (Code("50"), "My Expenses"),
    }
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA(rootspec=custom_rootspec)
    
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    # Verify custom codes and names
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "My Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "My Liabilities"
    assert result.find(Code("30")) is not None
    assert result.find(Code("30")).name == "My Equities"
    assert result.find(Code("40")) is not None
    assert result.find(Code("40")).name == "My Revenues"
    assert result.find(Code("50")) is not None
    assert result.find(Code("50")).name == "My Expenses"


# LLM-generated content at query #30
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance and verify it returns a COA
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that returns a customized COA."""
    
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            # Add custom accounts
            liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
            coa.add(liquidity.code, Code("1001"), "Bank Account")
            return coa
    
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")).name == "Bank Account"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class MultiCallReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = MultiCallReadChartOfAccounts()
    
    # Call multiple times
    result1 = reader()
    result2 = reader()
    
    # Both should be COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    
    # They should be different instances
    assert result1 is not result2
    
    # But have the same structure
    assert result1.find(Code("1")).name == result2.find(Code("1")).name
    assert result1.find(Code("5")).name == result2.find(Code("5")).name


# LLM-generated content at query #31
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    # Create a simple implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Call the function
    result = read_coa()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("1")).type == AccountType.ASSETS
    
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("4")).type == AccountType.REVENUES
    
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test that ReadChartOfAccounts.__call__ can return a custom COA."""
    # Create a custom implementation of ReadChartOfAccounts
    def read_custom_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    # Call the function
    result = read_custom_coa()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.code == Code("1000")


# LLM-generated content at query #32
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts protocol
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance and verify it returns a COA object
    result = reader()
    
    # Assert that the result is an instance of COA
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES
    
    # Verify the protocol is runtime checkable
    assert callable(reader)


# LLM-generated content at query #33
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can retrieve accounts from the returned COA
    assets = result.find(Code("1"))
    assert assets is not None
    assert assets.name == "Assets"
    assert assets.type == AccountType.ASSETS


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts with custom root specification."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("1")) is None


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # But have the same structure
    codes1 = [code for code, _ in coa1]
    codes2 = [code for code, _ in coa2]
    assert codes1 == codes2
    
    # Modify one COA and verify the other is not affected
    coa1.add(Code("1"), Code("1000"), "Test Account")
    assert coa1.find(Code("1000")) is not None
    assert coa2.find(Code("1000")) is None


# LLM-generated content at query #34
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that returns modified COA."""
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Custom Account")
        return coa
    
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom account was added
    custom_account = result.find(Code("1000"))
    assert custom_account is not None
    assert custom_account.name == "Custom Account"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    def read_coa() -> COA:
        return COA()
    
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call multiple times
    result1 = read_coa()
    result2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    
    # They should be different instances
    assert result1 is not result2
    
    # But with the same structure
    assert result1.find(Code("1")).name == result2.find(Code("1")).name


# LLM-generated content at query #35
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Create an instance that satisfies the ReadChartOfAccounts protocol
    reader: ReadChartOfAccounts = read_coa
    
    # Call the reader and verify it returns a COA instance
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
        AccountType.EQUITIES: (Code("E"), "My Equities"),
        AccountType.REVENUES: (Code("R"), "My Revenues"),
        AccountType.EXPENSES: (Code("X"), "My Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    reader: ReadChartOfAccounts = read_custom_coa
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")).name == "My Liabilities"
    assert result.find(Code("E")).name == "My Equities"
    assert result.find(Code("R")).name == "My Revenues"
    assert result.find(Code("X")).name == "My Expenses"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def counting_read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    reader: ReadChartOfAccounts = counting_read_coa
    
    result1 = reader()
    assert call_count == 1
    assert isinstance(result1, COA)
    
    result2 = reader()
    assert call_count == 2
    assert isinstance(result2, COA)
    
    # Each call should return a different COA instance
    assert result1 is not result2


# LLM-generated content at query #36
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """Test ReadChartOfAccounts with custom implementation."""
    
    # Define a custom implementation that adds accounts
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    # Call the function
    result = read_custom_coa()
    
    # Verify it returns a COA with custom accounts
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts callable can be invoked multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    result3 = read_coa_with_counter()
    
    # Verify each call created a new COA instance
    assert call_count == 3
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #37
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the COA contains the expected accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call__with_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    # Each call should return a new instance
    assert coa1 is not coa2


def test_ReadChartOfAccounts___call__returns_coa_with_root_accounts():
    """Test that ReadChartOfAccounts returns a COA with root accounts."""
    
    def read_coa() -> COA:
        return COA()
    
    coa = read_coa()
    
    # Verify all root accounts are present
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None
    
    # Verify account names
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #38
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can retrieve accounts from the returned COA
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"


def test_ReadChartOfAccounts___call__with_lambda():
    """
    Test ReadChartOfAccounts protocol with lambda implementation.
    """
    reader: ReadChartOfAccounts = lambda: COA()
    
    result = reader()
    assert isinstance(result, COA)
    assert result.find(Code("1")) is not None


def test_ReadChartOfAccounts___call__multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times independently.
    """
    def create_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    reader: ReadChartOfAccounts = create_coa
    
    # First call
    coa1 = reader()
    assert coa1.find(Code("1000")) is not None
    
    # Second call - should be independent
    coa2 = reader()
    assert coa2.find(Code("1000")) is not None
    assert coa1 is not coa2


# LLM-generated content at query #39
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify that the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("1")).type == AccountType.ASSETS
    
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("4")).type == AccountType.REVENUES
    
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"
    assert result.find(Code("5")).type == AccountType.EXPENSES
    
    # Test that multiple calls to the protocol work correctly
    result2 = read_coa()
    assert isinstance(result2, COA)
    assert result2.find(Code("1")).name == "Assets"
    
    # Verify protocol compatibility
    def another_read_coa() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("A"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("E"), "Custom Equities"),
            AccountType.REVENUES: (Code("R"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
        }
        return COA(rootspec=custom_rootspec)
    
    result3 = another_read_coa()
    assert isinstance(result3, COA)
    assert result3.find(Code("A")) is not None
    assert result3.find(Code("A")).name == "Custom Assets"


# LLM-generated content at query #40
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test ReadChartOfAccounts protocol __call__ method."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            return coa
    
    # Test that the protocol can be called
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    # Verify result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    
    # Test that __call__ returns COA with default root accounts
    reader2 = ConcreteReadChartOfAccounts()
    result2 = reader2()
    
    assert result2.find(Code("1")) is not None
    assert result2.find(Code("2")) is not None
    assert result2.find(Code("3")) is not None
    assert result2.find(Code("4")) is not None
    assert result2.find(Code("5")) is not None
    
    # Test multiple calls return independent COA instances
    result3 = reader()
    result4 = reader()
    
    assert result3 is not result4
    assert result3.find(Code("1")) is not None
    assert result4.find(Code("1")) is not None
    
    # Test that the protocol is runtime checkable
    assert callable(reader)
    
    # Test with lambda implementation
    lambda_reader: ReadChartOfAccounts = lambda: COA()
    lambda_result = lambda_reader()
    
    assert isinstance(lambda_result, COA)
    assert lambda_result.find(Code("1")) is not None


# LLM-generated content at query #41
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function
    result = read_coa()
    
    # Assert it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_rootspec():
    """
    Test ReadChartOfAccounts protocol with custom rootspec.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("E"), "Custom Equities"),
        AccountType.REVENUES: (Code("R"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    
    assert isinstance(result, COA)
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")).name == "Custom Liabilities"
    assert result.find(Code("E")).name == "Custom Equities"
    assert result.find(Code("R")).name == "Custom Revenues"
    assert result.find(Code("X")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times independently.
    """
    def read_coa() -> COA:
        return COA()
    
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name
    assert coa1.find(Code("5")).type == coa2.find(Code("5")).type


# LLM-generated content at query #42
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance and call it
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    # Verify the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the default 5 root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify the account types are in the expected order
    expected_types = [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]
    actual_types = [account.type for account in accounts]
    assert actual_types == expected_types


def test_ReadChartOfAccounts_protocol_compliance():
    """Test that ReadChartOfAccounts protocol is correctly implemented."""
    
    def create_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(create_coa)
    
    # Call the function and verify it returns a COA
    coa = create_coa()
    assert isinstance(coa, COA)


def test_ReadChartOfAccounts_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class MultiCallReader:
        def __init__(self):
            self.call_count = 0
        
        def __call__(self) -> COA:
            self.call_count += 1
            return COA()
    
    reader = MultiCallReader()
    
    # Call multiple times
    coa1 = reader()
    coa2 = reader()
    coa3 = reader()
    
    # Verify all returns are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    assert coa2 is not coa3
    
    # Verify call count
    assert reader.call_count == 3


# LLM-generated content at query #43
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    
    # Setup: Create a chart of accounts
    coa = COA()
    
    # Test 1: Add a sub-account under Assets (code "1")
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity.parent.name == "Assets"
    assert liquidity.type == AccountType.ASSETS
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Add a sub-account under the previously created sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.code == Code("1000")
    assert bank_account.parent.name == "Liquidity"
    assert bank_account.type == AccountType.ASSETS
    assert coa.find(Code("1001")) == bank_account
    
    # Test 3: Verify parent-child relationships
    assert coa.subaccounts(liquidity) == [bank_account]
    assert coa.subaccounts(coa.find(Code("1"))) == [liquidity]
    
    # Test 4: Add multiple sub-accounts under the same parent
    receivables = coa.add(Code("1"), Code("1100"), "Receivables")
    assert receivables.code == Code("1100")
    assert coa.subaccounts(coa.find(Code("1"))) == [liquidity, receivables]
    
    # Test 5: Try to add account where parent == code (should raise ValueError)
    try:
        coa.add(Code("2000"), Code("2000"), "Invalid Account")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "can not be the parent of itself" in str(e)
    
    # Test 6: Try to add account with non-existent parent (should raise ValueError)
    try:
        coa.add(Code("9999"), Code("3000"), "Invalid Parent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Parent account is not (yet) defined" in str(e)
    
    # Test 7: Add same account twice with matching details (should return existing account)
    liquidity_again = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity_again == liquidity
    assert liquidity_again.code == Code("1000")
    
    # Test 8: Try to add same code with different parent (should raise ValueError)
    try:
        coa.add(Code("2"), Code("1000"), "Liquidity")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "do not match existing chart of accounts member" in str(e)
    
    # Test 9: Try to add same code with different name (should raise ValueError)
    try:
        coa.add(Code("1"), Code("1000"), "Different Name")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "do not match existing chart of accounts member" in str(e)
    
    # Test 10: Add accounts under different account types
    debt = coa.add(Code("2"), Code("2000"), "Long-term Debt")
    assert debt.code == Code("2000")
    assert debt.type == AccountType.LIABILITIES
    
    equity_capital = coa.add(Code("3"), Code("3000"), "Capital")
    assert equity_capital.code == Code("3000")
    assert equity_capital.type == AccountType.EQUITIES
    
    # Test 11: Verify all accounts are accessible via find
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("1000")).name == "Liquidity"
    assert coa.find(Code("1001")).name == "Bank Account"
    assert coa.find(Code("2000")).name == "Long-term Debt"
    assert coa.find(Code("3000")).name == "Capital"


# LLM-generated content at query #44
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test the __call__ method of ReadChartOfAccounts protocol."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            return coa
    
    # Instantiate the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the __call__ method
    result = reader()
    
    # Assert that the result is a COA instance
    assert isinstance(result, COA)
    
    # Assert that the COA contains expected accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    
    # Assert that all root accounts are present
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___empty_coa():
    """Test __call__ method returning an empty COA with only root accounts."""
    
    class EmptyReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = EmptyReadChartOfAccounts()
    result = reader()
    
    # Assert that the result is a COA instance
    assert isinstance(result, COA)
    
    # Assert that only root accounts exist
    assert len(list(result.accounts)) == 5
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___complex_structure():
    """Test __call__ method with a complex account structure."""
    
    class ComplexReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
            coa.add(liquidity.code, Code("1001"), "Bank Account")
            coa.add(liquidity.code, Code("1002"), "Cash")
            expenses = coa.add(Code("5"), Code("5000"), "Operating Expenses")
            coa.add(expenses.code, Code("5001"), "Salaries")
            return coa
    
    reader = ComplexReadChartOfAccounts()
    result = reader()
    
    # Assert that the result is a COA instance
    assert isinstance(result, COA)
    
    # Assert hierarchy structure
    assert result.find(Code("1001")).parent.code == Code("1000")
    assert result.find(Code("1002")).parent.code == Code("1000")
    assert result.find(Code("5001")).parent.code == Code("5000")
    
    # Assert account types are inherited
    assert result.find(Code("1001")).type == AccountType.ASSETS
    assert result.find(Code("5001")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___callable():
    """Test that ReadChartOfAccounts is callable."""
    
    def simple_reader() -> COA:
        return COA()
    
    # The function should be callable and return a COA
    result = simple_reader()
    assert isinstance(result, COA)


# LLM-generated content at query #45
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both calls succeeded and returned COA instances
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    # Verify they are different instances
    assert coa1 is not coa2


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """
    Test that ReadChartOfAccounts can return COA with custom root specification.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("100"), "Current Assets"),
        AccountType.LIABILITIES: (Code("200"), "Current Liabilities"),
        AccountType.EQUITIES: (Code("300"), "Owner Equity"),
        AccountType.REVENUES: (Code("400"), "Operating Revenues"),
        AccountType.EXPENSES: (Code("500"), "Operating Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    assert isinstance(result, COA)
    assert result.find(Code("100")).name == "Current Assets"
    assert result.find(Code("200")).name == "Current Liabilities"
    assert result.find(Code("300")).name == "Owner Equity"
    assert result.find(Code("400")).name == "Operating Revenues"
    assert result.find(Code("500")).name == "Operating Expenses"


# LLM-generated content at query #46
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it matches the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result)
    assert len(accounts) == 5
    
    codes = [code for code, _ in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("100"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("300"), "Custom Equities"),
        AccountType.REVENUES: (Code("400"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    # Verify it matches the protocol
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    # Call and verify result
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    assert isinstance(read_coa_with_counter, ReadChartOfAccounts)
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #47
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts protocol
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Assert the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Assert the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___with_custom_coa():
    """
    Test that ReadChartOfAccounts protocol can return custom COA instances.
    """
    # Define a custom implementation
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1100"), "Cash")
            return coa
    
    # Create and call the reader
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    # Assert the result contains the custom account
    assert result.find(Code("1100")) is not None
    assert result.find(Code("1100")).name == "Cash"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    class CountingReadChartOfAccounts:
        def __init__(self):
            self.call_count = 0
        
        def __call__(self) -> COA:
            self.call_count += 1
            return COA()
    
    reader = CountingReadChartOfAccounts()
    
    # Call multiple times
    result1 = reader()
    result2 = reader()
    result3 = reader()
    
    # Assert all results are COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    
    # Assert call count is correct
    assert reader.call_count == 3
    
    # Assert each call returns a different COA instance
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #48
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test the __call__ method of ReadChartOfAccounts protocol."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    class SimpleChartOfAccountsReader:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            return coa
    
    # Instantiate the reader
    reader = SimpleChartOfAccountsReader()
    
    # Call the reader
    coa = reader()
    
    # Verify that it returns a COA instance
    assert isinstance(coa, COA)
    
    # Verify that the COA contains the expected accounts
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("1000")).name == "Liquidity"
    
    # Verify that the returned COA is callable multiple times
    coa2 = reader()
    assert isinstance(coa2, COA)
    assert coa2.find(Code("1000")) is not None


def test_ReadChartOfAccounts_protocol_compliance():
    """Test that ReadChartOfAccounts protocol is properly defined."""
    
    def custom_reader() -> COA:
        return COA()
    
    # Verify the callable signature matches the protocol
    coa = custom_reader()
    assert isinstance(coa, COA)
    
    # Verify default accounts are initialized
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("2")).name == "Liabilities"


def test_ReadChartOfAccounts_with_complex_structure():
    """Test ReadChartOfAccounts with a more complex COA structure."""
    
    def complex_reader() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        coa.add(Code("4"), Code("4000"), "Sales")
        return coa
    
    reader = complex_reader
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.name == "Liquidity"
    assert result.find(Code("4000")) is not None
    assert result.find(Code("4000")).name == "Sales"


# LLM-generated content at query #49
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the returned COA has the correct account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA implementation."""
    
    # Define a custom implementation that creates a COA with additional accounts
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA with custom accounts
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1000")).parent.code == Code("1")


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times and verify each returns a new COA instance
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2  # Different instances
    
    # Both should have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


# LLM-generated content at query #50
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it's callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test that ReadChartOfAccounts protocol works with custom COA implementations."""
    # Create a concrete implementation that returns a customized COA
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify it's callable
    assert callable(read_custom_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify the custom account was added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    result3 = read_coa_with_counter()
    
    # Verify all calls returned COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    
    # Verify the function was called the expected number of times
    assert call_count == 3


# LLM-generated content at query #51
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


# LLM-generated content at query #52
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_implementation():
    """
    Test ReadChartOfAccounts protocol with a custom implementation.
    """
    # Define a custom implementation that returns a COA with custom root accounts
    def read_custom_coa() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("A"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("E"), "Custom Equities"),
            AccountType.REVENUES: (Code("R"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
        }
        return COA(rootspec=custom_spec)
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")).name == "Custom Liabilities"
    assert result.find(Code("E")).name == "Custom Equities"
    assert result.find(Code("R")).name == "Custom Revenues"
    assert result.find(Code("X")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    # Verify all calls were made
    assert call_count == 3
    
    # Verify all results are valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #53
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify it implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call__returns_coa():
    """Test that calling ReadChartOfAccounts returns a valid COA object."""
    def simple_coa_reader() -> COA:
        return COA()
    
    coa = simple_coa_reader()
    
    # Verify basic COA functionality
    assert coa.find(Code("1")).type == AccountType.ASSETS
    assert coa.find(Code("2")).type == AccountType.LIABILITIES
    assert coa.find(Code("3")).type == AccountType.EQUITIES
    assert coa.find(Code("4")).type == AccountType.REVENUES
    assert coa.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_accounts():
    """Test ReadChartOfAccounts protocol with custom account setup."""
    def custom_coa_reader() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        coa.add(Code("1"), Code("1100"), "Fixed Assets")
        return coa
    
    coa = custom_coa_reader()
    
    # Verify custom accounts were added
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("1001")) is not None
    assert coa.find(Code("1100")) is not None
    assert coa.find(Code("1001")).parent.name == "Liquidity"


# LLM-generated content at query #54
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify the returned COA contains all core account types
    account_types = {account.type for account in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Both should be independent instances
    assert coa1 is not coa2
    
    # Both should have the same structure
    assert len(list(coa1.accounts)) == len(list(coa2.accounts))


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test that ReadChartOfAccounts can return COA with custom rootspec."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "My Assets"),
        AccountType.LIABILITIES: (Code("20"), "My Liabilities"),
        AccountType.EQUITIES: (Code("30"), "My Equities"),
        AccountType.REVENUES: (Code("40"), "My Revenues"),
        AccountType.EXPENSES: (Code("50"), "My Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "My Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "My Liabilities"
    assert result.find(Code("30")) is not None
    assert result.find(Code("30")).name == "My Equities"
    assert result.find(Code("40")) is not None
    assert result.find(Code("40")).name == "My Revenues"
    assert result.find(Code("50")) is not None
    assert result.find(Code("50")).name == "My Expenses"


# LLM-generated content at query #55
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable and returns a COA
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can retrieve accounts from the returned COA
    assets = result.find(Code("1"))
    assert assets is not None
    assert assets.name == "Assets"
    assert assets.type == AccountType.ASSETS


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times and verify each returns a valid COA
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2  # Different instances
    
    # Both should have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """
    Test that ReadChartOfAccounts can be implemented with custom root specifications.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes and names
    assets = result.find(Code("10"))
    assert assets is not None
    assert assets.name == "Custom Assets"
    
    liabilities = result.find(Code("20"))
    assert liabilities is not None
    assert liabilities.name == "Custom Liabilities"


# LLM-generated content at query #56
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function
    result = read_coa()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_implementation():
    """
    Test that ReadChartOfAccounts protocol works with custom implementations.
    """
    # Define a custom implementation
    def custom_read_coa() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("A"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("E"), "Custom Equities"),
            AccountType.REVENUES: (Code("R"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
        }
        return COA(rootspec=custom_spec)
    
    # Call the function
    result = custom_read_coa()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("A")) is not None
    assert result.find(Code("L")) is not None
    assert result.find(Code("E")) is not None
    assert result.find(Code("R")) is not None
    assert result.find(Code("X")) is not None
    
    # Verify custom names are used
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")).name == "Custom Liabilities"
    assert result.find(Code("E")).name == "Custom Equities"
    assert result.find(Code("R")).name == "Custom Revenues"
    assert result.find(Code("X")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts protocol can be called multiple times.
    """
    def read_coa_multiple() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa_multiple()
    coa2 = read_coa_multiple()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # But have same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


# LLM-generated content at query #57
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the implementation is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the COA contains expected accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1000")).parent.code == Code("1")


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = [0]
    
    def read_coa() -> COA:
        call_count[0] += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    coa3 = read_coa()
    
    # Verify all calls returned COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify call count
    assert call_count[0] == 3


def test_ReadChartOfAccounts___call___returns_different_instances():
    """Test that ReadChartOfAccounts returns different COA instances on each call."""
    def read_coa() -> COA:
        return COA()
    
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # But they should have the same structure
    codes1 = [code for code, _ in coa1]
    codes2 = [code for code, _ in coa2]
    assert codes1 == codes2


def test_ReadChartOfAccounts___call___complex_coa():
    """Test ReadChartOfAccounts with a more complex COA."""
    def read_complex_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        coa.add(liquidity.code, Code("1002"), "Cash")
        receivables = coa.add(Code("1"), Code("1100"), "Receivables")
        coa.add(receivables.code, Code("1101"), "Accounts Receivable")
        return coa
    
    result = read_complex_coa()
    
    # Verify structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1002")) is not None
    assert result.find(Code("1100")) is not None
    assert result.find(Code("1101")) is not None
    
    # Verify parent-child relationships
    assert result.find(Code("1001")).parent.code == Code("1000")
    assert result.find(Code("1101")).parent.code == Code("1100")


# LLM-generated content at query #58
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the callable implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA contains expected accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___returns_coa():
    """Test that ReadChartOfAccounts returns a valid COA with default accounts."""
    
    def read_default_coa() -> COA:
        return COA()
    
    result = read_default_coa()
    
    # Verify all default root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    # Verify they are different instances
    assert coa1 is not coa2


# LLM-generated content at query #59
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Instantiate the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance and verify it returns a COA
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have the correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """
    Test ReadChartOfAccounts protocol with a custom implementation that returns a customized COA.
    """
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            return coa
    
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1000")).type == AccountType.ASSETS


def test_ReadChartOfAccounts___call__protocol_compliance():
    """
    Test that any callable returning COA complies with ReadChartOfAccounts protocol.
    """
    def read_coa_function() -> COA:
        return COA()
    
    # Verify the function can be used where ReadChartOfAccounts is expected
    reader: ReadChartOfAccounts = read_coa_function
    result = reader()
    
    assert isinstance(result, COA)
    assert len(list(result.accounts)) == 5


# LLM-generated content at query #60
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    
    # Test 1: Add a sub-account to a root account
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.type == AccountType.ASSETS
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Add a nested sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.type == AccountType.ASSETS
    assert bankaccnt.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bankaccnt
    
    # Test 3: Verify sub-accounts are tracked correctly
    assert len(coa.subaccounts(coa.find(Code("1")))) == 1
    assert len(coa.subaccounts(coa.find(Code("1000")))) == 1
    
    # Test 4: Add account to different account type
    liability_sub = coa.add(Code("2"), Code("2000"), "Accounts Payable")
    assert liability_sub.type == AccountType.LIABILITIES
    assert liability_sub.parent.code == Code("2")
    
    # Test 5: Error when parent and code are the same
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1"), Code("1"), "Invalid")
    
    # Test 6: Error when parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not"):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")
    
    # Test 7: Return existing account if it already exists with same properties
    existing = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing == bankaccnt
    
    # Test 8: Error when trying to add existing code with different properties
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("1000"), Code("1001"), "Different Name")
    
    # Test 9: Verify account is in accounts buffer
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("1001")) is not None
    assert coa.find(Code("2000")) is not None
    
    # Test 10: Add multiple accounts to same parent
    savings = coa.add(Code("1"), Code("1002"), "Savings Account")
    assert len(coa.subaccounts(coa.find(Code("1")))) == 2
    assert savings in coa.subaccounts(coa.find(Code("1")))
    assert liquidity in coa.subaccounts(coa.find(Code("1")))


# LLM-generated content at query #61
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify that the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all root account types are present
    account_types = {a.type for a in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


def test_ReadChartOfAccounts_protocol_compliance():
    """
    Test that functions implementing ReadChartOfAccounts protocol are recognized.
    """
    def custom_read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    # The function should be callable and return a COA
    assert callable(custom_read_coa)
    result = custom_read_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None


def test_ReadChartOfAccounts_multiple_implementations():
    """
    Test that multiple different implementations can satisfy the protocol.
    """
    def reader_default() -> COA:
        return COA()
    
    def reader_custom() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "Fixed Assets"),
            AccountType.LIABILITIES: (Code("20"), "Current Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Share Capital"),
            AccountType.REVENUES: (Code("40"), "Operating Revenue"),
            AccountType.EXPENSES: (Code("50"), "Operating Expenses"),
        }
        return COA(rootspec=rootspec)
    
    # Both implementations should be callable and return COA
    coa1 = reader_default()
    coa2 = reader_custom()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Custom reader should have different root codes
    assert coa1.find(Code("1")) is not None
    assert coa2.find(Code("10")) is not None


# LLM-generated content at query #62
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times and returns new instances.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be COA instances but different objects
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """
    Test that ReadChartOfAccounts can return COA with custom root specifications.
    """
    custom_spec = {
        AccountType.ASSETS: (Code("100"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("300"), "Custom Equities"),
        AccountType.REVENUES: (Code("400"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_coa()
    
    # Verify custom codes and names
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"


# LLM-generated content at query #63
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    # Create a custom implementation
    def custom_read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    # Verify it's callable
    assert callable(custom_read_coa)
    
    # Call and verify results
    result = custom_read_coa()
    assert isinstance(result, COA)
    
    # Verify custom account was added
    test_account = result.find(Code("1000"))
    assert test_account is not None
    assert test_account.name == "Test Account"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def counting_read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = counting_read_coa()
    result2 = counting_read_coa()
    result3 = counting_read_coa()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    
    # Verify each call returns independent instances
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #64
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance
    result = reader()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            custom_spec = {
                AccountType.ASSETS: (Code("100"), "Custom Assets"),
                AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
                AccountType.EQUITIES: (Code("300"), "Custom Equities"),
                AccountType.REVENUES: (Code("400"), "Custom Revenues"),
                AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
            }
            return COA(rootspec=custom_spec)
    
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class MultiCallReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = MultiCallReadChartOfAccounts()
    
    coa1 = reader()
    coa2 = reader()
    
    # Both should be valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be independent instances
    assert coa1 is not coa2
    
    # Both should have the default root accounts
    assert coa1.find(Code("1")) is not None
    assert coa2.find(Code("1")) is not None


# LLM-generated content at query #65
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___returns_coa():
    """Test that ReadChartOfAccounts returns a valid COA with root accounts."""
    
    def read_default_coa() -> COA:
        return COA()
    
    result = read_default_coa()
    
    # Verify all root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    
    assert call_count == 2
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    # Each call should return a new instance
    assert result1 is not result2


# LLM-generated content at query #66
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify it matches the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("B"), "My Liabilities"),
        AccountType.EQUITIES: (Code("C"), "My Equities"),
        AccountType.REVENUES: (Code("D"), "My Revenues"),
        AccountType.EXPENSES: (Code("E"), "My Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("A")) is not None
    assert result.find(Code("B")) is not None
    assert result.find(Code("C")) is not None
    assert result.find(Code("D")) is not None
    assert result.find(Code("E")) is not None
    
    # Verify custom names are used
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("B")).name == "My Liabilities"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    assert isinstance(read_coa_with_counter, ReadChartOfAccounts)
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #67
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify it implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    assert accounts[0].type == AccountType.ASSETS
    assert accounts[1].type == AccountType.LIABILITIES
    assert accounts[2].type == AccountType.EQUITIES
    assert accounts[3].type == AccountType.REVENUES
    assert accounts[4].type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that modifies the COA."""
    def read_coa_custom() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify it implements the protocol
    assert isinstance(read_coa_custom, ReadChartOfAccounts)
    
    # Call the function
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify the custom modifications are present
    liquidity_account = result.find(Code("1000"))
    assert liquidity_account is not None
    assert liquidity_account.name == "Liquidity"
    assert liquidity_account.parent.code == Code("1")


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def read_coa_tracked() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    assert isinstance(read_coa_tracked, ReadChartOfAccounts)
    
    # Call multiple times
    result1 = read_coa_tracked()
    result2 = read_coa_tracked()
    result3 = read_coa_tracked()
    
    # Verify each call returns a COA instance
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    
    # Verify call count
    assert call_count == 3
    
    # Verify each COA is independent
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #68
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it conforms to the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify account types are correct
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation returning pre-configured COA."""
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    # Verify it conforms to the protocol
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    # Call and verify custom accounts are present
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")).name == "Bank Account"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Each call should return a new independent COA instance
    assert coa1 is not coa2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Both should have the same default structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name
    assert coa1.find(Code("5")).name == coa2.find(Code("5")).name


# LLM-generated content at query #69
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify we can iterate through the accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that modifies COA."""
    
    def read_custom_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")).name == "Bank Account"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    result1 = read_coa()
    result2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    
    # They should be different instances
    assert result1 is not result2
    
    # But have the same structure
    assert len(list(result1.accounts)) == len(list(result2.accounts))


# LLM-generated content at query #70
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all core account types are present
    account_types = {acc.type for acc in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that adds accounts."""
    
    def read_coa_custom() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    # Call the function
    result = read_coa_custom()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify the custom accounts were added
    liquidity_acct = result.find(Code("1000"))
    assert liquidity_acct is not None
    assert liquidity_acct.name == "Liquidity"
    
    bank_acct = result.find(Code("1001"))
    assert bank_acct is not None
    assert bank_acct.name == "Bank Account"
    assert bank_acct.parent == liquidity_acct


def test_ReadChartOfAccounts___call___protocol_compliance():
    """Test that ReadChartOfAccounts protocol is properly implemented."""
    
    def my_reader() -> COA:
        return COA()
    
    # Verify the function matches the protocol signature
    # A ReadChartOfAccounts callable should return a COA when called with no arguments
    result = my_reader()
    assert isinstance(result, COA)
    
    # Verify we can iterate over the returned COA
    accounts_list = list(result.accounts)
    assert len(accounts_list) > 0


# LLM-generated content at query #71
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Test that multiple calls return independent COA instances
    result2 = read_coa()
    assert result is not result2
    assert isinstance(result2, COA)
    
    # Test that we can add accounts to the returned COA
    new_account = result.add(Code("1"), Code("1000"), "Test Account")
    assert new_account.name == "Test Account"
    assert new_account.code == Code("1000")


# LLM-generated content at query #72
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance and call it
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    # Verify the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts_protocol_runtime_checkable():
    """Test that ReadChartOfAccounts protocol can be checked at runtime."""
    
    class ValidReader:
        def __call__(self) -> COA:
            return COA()
    
    class InvalidReader:
        pass
    
    valid_reader = ValidReader()
    invalid_reader = InvalidReader()
    
    # Verify that valid reader can be called
    assert callable(valid_reader)
    result = valid_reader()
    assert isinstance(result, COA)
    
    # Verify that invalid reader is not callable in the expected way
    assert not callable(invalid_reader)


def test_ReadChartOfAccounts_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class RepeatedReader:
        def __init__(self):
            self.call_count = 0
        
        def __call__(self) -> COA:
            self.call_count += 1
            return COA()
    
    reader = RepeatedReader()
    
    # Call multiple times
    coa1 = reader()
    coa2 = reader()
    coa3 = reader()
    
    # Verify all are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    assert coa2 is not coa3
    
    # Verify call count
    assert reader.call_count == 3


# LLM-generated content at query #73
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the ReadChartOfAccounts protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts_protocol_compliance():
    """Test that ReadChartOfAccounts protocol works with different implementations."""
    # Implementation 1: Simple COA factory
    def reader1() -> COA:
        return COA()
    
    # Implementation 2: COA with custom rootspec
    def reader2() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "Total Assets"),
            AccountType.LIABILITIES: (Code("20"), "Total Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Total Equity"),
            AccountType.REVENUES: (Code("40"), "Total Revenue"),
            AccountType.EXPENSES: (Code("50"), "Total Expenses"),
        }
        return COA(rootspec=rootspec)
    
    # Both should be callable and return COA instances
    coa1 = reader1()
    coa2 = reader2()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify reader2 uses custom codes
    assert coa2.find(Code("10")) is not None
    assert coa2.find(Code("10")).name == "Total Assets"
    assert coa2.find(Code("20")) is not None
    assert coa2.find(Code("20")).name == "Total Liabilities"


def test_ReadChartOfAccounts_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    coa3 = read_coa()
    
    # Verify all calls were made
    assert call_count == 3
    
    # Verify each returns a valid COA
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #74
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    # Setup
    coa = COA()
    
    # Test 1: Add a sub-account to a root account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.type == AccountType.ASSETS
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Add a nested sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.type == AccountType.ASSETS
    assert bank_account.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bank_account
    
    # Test 3: Add multiple sub-accounts to same parent
    savings = coa.add(Code("1"), Code("1002"), "Savings")
    assert savings.code == Code("1002")
    assert savings.parent.code == Code("1")
    
    # Test 4: Add account to different root account types
    payables = coa.add(Code("2"), Code("2000"), "Accounts Payable")
    assert payables.code == Code("2000")
    assert payables.type == AccountType.LIABILITIES
    
    retained_earnings = coa.add(Code("3"), Code("3000"), "Retained Earnings")
    assert retained_earnings.code == Code("3000")
    assert retained_earnings.type == AccountType.EQUITIES
    
    # Test 5: Error when parent and code are the same
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1"), Code("1"), "Invalid")
    
    # Test 6: Error when parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not"):
        coa.add(Code("9999"), Code("9998"), "Invalid")
    
    # Test 7: Adding existing account with matching details returns same account
    existing = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert existing == liquidity
    
    # Test 8: Error when adding existing account with different details
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    # Test 9: Verify subaccounts are properly tracked
    subaccounts_of_1 = coa.subaccounts(coa.find(Code("1")))
    assert len(subaccounts_of_1) == 2
    assert Code("1000") in [a.code for a in subaccounts_of_1]
    assert Code("1002") in [a.code for a in subaccounts_of_1]
    
    # Test 10: Verify nested subaccounts
    subaccounts_of_1000 = coa.subaccounts(coa.find(Code("1000")))
    assert len(subaccounts_of_1000) == 1
    assert subaccounts_of_1000[0].code == Code("1001")


# LLM-generated content at query #75
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected default root accounts
    codes_found = {code for code, _ in result}
    assert Code("1") in codes_found
    assert Code("2") in codes_found
    assert Code("3") in codes_found
    assert Code("4") in codes_found
    assert Code("5") in codes_found
    
    # Verify we can find accounts in the returned COA
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    # Call the function multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Both should have the added account
    assert coa1.find(Code("1000")) is not None
    assert coa2.find(Code("1000")) is not None


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts protocol with custom root specification."""
    
    def read_coa_custom() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("A"), "My Assets"),
            AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
            AccountType.EQUITIES: (Code("E"), "My Equities"),
            AccountType.REVENUES: (Code("R"), "My Revenues"),
            AccountType.EXPENSES: (Code("X"), "My Expenses"),
        }
        return COA(rootspec=custom_rootspec)
    
    # Call the function
    result = read_coa_custom()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify custom codes are present
    assert result.find(Code("A")) is not None
    assert result.find(Code("L")) is not None
    assert result.find(Code("E")) is not None
    assert result.find(Code("R")) is not None
    assert result.find(Code("X")) is not None
    
    # Verify custom names are set
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")).name == "My Liabilities"


# LLM-generated content at query #76
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify the root accounts have correct types
    account_types = {acc.type for acc in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


# LLM-generated content at query #77
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    # Verify they are different instances
    assert coa1 is not coa2


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts with custom root specifications."""
    
    def read_coa_custom() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "My Assets"),
            AccountType.LIABILITIES: (Code("20"), "My Liabilities"),
            AccountType.EQUITIES: (Code("30"), "My Equities"),
            AccountType.REVENUES: (Code("40"), "My Revenues"),
            AccountType.EXPENSES: (Code("50"), "My Expenses"),
        }
        return COA(rootspec=rootspec)
    
    result = read_coa_custom()
    
    # Verify custom codes and names
    assert result.find(Code("10")).name == "My Assets"
    assert result.find(Code("20")).name == "My Liabilities"
    assert result.find(Code("30")).name == "My Equities"
    assert result.find(Code("40")).name == "My Revenues"
    assert result.find(Code("50")).name == "My Expenses"


# LLM-generated content at query #78
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    # Create a simple implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify root account types are correct
    account_types = {acc.type for acc in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


def test_ReadChartOfAccounts___call__custom_rootspec():
    """Test that ReadChartOfAccounts.__call__ works with custom rootspec."""
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "Custom Liabilities"


def test_ReadChartOfAccounts___call__protocol_compliance():
    """Test that ReadChartOfAccounts protocol can be used as type hint."""
    def read_coa() -> COA:
        return COA()
    
    # Assign to protocol type
    reader: ReadChartOfAccounts = read_coa
    
    # Call through protocol
    coa = reader()
    assert isinstance(coa, COA)
    assert len(list(coa.accounts)) == 5


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts.__call__ can be called multiple times."""
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    reader: ReadChartOfAccounts = read_coa
    
    # Multiple calls should work
    coa1 = reader()
    coa2 = reader()
    
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    # Each call creates a new instance
    assert coa1 is not coa2


# LLM-generated content at query #79
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify the root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    
    # Define a custom implementation with state
    class CustomReader:
        def __init__(self, call_count: int = 0):
            self.call_count = call_count
        
        def __call__(self) -> COA:
            self.call_count += 1
            return COA()
    
    reader = CustomReader()
    
    # First call
    coa1 = reader()
    assert isinstance(coa1, COA)
    assert reader.call_count == 1
    
    # Second call
    coa2 = reader()
    assert isinstance(coa2, COA)
    assert reader.call_count == 2
    
    # Verify each call returns a new COA instance
    assert coa1 is not coa2


# LLM-generated content at query #80
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    
    # Create a function that implements the ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable and implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Test with a custom implementation
    def custom_read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Custom Account")
        return coa
    
    result2 = custom_read_coa()
    assert isinstance(result2, COA)
    assert result2.find(Code("1000")) is not None
    assert result2.find(Code("1000")).name == "Custom Account"
    
    # Verify protocol compliance
    assert isinstance(read_coa, ReadChartOfAccounts)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify we can iterate over the returned COA
    count = 0
    for code, acct in result:
        count += 1
        assert isinstance(code, str)
        assert hasattr(acct, 'name')
        assert hasattr(acct, 'code')
        assert hasattr(acct, 'type')
    assert count == 5


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """
    Test ReadChartOfAccounts protocol with a custom implementation that adds accounts.
    """
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Bank Accounts")
        coa.add(Code("1000"), Code("1001"), "Checking Account")
        return coa
    
    # Call and verify
    coa = read_custom_coa()
    assert isinstance(coa, COA)
    
    # Verify custom accounts were added
    bank_account = coa.find(Code("1000"))
    assert bank_account is not None
    assert bank_account.name == "Bank Accounts"
    
    checking_account = coa.find(Code("1001"))
    assert checking_account is not None
    assert checking_account.name == "Checking Account"
    assert checking_account.parent == bank_account


def test_ReadChartOfAccounts___call___protocol_compliance():
    """
    Test that any callable returning COA satisfies ReadChartOfAccounts protocol.
    """
    class CustomCOAReader:
        def __call__(self) -> COA:
            return COA()
    
    reader = CustomCOAReader()
    
    # Should be callable
    assert callable(reader)
    
    # Should return COA
    coa = reader()
    assert isinstance(coa, COA)


# LLM-generated content at query #2
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test the __call__ method of ReadChartOfAccounts protocol."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that the function is callable and returns a COA instance
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    root_accounts = list(result.toplevel)
    assert len(root_accounts) == 5
    
    # Verify account types are in correct order
    account_types = [acc.type for acc in root_accounts]
    expected_types = [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]
    assert account_types == expected_types


def test_ReadChartOfAccounts___call__custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation."""
    
    # Create a custom implementation that modifies the COA
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Custom Account")
        return coa
    
    # Verify callable
    assert callable(read_custom_coa)
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom account was added
    custom_account = result.find(Code("1000"))
    assert custom_account is not None
    assert custom_account.name == "Custom Account"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    
    # Verify both calls succeeded and returned COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert call_count == 2
    
    # Verify they are different instances
    assert result1 is not result2


# LLM-generated content at query #3
#--------------------------

```python
def test_COA_add():
    """Test COA.add method for adding accounts to chart of accounts."""
    
    # Setup
    coa = COA()
    
    # Test 1: Add a sub-account to a root account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity.type == AccountType.ASSETS
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Add a sub-sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.code == Code("1000")
    assert bank_account.type == AccountType.ASSETS
    
    # Test 3: Add multiple accounts to different parents
    debt = coa.add(Code("2"), Code("2000"), "Long-term Debt")
    assert debt.code == Code("2000")
    assert debt.parent.code == Code("2")
    assert debt.type == AccountType.LIABILITIES
    
    # Test 4: Re-adding the same account returns the existing account
    liquidity_again = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity_again == liquidity
    assert liquidity_again.code == Code("1000")
    
    # Test 5: Error when parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined"):
        coa.add(Code("9999"), Code("9000"), "Invalid Parent")
    
    # Test 6: Error when account is its own parent
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1000"), Code("1000"), "Self Parent")
    
    # Test 7: Error when re-adding account with inconsistent information
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    # Test 8: Error when re-adding account with different parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("2"), Code("1000"), "Liquidity")
    
    # Test 9: Verify all added accounts are retrievable
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("1001")) is not None
    assert coa.find(Code("2000")) is not None
    
    # Test 10: Verify sub-accounts are stored correctly
    subaccounts_of_1000 = coa.subaccounts(liquidity)
    assert len(subaccounts_of_1000) == 1
    assert subaccounts_of_1000[0].code == Code("1001")
    
    subaccounts_of_1 = coa.subaccounts(coa.find(Code("1")))
    assert len(subaccounts_of_1) == 1
    assert subaccounts_of_1[0].code == Code("1000")


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts.__call__ returns a COA instance.
    """
    # Create a simple implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify that the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify that the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify account types are present
    account_types = {acc.type for acc in accounts}
    assert AccountType.ASSETS in account_types
    assert AccountType.LIABILITIES in account_types
    assert AccountType.EQUITIES in account_types
    assert AccountType.REVENUES in account_types
    assert AccountType.EXPENSES in account_types


def test_ReadChartOfAccounts___call___with_custom_spec():
    """
    Test that ReadChartOfAccounts.__call__ works with custom rootspec.
    """
    # Create a custom rootspec
    custom_spec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    # Create an implementation that uses custom spec
    def read_coa_custom() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("30")).name == "Custom Equities"
    assert result.find(Code("40")).name == "Custom Revenues"
    assert result.find(Code("50")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___callable_protocol():
    """
    Test that ReadChartOfAccounts protocol is satisfied by callable implementations.
    """
    # Test that a function satisfies the protocol
    def my_reader() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    # Verify the function is callable
    assert callable(my_reader)
    
    # Call it and verify result
    result = my_reader()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Test Account"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts.__call__ can be called multiple times.
    """
    call_count = 0
    
    def read_coa_counted() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_counted()
    result2 = read_coa_counted()
    result3 = read_coa_counted()
    
    assert call_count == 3
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    # Each call should return a new instance
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #5
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify account types are present in order
    account_types = [acc.type for acc in accounts]
    assert account_types == [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that modifies the COA."""
    
    def read_coa_with_accounts() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Call the function
    result = read_coa_with_accounts()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify the custom account was added
    liquidity = result.find(Code("1000"))
    assert liquidity is not None
    assert liquidity.name == "Liquidity"
    assert liquidity.code == Code("1000")


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert len(list(coa1.accounts)) == len(list(coa2.accounts))


# LLM-generated content at query #6
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class TestReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance and call it
    reader = TestReadChartOfAccounts()
    result = reader()
    
    # Assert that the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify that the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify root account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES
    
    # Verify root account names
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with custom COA configuration."""
    
    # Define a custom implementation that returns a COA with custom root spec
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            custom_rootspec = {
                AccountType.ASSETS: (Code("100"), "Custom Assets"),
                AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
                AccountType.EQUITIES: (Code("300"), "Custom Equities"),
                AccountType.REVENUES: (Code("400"), "Custom Revenues"),
                AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
            }
            return COA(rootspec=custom_rootspec)
    
    # Create an instance and call it
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    # Assert that the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify custom codes and names
    assert result.find(Code("100")) is not None
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"
    
    # Verify default codes are not present
    assert result.find(Code("1")) is None
    assert result.find(Code("2")) is None


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class TestReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = TestReadChartOfAccounts()
    
    # Call multiple times
    result1 = reader()
    result2 = reader()
    
    # Both should be COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    
    # They should be different instances
    assert result1 is not result2
    
    # But both should have the same structure
    assert result1.find(Code("1")).name == result2.find(Code("1")).name


# LLM-generated content at query #7
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that read_coa implements the ReadChartOfAccounts protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts protocol with a custom COA configuration."""
    
    # Define a custom implementation that returns a COA with additional accounts
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    # Call the function and verify it returns the expected COA
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.code == Code("1000")


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call the function multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    result3 = read_coa_with_counter()
    
    # Verify all invocations returned COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    
    # Verify the function was called the expected number of times
    assert call_count == 3
    
    # Verify each returned COA is independent
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #8
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all account types are present
    account_types = {account.type for account in accounts}
    assert account_types == {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times independently.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are COA instances but different objects
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """
    Test that ReadChartOfAccounts can return COA with custom root specifications.
    """
    def read_coa_with_custom_spec() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "Fixed Assets"),
            AccountType.LIABILITIES: (Code("20"), "Current Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Share Capital"),
            AccountType.REVENUES: (Code("40"), "Operating Income"),
            AccountType.EXPENSES: (Code("50"), "Operating Costs"),
        }
        return COA(rootspec=rootspec)
    
    result = read_coa_with_custom_spec()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Fixed Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "Current Liabilities"


# LLM-generated content at query #9
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify we can iterate over the accounts
    for code, account in result:
        assert account is not None
        assert account.code == code
    
    # Test that the protocol is satisfied
    assert callable(read_coa)
    
    # Test multiple calls return independent COA instances
    result1 = read_coa()
    result2 = read_coa()
    assert result1 is not result2
    assert len(list(result1.accounts)) == len(list(result2.accounts))


# LLM-generated content at query #10
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    # Test basic addition of a sub-account
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.type == AccountType.ASSETS
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity
    
    # Test adding a sub-sub-account
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")
    
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.type == AccountType.ASSETS
    assert bankaccnt.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bankaccnt
    
    # Test that parent account can have multiple children
    savings = coa.add(Code("1"), Code("1002"), "Savings")
    assert savings.parent.code == Code("1")
    assert len(coa.subaccounts(coa.find(Code("1")))) == 2
    
    # Test error when parent is same as code
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1000"), Code("1000"), "Self Parent")
    
    # Test error when parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined"):
        coa.add(Code("9999"), Code("9998"), "Non-existent Parent")
    
    # Test adding an account that already exists with same properties returns existing
    existing = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert existing == liquidity
    
    # Test error when adding existing code with different properties
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("2"), Code("1000"), "Liquidity")
    
    # Test adding to different account types
    debt = coa.add(Code("2"), Code("2000"), "Debt")
    assert debt.type == AccountType.LIABILITIES
    assert debt.parent.code == Code("2")
    
    # Test that subaccounts are properly tracked
    assert len(coa.subaccounts(coa.find(Code("1")))) == 2
    assert coa.find(Code("1000")) in coa.subaccounts(coa.find(Code("1")))
    assert coa.find(Code("1002")) in coa.subaccounts(coa.find(Code("1")))


# LLM-generated content at query #11
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test the __call__ method of ReadChartOfAccounts protocol."""
    # Create a concrete implementation of ReadChartOfAccounts
    class SimpleCOAReader:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            return coa
    
    # Test that the implementation can be called and returns a COA instance
    reader = SimpleCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1000")).parent.code == Code("1")


def test_ReadChartOfAccounts___call___default_coa():
    """Test ReadChartOfAccounts with default COA initialization."""
    class DefaultCOAReader:
        def __call__(self) -> COA:
            return COA()
    
    reader = DefaultCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    # Check default root accounts exist
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___custom_spec():
    """Test ReadChartOfAccounts with custom root specification."""
    class CustomCOAReader:
        def __call__(self) -> COA:
            rootspec = {
                AccountType.ASSETS: (Code("100"), "My Assets"),
                AccountType.LIABILITIES: (Code("200"), "My Liabilities"),
                AccountType.EQUITIES: (Code("300"), "My Equities"),
                AccountType.REVENUES: (Code("400"), "My Revenues"),
                AccountType.EXPENSES: (Code("500"), "My Expenses"),
            }
            return COA(rootspec=rootspec)
    
    reader = CustomCOAReader()
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("100")).name == "My Assets"
    assert result.find(Code("200")).name == "My Liabilities"
    assert result.find(Code("300")).name == "My Equities"
    assert result.find(Code("400")).name == "My Revenues"
    assert result.find(Code("500")).name == "My Expenses"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    class MultiCOAReader:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1100"), "Bank")
            return coa
    
    reader = MultiCOAReader()
    
    # Call multiple times and verify each returns a valid COA
    result1 = reader()
    result2 = reader()
    
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert result1.find(Code("1100")) is not None
    assert result2.find(Code("1100")) is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call___with_custom_coa():
    """
    Test ReadChartOfAccounts with a custom COA configuration.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("100"), "Assets Custom"),
        AccountType.LIABILITIES: (Code("200"), "Liabilities Custom"),
        AccountType.EQUITIES: (Code("300"), "Equities Custom"),
        AccountType.REVENUES: (Code("400"), "Revenues Custom"),
        AccountType.EXPENSES: (Code("500"), "Expenses Custom"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance with custom codes
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("100")) is not None
    assert result.find(Code("100")).name == "Assets Custom"
    assert result.find(Code("200")) is not None
    assert result.find(Code("200")).name == "Liabilities Custom"
    assert result.find(Code("300")) is not None
    assert result.find(Code("300")).name == "Equities Custom"
    assert result.find(Code("400")) is not None
    assert result.find(Code("400")).name == "Revenues Custom"
    assert result.find(Code("500")) is not None
    assert result.find(Code("500")).name == "Expenses Custom"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times and returns independent instances.
    """
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Total Assets"),
        AccountType.LIABILITIES: (Code("20"), "Total Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Total Equities"),
        AccountType.REVENUES: (Code("40"), "Total Revenues"),
        AccountType.EXPENSES: (Code("50"), "Total Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("10")).name == "Total Assets"
    assert result.find(Code("20")).name == "Total Liabilities"
    assert result.find(Code("30")).name == "Total Equities"
    assert result.find(Code("40")).name == "Total Revenues"
    assert result.find(Code("50")).name == "Total Expenses"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify each instance is independent
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #14
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts callable returns a COA instance."""
    # Create a simple implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None


def test_ReadChartOfAccounts___call___with_custom_spec():
    """Test that ReadChartOfAccounts callable can return COA with custom rootspec."""
    # Create an implementation that returns a COA with custom rootspec
    def read_coa_custom() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Custom Equities"),
            AccountType.REVENUES: (Code("40"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
        }
        coa = COA(rootspec=rootspec)
        return coa
    
    # Call the function and verify the returned COA has correct structure
    result = read_coa_custom()
    assert isinstance(result, COA)
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts callable can be called multiple times."""
    call_count = 0
    
    def read_coa_counted() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call the function multiple times
    result1 = read_coa_counted()
    result2 = read_coa_counted()
    
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert call_count == 2
    # Each call should return a different instance
    assert result1 is not result2


def test_ReadChartOfAccounts___call___with_accounts():
    """Test that ReadChartOfAccounts callable can return COA with added accounts."""
    def read_coa_with_accounts() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    # Call the function and verify the returned COA has correct structure
    result = read_coa_with_accounts()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).parent.name == "Liquidity"


# LLM-generated content at query #15
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can find accounts in the returned COA
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    
    # Define a custom implementation that adds accounts
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Cash")
        coa.add(Code("1000"), Code("1001"), "Checking Account")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Cash"
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Checking Account"
    assert result.find(Code("1001")).parent.code == Code("1000")


def test_ReadChartOfAccounts___call__protocol_compliance():
    """Test that ReadChartOfAccounts protocol is properly defined and callable."""
    
    # Create instances that comply with the protocol
    implementations = []
    
    def impl1() -> COA:
        return COA()
    
    def impl2() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1100"), "Current Assets")
        return coa
    
    implementations.append(impl1)
    implementations.append(impl2)
    
    # Verify all implementations are callable and return COA
    for impl in implementations:
        assert callable(impl)
        result = impl()
        assert isinstance(result, COA)
        assert result is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Verify both have the same structure
    for (code1, acct1), (code2, acct2) in zip(coa1, coa2):
        assert code1 == code2
        assert acct1.name == acct2.name
        assert acct1.type == acct2.type


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts with custom root account specifications."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
        AccountType.EQUITIES: (Code("E"), "My Equities"),
        AccountType.REVENUES: (Code("R"), "My Revenues"),
        AccountType.EXPENSES: (Code("X"), "My Expenses"),
    }
    
    def read_coa_custom() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa_custom()
    
    # Verify custom codes and names
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")).name == "My Liabilities"
    assert result.find(Code("E")).name == "My Equities"
    assert result.find(Code("R")).name == "My Revenues"
    assert result.find(Code("X")).name == "My Expenses"
    
    # Verify types are correct
    assert result.find(Code("A")).type == AccountType.ASSETS
    assert result.find(Code("L")).type == AccountType.LIABILITIES
    assert result.find(Code("E")).type == AccountType.EQUITIES
    assert result.find(Code("R")).type == AccountType.REVENUES
    assert result.find(Code("X")).type == AccountType.EXPENSES


# LLM-generated content at query #17
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    # Create a simple implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify it implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call__with_custom_spec():
    """Test ReadChartOfAccounts.__call__ with custom rootspec."""
    custom_spec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("E"), "Custom Equities"),
        AccountType.REVENUES: (Code("R"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
    }
    
    def read_coa_custom() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_coa_custom()
    
    assert isinstance(result, COA)
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")).name == "Custom Liabilities"
    assert result.find(Code("E")).name == "Custom Equities"
    assert result.find(Code("R")).name == "Custom Revenues"
    assert result.find(Code("X")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts.__call__ can be called multiple times."""
    call_count = 0
    
    def read_coa_counted() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    result1 = read_coa_counted()
    result2 = read_coa_counted()
    
    assert call_count == 2
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    # Each call should return a different instance
    assert result1 is not result2


# LLM-generated content at query #18
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify it satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call__with_custom_rootspec():
    """Test ReadChartOfAccounts with custom root specification."""
    
    custom_spec = {
        AccountType.ASSETS: (Code("A"), "All Assets"),
        AccountType.LIABILITIES: (Code("L"), "All Liabilities"),
        AccountType.EQUITIES: (Code("E"), "All Equities"),
        AccountType.REVENUES: (Code("R"), "All Revenues"),
        AccountType.EXPENSES: (Code("X"), "All Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_custom_coa()
    
    # Verify custom codes exist
    assert result.find(Code("A")) is not None
    assert result.find(Code("L")) is not None
    assert result.find(Code("E")) is not None
    assert result.find(Code("R")) is not None
    assert result.find(Code("X")) is not None
    
    # Verify custom names
    assert result.find(Code("A")).name == "All Assets"
    assert result.find(Code("L")).name == "All Liabilities"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_count() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_count()
    coa2 = read_coa_with_count()
    coa3 = read_coa_with_count()
    
    # Verify each call returned a COA
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify call count
    assert call_count == 3
    
    # Verify each COA is independent
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #19
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected default root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can access accounts from the returned COA
    assets = result.find(Code("1"))
    assert assets is not None
    assert assets.name == "Assets"
    assert assets.type == AccountType.ASSETS


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that adds sub-accounts."""
    
    def read_coa_with_subaccounts() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    # Call the function
    result = read_coa_with_subaccounts()
    
    # Verify custom accounts were added
    liquidity_account = result.find(Code("1000"))
    assert liquidity_account is not None
    assert liquidity_account.name == "Liquidity"
    
    bank_account = result.find(Code("1001"))
    assert bank_account is not None
    assert bank_account.name == "Bank Account"
    assert bank_account.parent == liquidity_account


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Verify both have the same root accounts
    for code1, account1 in coa1:
        account2 = coa2.find(code1)
        assert account2 is not None
        assert account1.name == account2.name
        assert account1.type == account2.type


# LLM-generated content at query #20
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify that the function is callable and conforms to the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts protocol with custom COA configuration."""
    
    # Define a concrete implementation with custom rootspec
    def read_custom_coa() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("100"), "Current Assets"),
            AccountType.LIABILITIES: (Code("200"), "Current Liabilities"),
            AccountType.EQUITIES: (Code("300"), "Shareholders Equity"),
            AccountType.REVENUES: (Code("400"), "Sales"),
            AccountType.EXPENSES: (Code("500"), "Operating Costs"),
        }
        return COA(rootspec=custom_spec)
    
    # Call the function and verify custom COA
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("100")).name == "Current Assets"
    assert result.find(Code("200")).name == "Current Liabilities"
    assert result.find(Code("300")).name == "Shareholders Equity"
    assert result.find(Code("400")).name == "Sales"
    assert result.find(Code("500")).name == "Operating Costs"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa()
    result2 = read_coa()
    result3 = read_coa()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert all(isinstance(r, COA) for r in [result1, result2, result3])
    
    # Verify each call returns a separate COA instance
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #21
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts protocol can be called multiple times.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times and verify each returns a valid COA
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify both instances are independent
    assert coa1 is not coa2
    
    # Both should have the same root structure
    assert len(list(coa1)) == len(list(coa2))


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """
    Test that ReadChartOfAccounts protocol works with custom rootspec.
    """
    def read_coa_custom() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        }
        return COA(rootspec=custom_spec)
    
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify custom codes exist
    assert result.find(Code("10")) is not None
    assert result.find(Code("20")) is not None
    
    # Verify custom names
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"


# LLM-generated content at query #22
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    coa = read_coa()
    assert isinstance(coa, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None


def test_ReadChartOfAccounts___call__with_custom_coa():
    """
    Test that ReadChartOfAccounts protocol can return custom COA instances.
    """
    # Define a custom implementation
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Call and verify
    coa = read_custom_coa()
    assert isinstance(coa, COA)
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call__multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    custom_spec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_spec)
    
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes and names
    assert result.find(Code("A")) is not None
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")) is not None
    assert result.find(Code("L")).name == "Custom Liabilities"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    assert isinstance(read_coa_with_counter, ReadChartOfAccounts)
    
    result1 = read_coa_with_counter()
    assert call_count == 1
    assert isinstance(result1, COA)
    
    result2 = read_coa_with_counter()
    assert call_count == 2
    assert isinstance(result2, COA)
    
    # Verify they are different instances
    assert result1 is not result2


# LLM-generated content at query #24
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    # Setup
    coa = COA()
    
    # Test adding a sub-account to a root account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.type == AccountType.ASSETS
    assert liquidity.parent.code == Code("1")
    assert liquidity.coa is coa
    
    # Test that the account is now in the COA
    found = coa.find(Code("1000"))
    assert found is not None
    assert found.code == Code("1000")
    assert found.name == "Liquidity"
    
    # Test adding a sub-account to a sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.code == Code("1000")
    assert bank_account.type == AccountType.ASSETS
    
    # Test that nested account is found
    found = coa.find(Code("1001"))
    assert found is not None
    assert found.parent.name == "Liquidity"
    
    # Test adding to different account types
    debt = coa.add(Code("2"), Code("2000"), "Long-term Debt")
    assert debt.type == AccountType.LIABILITIES
    
    # Test error when parent and code are the same
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1"), Code("1"), "Invalid")
    
    # Test error when parent does not exist
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined"):
        coa.add(Code("9999"), Code("9998"), "Invalid")
    
    # Test adding duplicate account with same properties returns existing account
    duplicate = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert duplicate is liquidity
    assert duplicate.code == Code("1000")
    
    # Test error when adding duplicate with different properties
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    # Test subaccounts are properly tracked
    subaccounts_of_1000 = coa.subaccounts(liquidity)
    assert len(subaccounts_of_1000) == 1
    assert subaccounts_of_1000[0].code == Code("1001")
    
    # Test multiple sub-accounts under same parent
    cash = coa.add(Code("1"), Code("1002"), "Cash")
    subaccounts_of_1 = coa.subaccounts(coa.find(Code("1")))
    assert len(subaccounts_of_1) == 2
    assert any(a.code == Code("1000") for a in subaccounts_of_1)
    assert any(a.code == Code("1002") for a in subaccounts_of_1)


# LLM-generated content at query #25
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify that the function is callable and matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result)
    assert len(accounts) == 5
    
    codes = [code for code, _ in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts_protocol_implementation():
    """
    Test that different implementations can satisfy the ReadChartOfAccounts protocol.
    """
    # First implementation
    def read_default_coa() -> COA:
        return COA()
    
    # Second implementation with custom rootspec
    def read_custom_coa() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("10"), "My Assets"),
            AccountType.LIABILITIES: (Code("20"), "My Liabilities"),
            AccountType.EQUITIES: (Code("30"), "My Equities"),
            AccountType.REVENUES: (Code("40"), "My Revenues"),
            AccountType.EXPENSES: (Code("50"), "My Expenses"),
        }
        return COA(rootspec=custom_spec)
    
    # Both should be callable and return COA instances
    coa1 = read_default_coa()
    coa2 = read_custom_coa()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they have different codes
    default_codes = [code for code, _ in coa1]
    custom_codes = [code for code, _ in coa2]
    
    assert Code("1") in default_codes
    assert Code("10") in custom_codes


# LLM-generated content at query #26
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can access accounts from the returned COA
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    
    # Test with another implementation
    def read_custom_coa() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("A"), "My Assets"),
            AccountType.LIABILITIES: (Code("B"), "My Liabilities"),
            AccountType.EQUITIES: (Code("C"), "My Equities"),
            AccountType.REVENUES: (Code("D"), "My Revenues"),
            AccountType.EXPENSES: (Code("E"), "My Expenses"),
        }
        return COA(rootspec=custom_spec)
    
    result2 = read_custom_coa()
    assert isinstance(result2, COA)
    assert result2.find(Code("A")) is not None
    assert result2.find(Code("A")).name == "My Assets"


# LLM-generated content at query #27
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test the __call__ method of ReadChartOfAccounts protocol."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    
    # Test with another implementation
    def read_empty_coa() -> COA:
        return COA()
    
    result2 = read_empty_coa()
    assert isinstance(result2, COA)
    assert result2.find(Code("1")) is not None
    assert result2.find(Code("1")).name == "Assets"
    
    # Test that multiple calls return different instances
    result3 = read_coa()
    result4 = read_coa()
    assert result3 is not result4
    assert result3.find(Code("1000")).name == result4.find(Code("1000")).name


# LLM-generated content at query #28
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of the ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    coa = read_coa()
    assert isinstance(coa, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(coa)
    assert len(accounts) == 5
    
    codes = [code for code, _ in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify account types are correct
    assert coa.find(Code("1")).type == AccountType.ASSETS
    assert coa.find(Code("2")).type == AccountType.LIABILITIES
    assert coa.find(Code("3")).type == AccountType.EQUITIES
    assert coa.find(Code("4")).type == AccountType.REVENUES
    assert coa.find(Code("5")).type == AccountType.EXPENSES


# LLM-generated content at query #29
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts_protocol_runtime_checkable():
    """Test that ReadChartOfAccounts protocol works with runtime checking."""
    
    def custom_reader() -> COA:
        return COA()
    
    # Verify the function matches the protocol
    assert callable(custom_reader)
    
    # Call and verify result
    coa = custom_reader()
    assert isinstance(coa, COA)
    assert len(list(coa.accounts)) == 5


def test_ReadChartOfAccounts_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    
    # Verify both have the same root accounts
    for code in [Code("1"), Code("2"), Code("3"), Code("4"), Code("5")]:
        assert coa1.find(code) is not None
        assert coa2.find(code) is not None


# LLM-generated content at query #30
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that read_coa implements the ReadChartOfAccounts protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
        AccountType.EQUITIES: (Code("E"), "My Equities"),
        AccountType.REVENUES: (Code("R"), "My Revenues"),
        AccountType.EXPENSES: (Code("X"), "My Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")).name == "My Liabilities"
    assert result.find(Code("E")).name == "My Equities"
    assert result.find(Code("R")).name == "My Revenues"
    assert result.find(Code("X")).name == "My Expenses"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    assert isinstance(read_coa_with_counter, ReadChartOfAccounts)
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    assert call_count == 3
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)


# LLM-generated content at query #31
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class TestReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance
    reader = TestReadChartOfAccounts()
    
    # Call the instance
    result = reader()
    
    # Assert that result is a COA instance
    assert isinstance(result, COA)
    
    # Verify that the COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify account types are in correct order
    account_types = [acct.type for acct in accounts]
    assert account_types == [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]


# LLM-generated content at query #32
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify it conforms to the ReadChartOfAccounts protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Test with a more complex implementation
    def read_custom_coa() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("A"), "My Assets"),
            AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
            AccountType.EQUITIES: (Code("E"), "My Equities"),
            AccountType.REVENUES: (Code("R"), "My Revenues"),
            AccountType.EXPENSES: (Code("X"), "My Expenses"),
        }
        coa = COA(rootspec=custom_spec)
        return coa
    
    # Verify custom implementation works
    custom_result = read_custom_coa()
    assert isinstance(custom_result, COA)
    assert custom_result.find(Code("A")) is not None
    assert custom_result.find(Code("A")).name == "My Assets"
    assert custom_result.find(Code("L")) is not None
    assert custom_result.find(Code("E")) is not None
    assert custom_result.find(Code("R")) is not None
    assert custom_result.find(Code("X")) is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that the function is callable and returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify that the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify that accounts have correct types and names
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).name == "Expenses"
    
    # Test that the protocol works with different implementations
    def read_custom_coa() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Custom Equities"),
            AccountType.REVENUES: (Code("40"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
        }
        return COA(rootspec=custom_rootspec)
    
    custom_result = read_custom_coa()
    assert isinstance(custom_result, COA)
    assert custom_result.find(Code("10")).name == "Custom Assets"
    assert custom_result.find(Code("20")).name == "Custom Liabilities"


# LLM-generated content at query #34
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa()
    result2 = read_coa()
    
    # Verify both calls succeeded
    assert call_count == 2
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    # Verify they are different instances
    assert result1 is not result2


def test_ReadChartOfAccounts___call__with_custom_rootspec():
    """Test ReadChartOfAccounts with custom root specification."""
    def read_coa_custom() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("100"), "Company Assets"),
            AccountType.LIABILITIES: (Code("200"), "Company Liabilities"),
            AccountType.EQUITIES: (Code("300"), "Company Equities"),
            AccountType.REVENUES: (Code("400"), "Company Revenues"),
            AccountType.EXPENSES: (Code("500"), "Company Expenses"),
        }
        return COA(rootspec=rootspec)
    
    # Call the function
    result = read_coa_custom()
    
    # Verify custom codes and names
    assert result.find(Code("100")).name == "Company Assets"
    assert result.find(Code("200")).name == "Company Liabilities"
    assert result.find(Code("300")).name == "Company Equities"
    assert result.find(Code("400")).name == "Company Revenues"
    assert result.find(Code("500")).name == "Company Expenses"


# LLM-generated content at query #35
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance
    result = reader()
    
    # Assert that result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that returns a modified COA."""
    
    # Define a custom implementation that returns a COA with added accounts
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            coa.add(Code("1000"), Code("1001"), "Bank Account")
            return coa
    
    # Create and call the reader
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    # Assert that result is a COA instance
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.code == Code("1000")


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class MultiCallReader:
        def __call__(self) -> COA:
            return COA()
    
    reader = MultiCallReader()
    
    # Call multiple times
    coa1 = reader()
    coa2 = reader()
    coa3 = reader()
    
    # All should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    assert coa2 is not coa3
    
    # But have the same structure
    assert len(list(coa1.accounts)) == len(list(coa2.accounts)) == len(list(coa3.accounts))


# LLM-generated content at query #36
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteChartReader:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the concrete implementation
    reader = ConcreteChartReader()
    
    # Call the reader and verify it returns a COA instance
    result = reader()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """
    Test that ReadChartOfAccounts can return a custom configured COA.
    """
    # Define a concrete implementation that returns a custom COA
    class CustomChartReader:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            return coa
    
    # Create an instance and call it
    reader = CustomChartReader()
    result = reader()
    
    # Verify the custom account was added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1000")).parent.code == Code("1")


def test_ReadChartOfAccounts_protocol_compliance():
    """
    Test that ReadChartOfAccounts protocol is correctly implemented.
    """
    # Define a callable that implements the protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify it's callable
    assert callable(read_coa)
    
    # Call it and verify return type
    result = read_coa()
    assert isinstance(result, COA)


# LLM-generated content at query #37
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Call the function
    result = read_coa()
    
    # Assert it returns a COA instance
    assert isinstance(result, COA)
    
    # Assert the COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___empty_coa():
    """Test that ReadChartOfAccounts can return an empty COA with only root accounts."""
    
    def read_empty_coa() -> COA:
        return COA()
    
    result = read_empty_coa()
    
    assert isinstance(result, COA)
    assert result.find(Code("1")) is not None
    assert result.find(Code("1").name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___complex_structure():
    """Test that ReadChartOfAccounts can return a complex COA structure."""
    
    def read_complex_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        coa.add(liquidity.code, Code("1002"), "Cash")
        receivables = coa.add(Code("1"), Code("1100"), "Receivables")
        coa.add(receivables.code, Code("1101"), "Accounts Receivable")
        return coa
    
    result = read_complex_coa()
    
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1002")) is not None
    assert result.find(Code("1100")) is not None
    assert result.find(Code("1101")) is not None
    assert result.find(Code("1001")).parent.code == Code("1000")
    assert result.find(Code("1101")).parent.code == Code("1100")


def test_ReadChartOfAccounts___call___is_callable():
    """Test that ReadChartOfAccounts protocol instances are callable."""
    
    def read_coa() -> COA:
        return COA()
    
    # Verify it's callable
    assert callable(read_coa)
    
    # Call it and verify result
    result = read_coa()
    assert isinstance(result, COA)


# LLM-generated content at query #38
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify it's callable and returns a COA
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify root accounts are in correct order
    codes = [acc.code for acc in result.toplevel]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call__custom_coa():
    """Test that ReadChartOfAccounts.__call__ can return custom COA instances."""
    # Create a callable that returns a custom COA
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Custom Account")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Custom Account"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts.__call__ can be called multiple times."""
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify each call returns a new COA instance
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2


# LLM-generated content at query #39
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable and implements ReadChartOfAccounts protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct names
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).name == "Expenses"
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts protocol with custom COA initialization."""
    # Define a concrete implementation that returns a custom COA
    def read_custom_coa() -> COA:
        custom_spec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Custom Equities"),
            AccountType.REVENUES: (Code("40"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
        }
        coa = COA(rootspec=custom_spec)
        return coa
    
    # Call the function and verify it returns a COA instance
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify the custom root accounts are present
    assert result.find(Code("10")) is not None
    assert result.find(Code("20")) is not None
    assert result.find(Code("30")) is not None
    assert result.find(Code("40")) is not None
    assert result.find(Code("50")) is not None
    
    # Verify the custom names
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("30")).name == "Custom Equities"
    assert result.find(Code("40")).name == "Custom Revenues"
    assert result.find(Code("50")).name == "Custom Expenses"


# LLM-generated content at query #40
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    
    # Initialize a chart of accounts
    coa = COA()
    
    # Test 1: Add a sub-account to a root account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity.parent.name == "Assets"
    assert liquidity.type == AccountType.ASSETS
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Add a nested sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.code == Code("1000")
    assert bank_account.parent.name == "Liquidity"
    assert bank_account.type == AccountType.ASSETS
    
    # Test 3: Add multiple sub-accounts to different root accounts
    debt = coa.add(Code("2"), Code("2000"), "Long-term Debt")
    assert debt.code == Code("2000")
    assert debt.parent.code == Code("2")
    assert debt.type == AccountType.LIABILITIES
    
    # Test 4: Retrieve added account
    retrieved = coa.find(Code("1001"))
    assert retrieved == bank_account
    assert retrieved.name == "Bank Account"
    
    # Test 5: Adding to non-existent parent raises ValueError
    try:
        coa.add(Code("9999"), Code("9001"), "Invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Parent account is not (yet) defined" in str(e)
    
    # Test 6: Account cannot be its own parent
    try:
        coa.add(Code("1000"), Code("1000"), "Self Parent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "An account can not be the parent of itself" in str(e)
    
    # Test 7: Adding duplicate account with same parameters returns existing account
    duplicate = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert duplicate == liquidity
    assert duplicate.code == Code("1000")
    
    # Test 8: Adding duplicate with different parameters raises ValueError
    try:
        coa.add(Code("1"), Code("1000"), "Different Name")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Account name, code and parent do not match" in str(e)
    
    # Test 9: Verify subaccounts are tracked
    subaccounts_of_1000 = coa.subaccounts(liquidity)
    assert len(subaccounts_of_1000) == 1
    assert subaccounts_of_1000[0].code == Code("1001")
    
    # Test 10: Add multiple children to same parent
    savings = coa.add(Code("1"), Code("1002"), "Savings Account")
    subaccounts_of_1 = coa.subaccounts(coa.find(Code("1")))
    assert len(subaccounts_of_1) == 2
    assert Code("1000") in [acc.code for acc in subaccounts_of_1]
    assert Code("1002") in [acc.code for acc in subaccounts_of_1]
    
    # Test 11: Verify account is findable after addition
    assert coa.find(Code("1002")) == savings
    assert coa.find(Code("2000")) == debt


# LLM-generated content at query #41
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert len(list(result.toplevel)) == 5
    
    # Verify we can iterate over accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify each account type is present
    account_types = {account.type for account in accounts}
    assert AccountType.ASSETS in account_types
    assert AccountType.LIABILITIES in account_types
    assert AccountType.EQUITIES in account_types
    assert AccountType.REVENUES in account_types
    assert AccountType.EXPENSES in account_types


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that modifies COA."""
    def read_custom_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    # Verify it satisfies the protocol
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    # Call and verify the result
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    liquidity_account = result.find(Code("1000"))
    assert liquidity_account is not None
    assert liquidity_account.name == "Liquidity"
    
    bank_account = result.find(Code("1001"))
    assert bank_account is not None
    assert bank_account.name == "Bank Account"
    assert bank_account.parent == liquidity_account


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Verify protocol implementation
    assert isinstance(read_coa_with_counter, ReadChartOfAccounts)
    
    # Call multiple times
    result1 = read_coa_with_counter()
    assert call_count == 1
    assert isinstance(result1, COA)
    
    result2 = read_coa_with_counter()
    assert call_count == 2
    assert isinstance(result2, COA)
    
    # Verify they are different instances
    assert result1 is not result2


# LLM-generated content at query #42
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance
    reader = ConcreteReadChartOfAccounts()
    
    # Call it and verify it returns a COA instance
    result = reader()
    
    assert isinstance(result, COA)
    assert result is not None


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts implementation with custom rootspec."""
    
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            custom_spec = {
                AccountType.ASSETS: (Code("100"), "Custom Assets"),
                AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
                AccountType.EQUITIES: (Code("300"), "Custom Equities"),
                AccountType.REVENUES: (Code("400"), "Custom Revenues"),
                AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
            }
            return COA(rootspec=custom_spec)
    
    # Create an instance and call it
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    # Verify it returns a COA with custom accounts
    assert isinstance(result, COA)
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___protocol_compliance():
    """Test that ReadChartOfAccounts protocol is properly defined and usable."""
    
    def create_coa_reader() -> ReadChartOfAccounts:
        """Factory function that returns a ReadChartOfAccounts compliant callable."""
        def read_coa() -> COA:
            return COA()
        return read_coa
    
    # Get a reader
    reader = create_coa_reader()
    
    # Verify it's callable
    assert callable(reader)
    
    # Call it and verify result
    coa = reader()
    assert isinstance(coa, COA)


def test_ReadChartOfAccounts___call___returns_populated_coa():
    """Test that a ReadChartOfAccounts implementation returns a properly populated COA."""
    
    class PopulatedReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
            coa.add(liquidity.code, Code("1001"), "Bank Account")
            return coa
    
    reader = PopulatedReadChartOfAccounts()
    result = reader()
    
    # Verify the COA has the expected accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).parent.name == "Liquidity"


# LLM-generated content at query #43
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts is callable and returns a COA instance."""
    # Create a simple implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify it's callable
    assert callable(read_coa)
    
    # Call it and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call__with_custom_rootspec():
    """Test ReadChartOfAccounts with a custom rootspec."""
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("E"), "Custom Equities"),
        AccountType.REVENUES: (Code("R"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("A")) is not None
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")) is not None
    assert result.find(Code("L")).name == "Custom Liabilities"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    codes1 = [code for code, _ in coa1]
    codes2 = [code for code, _ in coa2]
    assert codes1 == codes2


# LLM-generated content at query #44
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default structure
    accounts = list(result)
    assert len(accounts) == 5
    assert accounts[0][0] == Code("1")
    assert accounts[0][1].name == "Assets"
    assert accounts[1][0] == Code("2")
    assert accounts[1][1].name == "Liabilities"
    assert accounts[2][0] == Code("3")
    assert accounts[2][1].name == "Equities"
    assert accounts[3][0] == Code("4")
    assert accounts[3][1].name == "Revenues"
    assert accounts[4][0] == Code("5")
    assert accounts[4][1].name == "Expenses"


def test_ReadChartOfAccounts___call___with_custom_spec():
    """Test ReadChartOfAccounts with custom root specification."""
    
    custom_spec = {
        AccountType.ASSETS: (Code("10"), "Total Assets"),
        AccountType.LIABILITIES: (Code("20"), "Total Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Total Equities"),
        AccountType.REVENUES: (Code("40"), "Total Revenues"),
        AccountType.EXPENSES: (Code("50"), "Total Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Total Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "Total Liabilities"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times and verify each returns a new COA instance
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2  # Different instances
    
    # Both should have the same structure
    accounts1 = list(coa1)
    accounts2 = list(coa2)
    assert len(accounts1) == len(accounts2)


# LLM-generated content at query #45
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance and verify it returns a COA
    result = reader()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify the accounts are in the correct order
    codes = [a.code for a in accounts]
    assert codes == [Code("1"), Code("2"), Code("3"), Code("4"), Code("5")]
    
    # Verify account types
    types = [a.type for a in accounts]
    assert types == [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]


def test_ReadChartOfAccounts_protocol_compliance():
    """Test that ReadChartOfAccounts protocol is properly defined."""
    
    # Verify ReadChartOfAccounts is a Protocol
    assert hasattr(ReadChartOfAccounts, '__call__')
    
    # Define a function that matches the protocol
    def read_coa() -> COA:
        return COA()
    
    # The function should be callable
    result = read_coa()
    assert isinstance(result, COA)


def test_ReadChartOfAccounts_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class MultiCallReader:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Test Account")
            return coa
    
    reader = MultiCallReader()
    
    # Call multiple times
    coa1 = reader()
    coa2 = reader()
    
    # Both should be valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # Both should have the added account
    assert coa1.find(Code("1000")) is not None
    assert coa2.find(Code("1000")) is not None


# LLM-generated content at query #46
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance and verify it returns a COA
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have the correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = ConcreteReadChartOfAccounts()
    
    # Call multiple times
    coa1 = reader()
    coa2 = reader()
    
    # Both should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But with same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


def test_ReadChartOfAccounts___call__custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that modifies COA."""
    
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1000"), "Liquidity")
            return coa
    
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1000")).parent.code == Code("1")


def test_ReadChartOfAccounts_protocol_compliance():
    """Test that an object can be used as ReadChartOfAccounts if it has __call__ returning COA."""
    
    def create_coa() -> COA:
        return COA()
    
    # Functions with __call__ should work as ReadChartOfAccounts
    reader: ReadChartOfAccounts = create_coa
    
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("1")) is not None


# LLM-generated content at query #47
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert len(list(result.toplevel)) == 5
    
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """
    Test ReadChartOfAccounts with a custom implementation.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
        AccountType.EQUITIES: (Code("E"), "My Equities"),
        AccountType.REVENUES: (Code("R"), "My Revenues"),
        AccountType.EXPENSES: (Code("X"), "My Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are present
    assert result.find(Code("A")) is not None
    assert result.find(Code("L")) is not None
    assert result.find(Code("E")) is not None
    assert result.find(Code("R")) is not None
    assert result.find(Code("X")) is not None


def test_ReadChartOfAccounts___call__multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    coa3 = read_coa()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify they are separate instances
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #48
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("1")).type == AccountType.ASSETS
    
    # Verify all root accounts are present
    root_accounts = list(result.toplevel)
    assert len(root_accounts) == 5
    
    # Test with another implementation that customizes the COA
    def read_custom_coa() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("A"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("E"), "Custom Equities"),
            AccountType.REVENUES: (Code("R"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
        }
        return COA(rootspec=custom_rootspec)
    
    # Verify custom implementation works
    custom_result = read_custom_coa()
    assert isinstance(custom_result, COA)
    assert custom_result.find(Code("A")) is not None
    assert custom_result.find(Code("A")).name == "Custom Assets"
    assert custom_result.find(Code("1")) is None


# LLM-generated content at query #49
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts is callable and returns a COA instance."""
    
    def read_coa() -> COA:
        """Sample implementation of ReadChartOfAccounts."""
        return COA()
    
    # Verify that the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts with custom COA initialization."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Current Assets"),
        AccountType.LIABILITIES: (Code("20"), "Current Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Shareholders Equity"),
        AccountType.REVENUES: (Code("40"), "Operating Revenues"),
        AccountType.EXPENSES: (Code("50"), "Operating Expenses"),
    }
    
    def read_custom_coa() -> COA:
        """Sample implementation returning custom COA."""
        return COA(rootspec=custom_rootspec)
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("10")).name == "Current Assets"
    assert result.find(Code("20")).name == "Current Liabilities"
    assert result.find(Code("30")).name == "Shareholders Equity"
    assert result.find(Code("40")).name == "Operating Revenues"
    assert result.find(Code("50")).name == "Operating Expenses"


def test_ReadChartOfAccounts___call__protocol_compliance():
    """Test that a function complies with ReadChartOfAccounts protocol."""
    
    def my_coa_reader() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1001"), "Cash")
        return coa
    
    # Verify the function matches the protocol
    assert callable(my_coa_reader)
    
    # Call and verify result
    coa = my_coa_reader()
    assert isinstance(coa, COA)
    assert coa.find(Code("1001")) is not None
    assert coa.find(Code("1001")).name == "Cash"
    assert coa.find(Code("1001")).parent.code == Code("1")


# LLM-generated content at query #50
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable and returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES
    
    # Verify the function can be called multiple times
    result2 = read_coa()
    assert isinstance(result2, COA)
    assert result is not result2  # Different instances


# LLM-generated content at query #51
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the account types are correct
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts.__call__ with a custom COA."""
    
    # Create a custom COA
    custom_coa = COA()
    custom_coa.add(Code("1"), Code("1000"), "Custom Account")
    
    # Create a callable that returns the custom COA
    def read_custom_coa() -> COA:
        return custom_coa
    
    # Call the function and verify it returns the custom COA
    result = read_custom_coa()
    assert result is custom_coa
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Custom Account"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts.__call__ can be called multiple times."""
    
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa()
    result2 = read_coa()
    result3 = read_coa()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert all(isinstance(r, COA) for r in [result1, result2, result3])
    
    # Verify each call returns a separate instance
    assert result1 is not result2
    assert result2 is not result3


# LLM-generated content at query #52
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a simple implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default accounts
    accounts = list(result)
    assert len(accounts) == 5
    
    expected_codes = [Code("1"), Code("2"), Code("3"), Code("4"), Code("5")]
    actual_codes = [code for code, _ in accounts]
    assert actual_codes == expected_codes
    
    # Verify the returned COA has the expected account types
    expected_types = [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]
    actual_types = [acct.type for _, acct in accounts]
    assert actual_types == expected_types


def test_ReadChartOfAccounts_protocol_compliance():
    """
    Test that a function implementing ReadChartOfAccounts protocol works correctly.
    """
    # Create a custom COA with specific root accounts
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    # Call and verify
    coa = read_custom_coa()
    assert coa.find(Code("10")).name == "Custom Assets"
    assert coa.find(Code("20")).name == "Custom Liabilities"
    assert coa.find(Code("30")).name == "Custom Equities"
    assert coa.find(Code("40")).name == "Custom Revenues"
    assert coa.find(Code("50")).name == "Custom Expenses"


def test_ReadChartOfAccounts_multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Modify coa1
    coa1.add(Code("1"), Code("1000"), "Test Account")
    
    # Verify coa2 is unaffected
    assert coa2.find(Code("1000")) is None
    assert coa1.find(Code("1000")) is not None


# LLM-generated content at query #53
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all default account types are present
    account_types = {acc.type for acc in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation that modifies the COA."""
    
    def read_coa_with_custom_accounts() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Call and verify the custom implementation
    result = read_coa_with_custom_accounts()
    assert isinstance(result, COA)
    
    # Verify custom account was added
    liquidity = result.find(Code("1000"))
    assert liquidity is not None
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    
    # Modify one and verify the other is unaffected
    coa1.add(Code("1"), Code("1000"), "Test Account")
    assert coa1.find(Code("1000")) is not None
    assert coa2.find(Code("1000")) is None


# LLM-generated content at query #54
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the expected default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call___with_custom_coa():
    """
    Test ReadChartOfAccounts protocol with custom COA initialization.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("E"), "Custom Equities"),
        AccountType.REVENUES: (Code("R"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("A")) is not None
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")) is not None
    assert result.find(Code("L")).name == "Custom Liabilities"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times and returns independent instances.
    """
    def read_coa() -> COA:
        return COA()
    
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert coa1 is not coa2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Both should have the same default structure
    assert len(list(coa1.accounts)) == len(list(coa2.accounts))


# LLM-generated content at query #55
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify root accounts are present with correct types
    root_types = {account.type for account in result.toplevel}
    assert root_types == {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }


def test_ReadChartOfAccounts___call___custom_implementation():
    """
    Test that ReadChartOfAccounts protocol works with custom implementations.
    """
    # Create a custom implementation
    def custom_read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify it conforms to the protocol
    assert callable(custom_read_coa)
    
    # Call and verify the result
    result = custom_read_coa()
    assert isinstance(result, COA)
    
    # Verify custom account was added
    liquidity = result.find(Code("1000"))
    assert liquidity is not None
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times independently.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Verify both have the same structure
    assert len(list(coa1.accounts)) == len(list(coa2.accounts))


# LLM-generated content at query #56
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of the ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify that the function conforms to the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify root accounts are present
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Each call should return a distinct COA instance
    assert coa1 is not coa2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts protocol with custom initialization."""
    
    def read_coa_custom() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("A"), "My Assets"),
            AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
            AccountType.EQUITIES: (Code("E"), "My Equities"),
            AccountType.REVENUES: (Code("R"), "My Revenues"),
            AccountType.EXPENSES: (Code("X"), "My Expenses"),
        }
        return COA(rootspec=rootspec)
    
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("A")) is not None
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")) is not None
    assert result.find(Code("L")).name == "My Liabilities"


# LLM-generated content at query #57
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function
    result = read_coa()
    
    # Assert it returns a COA instance
    assert isinstance(result, COA)
    
    # Assert the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("1")).type == AccountType.ASSETS
    
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("4")).type == AccountType.REVENUES
    
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts protocol with a custom COA implementation."""
    
    # Define a callable that returns a custom COA with additional accounts
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    # Call the function
    result = read_custom_coa()
    
    # Assert it returns a COA instance
    assert isinstance(result, COA)
    
    # Assert the custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.code == Code("1000")


def test_ReadChartOfAccounts___call__protocol_compliance():
    """Test that any callable returning COA satisfies ReadChartOfAccounts protocol."""
    
    # Create a callable that returns COA
    def my_coa_reader() -> COA:
        return COA()
    
    # Verify it satisfies the protocol by calling it
    result = my_coa_reader()
    
    # Verify the protocol requirement: __call__ returns COA
    assert callable(my_coa_reader)
    assert isinstance(result, COA)


# LLM-generated content at query #58
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance and verify it returns a COA
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA contains the default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """
    Test that ReadChartOfAccounts protocol can return custom COA configurations.
    """
    # Define a concrete implementation that returns a custom COA
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1100"), "Current Assets")
            coa.add(Code("1100"), Code("1101"), "Cash")
            return coa
    
    # Create and call the reader
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    # Assert result is a COA
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1100")) is not None
    assert result.find(Code("1101")) is not None
    assert result.find(Code("1100")).name == "Current Assets"
    assert result.find(Code("1101")).name == "Cash"
    assert result.find(Code("1101")).parent.code == Code("1100")


def test_ReadChartOfAccounts___call__multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    class MultiCallReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = MultiCallReadChartOfAccounts()
    
    # Call multiple times
    coa1 = reader()
    coa2 = reader()
    
    # Both should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


# LLM-generated content at query #59
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the returned COA has correct account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """
    Test that ReadChartOfAccounts can return a COA with custom rootspec.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Current Assets"),
        AccountType.LIABILITIES: (Code("20"), "Current Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Owner Equity"),
        AccountType.REVENUES: (Code("40"), "Operating Revenues"),
        AccountType.EXPENSES: (Code("50"), "Operating Expenses"),
    }
    
    def read_coa_custom() -> COA:
        coa = COA(rootspec=custom_rootspec)
        return coa
    
    # Call the function and verify it returns a COA instance
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the custom root accounts
    assert result.find(Code("10")) is not None
    assert result.find(Code("20")) is not None
    assert result.find(Code("30")) is not None
    assert result.find(Code("40")) is not None
    assert result.find(Code("50")) is not None
    
    # Verify custom names
    assert result.find(Code("10")).name == "Current Assets"
    assert result.find(Code("20")).name == "Current Liabilities"
    assert result.find(Code("30")).name == "Owner Equity"
    assert result.find(Code("40")).name == "Operating Revenues"
    assert result.find(Code("50")).name == "Operating Expenses"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.
    """
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both calls were made
    assert call_count == 2
    
    # Verify both return COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    
    # Verify both have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


# LLM-generated content at query #60
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test the __call__ method of ReadChartOfAccounts protocol."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Verify the function implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    
    # Test another implementation
    def read_empty_coa() -> COA:
        return COA()
    
    assert isinstance(read_empty_coa, ReadChartOfAccounts)
    result2 = read_empty_coa()
    assert isinstance(result2, COA)
    assert result2.find(Code("1")) is not None
    assert result2.find(Code("2")) is not None
    assert result2.find(Code("3")) is not None
    assert result2.find(Code("4")) is not None
    assert result2.find(Code("5")) is not None
    
    # Test that __call__ can be invoked on callable objects
    class CustomReader:
        def __call__(self) -> COA:
            coa = COA()
            coa.add(Code("1"), Code("1100"), "Cash")
            return coa
    
    reader = CustomReader()
    assert isinstance(reader, ReadChartOfAccounts)
    result3 = reader()
    assert isinstance(result3, COA)
    assert result3.find(Code("1100")) is not None
    assert result3.find(Code("1100")).name == "Cash"


# LLM-generated content at query #61
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        """Concrete implementation that returns a COA instance."""
        return COA()
    
    # Verify the callable matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all root account types are present
    account_types = {acct.type for acct in accounts}
    expected_types = {
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    }
    assert account_types == expected_types


def test_ReadChartOfAccounts___call___custom_implementation():
    """Test ReadChartOfAccounts with a custom implementation."""
    
    # Define a custom implementation
    def read_custom_coa() -> COA:
        """Custom implementation that creates a COA with sub-accounts."""
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.name == "Liquidity"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        """Implementation that tracks call count."""
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Verify protocol compliance
    assert callable(read_coa_with_counter)
    
    # Call multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    result3 = read_coa_with_counter()
    
    # Verify all calls returned COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)
    
    # Verify call count
    assert call_count == 3


# LLM-generated content at query #62
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a callable that implements the ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify that the callable can be assigned to ReadChartOfAccounts type
    reader: ReadChartOfAccounts = read_coa
    
    # Call the reader and verify it returns a COA instance
    result = reader()
    
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the expected default root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call___with_custom_coa():
    """
    Test that ReadChartOfAccounts can return customized COA instances.
    """
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    reader: ReadChartOfAccounts = read_custom_coa
    result = reader()
    
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    call_count = 0
    
    def read_coa_with_count() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    reader: ReadChartOfAccounts = read_coa_with_count
    
    result1 = reader()
    assert call_count == 1
    assert isinstance(result1, COA)
    
    result2 = reader()
    assert call_count == 2
    assert isinstance(result2, COA)
    
    # Each call should return a different instance
    assert result1 is not result2


# LLM-generated content at query #63
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify it's callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


def test_ReadChartOfAccounts_protocol_compliance():
    """
    Test that ReadChartOfAccounts protocol enforces __call__ method.
    """
    # Create an implementation that returns different COA configurations
    def read_custom_coa() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("10"), "Assets Custom"),
            AccountType.LIABILITIES: (Code("20"), "Liabilities Custom"),
            AccountType.EQUITIES: (Code("30"), "Equities Custom"),
            AccountType.REVENUES: (Code("40"), "Revenues Custom"),
            AccountType.EXPENSES: (Code("50"), "Expenses Custom"),
        }
        return COA(rootspec=custom_rootspec)
    
    # Call and verify the custom configuration
    result = read_custom_coa()
    assert result.find(Code("10")).name == "Assets Custom"
    assert result.find(Code("20")).name == "Liabilities Custom"
    assert result.find(Code("30")).name == "Equities Custom"
    assert result.find(Code("40")).name == "Revenues Custom"
    assert result.find(Code("50")).name == "Expenses Custom"


def test_ReadChartOfAccounts_multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times independently.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Verify both have valid structure
    assert coa1.find(Code("1")) is not None
    assert coa2.find(Code("1")) is not None
    
    # Modify one and verify the other is unaffected
    coa1.add(Code("1"), Code("1000"), "Test Account")
    assert coa1.find(Code("1000")) is not None
    assert coa2.find(Code("1000")) is None


# LLM-generated content at query #64
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function satisfies the ReadChartOfAccounts protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify all account types are present
    account_types = {acc.type for acc in accounts}
    assert AccountType.ASSETS in account_types
    assert AccountType.LIABILITIES in account_types
    assert AccountType.EQUITIES in account_types
    assert AccountType.REVENUES in account_types
    assert AccountType.EXPENSES in account_types


def test_ReadChartOfAccounts___call___with_custom_coa():
    """
    Test that ReadChartOfAccounts protocol works with custom COA configurations.
    """
    # Create a custom implementation that modifies the COA
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1100"), "Cash")
        return coa
    
    # Verify the function satisfies the ReadChartOfAccounts protocol
    assert isinstance(read_custom_coa, ReadChartOfAccounts)
    
    # Call and verify the custom account was added
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    cash_account = result.find(Code("1100"))
    assert cash_account is not None
    assert cash_account.name == "Cash"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times independently.
    """
    def read_coa() -> COA:
        return COA()
    
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are independent instances
    assert coa1 is not coa2
    
    # Modify one and verify the other is unaffected
    coa1.add(Code("1"), Code("1200"), "Savings")
    assert coa1.find(Code("1200")) is not None
    assert coa2.find(Code("1200")) is None


# LLM-generated content at query #65
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify it satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected structure
    accounts = list(result.accounts)
    assert len(accounts) == 5
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___with_custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    
    def custom_read_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    # Call and verify
    result = custom_read_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1000")).name == "Liquidity"
    assert result.find(Code("1001")).name == "Bank Account"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be invoked multiple times."""
    
    invocation_count = 0
    
    def counting_read_coa() -> COA:
        nonlocal invocation_count
        invocation_count += 1
        return COA()
    
    # Call multiple times
    coa1 = counting_read_coa()
    coa2 = counting_read_coa()
    coa3 = counting_read_coa()
    
    # Verify all returned valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify invocation count
    assert invocation_count == 3
    
    # Verify each is independent
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #66
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function
    result = read_coa()
    
    # Assert it returns a COA instance
    assert isinstance(result, COA)
    
    # Assert the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Assert root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call__with_custom_coa():
    """
    Test ReadChartOfAccounts protocol with custom COA configuration.
    """
    # Define a custom implementation
    def read_custom_coa() -> COA:
        rootspec = {
            AccountType.ASSETS: (Code("10"), "My Assets"),
            AccountType.LIABILITIES: (Code("20"), "My Liabilities"),
            AccountType.EQUITIES: (Code("30"), "My Equities"),
            AccountType.REVENUES: (Code("40"), "My Revenues"),
            AccountType.EXPENSES: (Code("50"), "My Expenses"),
        }
        return COA(rootspec=rootspec)
    
    # Call the function
    result = read_custom_coa()
    
    # Assert custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "My Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "My Liabilities"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times independently.
    """
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Call multiple times
    result1 = read_coa()
    result2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    
    # Both should have the added account
    assert result1.find(Code("1000")) is not None
    assert result2.find(Code("1000")) is not None


# LLM-generated content at query #67
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    # Verify it conforms to the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Test Account"


def test_ReadChartOfAccounts___call___returns_coa():
    """Test that ReadChartOfAccounts callable returns a valid COA."""
    def custom_reader() -> COA:
        return COA()
    
    coa = custom_reader()
    
    assert isinstance(coa, COA)
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).type == AccountType.ASSETS


def test_ReadChartOfAccounts___call___with_custom_coa():
    """Test ReadChartOfAccounts with custom initialized COA."""
    def reader_with_custom_accounts() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    result = reader_with_custom_accounts()
    
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.name == "Liquidity"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = 0
    
    def counting_reader() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    coa1 = counting_reader()
    coa2 = counting_reader()
    
    assert call_count == 2
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert coa1 is not coa2


# LLM-generated content at query #68
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    
    # Create a basic COA
    coa = COA()
    
    # Test 1: Add a sub-account to root account (Assets)
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity.type == AccountType.ASSETS
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Add a sub-sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.code == Code("1000")
    assert bank_account.type == AccountType.ASSETS
    assert coa.find(Code("1001")) == bank_account
    
    # Test 3: Add multiple accounts to different root accounts
    debt = coa.add(Code("2"), Code("2000"), "Long-term Debt")
    assert debt.code == Code("2000")
    assert debt.type == AccountType.LIABILITIES
    
    retained_earnings = coa.add(Code("3"), Code("3000"), "Retained Earnings")
    assert retained_earnings.type == AccountType.EQUITIES
    
    # Test 4: Retrieve existing account with same parameters (should return existing)
    existing = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert existing == liquidity
    
    # Test 5: Error when parent and code are the same
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("1"), Code("1"), "Invalid")
    
    # Test 6: Error when parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not"):
        coa.add(Code("9999"), Code("9998"), "Invalid")
    
    # Test 7: Error when trying to add account with inconsistent data
    with pytest.raises(ValueError, match="Account name, code and parent do not match"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    # Test 8: Verify subaccounts are correctly tracked
    subaccounts_of_1 = coa.subaccounts(coa.find(Code("1")))
    assert liquidity in subaccounts_of_1
    assert len(subaccounts_of_1) == 1
    
    subaccounts_of_1000 = coa.subaccounts(coa.find(Code("1000")))
    assert bank_account in subaccounts_of_1000
    assert len(subaccounts_of_1000) == 1
    
    # Test 9: Verify iteration includes new accounts
    codes = [code for code, _ in coa]
    assert Code("1000") in codes
    assert Code("1001") in codes
    assert Code("2000") in codes
    assert Code("3000") in codes


# LLM-generated content at query #69
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function implements the protocol
    assert isinstance(read_coa, ReadChartOfAccounts)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts protocol with custom COA initialization."""
    
    def read_custom_coa() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
            AccountType.EQUITIES: (Code("30"), "Custom Equities"),
            AccountType.REVENUES: (Code("40"), "Custom Revenues"),
            AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
        }
        coa = COA(rootspec=custom_rootspec)
        return coa
    
    # Call the function
    result = read_custom_coa()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("30")) is not None
    assert result.find(Code("30")).name == "Custom Equities"
    assert result.find(Code("40")) is not None
    assert result.find(Code("40")).name == "Custom Revenues"
    assert result.find(Code("50")) is not None
    assert result.find(Code("50")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa_with_counter()
    coa2 = read_coa_with_counter()
    coa3 = read_coa_with_counter()
    
    # Verify each call returned a COA instance
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify call count
    assert call_count == 3
    
    # Verify each COA is independent
    assert coa1 is not coa2
    assert coa2 is not coa3


# LLM-generated content at query #70
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the callable satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result)
    assert len(accounts) == 5
    assert accounts[0][0] == Code("1")
    assert accounts[1][0] == Code("2")
    assert accounts[2][0] == Code("3")
    assert accounts[3][0] == Code("4")
    assert accounts[4][0] == Code("5")


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    
    # Define a custom implementation that initializes with specific accounts
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom account was added
    liquidity = result.find(Code("1000"))
    assert liquidity is not None
    assert liquidity.name == "Liquidity"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert list(coa1) == list(coa2)


# LLM-generated content at query #71
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance of the concrete implementation
    reader = ConcreteReadChartOfAccounts()
    
    # Call the instance and verify it returns a COA
    result = reader()
    
    # Assert the result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES
    
    # Test that the protocol works with runtime_checkable
    from typing import runtime_checkable
    assert isinstance(reader, ReadChartOfAccounts)


def test_ReadChartOfAccounts___call___multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.
    """
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = ConcreteReadChartOfAccounts()
    
    # Call multiple times
    coa1 = reader()
    coa2 = reader()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Verify both have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


def test_ReadChartOfAccounts___call___custom_rootspec():
    """
    Test that ReadChartOfAccounts can return COA with custom rootspec.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA(rootspec=custom_rootspec)
    
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    # Verify custom codes and names
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("30")).name == "Custom Equities"
    assert result.find(Code("40")).name == "Custom Revenues"
    assert result.find(Code("50")).name == "Custom Expenses"


# LLM-generated content at query #72
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    # Call the function
    result = read_coa()
    
    # Assert that it returns a COA instance
    assert isinstance(result, COA)
    
    # Assert that the COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1000")).name == "Liquidity"


def test_ReadChartOfAccounts___call___empty_coa():
    """Test that ReadChartOfAccounts protocol can return an empty COA with only root accounts."""
    
    def read_empty_coa() -> COA:
        return COA()
    
    result = read_empty_coa()
    
    assert isinstance(result, COA)
    assert result.find(Code("1")) is not None
    assert result.find(Code("1").name) == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts protocol can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert call_count == 2
    assert result1 is not result2


# LLM-generated content at query #73
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts can be called multiple times.
    """
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times and verify each returns a valid COA
    coa1 = read_coa()
    coa2 = read_coa()
    
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


def test_ReadChartOfAccounts___call___with_custom_spec():
    """
    Test that ReadChartOfAccounts can return COA with custom rootspec.
    """
    custom_spec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Custom Equities"),
        AccountType.REVENUES: (Code("40"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_coa()
    
    assert result.find(Code("10")).name == "Custom Assets"
    assert result.find(Code("20")).name == "Custom Liabilities"
    assert result.find(Code("30")).name == "Custom Equities"
    assert result.find(Code("40")).name == "Custom Revenues"
    assert result.find(Code("50")).name == "Custom Expenses"


# LLM-generated content at query #74
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts is callable and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the default 5 root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    # Verify the root accounts have the correct types in order
    expected_types = [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]
    actual_types = [account.type for account in accounts]
    assert actual_types == expected_types


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Current Assets"),
        AccountType.LIABILITIES: (Code("20"), "Current Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Owner Equity"),
        AccountType.REVENUES: (Code("40"), "Operating Revenues"),
        AccountType.EXPENSES: (Code("50"), "Operating Expenses"),
    }
    
    def read_custom_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_custom_coa()
    
    # Verify custom codes are used
    assert result.find(Code("10")).name == "Current Assets"
    assert result.find(Code("20")).name == "Current Liabilities"
    assert result.find(Code("30")).name == "Owner Equity"
    assert result.find(Code("40")).name == "Operating Revenues"
    assert result.find(Code("50")).name == "Operating Expenses"


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    def read_coa() -> COA:
        return COA()
    
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert list(c.code for c, _ in coa1) == list(c.code for c, _ in coa2)


# LLM-generated content at query #75
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts.__call__ returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function
    result = read_coa()
    
    # Assert that the result is a COA instance
    assert isinstance(result, COA)
    
    # Assert that the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Assert that root accounts have correct names and types
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("1")).type == AccountType.ASSETS
    
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("3")).type == AccountType.EQUITIES
    
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("4")).type == AccountType.REVENUES
    
    assert result.find(Code("5")).name == "Expenses"
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_rootspec():
    """Test ReadChartOfAccounts.__call__ with custom rootspec."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("L"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("E"), "Custom Equities"),
        AccountType.REVENUES: (Code("R"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("X"), "Custom Expenses"),
    }
    
    def read_coa() -> COA:
        coa = COA(rootspec=custom_rootspec)
        return coa
    
    result = read_coa()
    
    assert isinstance(result, COA)
    assert result.find(Code("A")).name == "Custom Assets"
    assert result.find(Code("L")).name == "Custom Liabilities"
    assert result.find(Code("E")).name == "Custom Equities"
    assert result.find(Code("R")).name == "Custom Revenues"
    assert result.find(Code("X")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___with_subaccounts():
    """Test ReadChartOfAccounts.__call__ that creates and returns COA with subaccounts."""
    
    def read_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    result = read_coa()
    
    assert isinstance(result, COA)
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"
    assert result.find(Code("1001")).parent.name == "Liquidity"


# LLM-generated content at query #76
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the callable returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    assert result is not None
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result)
    assert len(accounts) == 5
    
    codes = [code for code, _ in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """
    Test that ReadChartOfAccounts protocol works with custom rootspec.
    """
    custom_spec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
        AccountType.EQUITIES: (Code("E"), "My Equities"),
        AccountType.REVENUES: (Code("R"), "My Revenues"),
        AccountType.EXPENSES: (Code("X"), "My Expenses"),
    }
    
    def read_coa_custom() -> COA:
        return COA(rootspec=custom_spec)
    
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify custom codes are used
    assert result.find(Code("A")) is not None
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")) is not None
    assert result.find(Code("L")).name == "My Liabilities"
    assert result.find(Code("E")) is not None
    assert result.find(Code("E")).name == "My Equities"
    assert result.find(Code("R")) is not None
    assert result.find(Code("R")).name == "My Revenues"
    assert result.find(Code("X")) is not None
    assert result.find(Code("X")).name == "My Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """
    Test that ReadChartOfAccounts protocol can be called multiple times.
    """
    call_count = 0
    
    def read_coa_counted() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_counted()
    result2 = read_coa_counted()
    
    assert call_count == 2
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    # Each call should return a new instance
    assert result1 is not result2


# LLM-generated content at query #77
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify the function matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    accounts = list(result)
    assert len(accounts) == 5
    assert accounts[0][0] == Code("1")
    assert accounts[0][1].type == AccountType.ASSETS
    assert accounts[1][0] == Code("2")
    assert accounts[1][1].type == AccountType.LIABILITIES
    assert accounts[2][0] == Code("3")
    assert accounts[2][1].type == AccountType.EQUITIES
    assert accounts[3][0] == Code("4")
    assert accounts[3][1].type == AccountType.REVENUES
    assert accounts[4][0] == Code("5")
    assert accounts[4][1].type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_coa():
    """Test ReadChartOfAccounts protocol with custom COA configuration."""
    
    # Create a concrete implementation that returns a custom COA
    def read_custom_coa() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("A"), "My Assets"),
            AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
            AccountType.EQUITIES: (Code("E"), "My Equities"),
            AccountType.REVENUES: (Code("R"), "My Revenues"),
            AccountType.EXPENSES: (Code("X"), "My Expenses"),
        }
        return COA(rootspec=custom_rootspec)
    
    # Call the function and verify it returns a COA instance
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom codes and names
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")).name == "My Liabilities"
    assert result.find(Code("E")).name == "My Equities"
    assert result.find(Code("R")).name == "My Revenues"
    assert result.find(Code("X")).name == "My Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    
    # Verify both calls succeeded
    assert call_count == 2
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    # Verify they are different instances
    assert result1 is not result2


# LLM-generated content at query #78
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that the function is callable and conforms to the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"
    
    # Test that multiple calls return different COA instances
    result2 = read_coa()
    assert isinstance(result2, COA)
    assert result is not result2
    
    # Test with a more complex implementation that adds sub-accounts
    def read_coa_with_subaccounts() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    result3 = read_coa_with_subaccounts()
    assert isinstance(result3, COA)
    assert result3.find(Code("1000")) is not None
    assert result3.find(Code("1000")).name == "Liquidity"


# LLM-generated content at query #79
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify that the function is callable and returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify that the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify that the returned COA is a valid instance with accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")) is not None
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")) is not None
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")) is not None
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")) is not None
    assert result.find(Code("5")).name == "Expenses"


# LLM-generated content at query #80
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the callable matches the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected structure
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call__with_custom_implementation():
    """Test ReadChartOfAccounts protocol with a custom implementation."""
    
    # Define a custom implementation that creates a COA with sub-accounts
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(Code("1000"), Code("1001"), "Bank Account")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).parent.name == "Liquidity"


def test_ReadChartOfAccounts___call__multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa_with_counter() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    result1 = read_coa_with_counter()
    result2 = read_coa_with_counter()
    result3 = read_coa_with_counter()
    
    # Verify each call was executed
    assert call_count == 3
    assert isinstance(result1, COA)
    assert isinstance(result2, COA)
    assert isinstance(result3, COA)


# LLM-generated content at query #81
#--------------------------

```python
def test_COA_add():
    """Test the add method of COA class."""
    
    # Setup
    coa = COA()
    
    # Test 1: Successfully add a sub-account to a root account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity.type == AccountType.ASSETS
    assert coa.find(Code("1000")) == liquidity
    
    # Test 2: Successfully add a nested sub-account
    bank_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bank_account.code == Code("1001")
    assert bank_account.name == "Bank Account"
    assert bank_account.parent.code == Code("1000")
    assert bank_account.type == AccountType.ASSETS
    assert coa.find(Code("1001")) == bank_account
    
    # Test 3: Add multiple accounts to different parents
    liabilities = coa.add(Code("2"), Code("2000"), "Current Liabilities")
    assert liabilities.code == Code("2000")
    assert liabilities.type == AccountType.LIABILITIES
    assert coa.find(Code("2000")) == liabilities
    
    # Test 4: Return existing account if already exists with same details
    existing = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert existing == liquidity
    assert existing.code == Code("1000")
    assert existing.name == "Liquidity"
    
    # Test 5: Raise error when parent code equals code (self-parent)
    with pytest.raises(ValueError, match="An account can not be the parent of itself"):
        coa.add(Code("3000"), Code("3000"), "Invalid")
    
    # Test 6: Raise error when parent account does not exist
    with pytest.raises(ValueError, match="Parent account is not \\(yet\\) defined"):
        coa.add(Code("9999"), Code("9000"), "Non-existent Parent")
    
    # Test 7: Raise error when adding same code with different parent
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("2"), Code("1000"), "Liquidity")
    
    # Test 8: Raise error when adding same code with different name
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member"):
        coa.add(Code("1"), Code("1000"), "Different Name")
    
    # Test 9: Verify subaccounts are properly tracked
    subaccounts_of_1000 = coa.subaccounts(liquidity)
    assert len(subaccounts_of_1000) == 1
    assert bank_account in subaccounts_of_1000
    
    # Test 10: Add account to revenue account
    revenue = coa.add(Code("4"), Code("4000"), "Sales Revenue")
    assert revenue.type == AccountType.REVENUES
    assert revenue.parent.code == Code("4")


# LLM-generated content at query #82
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts protocol
    def read_coa() -> COA:
        return COA()
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected default root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts_protocol_implementation():
    """Test that ReadChartOfAccounts protocol works with different implementations."""
    # Implementation 1: Simple COA creation
    def reader1() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Test Account")
        return coa
    
    # Implementation 2: Custom root spec
    custom_spec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
        AccountType.EQUITIES: (Code("E"), "My Equities"),
        AccountType.REVENUES: (Code("R"), "My Revenues"),
        AccountType.EXPENSES: (Code("X"), "My Expenses"),
    }
    
    def reader2() -> COA:
        return COA(rootspec=custom_spec)
    
    # Both should be valid implementations
    coa1 = reader1()
    assert isinstance(coa1, COA)
    assert coa1.find(Code("1000")) is not None
    
    coa2 = reader2()
    assert isinstance(coa2, COA)
    assert coa2.find(Code("A")) is not None
    assert coa2.find(Code("A")).name == "My Assets"


def test_ReadChartOfAccounts_multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    call_count = [0]
    
    def read_coa() -> COA:
        call_count[0] += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    coa3 = read_coa()
    
    # Verify all are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    assert isinstance(coa3, COA)
    
    # Verify call count
    assert call_count[0] == 3
    
    # Verify each COA is independent
    coa1.add(Code("1"), Code("1001"), "Account 1")
    assert coa2.find(Code("1001")) is None
    assert coa3.find(Code("1001")) is None


# LLM-generated content at query #83
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts can be called and returns a COA instance."""
    
    # Create a simple implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Call the function
    result = read_coa()
    
    # Verify it returns a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify the root accounts have correct types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___custom_rootspec():
    """Test ReadChartOfAccounts with custom root specification."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("A"), "My Assets"),
        AccountType.LIABILITIES: (Code("L"), "My Liabilities"),
        AccountType.EQUITIES: (Code("E"), "My Equities"),
        AccountType.REVENUES: (Code("R"), "My Revenues"),
        AccountType.EXPENSES: (Code("X"), "My Expenses"),
    }
    
    def read_coa_custom() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa_custom()
    
    # Verify custom codes and names
    assert result.find(Code("A")).name == "My Assets"
    assert result.find(Code("L")).name == "My Liabilities"
    assert result.find(Code("E")).name == "My Equities"
    assert result.find(Code("R")).name == "My Revenues"
    assert result.find(Code("X")).name == "My Expenses"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name


# LLM-generated content at query #84
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can find accounts in the returned COA
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"


def test_ReadChartOfAccounts___call__with_custom_coa():
    """Test ReadChartOfAccounts with a custom COA configuration."""
    def read_custom_coa() -> COA:
        coa = COA()
        liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
        coa.add(liquidity.code, Code("1001"), "Bank Account")
        return coa
    
    result = read_custom_coa()
    assert isinstance(result, COA)
    
    # Verify custom accounts were added
    assert result.find(Code("1000")) is not None
    assert result.find(Code("1001")) is not None
    assert result.find(Code("1001")).name == "Bank Account"


def test_ReadChartOfAccounts___call__protocol_compliance():
    """Test that implementations comply with ReadChartOfAccounts protocol."""
    def read_coa_impl() -> COA:
        return COA()
    
    # Verify the implementation satisfies the protocol
    assert isinstance(read_coa_impl, ReadChartOfAccounts)
    
    # Call it and verify return type
    coa = read_coa_impl()
    assert isinstance(coa, COA)


# LLM-generated content at query #85
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the function implements the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected properties
    assert len(list(result.accounts)) == 5
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None


def test_ReadChartOfAccounts___call___with_custom_coa():
    """Test that ReadChartOfAccounts can return customized COA instances."""
    
    # Define a concrete implementation that returns a customized COA
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1100"), "Current Assets")
        return coa
    
    # Call and verify
    result = read_custom_coa()
    assert isinstance(result, COA)
    assert result.find(Code("1100")) is not None
    assert result.find(Code("1100")).name == "Current Assets"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Each should be a valid COA instance
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be separate instances
    assert coa1 is not coa2


# LLM-generated content at query #86
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation satisfies the protocol
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has expected root accounts
    root_accounts = list(result.toplevel)
    assert len(root_accounts) == 5
    
    # Verify account types are in correct order
    account_types = [acc.type for acc in root_accounts]
    assert account_types == [
        AccountType.ASSETS,
        AccountType.LIABILITIES,
        AccountType.EQUITIES,
        AccountType.REVENUES,
        AccountType.EXPENSES,
    ]


def test_ReadChartOfAccounts___call__with_custom_rootspec():
    """Test ReadChartOfAccounts with custom root specification."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("100"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
    }
    
    def read_coa_custom() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa_custom()
    assert isinstance(result, COA)
    
    # Verify custom codes and names
    assets_account = result.find(Code("100"))
    assert assets_account is not None
    assert assets_account.name == "Custom Assets"
    assert assets_account.type == AccountType.ASSETS
    
    liabilities_account = result.find(Code("200"))
    assert liabilities_account is not None
    assert liabilities_account.name == "Custom Liabilities"
    assert liabilities_account.type == AccountType.LIABILITIES


def test_ReadChartOfAccounts___call__multiple_invocations():
    """Test that ReadChartOfAccounts can be called multiple times independently."""
    
    def read_coa() -> COA:
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    
    # Both should be valid COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # They should be different instances
    assert coa1 is not coa2
    
    # But have the same structure
    assert len(list(coa1.toplevel)) == len(list(coa2.toplevel))


# LLM-generated content at query #87
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """Test that ReadChartOfAccounts protocol can be called and returns a COA instance."""
    
    # Define a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        coa = COA()
        return coa
    
    # Verify the implementation is callable
    assert callable(read_coa)
    
    # Call the function and verify it returns a COA instance
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    codes = [code for code, _ in result]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify we can find accounts in the returned COA
    assert result.find(Code("1")) is not None
    assert result.find(Code("1")).name == "Assets"


def test_ReadChartOfAccounts___call___multiple_calls():
    """Test that ReadChartOfAccounts can be called multiple times."""
    
    call_count = 0
    
    def read_coa() -> COA:
        nonlocal call_count
        call_count += 1
        return COA()
    
    # Call multiple times
    coa1 = read_coa()
    coa2 = read_coa()
    coa3 = read_coa()
    
    # Verify all calls succeeded
    assert call_count == 3
    assert all(isinstance(coa, COA) for coa in [coa1, coa2, coa3])
    
    # Verify each returned COA is independent
    assert coa1 is not coa2
    assert coa2 is not coa3


def test_ReadChartOfAccounts___call___with_custom_rootspec():
    """Test ReadChartOfAccounts with a custom rootspec."""
    
    custom_rootspec = {
        AccountType.ASSETS: (Code("10"), "Current Assets"),
        AccountType.LIABILITIES: (Code("20"), "Current Liabilities"),
        AccountType.EQUITIES: (Code("30"), "Owner Equity"),
        AccountType.REVENUES: (Code("40"), "Sales Revenue"),
        AccountType.EXPENSES: (Code("50"), "Operating Expenses"),
    }
    
    def read_coa() -> COA:
        return COA(rootspec=custom_rootspec)
    
    result = read_coa()
    
    # Verify custom codes are used
    assert result.find(Code("10")) is not None
    assert result.find(Code("10")).name == "Current Assets"
    assert result.find(Code("20")) is not None
    assert result.find(Code("20")).name == "Current Liabilities"
    assert result.find(Code("30")) is not None
    assert result.find(Code("30")).name == "Owner Equity"


# LLM-generated content at query #88
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test that ReadChartOfAccounts protocol can be called and returns a COA instance.
    """
    # Define a concrete implementation of ReadChartOfAccounts
    class ConcreteReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    # Create an instance and call it
    reader = ConcreteReadChartOfAccounts()
    result = reader()
    
    # Assert that result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the COA has the expected root accounts
    assert result.find(Code("1")) is not None
    assert result.find(Code("2")) is not None
    assert result.find(Code("3")) is not None
    assert result.find(Code("4")) is not None
    assert result.find(Code("5")) is not None
    
    # Verify root account types
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES


def test_ReadChartOfAccounts___call___with_custom_coa():
    """
    Test that ReadChartOfAccounts protocol can return custom COA instances.
    """
    custom_rootspec = {
        AccountType.ASSETS: (Code("100"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("200"), "Custom Liabilities"),
        AccountType.EQUITIES: (Code("300"), "Custom Equities"),
        AccountType.REVENUES: (Code("400"), "Custom Revenues"),
        AccountType.EXPENSES: (Code("500"), "Custom Expenses"),
    }
    
    class CustomReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA(rootspec=custom_rootspec)
    
    reader = CustomReadChartOfAccounts()
    result = reader()
    
    # Assert that result is a COA instance
    assert isinstance(result, COA)
    
    # Verify the custom codes and names
    assert result.find(Code("100")).name == "Custom Assets"
    assert result.find(Code("200")).name == "Custom Liabilities"
    assert result.find(Code("300")).name == "Custom Equities"
    assert result.find(Code("400")).name == "Custom Revenues"
    assert result.find(Code("500")).name == "Custom Expenses"


def test_ReadChartOfAccounts___call___multiple_invocations():
    """
    Test that ReadChartOfAccounts can be called multiple times and returns independent COA instances.
    """
    class MultiCallReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()
    
    reader = MultiCallReadChartOfAccounts()
    coa1 = reader()
    coa2 = reader()
    
    # Verify both are COA instances
    assert isinstance(coa1, COA)
    assert isinstance(coa2, COA)
    
    # Verify they are different instances
    assert coa1 is not coa2
    
    # Verify both have the same structure
    assert coa1.find(Code("1")).name == coa2.find(Code("1")).name
    assert coa1.find(Code("5")).name == coa2.find(Code("5")).name


# LLM-generated content at query #89
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    """
    Test the __call__ method of ReadChartOfAccounts protocol.
    """
    # Create a concrete implementation of ReadChartOfAccounts
    def read_coa() -> COA:
        return COA()
    
    # Verify that the function is callable and returns a COA instance
    assert callable(read_coa)
    result = read_coa()
    assert isinstance(result, COA)
    
    # Verify the returned COA has the expected root accounts
    accounts = list(result.accounts)
    assert len(accounts) == 5
    
    codes = [acc.code for acc in accounts]
    assert Code("1") in codes
    assert Code("2") in codes
    assert Code("3") in codes
    assert Code("4") in codes
    assert Code("5") in codes
    
    # Verify account types are correct
    assert result.find(Code("1")).type == AccountType.ASSETS
    assert result.find(Code("2")).type == AccountType.LIABILITIES
    assert result.find(Code("3")).type == AccountType.EQUITIES
    assert result.find(Code("4")).type == AccountType.REVENUES
    assert result.find(Code("5")).type == AccountType.EXPENSES
    
    # Test with a more complex implementation
    def read_custom_coa() -> COA:
        coa = COA()
        coa.add(Code("1"), Code("1000"), "Liquidity")
        return coa
    
    assert callable(read_custom_coa)
    custom_result = read_custom_coa()
    assert isinstance(custom_result, COA)
    assert custom_result.find(Code("1000")) is not None
    assert custom_result.find(Code("1000")).name == "Liquidity"


