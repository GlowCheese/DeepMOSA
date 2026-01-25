####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_COA_add():
    coa = COA()

    # Test adding a new account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")

    # Test adding a sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")

    # Test adding an existing account with same details
    existing_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing_account.code == Code("1001")
    assert existing_account.name == "Bank Account"
    assert existing_account.parent.code == Code("1000")

    # Test adding an account with different details (should raise ValueError)
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding an account with non-existent parent (should raise ValueError)
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code (should raise ValueError)
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Same Parent and Code")


# LLM-generated content at query #2
#--------------------------

```python
def test_COA_add():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity

    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bankaccnt

    with pytest.raises(ValueError):
        coa.add(Code("1"), Code("1"), "Invalid")

    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    existing = coa.add(Code("1000"), Code("1000"), "Liquidity")
    assert existing == liquidity

    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1000"), "Different Name")


# LLM-generated content at query #3
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements the ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Assign the mock function to a variable with the ReadChartOfAccounts type
    read_coa: ReadChartOfAccounts = mock_read_coa

    # Call the function
    result = read_coa()

    # Assert that the result is an instance of COA
    assert isinstance(result, COA)

    # Assert that the COA has the default root accounts
    assert len(list(result)) == 5
    assert result.find(Code("1")).name == "Assets"
    assert result.find(Code("2")).name == "Liabilities"
    assert result.find(Code("3")).name == "Equities"
    assert result.find(Code("4")).name == "Revenues"
    assert result.find(Code("5")).name == "Expenses"


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA has the default root accounts
    assert len(list(coa)) == 5
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value[0]))) is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_COA___iter__():
    coa = COA()
    codes_and_accounts = list(coa)
    assert len(codes_and_accounts) == 5
    for code, account in codes_and_accounts:
        assert isinstance(code, Code)
        assert isinstance(account, Account)
        assert account.coa == coa
    assert codes_and_accounts[0][0] == Code("1")
    assert codes_and_accounts[0][1].name == "Assets"
    assert codes_and_accounts[1][0] == Code("2")
    assert codes_and_accounts[1][1].name == "Liabilities"
    assert codes_and_accounts[2][0] == Code("3")
    assert codes_and_accounts[2][1].name == "Equities"
    assert codes_and_accounts[3][0] == Code("4")
    assert codes_and_accounts[3][1].name == "Revenues"
    assert codes_and_accounts[4][0] == Code("5")
    assert codes_and_accounts[4][1].name == "Expenses"


# LLM-generated content at query #6
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock implementation of ReadChartOfAccounts
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    # Instantiate the mock
    reader = MockReadChartOfAccounts()

    # Call the method
    result = reader()

    # Assert the result is an instance of COA
    assert isinstance(result, COA)

    # Assert the COA has the expected default accounts
    expected_accounts = {
        Code("1"): "Assets",
        Code("2"): "Liabilities",
        Code("3"): "Equities",
        Code("4"): "Revenues",
        Code("5"): "Expenses"
    }

    for code, name in result:
        assert expected_accounts[code] == name


# LLM-generated content at query #7
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA has the default root accounts
    assert len(list(coa)) == 5
    expected_codes = [Code("1"), Code("2"), Code("3"), Code("4"), Code("5")]
    expected_names = ["Assets", "Liabilities", "Equities", "Revenues", "Expenses"]
    for (code, account), expected_code, expected_name in zip(coa, expected_codes, expected_names):
        assert code == expected_code
        assert account.name == expected_name


# LLM-generated content at query #8
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that adheres to ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA has the default root accounts
    assert len(list(coa)) == 5
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value[0]))) is not None

    # Test that the COA is empty of sub-accounts initially
    for account in coa.accounts:
        assert len(coa.subaccounts(account)) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    reader = MockReadChartOfAccounts()
    coa = reader()

    assert isinstance(coa, COA)
    assert len(list(coa)) == 5

    expected_codes = [Code("1"), Code("2"), Code("3"), Code("4"), Code("5")]
    expected_names = ["Assets", "Liabilities", "Equities", "Revenues", "Expenses"]

    for (code, account), expected_code, expected_name in zip(coa, expected_codes, expected_names):
        assert code == expected_code
        assert account.name == expected_name


# LLM-generated content at query #10
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Verify that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Verify that the COA has the default root accounts
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"

    # Verify that the function can be called multiple times and returns a new COA each time
    coa2 = mock_read_coa()
    assert coa2 is not coa
    assert len(list(coa2)) == 5


# LLM-generated content at query #11
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Verify that the mock function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Verify that the returned COA has the expected structure
    expected_accounts = [
        (Code("1"), "Assets"),
        (Code("2"), "Liabilities"),
        (Code("3"), "Equities"),
        (Code("4"), "Revenues"),
        (Code("5"), "Expenses"),
    ]

    for code, name in expected_accounts:
        account = coa.find(code)
        assert account is not None
        assert account.name == name
        assert account.parent is None


# LLM-generated content at query #12
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the returned COA has the default root accounts
    assert len(list(coa)) == 5
    expected_codes = [Code("1"), Code("2"), Code("3"), Code("4"), Code("5")]
    expected_names = ["Assets", "Liabilities", "Equities", "Revenues", "Expenses"]
    for (code, account), expected_code, expected_name in zip(coa, expected_codes, expected_names):
        assert code == expected_code
        assert account.name == expected_name


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements the ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Assign the mock function to a variable of type ReadChartOfAccounts
    read_coa: ReadChartOfAccounts = mock_read_coa

    # Call the function and verify it returns a COA instance
    coa = read_coa()
    assert isinstance(coa, COA)

    # Verify the COA has the default root accounts
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #14
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Mock implementation of ReadChartOfAccounts
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    # Create an instance of the mock
    mock_reader = MockReadChartOfAccounts()

    # Call the method
    coa = mock_reader()

    # Verify the result is a COA instance
    assert isinstance(coa, COA)

    # Verify the COA has the default root accounts
    assert len(list(coa)) == 5
    expected_codes = {Code("1"), Code("2"), Code("3"), Code("4"), Code("5")}
    actual_codes = {code for code, _ in coa}
    assert actual_codes == expected_codes

    # Verify the root accounts have correct names and types
    for code, account in coa:
        if code == Code("1"):
            assert account.name == "Assets"
            assert account.type == AccountType.ASSETS
        elif code == Code("2"):
            assert account.name == "Liabilities"
            assert account.type == AccountType.LIABILITIES
        elif code == Code("3"):
            assert account.name == "Equities"
            assert account.type == AccountType.EQUITIES
        elif code == Code("4"):
            assert account.name == "Revenues"
            assert account.type == AccountType.REVENUES
        elif code == Code("5"):
            assert account.name == "Expenses"
            assert account.type == AccountType.EXPENSES


# LLM-generated content at query #15
#--------------------------

```python
def test_COA_nodify():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")

    root_account = coa.find(Code("1"))
    node = coa.nodify(root_account)

    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == liquidity
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == bankaccnt
    assert len(node.children[0].children[0].children) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    result = mock_read_coa()
    assert isinstance(result, COA)

    # Test that the returned COA has the default root accounts
    expected_codes = [Code("1"), Code("2"), Code("3"), Code("4"), Code("5")]
    expected_names = ["Assets", "Liabilities", "Equities", "Revenues", "Expenses"]
    for code, name in zip(expected_codes, expected_names):
        account = result.find(code)
        assert account is not None
        assert account.name == name


# LLM-generated content at query #17
#--------------------------

```python
def test_COA_nodify():
    # Create a new COA instance
    coa = COA()

    # Add some accounts to the COA
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_accnt = coa.add(liquidity.code, Code("1001"), "Bank Account")
    cash_accnt = coa.add(liquidity.code, Code("1002"), "Cash Account")

    # Get the root account (Assets)
    root_account = coa.find(Code("1"))

    # Test nodify with root account
    root_node = coa.nodify(root_account)
    assert root_node.account == root_account
    assert len(root_node.children) == 1  # Only Liquidity is a direct child
    assert root_node.children[0].account == liquidity

    # Test nodify with sub-account (Liquidity)
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account == liquidity
    assert len(liquidity_node.children) == 2  # Bank Account and Cash Account
    assert liquidity_node.children[0].account == bank_accnt
    assert liquidity_node.children[1].account == cash_accnt

    # Test nodify with leaf account (Bank Account)
    bank_node = coa.nodify(bank_accnt)
    assert bank_node.account == bank_accnt
    assert len(bank_node.children) == 0  # No children

    # Test nodify with non-existent account (should still work but return empty children)
    # This is more of a structure test since we're using existing accounts
    cash_node = coa.nodify(cash_accnt)
    assert cash_node.account == cash_accnt
    assert len(cash_node.children) == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_COA_add():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity
    assert liquidity in coa.subaccounts(coa.find(Code("1")))

    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bankaccnt
    assert bankaccnt in coa.subaccounts(coa.find(Code("1000")))

    with pytest.raises(ValueError):
        coa.add(Code("1"), Code("1"), "Invalid")

    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    existing = coa.add(Code("1000"), Code("1000"), "Liquidity")
    assert existing == liquidity

    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1000"), "Different Name")


# LLM-generated content at query #19
#--------------------------

```python
def test_COA_add():
    coa = COA()
    parent_code = Code("1")
    code = Code("1000")
    name = "Liquidity"

    # Test adding a new account
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert coa.find(code) == account

    # Test adding an existing account with same details
    account2 = coa.add(parent_code, code, name)
    assert account2 == account

    # Test adding an account with different details
    with pytest.raises(ValueError):
        coa.add(parent_code, code, "Different Name")

    # Test adding an account with non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1001"), "Orphan Account")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(code, code, "Self Parent")

    # Test adding a sub-account
    sub_account = coa.add(code, Code("1001"), "Bank Account")
    assert sub_account.parent.code == code
    assert coa.find(Code("1001")) == sub_account


# LLM-generated content at query #20
#--------------------------

```python
def test_COA_nodify():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")

    structure = list(coa.structure)

    assert len(structure) == 5

    assets_node = structure[0]
    assert assets_node.account.code == Code("1")
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1

    liquidity_node = assets_node.children[0]
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1

    bankaccnt_node = liquidity_node.children[0]
    assert bankaccnt_node.account.code == Code("1001")
    assert bankaccnt_node.account.name == "Bank Account"
    assert len(bankaccnt_node.children) == 0

    for i, account_type in enumerate([AccountType.LIABILITIES, AccountType.EQUITIES, AccountType.REVENUES, AccountType.EXPENSES], start=1):
        node = structure[i]
        assert node.account.code == Code(str(i + 1))
        assert node.account.name == account_type.name.capitalize()
        assert len(node.children) == 0


# LLM-generated content at query #21
#--------------------------

```python
def test_COA_add():
    # Test basic account addition
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity

    # Test adding sub-account to newly created account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bankaccnt

    # Test adding existing account with same parameters
    existing = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing == bankaccnt

    # Test adding account with same code but different parameters
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding account with non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding account with same parent and code
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Self Parent")

    # Test adding account to different root accounts
    liability = coa.add(Code("2"), Code("2000"), "Long-term Liabilities")
    assert liability.parent.code == Code("2")
    assert liability.type == AccountType.LIABILITIES

    # Test that subaccounts are properly tracked
    assert len(coa.subaccounts(liquidity)) == 1
    assert coa.subaccounts(liquidity)[0] == bankaccnt
    assert len(coa.subaccounts(liability)) == 0

    # Test that accounts are in the iteration
    codes = [code for code, _ in coa]
    assert Code("1000") in codes
    assert Code("1001") in codes
    assert Code("2000") in codes


# LLM-generated content at query #22
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA has the default root accounts
    assert len(list(coa)) == 5
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value[0]))) is not None


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Define a simple implementation of ReadChartOfAccounts for testing
    class SimpleCOAReader:
        def __call__(self) -> COA:
            return COA()

    # Create an instance of the reader
    reader = SimpleCOAReader()

    # Call the reader and verify it returns a COA instance
    coa = reader()
    assert isinstance(coa, COA)

    # Verify the COA has the default root accounts
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #24
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA instance has the expected root accounts
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

    # Test that the root accounts have the correct names
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #25
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Mock a ReadChartOfAccounts instance
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    # Create an instance of the mock
    reader = MockReadChartOfAccounts()

    # Call the method
    result = reader()

    # Assert the result is an instance of COA
    assert isinstance(result, COA)

    # Assert the COA has the default root accounts
    assert len(list(result)) == 5
    for code, account in result:
        assert isinstance(code, Code)
        assert isinstance(account, Account)
        assert account.parent is None


# LLM-generated content at query #26
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the returned COA has the default accounts
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    reader = MockReadChartOfAccounts()
    coa = reader()

    assert isinstance(coa, COA)
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #2
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the returned COA has the expected default accounts
    expected_accounts = {
        Code("1"): "Assets",
        Code("2"): "Liabilities",
        Code("3"): "Equities",
        Code("4"): "Revenues",
        Code("5"): "Expenses",
    }

    for code, name in expected_accounts.items():
        account = coa.find(code)
        assert account is not None
        assert account.name == name
        assert account.code == code

    # Test that the function can be called multiple times and returns new instances
    coa2 = mock_read_coa()
    assert coa is not coa2
    assert isinstance(coa2, COA)

    # Test that the function can be used with different implementations
    def another_read_coa() -> COA:
        custom_rootspec = {
            AccountType.ASSETS: (Code("10"), "Custom Assets"),
            AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        }
        return COA(custom_rootspec)

    coa3 = another_read_coa()
    assert isinstance(coa3, COA)
    assert coa3.find(Code("10")).name == "Custom Assets"
    assert coa3.find(Code("20")).name == "Custom Liabilities"


# LLM-generated content at query #3
#--------------------------

```python
def test_COA_add():
    coa = COA()

    # Test adding a valid sub-account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity

    # Test adding a sub-account to a non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("999"), Code("1001"), "Invalid Parent")

    # Test adding an account with the same code as parent
    with pytest.raises(ValueError):
        coa.add(Code("1"), Code("1"), "Same Code")

    # Test adding an existing account with consistent data
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")

    # Test adding an existing account with inconsistent data
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding multiple levels of sub-accounts
    cash = coa.add(Code("1001"), Code("1002"), "Cash")
    assert cash.code == Code("1002")
    assert cash.name == "Cash"
    assert cash.parent.code == Code("1001")

    # Verify the structure
    assert len(list(coa)) == 8  # 5 root + 3 added
    assert coa.find(Code("1002")).parent.name == "Bank Account"


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA has the default root accounts
    assert len(list(coa)) == 5
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value[0]))) is not None

    # Test that the function can be called multiple times
    coa2 = mock_read_coa()
    assert isinstance(coa2, COA)
    assert coa is not coa2


# LLM-generated content at query #5
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that returns a COA instance
    def mock_read_coa() -> COA:
        return COA()

    # Ensure the mock function adheres to the ReadChartOfAccounts protocol
    read_coa: ReadChartOfAccounts = mock_read_coa

    # Call the function
    coa = read_coa()

    # Verify the returned object is a COA instance
    assert isinstance(coa, COA)

    # Verify the COA has the expected default accounts
    expected_accounts = {
        Code("1"): "Assets",
        Code("2"): "Liabilities",
        Code("3"): "Equities",
        Code("4"): "Revenues",
        Code("5"): "Expenses",
    }

    for code, name in expected_accounts.items():
        account = coa.find(code)
        assert account is not None
        assert account.name == name


# LLM-generated content at query #6
#--------------------------

```python
def test_COA_nodify():
    # Initialize a chart of accounts
    coa = COA()

    # Add some accounts to create a hierarchy
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bank_accnt = coa.add(liquidity.code, Code("1001"), "Bank Account")
    cash_accnt = coa.add(liquidity.code, Code("1002"), "Cash Account")

    # Test nodify for a root account with subaccounts
    assets_node = coa.nodify(coa.find(Code("1")))
    assert assets_node.account.code == Code("1")
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account.code == Code("1000")

    # Test nodify for a subaccount with further subaccounts
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 2
    assert liquidity_node.children[0].account.code == Code("1001")
    assert liquidity_node.children[1].account.code == Code("1002")

    # Test nodify for a leaf account (no children)
    bank_node = coa.nodify(bank_accnt)
    assert bank_node.account.code == Code("1001")
    assert bank_node.account.name == "Bank Account"
    assert len(bank_node.children) == 0

    # Test nodify for a root account without subaccounts
    liabilities_node = coa.nodify(coa.find(Code("2")))
    assert liabilities_node.account.code == Code("2")
    assert liabilities_node.account.name == "Liabilities"
    assert len(liabilities_node.children) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_COA_nodify():
    # Create a COA instance
    coa = COA()

    # Get a root account
    assets = coa.find(Code("1"))

    # Create a sub-account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")

    # Create a sub-sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")

    # Test nodify on root account
    assets_node = coa.nodify(assets)
    assert assets_node.account == assets
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account == liquidity

    # Test nodify on sub-account
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account == liquidity
    assert len(liquidity_node.children) == 1
    assert liquidity_node.children[0].account == bankaccnt

    # Test nodify on leaf account
    bankaccnt_node = coa.nodify(bankaccnt)
    assert bankaccnt_node.account == bankaccnt
    assert len(bankaccnt_node.children) == 0

    # Test nodify on all top-level accounts
    for account in coa.toplevel:
        node = coa.nodify(account)
        assert node.account == account
        assert isinstance(node.children, list)


# LLM-generated content at query #8
#--------------------------

```python
def test_COA_nodify():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")

    # Test nodify on root account
    root_node = coa.nodify(coa.find(Code("1")))
    assert root_node.account.code == Code("1")
    assert root_node.account.name == "Assets"
    assert len(root_node.children) == 1
    assert root_node.children[0].account.code == Code("1000")

    # Test nodify on sub-account
    sub_node = coa.nodify(liquidity)
    assert sub_node.account.code == Code("1000")
    assert sub_node.account.name == "Liquidity"
    assert len(sub_node.children) == 1
    assert sub_node.children[0].account.code == Code("1001")

    # Test nodify on leaf account
    leaf_node = coa.nodify(bankaccnt)
    assert leaf_node.account.code == Code("1001")
    assert leaf_node.account.name == "Bank Account"
    assert len(leaf_node.children) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_COA_add():
    coa = COA()
    parent_code = Code("1")
    code = Code("1000")
    name = "Test Account"

    # Test adding a new account
    new_account = coa.add(parent_code, code, name)
    assert new_account.code == code
    assert new_account.name == name
    assert new_account.parent.code == parent_code
    assert coa.find(code) == new_account

    # Test adding an existing account with same details
    existing_account = coa.add(parent_code, code, name)
    assert existing_account == new_account

    # Test adding an account with different details
    with pytest.raises(ValueError):
        coa.add(parent_code, code, "Different Name")

    # Test adding an account with non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1001"), "Orphan Account")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(code, code, "Self Parent Account")


# LLM-generated content at query #10
#--------------------------

```python
def test_COA_nodify():
    # Create a chart of accounts
    coa = COA()

    # Add some accounts to create a hierarchy
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")
    cashaccnt = coa.add(liquidity.code, Code("1002"), "Cash Account")

    # Test nodify with a root account (Assets)
    assets = coa.find(Code("1"))
    assets_node = coa.nodify(assets)
    assert assets_node.account == assets
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account == liquidity

    # Test nodify with a sub-account (Liquidity)
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account == liquidity
    assert len(liquidity_node.children) == 2
    assert liquidity_node.children[0].account == bankaccnt
    assert liquidity_node.children[1].account == cashaccnt

    # Test nodify with a leaf account (Bank Account)
    bankaccnt_node = coa.nodify(bankaccnt)
    assert bankaccnt_node.account == bankaccnt
    assert len(bankaccnt_node.children) == 0

    # Test nodify with a non-existent account (should raise KeyError)
    try:
        coa.nodify(SubAccount(Code("9999"), "Non-existent", liquidity))
    except KeyError:
        pass
    else:
        assert False, "Expected KeyError for non-existent account"


# LLM-generated content at query #11
#--------------------------

```python
def test_COA_add():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity
    assert coa.subaccounts(coa.find(Code("1"))) == [liquidity]

    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bankaccnt
    assert coa.subaccounts(coa.find(Code("1000"))) == [bankaccnt]

    with pytest.raises(ValueError):
        coa.add(Code("1"), Code("1"), "Invalid")

    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    existing_account = coa.add(Code("1001"), Code("1001"), "Bank Account")
    assert existing_account == bankaccnt

    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Different Name")


# LLM-generated content at query #12
#--------------------------

```python
def test_COA_add():
    # Test adding a new account to the chart of accounts
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity.type == AccountType.ASSETS

    # Test adding a sub-account to an existing account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert bankaccnt.type == AccountType.ASSETS

    # Test adding an account with the same code as parent
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Invalid Account")

    # Test adding an account with a non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent Account")

    # Test adding an account that already exists with consistent information
    existing_account = coa.add(Code("1000"), Code("1000"), "Liquidity")
    assert existing_account.code == Code("1000")
    assert existing_account.name == "Liquidity"
    assert existing_account.parent.code == Code("1")

    # Test adding an account that already exists with inconsistent information
    with pytest.raises(ValueError):
        coa.add(Code("1"), Code("1000"), "Different Name")

    # Test that the account is added to the COA's internal buffers
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("1001")) is not None
    assert liquidity in coa.subaccounts(coa.find(Code("1")))
    assert bankaccnt in coa.subaccounts(coa.find(Code("1000")))


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that returns a COA instance
    def mock_read_coa() -> COA:
        return COA()

    # Check if the mock function adheres to the ReadChartOfAccounts protocol
    assert isinstance(mock_read_coa, ReadChartOfAccounts)

    # Call the function and verify it returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Verify the COA has the expected initial accounts
    expected_accounts = {
        Code("1"): "Assets",
        Code("2"): "Liabilities",
        Code("3"): "Equities",
        Code("4"): "Revenues",
        Code("5"): "Expenses",
    }
    for code, name in expected_accounts.items():
        account = coa.find(code)
        assert account is not None
        assert account.name == name


# LLM-generated content at query #14
#--------------------------

```python
def test_COA_nodify():
    # Create a COA instance
    coa = COA()

    # Add some accounts to the COA
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")

    # Test nodify for a root account (Assets)
    assets_node = coa.nodify(coa.find(Code("1")))
    assert assets_node.account.code == Code("1")
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account.code == Code("1000")

    # Test nodify for a sub-account (Liquidity)
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    assert liquidity_node.children[0].account.code == Code("1001")

    # Test nodify for a leaf account (Bank Account)
    bankaccnt_node = coa.nodify(bankaccnt)
    assert bankaccnt_node.account.code == Code("1001")
    assert bankaccnt_node.account.name == "Bank Account"
    assert len(bankaccnt_node.children) == 0

    # Test nodify for a root account with no children (Liabilities)
    liabilities_node = coa.nodify(coa.find(Code("2")))
    assert liabilities_node.account.code == Code("2")
    assert liabilities_node.account.name == "Liabilities"
    assert len(liabilities_node.children) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_COA_add():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert coa.find(Code("1000")) == liquidity
    assert coa.subaccounts(coa.find(Code("1"))) == [liquidity]

    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert coa.find(Code("1001")) == bankaccnt
    assert coa.subaccounts(coa.find(Code("1000"))) == [bankaccnt]

    with pytest.raises(ValueError):
        coa.add(Code("1"), Code("1"), "Invalid")

    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    existing = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing == bankaccnt

    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")


# LLM-generated content at query #16
#--------------------------

```python
def test_COA_add():
    coa = COA()

    # Test adding a new account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity in coa._accounts.values()

    # Test adding a sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert bankaccnt in coa._accounts.values()

    # Test adding an existing account with same details
    existing_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing_account == bankaccnt

    # Test adding an account with same code but different details
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding an account with non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Self Parent")


# LLM-generated content at query #17
#--------------------------

```python
def test_COA_add():
    coa = COA()

    # Test adding a new account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")

    # Test adding a sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")

    # Test adding an existing account with same details
    liquidity_again = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity_again.code == Code("1000")
    assert liquidity_again.name == "Liquidity"
    assert liquidity_again.parent.code == Code("1")

    # Test adding an account with wrong parent
    with pytest.raises(ValueError):
        coa.add(Code("999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1000"), "Same Parent and Code")

    # Test adding an account with different details than existing
    with pytest.raises(ValueError):
        coa.add(Code("1"), Code("1000"), "Different Name")


# LLM-generated content at query #18
#--------------------------

```python
def test_COA_add():
    coa = COA()

    # Test adding a new account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")

    # Test adding a sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")

    # Test adding an existing account with same details
    existing = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing.code == Code("1001")
    assert existing.name == "Bank Account"
    assert existing.parent.code == Code("1000")

    # Test adding an account with same code but different details
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding an account with non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Same Parent and Code")


# LLM-generated content at query #19
#--------------------------

```python
def test_COA_add():
    coa = COA()

    # Test adding a new account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")

    # Test adding a sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")

    # Test adding an existing account with same details
    existing_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing_account.code == Code("1001")
    assert existing_account.name == "Bank Account"

    # Test adding an account with different details should raise ValueError
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding an account with non-existent parent should raise ValueError
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code should raise ValueError
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Self Parent")


# LLM-generated content at query #20
#--------------------------

```python
def test_COA_add():
    # Test adding a new account
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")

    # Test adding a sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")

    # Test adding an existing account with same details
    existing_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing_account.code == Code("1001")
    assert existing_account.name == "Bank Account"
    assert existing_account.parent.code == Code("1000")

    # Test adding an account with different details
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding an account with non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Same Parent and Code")


# LLM-generated content at query #21
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the returned COA has the default root accounts
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"

    # Test that the function can be called multiple times and returns new instances
    coa2 = mock_read_coa()
    assert coa is not coa2
    assert isinstance(coa2, COA)


# LLM-generated content at query #22
#--------------------------

```python
def test_COA_add():
    # Test adding a sub-account to a root account
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")
    assert liquidity in coa.subaccounts(coa.find(Code("1")))

    # Test adding a sub-account to a sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")
    assert bankaccnt in coa.subaccounts(coa.find(Code("1000")))

    # Test adding an existing account with consistent information
    existing_acct = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing_acct == bankaccnt

    # Test adding an account with parent not in COA
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Invalid Parent")

    # Test adding an account with inconsistent existing information
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA has the expected default accounts
    expected_accounts = {
        Code("1"): "Assets",
        Code("2"): "Liabilities",
        Code("3"): "Equities",
        Code("4"): "Revenues",
        Code("5"): "Expenses",
    }

    for code, name in expected_accounts.items():
        account = coa.find(code)
        assert account is not None
        assert account.name == name
        assert account.parent is None

    # Test that the COA is iterable and contains the expected accounts
    accounts = list(coa)
    assert len(accounts) == len(expected_accounts)
    for code, account in accounts:
        assert code in expected_accounts
        assert account.name == expected_accounts[code]


# LLM-generated content at query #24
#--------------------------

```python
def test_COA_nodify():
    coa = COA()
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")

    # Test nodify for root account
    assets_node = coa.nodify(coa.find(Code("1")))
    assert assets_node.account.code == Code("1")
    assert assets_node.account.name == "Assets"
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account.code == Code("1000")

    # Test nodify for subaccount
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account.code == Code("1000")
    assert liquidity_node.account.name == "Liquidity"
    assert len(liquidity_node.children) == 1
    assert liquidity_node.children[0].account.code == Code("1001")

    # Test nodify for leaf account
    bankaccnt_node = coa.nodify(bankaccnt)
    assert bankaccnt_node.account.code == Code("1001")
    assert bankaccnt_node.account.name == "Bank Account"
    assert len(bankaccnt_node.children) == 0

    # Test nodify for account with no children
    liabilities_node = coa.nodify(coa.find(Code("2")))
    assert liabilities_node.account.code == Code("2")
    assert liabilities_node.account.name == "Liabilities"
    assert len(liabilities_node.children) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_COA_nodify():
    # Create a COA instance
    coa = COA()

    # Add some accounts to the COA
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")

    # Get the root account
    root_account = coa.find(Code("1"))

    # Test nodify with root account
    node = coa.nodify(root_account)
    assert node.account == root_account
    assert len(node.children) == 1
    assert node.children[0].account == liquidity

    # Test nodify with sub-account
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account == liquidity
    assert len(liquidity_node.children) == 1
    assert liquidity_node.children[0].account == bankaccnt

    # Test nodify with leaf account
    bankaccnt_node = coa.nodify(bankaccnt)
    assert bankaccnt_node.account == bankaccnt
    assert len(bankaccnt_node.children) == 0

    # Test nodify with non-existent account
    non_existent_account = RootAccount(Code("999"), "Non-existent", AccountType.ASSETS, coa)
    non_existent_node = coa.nodify(non_existent_account)
    assert non_existent_node.account == non_existent_account
    assert len(non_existent_node.children) == 0


# LLM-generated content at query #26
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    class MockReadChartOfAccounts:
        def __call__(self) -> COA:
            return COA()

    reader = MockReadChartOfAccounts()
    coa = reader()

    assert isinstance(coa, COA)
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equities"
    assert coa.find(Code("4")).name == "Revenues"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #27
#--------------------------

```python
def test_COA_nodify():
    # Create a new COA instance
    coa = COA()

    # Add some accounts to the COA
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    bankaccnt = coa.add(liquidity.code, Code("1001"), "Bank Account")

    # Test nodify with a root account (Assets)
    assets_account = coa.find(Code("1"))
    assets_node = coa.nodify(assets_account)
    assert assets_node.account == assets_account
    assert len(assets_node.children) == 1
    assert assets_node.children[0].account == liquidity

    # Test nodify with a sub-account (Liquidity)
    liquidity_node = coa.nodify(liquidity)
    assert liquidity_node.account == liquidity
    assert len(liquidity_node.children) == 1
    assert liquidity_node.children[0].account == bankaccnt

    # Test nodify with a leaf account (Bank Account)
    bankaccnt_node = coa.nodify(bankaccnt)
    assert bankaccnt_node.account == bankaccnt
    assert len(bankaccnt_node.children) == 0

    # Test nodify with an account that has no children (Liabilities)
    liabilities_account = coa.find(Code("2"))
    liabilities_node = coa.nodify(liabilities_account)
    assert liabilities_node.account == liabilities_account
    assert len(liabilities_node.children) == 0


# LLM-generated content at query #28
#--------------------------

```python
def test_COA_add():
    # Initialize a COA
    coa = COA()

    # Test adding a valid sub-account
    liquidity = coa.add(Code("1"), Code("1000"), "Liquidity")
    assert liquidity.code == Code("1000")
    assert liquidity.name == "Liquidity"
    assert liquidity.parent.code == Code("1")

    # Test adding a nested sub-account
    bankaccnt = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert bankaccnt.code == Code("1001")
    assert bankaccnt.name == "Bank Account"
    assert bankaccnt.parent.code == Code("1000")

    # Test adding an existing account with same details
    existing_account = coa.add(Code("1000"), Code("1001"), "Bank Account")
    assert existing_account.code == Code("1001")
    assert existing_account.name == "Bank Account"

    # Test adding an account with same code but different details
    with pytest.raises(ValueError):
        coa.add(Code("1000"), Code("1001"), "Different Name")

    # Test adding an account with non-existent parent
    with pytest.raises(ValueError):
        coa.add(Code("9999"), Code("1002"), "Invalid Parent")

    # Test adding an account with same parent and code
    with pytest.raises(ValueError):
        coa.add(Code("1001"), Code("1001"), "Self Parent")

    # Verify the accounts are in the COA
    assert coa.find(Code("1000")) == liquidity
    assert coa.find(Code("1001")) == bankaccnt


# LLM-generated content at query #29
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Mock a ReadChartOfAccounts callable
    def mock_read_coa() -> COA:
        return COA()

    # Test that the callable returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the COA has the expected default accounts
    expected_accounts = {
        Code("1"): "Assets",
        Code("2"): "Liabilities",
        Code("3"): "Equities",
        Code("4"): "Revenues",
        Code("5"): "Expenses",
    }

    for code, name in expected_accounts.items():
        account = coa.find(code)
        assert account is not None
        assert account.name == name
        assert account.type == AccountType[name.upper()]

    # Test that the COA is iterable and contains the expected accounts
    accounts = list(coa)
    assert len(accounts) == len(expected_accounts)
    for code, account in accounts:
        assert code in expected_accounts
        assert account.name == expected_accounts[code]


# LLM-generated content at query #30
#--------------------------

```python
def test_ReadChartOfAccounts___call__():
    # Create a mock function that implements ReadChartOfAccounts protocol
    def mock_read_coa() -> COA:
        return COA()

    # Test that the function returns a COA instance
    coa = mock_read_coa()
    assert isinstance(coa, COA)

    # Test that the returned COA has the default root accounts
    expected_codes = {Code("1"), Code("2"), Code("3"), Code("4"), Code("5")}
    actual_codes = {code for code, _ in coa}
    assert actual_codes == expected_codes

    # Test that the returned COA has the correct account names
    expected_names = {
        Code("1"): "Assets",
        Code("2"): "Liabilities",
        Code("3"): "Equities",
        Code("4"): "Revenues",
        Code("5"): "Expenses",
    }
    for code, account in coa:
        assert account.name == expected_names[code]

    # Test that the returned COA has the correct account types
    expected_types = {
        Code("1"): AccountType.ASSETS,
        Code("2"): AccountType.LIABILITIES,
        Code("3"): AccountType.EQUITIES,
        Code("4"): AccountType.REVENUES,
        Code("5"): AccountType.EXPENSES,
    }
    for code, account in coa:
        assert account.type == expected_types[code]


