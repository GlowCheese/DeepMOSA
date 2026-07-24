####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_reader = lambda: COA(accounts={"1000": "Cash", "2000": "Accounts Receivable"})
    assert isinstance(mock_reader(), COA)


# LLM-generated content at query #2
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value))) is not None

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2"), "Custom Liabilities")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")).name == "Custom Assets"
    assert coa.find(Code("2")).name == "Custom Liabilities"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #3
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert coa.find(code) == account
    assert account in coa.subaccounts(coa.find(parent_code))

def test_add_parent_not_defined():
    coa = COA()
    parent_code = Code("999")
    code = Code("999.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_self_parent():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Account")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name1 = "Test Account 1"
    name2 = "Test Account 2"
    coa.add(parent_code, code, name1)
    try:
        coa.add(parent_code, code, name2)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #4
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert all(isinstance(account, RootAccount) for _, account in coa)
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("10"), "Custom Asset"),
        AccountType.LIABILITY: (Code("20"), "Custom Liability"),
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("10")) is not None
    assert coa.find(Code("20")) is not None
    assert coa.find(Code("10")).name == "Custom Asset"
    assert coa.find(Code("20")).name == "Custom Liability"
    assert coa.find(Code("3")) is not None  # Default for AccountType.EQUITY
    assert coa.find(Code("4")) is not None  # Default for AccountType.INCOME
    assert coa.find(Code("5")) is not None  # Default for AccountType.EXPENSE

def test_coa_constructor_immutability():
    coa = COA()
    with pytest.raises(FrozenInstanceError):
        coa._accounts = {}
    with pytest.raises(FrozenInstanceError):
        coa._subaccounts = {}


# LLM-generated content at query #5
#--------------------------

```python
def test_parent_instance_exists():
    coa = COA()
    parent_code = Code("1")
    account_code = Code("1.1")
    account_name = "Test Account"
    result = coa.add(parent_code, account_code, account_name)
    assert result is not None


# LLM-generated content at query #6
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test Account"
    parent = Account(code=Code("456"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #7
#--------------------------

```python
def test_coa_constructor_initializes_root_accounts():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None


# LLM-generated content at query #8
#--------------------------

```python
def test_code_in_accounts():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    assert Code("1.1") in coa._accounts


# LLM-generated content at query #9
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.toplevel)

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.INCOME: (Code("4"), "Income"),
        AccountType.EXPENSE: (Code("5"), "Expenses")
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa)) == 5
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #10
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    result = coa.add(parent_code, code, name)
    assert result.code == code
    assert result.name == name
    assert result.parent.code == parent_code
    assert coa.find(code) == result
    assert result in coa.subaccounts(coa.find(parent_code))

def test_add_parent_not_found():
    coa = COA()
    parent_code = Code("999")
    code = Code("1.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError for non-existent parent"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_self_parent():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Account")
        assert False, "Expected ValueError for self-parenting"
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    first_call = coa.add(parent_code, code, name)
    second_call = coa.add(parent_code, code, name)
    assert first_call == second_call

def test_add_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    first_name = "Test Account"
    second_name = "Different Name"
    coa.add(parent_code, code, first_name)
    try:
        coa.add(parent_code, code, second_name)
        assert False, "Expected ValueError for inconsistent account info"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #11
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #12
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #13
#--------------------------

```python
def test_nodify_creates_node_with_account_and_empty_children():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_creates_node_with_account_and_subaccounts():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    coa.add(parent_code, child_code, "Child Account")
    parent_account = coa._accounts[parent_code]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == coa._accounts[child_code]


# LLM-generated content at query #14
#--------------------------

```python
def test_add_existing_account_with_consistent_info():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    coa.add(parent_code, code, name)
    result = coa.add(parent_code, code, name)
    assert result is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    result = coa.add(parent_code, code, name)
    assert result.code == code
    assert result.name == name
    assert result.parent.code == parent_code

def test_add_parent_not_found():
    coa = COA()
    parent_code = Code("999")
    code = Code("999.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_self_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    result = coa.add(parent_code, code, name)
    assert result.code == code
    assert result.name == name
    assert result.parent.code == parent_code

def test_add_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    try:
        coa.add(Code("2"), code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #16
#--------------------------

```python
def test_subaccount_constructor():
    code = "1001"
    name = "Cash"
    parent = Account(code="1000", name="Assets", coa=COA(name="Main COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #17
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #18
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert all(isinstance(account, RootAccount) for _, account in coa)
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1000"), "Assets"),
        AccountType.LIABILITIES: (Code("2000"), "Liabilities"),
        AccountType.EQUITY: (Code("3000"), "Equity"),
        AccountType.INCOME: (Code("4000"), "Income"),
        AccountType.EXPENSES: (Code("5000"), "Expenses")
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa)) == 5
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("2000")) is not None
    assert coa.find(Code("3000")) is not None
    assert coa.find(Code("4000")) is not None
    assert coa.find(Code("5000")) is not None
    assert coa.find(Code("1000")).name == "Assets"
    assert coa.find(Code("2000")).name == "Liabilities"
    assert coa.find(Code("3000")).name == "Equity"
    assert coa.find(Code("4000")).name == "Income"
    assert coa.find(Code("5000")).name == "Expenses"


# LLM-generated content at query #19
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa)) == len(AccountType)
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value))) is not None
        account = coa.find(Code(str(account_type.value)))
        assert account.name == account_type.name.capitalize()
        assert isinstance(account, RootAccount)
        assert account.parent is None

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("2")).name == "Custom Liability"
    assert coa.find(Code("3")) is not None  # Default for AccountType.EQUITY
    assert coa.find(Code("3")).name == "Equity"


# LLM-generated content at query #20
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=code, name="Parent Account", coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #21
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    account = coa.add(Code("1"), Code("1.1"), "Test Account")
    assert account.code == Code("1.1")
    assert account.name == "Test Account"
    assert account.parent.code == Code("1")

def test_add_existing_account_consistent():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    account = coa.add(Code("1"), Code("1.1"), "Test Account")
    assert account.code == Code("1.1")
    assert account.name == "Test Account"
    assert account.parent.code == Code("1")

def test_add_existing_account_inconsistent():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    try:
        coa.add(Code("1"), Code("1.1"), "Different Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_parent_not_found():
    coa = COA()
    try:
        coa.add(Code("99"), Code("99.1"), "Test Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_self_parent():
    coa = COA()
    try:
        coa.add(Code("1.1"), Code("1.1"), "Test Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("1000"), name="Assets", coa=COA("US-GAAP"), type=AccountType.ASSET)
    subaccount = SubAccount(code=Code("1010"), name="Cash", parent=parent_account)

    assert subaccount.code == Code("1010")
    assert subaccount.name == "Cash"
    assert subaccount.parent == parent_account


# LLM-generated content at query #23
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #24
#--------------------------

```python
def test_add_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    result = coa.add(parent_code, code, name)
    assert isinstance(result, SubAccount)
    assert result.code == code
    assert result.name == name
    assert result.parent == coa.find(parent_code)

def test_add_existing_subaccount():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    coa.add(parent_code, code, name)
    result = coa.add(parent_code, code, name)
    assert isinstance(result, SubAccount)
    assert result.code == code
    assert result.name == name
    assert result.parent == coa.find(parent_code)

def test_add_subaccount_with_inconsistent_data():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    coa.add(parent_code, code, name)
    try:
        coa.add(parent_code, code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_subaccount_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("99")
    code = Code("99.1")
    name = "Test Subaccount"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_subaccount_with_same_parent_and_code():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Subaccount")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA(), type=AccountType.ASSET)

    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #26
#--------------------------

```python
def test_add_new_account():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent == coa.find(parent_code)
    assert coa.find(code) == account
    assert account in coa.subaccounts(coa.find(parent_code))

def test_add_existing_account_with_consistent_info():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_account_with_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name1 = "Test Account"
    name2 = "Different Account"
    coa.add(parent_code, code, name1)
    try:
        coa.add(parent_code, code, name2)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("999")
    code = Code("999.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_self_as_parent():
    coa = COA()
    code = Code("1")
    try:
        coa.add(code, code, "Test Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #28
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader.__call__()
    assert isinstance(result, COA)


# LLM-generated content at query #29
#--------------------------

```python
def test_add_existing_account_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    coa.add(parent_code, code, name)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, code, "Different Name")


# LLM-generated content at query #30
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test Account"
    parent = Account(code, "Parent Account", AccountType.ASSET, coa)
    subaccount = SubAccount(code, name, parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #31
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("456"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #32
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #33
#--------------------------

```python
def test_add_new_account():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert coa.find(code) == account
    assert account in coa.subaccounts(account.parent)

def test_add_existing_account_with_consistent_data():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_account_with_inconsistent_data():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    try:
        coa.add(parent_code, code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("999")
    code = Code("999.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_self_as_parent():
    coa = COA()
    parent_code = Code("1.1")
    code = Code("1.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_add_existing_account_with_different_parent():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), Code("1.1"), "Test Account")


# LLM-generated content at query #35
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", type=AccountType.ASSET, coa=COA("Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #36
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa_instance():
    reader = ReadChartOfAccounts()
    result = reader.__call__()
    assert isinstance(result, COA)


# LLM-generated content at query #37
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("456"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #38
#--------------------------

```python
def test_subaccount_constructor():
    coa = COA(code="COA001", name="Test COA")
    parent_account = Account(code=Code("ACCT001"), name="Parent Account", coa=coa, type=AccountType.ASSET)
    subaccount = SubAccount(code=Code("SUB001"), name="Test SubAccount", parent=parent_account)

    assert subaccount.code == Code("SUB001")
    assert subaccount.name == "Test SubAccount"
    assert subaccount.parent == parent_account


# LLM-generated content at query #39
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA(), type=AccountType.ASSET)
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #40
#--------------------------

```python
def test_add_new_account():
    coa = COA()
    account = coa.add(Code("1"), Code("1.1"), "Test Account")
    assert account.code == Code("1.1")
    assert account.name == "Test Account"
    assert account.parent.code == Code("1")

def test_add_existing_account():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    account = coa.add(Code("1"), Code("1.1"), "Test Account")
    assert account.code == Code("1.1")
    assert account.name == "Test Account"
    assert account.parent.code == Code("1")

def test_add_account_with_inconsistent_data():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    try:
        coa.add(Code("2"), Code("1.1"), "Different Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_nonexistent_parent():
    coa = COA()
    try:
        coa.add(Code("99"), Code("99.1"), "Test Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_self_as_parent():
    coa = COA()
    try:
        coa.add(Code("1.1"), Code("1.1"), "Test Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("456"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #2
#--------------------------

```python
def test_nodify_creates_node_with_correct_account():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert node.account == account

def test_nodify_creates_node_with_empty_children_list():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert node.children == []

def test_nodify_creates_node_with_subaccounts():
    coa = COA()
    parent_account = coa._accounts[Code("1")]
    child_account = coa.add(Code("1"), Code("1.1"), "Child Account")
    node = coa.nodify(parent_account)
    assert len(node.children) == 1
    assert node.children[0].account == child_account

def test_nodify_creates_node_with_nested_subaccounts():
    coa = COA()
    parent_account = coa._accounts[Code("1")]
    child_account = coa.add(Code("1"), Code("1.1"), "Child Account")
    grandchild_account = coa.add(Code("1.1"), Code("1.1.1"), "Grandchild Account")
    node = coa.nodify(parent_account)
    assert len(node.children) == 1
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account == grandchild_account


# LLM-generated content at query #3
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert coa._subaccounts == {}

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec=rootspec)
    assert coa._accounts[Code("1")].name == "Custom Asset"
    assert coa._accounts[Code("2")].name == "Custom Liability"
    assert coa._accounts[Code("3")].name == "Equity"  # Default for EQUITY
    assert coa._accounts[Code("4")].name == "Revenue"  # Default for REVENUE
    assert coa._accounts[Code("5")].name == "Expense"  # Default for EXPENSE

def test_coa_constructor_immutability():
    coa = COA()
    with pytest.raises(FrozenInstanceError):
        coa._accounts = {}


# LLM-generated content at query #4
#--------------------------

```python
def test___iter___returns_iterable_of_code_account_tuples():
    coa = COA()
    accounts = list(coa)
    assert all(isinstance(code, Code) and isinstance(account, Account) for code, account in accounts)


# LLM-generated content at query #5
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #6
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert all(account.parent is None for account in coa._accounts.values())

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2"), "Custom Liabilities"),
        AccountType.EQUITY: (Code("3"), "Custom Equity"),
        AccountType.INCOME: (Code("4"), "Custom Income"),
        AccountType.EXPENSES: (Code("5"), "Custom Expenses")
    }
    coa = COA(rootspec=rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Custom Assets"
    assert coa._accounts[Code("2")].name == "Custom Liabilities"
    assert coa._accounts[Code("3")].name == "Custom Equity"
    assert coa._accounts[Code("4")].name == "Custom Income"
    assert coa._accounts[Code("5")].name == "Custom Expenses"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2"), "Custom Liabilities")
    }
    coa = COA(rootspec=rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Custom Assets"
    assert coa._accounts[Code("2")].name == "Custom Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expenses"


# LLM-generated content at query #7
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == len(AccountType)
    for account_type in AccountType:
        expected_code = Code(str(account_type.value))
        expected_name = account_type.name.capitalize()
        assert coa._accounts[expected_code].code == expected_code
        assert coa._accounts[expected_code].name == expected_name
        assert coa._accounts[expected_code].type == account_type

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec=rootspec)
    assert coa._accounts[Code("1")].name == "Custom Asset"
    assert coa._accounts[Code("2")].name == "Custom Liability"
    assert len(coa._accounts) == len(AccountType)

def test_coa_constructor_with_partial_rootspec():
    rootspec = {AccountType.ASSET: (Code("1"), "Custom Asset")}
    coa = COA(rootspec=rootspec)
    assert coa._accounts[Code("1")].name == "Custom Asset"
    assert coa._accounts[Code("2")].name == "Liability"
    assert len(coa._accounts) == len(AccountType)


# LLM-generated content at query #8
#--------------------------

```python
def test_nodify_creates_node_with_correct_account():
    coa = COA()
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_creates_node_with_subaccounts():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Sub Account 1")
    coa.add(Code("1"), Code("1.2"), "Sub Account 2")
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert node.account == account
    assert len(node.children) == 2
    assert node.children[0].account.code == Code("1.1")
    assert node.children[1].account.code == Code("1.2")

def test_nodify_creates_nested_node_structure():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Sub Account 1")
    coa.add(Code("1.1"), Code("1.1.1"), "Sub Sub Account 1")
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert node.account == account
    assert len(node.children) == 1
    assert node.children[0].account.code == Code("1.1")
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account.code == Code("1.1.1")


# LLM-generated content at query #9
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert account in coa._accounts.values()
    assert account in coa._subaccounts[coa._accounts[parent_code]]

def test_add_parent_not_defined():
    coa = COA()
    parent_code = Code("999")
    code = Code("1.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_self_parent():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Account")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    try:
        coa.add(parent_code, code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #10
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #11
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #12
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Asset"
    assert coa._accounts[Code("2")].name == "Liability"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expense"

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Assets"),
        AccountType.LIABILITY: (Code("2000"), "Liabilities"),
        AccountType.EQUITY: (Code("3000"), "Equities"),
        AccountType.INCOME: (Code("4000"), "Incomes"),
        AccountType.EXPENSE: (Code("5000"), "Expenses")
    }
    coa = COA(rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1000")].name == "Assets"
    assert coa._accounts[Code("2000")].name == "Liabilities"
    assert coa._accounts[Code("3000")].name == "Equities"
    assert coa._accounts[Code("4000")].name == "Incomes"
    assert coa._accounts[Code("5000")].name == "Expenses"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Assets"),
        AccountType.LIABILITY: (Code("2000"), "Liabilities")
    }
    coa = COA(rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1000")].name == "Assets"
    assert coa._accounts[Code("2000")].name == "Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expense"


# LLM-generated content at query #13
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa_instance():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #14
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA(), type=AccountType.ASSET)
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #15
#--------------------------

```python
def test_nodify_returns_node_with_account_and_children():
    coa = COA()
    parent_account = coa._accounts[Code("1")]
    child_account = coa.add(Code("1"), Code("1.1"), "Test Child Account")
    node = coa.nodify(parent_account)
    assert isinstance(node, COA.Node)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == child_account


# LLM-generated content at query #16
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert coa._subaccounts == {}

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
        AccountType.EQUITY: (Code("3"), "Custom Equity"),
        AccountType.INCOME: (Code("4"), "Custom Income"),
        AccountType.EXPENSE: (Code("5"), "Custom Expense")
    }
    coa = COA(rootspec=rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Custom Asset"
    assert coa._accounts[Code("2")].name == "Custom Liability"
    assert coa._accounts[Code("3")].name == "Custom Equity"
    assert coa._accounts[Code("4")].name == "Custom Income"
    assert coa._accounts[Code("5")].name == "Custom Expense"
    assert coa._subaccounts == {}

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset")
    }
    coa = COA(rootspec=rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Custom Asset"
    assert coa._accounts[Code("2")].name == "Liability"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expense"
    assert coa._subaccounts == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa)) == 5  # 5 default root accounts
    assert all(isinstance(account, RootAccount) for _, account in coa)
    assert all(account.parent is None for _, account in coa)

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Custom Liability"
    assert coa.find(Code("3")).name == "Equity"  # Default name for EQUITY
    assert coa.find(Code("4")).name == "Revenue"  # Default name for REVENUE
    assert coa.find(Code("5")).name == "Expense"  # Default name for EXPENSE

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa)) == 5  # 5 default root accounts
    assert coa.find(Code("1")).name == "Asset"
    assert coa.find(Code("2")).name == "Liability"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Revenue"
    assert coa.find(Code("5")).name == "Expense"


# LLM-generated content at query #18
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("123"), name="Parent Account", coa=COA())
    subaccount = SubAccount(code=Code("456"), name="Sub Account", parent=parent_account)
    assert subaccount.code == Code("456")
    assert subaccount.name == "Sub Account"
    assert subaccount.parent == parent_account


# LLM-generated content at query #19
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert coa._subaccounts == {}

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Custom Liability"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expense"

def test_coa_constructor_immutable():
    coa = COA()
    with pytest.raises(Exception):  # Expecting a FrozenInstanceError or similar
        coa._accounts = {}


# LLM-generated content at query #20
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", type=AccountType.ASSET, coa=COA(code="COA001", name="Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #21
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa_instance():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #22
#--------------------------

```python
def test_parent_instance_not_none():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    assert coa._accounts.get(parent_code) is not None


# LLM-generated content at query #23
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert coa._subaccounts == {}

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.INCOME: (Code("4"), "Income"),
        AccountType.EXPENSE: (Code("5"), "Expenses")
    }
    coa = COA(rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Assets"
    assert coa._accounts[Code("2")].name == "Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expenses"
    assert coa._subaccounts == {}


# LLM-generated content at query #24
#--------------------------

```python
def test_add_new_account():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert coa.find(code) == account

def test_add_existing_account():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_inconsistent_account():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    try:
        coa.add(parent_code, code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_nonexistent_parent():
    coa = COA()
    parent_code = Code("99")
    code = Code("99.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_self_parent():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("456"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #26
#--------------------------

```python
def test_add_new_account():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert coa.find(code) == account
    assert account in coa.subaccounts(coa.find(parent_code))

def test_add_existing_account():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_account_inconsistent_info():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name1 = "Test Account 1"
    name2 = "Test Account 2"
    coa.add(parent_code, code, name1)
    with pytest.raises(ValueError):
        coa.add(parent_code, code, name2)

def test_add_account_parent_not_found():
    coa = COA()
    parent_code = Code("999")
    code = Code("999.1")
    name = "Test Account"
    with pytest.raises(ValueError):
        coa.add(parent_code, code, name)

def test_add_account_self_parent():
    coa = COA()
    parent_code = Code("1.1")
    code = Code("1.1")
    name = "Test Account"
    with pytest.raises(ValueError):
        coa.add(parent_code, code, name)


# LLM-generated content at query #27
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_reader = lambda: COA(accounts=[Account(code="1000", name="Cash")])
    assert isinstance(mock_reader(), COA)


# LLM-generated content at query #28
#--------------------------

```python
def test_subaccount_constructor():
    coa = COA(code=Code("COA001"), name="Test COA")
    parent_account = Account(code=Code("ACC001"), name="Parent Account", coa=coa, type=AccountType.ASSET)
    sub_account = SubAccount(code=Code("SUB001"), name="Sub Account", parent=parent_account)

    assert sub_account.code == Code("SUB001")
    assert sub_account.name == "Sub Account"
    assert sub_account.parent == parent_account
    assert sub_account.type == AccountType.ASSET
    assert sub_account.coa == coa


# LLM-generated content at query #29
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #30
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("456"), name="Parent Account", coa=COA(name="Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #31
#--------------------------

```python
def test_coa_constructor_default_initialization():
    coa = COA()
    assert len(list(coa)) == len(AccountType)
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value))) is not None
        account = coa.find(Code(str(account_type.value)))
        assert account.name == account_type.name.capitalize()
        assert isinstance(account, RootAccount)

def test_coa_constructor_custom_rootspec():
    custom_rootspec = {
        AccountType.ASSET: (Code("100"), "Custom Asset"),
        AccountType.LIABILITY: (Code("200"), "Custom Liability")
    }
    coa = COA(rootspec=custom_rootspec)
    assert coa.find(Code("100")).name == "Custom Asset"
    assert coa.find(Code("200")).name == "Custom Liability"
    assert coa.find(Code("100")).type == AccountType.ASSET
    assert coa.find(Code("200")).type == AccountType.LIABILITY


# LLM-generated content at query #32
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa_instance():
    mock_read_chart_of_accounts = lambda: COA()
    result = mock_read_chart_of_accounts()
    assert isinstance(result, COA)


# LLM-generated content at query #33
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    root_account = next(coa.toplevel)
    node = coa.nodify(root_account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #34
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert coa._subaccounts == {}

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Custom Liability"
    assert len(coa._accounts) == 5


# LLM-generated content at query #35
#--------------------------

```python
def test_parent_instance_exists():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    assert coa.find(parent_code) is not None
    coa.add(parent_code, code, name)


# LLM-generated content at query #36
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA(name="Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #37
#--------------------------

```python
def test_add_raises_value_error_when_parent_account_not_defined():
    coa = COA()
    with pytest.raises(ValueError, match="Parent account is not (yet) defined."):
        coa.add(Code("999"), Code("001"), "Test Account")


# LLM-generated content at query #38
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 5
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("100"), "Custom Asset"),
        AccountType.LIABILITY: (Code("200"), "Custom Liability"),
        AccountType.EQUITY: (Code("300"), "Custom Equity"),
        AccountType.INCOME: (Code("400"), "Custom Income"),
        AccountType.EXPENSE: (Code("500"), "Custom Expense"),
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("100")).name == "Custom Asset"
    assert coa.find(Code("200")).name == "Custom Liability"
    assert coa.find(Code("300")).name == "Custom Equity"
    assert coa.find(Code("400")).name == "Custom Income"
    assert coa.find(Code("500")).name == "Custom Expense"


# LLM-generated content at query #39
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("67890"), name="Parent Account", coa=COA(name="Test COA", code=Code("COA123")))
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #40
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("PARENT"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=Code("SUB"), name="Sub Account", parent=parent_account)
    assert subaccount.code == Code("SUB")
    assert subaccount.name == "Sub Account"
    assert subaccount.parent == parent_account


# LLM-generated content at query #41
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code

def test_add_parent_not_defined():
    coa = COA()
    parent_code = Code("99")
    code = Code("99.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_self_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name1 = "Test Account 1"
    name2 = "Test Account 2"
    coa.add(parent_code, code, name1)
    try:
        coa.add(parent_code, code, name2)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #42
#--------------------------

```python
def test_add_new_subaccount():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Sub Account"
    account = coa.add(parent_code, code, name)
    assert account.code == code
    assert account.name == name
    assert account.parent.code == parent_code
    assert coa.find(code) == account
    assert account in coa.subaccounts(coa.find(parent_code))

def test_add_existing_subaccount():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Sub Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 == account2

def test_add_inconsistent_subaccount():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Sub Account"
    coa.add(parent_code, code, name)
    try:
        coa.add(parent_code, code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_self_parent_subaccount():
    coa = COA()
    code = Code("1.1")
    try:
        coa.add(code, code, "Sub Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_nonexistent_parent_subaccount():
    coa = COA()
    parent_code = Code("99")
    code = Code("99.1")
    try:
        coa.add(parent_code, code, "Sub Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass


