####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Assets"
    assert coa._accounts[Code("2")].name == "Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expenses"

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1000"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2000"), "Custom Liabilities"),
    }
    coa = COA(rootspec)
    assert coa._accounts[Code("1000")].name == "Custom Assets"
    assert coa._accounts[Code("2000")].name == "Custom Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expenses"


# LLM-generated content at query #2
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("456"), name="Parent Account", coa=COA("Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent
    assert subaccount.type == parent.type
    assert subaccount.coa == parent.coa


# LLM-generated content at query #3
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    result = coa.add(parent_code, code, name)
    assert isinstance(result, SubAccount)
    assert result.code == code
    assert result.name == name
    assert result.parent.code == parent_code

def test_add_parent_not_found():
    coa = COA()
    parent_code = Code("999")
    code = Code("1.1")
    name = "Test Subaccount"
    with pytest.raises(ValueError, match="Parent account is not (yet) defined."):
        coa.add(parent_code, code, name)

def test_add_same_parent_and_code():
    coa = COA()
    parent_code = Code("1")
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(parent_code, parent_code, "Test")

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    first_call = coa.add(parent_code, code, name)
    second_call = coa.add(parent_code, code, name)
    assert first_call == second_call

def test_add_existing_account_inconsistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    first_name = "First Name"
    second_name = "Second Name"
    coa.add(parent_code, code, first_name)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, code, second_name)


# LLM-generated content at query #4
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert coa._subaccounts == {}

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2"), "Custom Liabilities")
    }
    coa = COA(rootspec=rootspec)
    assert coa._accounts[Code("1")].name == "Custom Assets"
    assert coa._accounts[Code("2")].name == "Custom Liabilities"
    assert len(coa._accounts) == 5


# LLM-generated content at query #5
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("1000")
    name = "Cash"
    parent = Account(code=Code("1000"), name="Assets", coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #6
#--------------------------

```python
def test_nodify_creates_node_with_account_and_children():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_creates_node_with_nested_children():
    coa = COA()
    parent_code = Code("1")
    child_code = Code("1.1")
    coa.add(parent_code, child_code, "Child Account")
    parent_account = coa._accounts[parent_code]
    node = coa.nodify(parent_account)
    assert node.account == parent_account
    assert len(node.children) == 1
    assert node.children[0].account == coa._accounts[child_code]
    assert node.children[0].children == []


# LLM-generated content at query #7
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


# LLM-generated content at query #8
#--------------------------

```python
def test_parent_instance_not_none():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    coa.add(parent_code, code, name)
    parentinstance = coa._accounts.get(parent_code)
    assert parentinstance is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #10
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert all(isinstance(account, RootAccount) for _, account in coa)
    assert all(account.parent is None for _, account in coa)

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


# LLM-generated content at query #11
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
    coa = COA(rootspec)
    assert coa._accounts[Code("1")].name == "Custom Asset"
    assert coa._accounts[Code("2")].name == "Custom Liability"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expense"


# LLM-generated content at query #12
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_read_coa = lambda: COA(accounts=[Account(code="1000", name="Cash")])
    result = mock_read_coa()
    assert isinstance(result, COA)
    assert len(result.accounts) == 1
    assert result.accounts[0].code == "1000"
    assert result.accounts[0].name == "Cash"


# LLM-generated content at query #13
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

def test_add_inconsistent_account():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    try:
        coa.add(Code("1"), Code("1.1"), "Different Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_nonexistent_parent():
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


# LLM-generated content at query #14
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1000"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2000"), "Custom Liabilities"),
        AccountType.EQUITY: (Code("3000"), "Custom Equity"),
        AccountType.INCOME: (Code("4000"), "Custom Income"),
        AccountType.EXPENSES: (Code("5000"), "Custom Expenses")
    }
    coa = COA(rootspec)
    assert coa.find(Code("1000")).name == "Custom Assets"
    assert coa.find(Code("2000")).name == "Custom Liabilities"
    assert coa.find(Code("3000")).name == "Custom Equity"
    assert coa.find(Code("4000")).name == "Custom Income"
    assert coa.find(Code("5000")).name == "Custom Expenses"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1000"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2000"), "Custom Liabilities")
    }
    coa = COA(rootspec)
    assert coa.find(Code("1000")).name == "Custom Assets"
    assert coa.find(Code("2000")).name == "Custom Liabilities"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #15
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("67890"), name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #16
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


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
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert all(isinstance(account, RootAccount) for _, account in coa)
    assert all(account.parent is None for _, account in coa)

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

def test_coa_constructor_with_partial_rootspec():
    rootspec = {AccountType.ASSET: (Code("10"), "Assets")}
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("10")).name == "Assets"
    assert coa.find(Code("2")).name == "Liability"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expense"


# LLM-generated content at query #19
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #20
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa)) == len(AccountType)
    for account_type in AccountType:
        assert coa.find(Code(str(account_type.value))) is not None

def test_coa_constructor_with_rootspec():
    rootspec = {AccountType.ASSET: (Code("1"), "Assets"), AccountType.LIABILITY: (Code("2"), "Liabilities")}
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"


# LLM-generated content at query #21
#--------------------------

```python
def test_subaccount_constructor():
    parent_account = Account(code=Code("1000"), name="Assets", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=Code("1010"), name="Cash", parent=parent_account)

    assert subaccount.code == Code("1010")
    assert subaccount.name == "Cash"
    assert subaccount.parent == parent_account


# LLM-generated content at query #22
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

def test_add_parent_not_found():
    coa = COA()
    parent_code = Code("999")
    code = Code("999.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_same_parent_and_code():
    coa = COA()
    parent_code = Code("1")
    code = Code("1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_existing_account_consistent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    account1 = coa.add(parent_code, code, name)
    account2 = coa.add(parent_code, code, name)
    assert account1 is account2

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
    except ValueError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_add_existing_account_with_different_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    new_parent_code = Code("2")
    try:
        coa.add(new_parent_code, code, name)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #24
#--------------------------

```python
def test_subaccount_constructor():
    coa = COA(code="COA1", name="Test COA")
    parent_account = Account(code=Code("PARENT"), name="Parent Account", coa=coa, type=AccountType.ASSET)
    subaccount = SubAccount(code=Code("SUB"), name="Sub Account", parent=parent_account)

    assert subaccount.code == Code("SUB")
    assert subaccount.name == "Sub Account"
    assert subaccount.parent == parent_account
    assert subaccount.type == AccountType.ASSET
    assert subaccount.coa == coa


# LLM-generated content at query #25
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


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

def test_add_account_with_invalid_parent():
    coa = COA()
    parent_code = Code("999")
    code = Code("1.1")
    name = "Test Account"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_add_account_with_same_parent_and_code():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Account")
        assert False, "Expected ValueError"
    except ValueError:
        pass

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


# LLM-generated content at query #27
#--------------------------

```python
def test_add_existing_account_with_different_parent():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), Code("1.1"), "Test Account")


# LLM-generated content at query #28
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("67890"), name="Parent Account", type=AccountType.ASSET, coa=COA("Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #29
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("1000")
    name = "Test SubAccount"
    parent_account = Account(code=Code("1000"), name="Parent Account", coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent_account)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent_account


# LLM-generated content at query #30
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_reader = lambda: COA(accounts={"1000": "Cash", "2000": "Accounts Receivable"})
    assert isinstance(mock_reader(), COA)
    assert mock_reader().accounts == {"1000": "Cash", "2000": "Accounts Receivable"}


# LLM-generated content at query #31
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA("Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #32
#--------------------------

```python
def test_add_existing_account_with_inconsistent_info():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Sub Account 1")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("1"), Code("1.1"), "Different Name")


# LLM-generated content at query #33
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
    assert subaccount.type == parent.type
    assert subaccount.coa == parent.coa


# LLM-generated content at query #34
#--------------------------

```python
def test_add_existing_account_with_different_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"

    # Add the account initially
    coa.add(parent_code, code, name)

    # Try to add the same account with a different parent
    different_parent_code = Code("3")
    coa.add(different_parent_code, code, name)


# LLM-generated content at query #35
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #36
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("1000")
    name = "Test SubAccount"
    parent = Account(code=Code("1000"), name="Test Account", coa=COA(code="COA1", name="Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


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
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("123"), name="Parent Account", coa=COA("Test COA"), type=AccountType.ASSET)
    subaccount = SubAccount(code, name, parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #39
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("456"), name="Parent Account", coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #40
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #41
#--------------------------

```python
def test_add_existing_account_with_different_parent():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account 1")
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(Code("2"), Code("1.1"), "Test Account 1")


# LLM-generated content at query #42
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


# LLM-generated content at query #43
#--------------------------

```python
def test_add_successful_creation():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    result = coa.add(parent_code, code, name)
    assert isinstance(result, SubAccount)
    assert result.code == code
    assert result.name == name
    assert result.parent.code == parent_code

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

def test_add_account_is_its_own_parent():
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
    assert isinstance(result, SubAccount)
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
        coa.add(parent_code, code, "Different Name")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #44
#--------------------------

```python
def test_subaccount_constructor():
    coa = COA()
    parent = Account(code=Code("123"), name="Parent Account", coa=coa)
    subaccount = SubAccount(code=Code("456"), name="Sub Account", parent=parent)
    assert subaccount.code == Code("456")
    assert subaccount.name == "Sub Account"
    assert subaccount.parent == parent


# LLM-generated content at query #45
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_reader = ReadChartOfAccounts()
    result = mock_reader()
    assert isinstance(result, COA)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_coa = COA()
    mock_reader = lambda: mock_coa
    assert isinstance(mock_reader(), COA)


# LLM-generated content at query #2
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 5
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2000"), "Custom Liability"),
        AccountType.EQUITY: (Code("3000"), "Custom Equity"),
        AccountType.REVENUE: (Code("4000"), "Custom Revenue"),
        AccountType.EXPENSE: (Code("5000"), "Custom Expense")
    }
    coa = COA(rootspec)
    assert coa.find(Code("1000")).name == "Custom Asset"
    assert coa.find(Code("2000")).name == "Custom Liability"
    assert coa.find(Code("3000")).name == "Custom Equity"
    assert coa.find(Code("4000")).name == "Custom Revenue"
    assert coa.find(Code("5000")).name == "Custom Expense"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2000"), "Custom Liability")
    }
    coa = COA(rootspec)
    assert coa.find(Code("1000")).name == "Custom Asset"
    assert coa.find(Code("2000")).name == "Custom Liability"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Revenue"
    assert coa.find(Code("5")).name == "Expense"


# LLM-generated content at query #3
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
    parent = coa._accounts[Code("1")]
    child = coa.add(Code("1"), Code("1.1"), "Test Account")
    node = coa.nodify(parent)
    assert node.account == parent
    assert len(node.children) == 1
    assert node.children[0].account == child


# LLM-generated content at query #4
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("67890"), name="Parent Account", coa=COA(code="COA1"), type=AccountType.ASSET)
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #5
#--------------------------

```python
def test_add_subaccount_success():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    result = coa.add(parent_code, code, name)
    assert isinstance(result, SubAccount)
    assert result.code == code
    assert result.name == name
    assert result.parent.code == parent_code

def test_add_existing_subaccount_success():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    first_call = coa.add(parent_code, code, name)
    second_call = coa.add(parent_code, code, name)
    assert first_call is second_call

def test_add_subaccount_invalid_parent():
    coa = COA()
    parent_code = Code("999")
    code = Code("1.1")
    name = "Test Subaccount"
    with pytest.raises(ValueError, match="Parent account is not (yet) defined."):
        coa.add(parent_code, code, name)

def test_add_subaccount_same_as_parent():
    coa = COA()
    parent_code = Code("1")
    code = Code("1")
    name = "Test Subaccount"
    with pytest.raises(ValueError, match="An account can not be the parent of itself."):
        coa.add(parent_code, code, name)

def test_add_subaccount_inconsistent_data():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    coa.add(parent_code, code, name)
    with pytest.raises(ValueError, match="Account name, code and parent do not match existing chart of accounts member."):
        coa.add(parent_code, code, "Different Name")


# LLM-generated content at query #6
#--------------------------

```python
def test_nodify_creates_node_with_account_and_empty_children():
    coa = COA()
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_creates_node_with_account_and_subaccounts():
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


# LLM-generated content at query #7
#--------------------------

```python
def test_subaccount_constructor():
    code = "12345"
    name = "Test SubAccount"
    parent = Account(code="1000", name="Parent Account", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #8
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
    assert result.parent.code == parent_code
    assert coa.find(code) == result
    assert result in coa.subaccounts(coa.find(parent_code))

def test_add_existing_subaccount_with_same_details():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    first_call_result = coa.add(parent_code, code, name)
    second_call_result = coa.add(parent_code, code, name)
    assert first_call_result == second_call_result
    assert coa.find(code) == first_call_result

def test_add_subaccount_with_nonexistent_parent():
    coa = COA()
    parent_code = Code("999")
    code = Code("999.1")
    name = "Test Subaccount"
    try:
        coa.add(parent_code, code, name)
        assert False, "Expected ValueError for nonexistent parent"
    except ValueError as e:
        assert str(e) == "Parent account is not (yet) defined."

def test_add_subaccount_with_same_parent_and_code():
    coa = COA()
    parent_code = Code("1")
    try:
        coa.add(parent_code, parent_code, "Test Subaccount")
        assert False, "Expected ValueError for same parent and code"
    except ValueError as e:
        assert str(e) == "An account can not be the parent of itself."

def test_add_subaccount_with_inconsistent_details():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Subaccount"
    coa.add(parent_code, code, name)
    try:
        coa.add(parent_code, code, "Different Name")
        assert False, "Expected ValueError for inconsistent details"
    except ValueError as e:
        assert str(e) == "Account name, code and parent do not match existing chart of accounts member."


# LLM-generated content at query #9
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert all(isinstance(account, RootAccount) for _, account in coa)
    assert coa.find(Code("1")).name == "Assets"
    assert coa.find(Code("2")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expenses"

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
        AccountType.EQUITY: (Code("30"), "Custom Equity"),
        AccountType.INCOME: (Code("40"), "Custom Income"),
        AccountType.EXPENSES: (Code("50"), "Custom Expenses"),
    }
    coa = COA(rootspec)
    assert len(list(coa)) == 5
    assert all(isinstance(account, RootAccount) for _, account in coa)
    assert coa.find(Code("10")).name == "Custom Assets"
    assert coa.find(Code("20")).name == "Custom Liabilities"
    assert coa.find(Code("30")).name == "Custom Equity"
    assert coa.find(Code("40")).name == "Custom Income"
    assert coa.find(Code("50")).name == "Custom Expenses"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("10"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("20"), "Custom Liabilities"),
    }
    coa = COA(rootspec)
    assert len(list(coa)) == 5
    assert coa.find(Code("10")).name == "Custom Assets"
    assert coa.find(Code("20")).name == "Custom Liabilities"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expenses"


# LLM-generated content at query #10
#--------------------------

```python
def test_parent_instance_exists():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Test Account")
    parentinstance = coa._accounts.get(Code("1"))
    assert parentinstance is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert all(isinstance(account, RootAccount) for account in coa._accounts.values())
    assert all(account.parent is None for account in coa._accounts.values())

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

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec=rootspec)
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Custom Asset"
    assert coa._accounts[Code("2")].name == "Custom Liability"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expense"


# LLM-generated content at query #12
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_reader = lambda: COA(accounts=[Account(code="1000", name="Cash")])
    assert isinstance(mock_reader(), COA)


# LLM-generated content at query #13
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(list(coa)) == 5
    for c, a in coa:
        assert isinstance(a, RootAccount)
        assert a.parent is None

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
    for c, a in coa:
        assert isinstance(a, RootAccount)
        assert a.parent is None
        assert a.code in rootspec
        assert a.name == rootspec[a.code][1]


# LLM-generated content at query #14
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_reader = lambda: COA(accounts=[Account(id="1", name="Cash")])
    assert isinstance(mock_reader(), COA)


# LLM-generated content at query #15
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    for account_type in AccountType:
        code = Code(str(account_type.value))
        account = coa.find(code)
        assert account is not None
        assert account.name == account_type.name.capitalize()
        assert isinstance(account, RootAccount)
        assert account.parent is None

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("100"), "Custom Assets"),
        AccountType.LIABILITY: (Code("200"), "Custom Liabilities"),
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("100")).name == "Custom Assets"
    assert coa.find(Code("200")).name == "Custom Liabilities"
    assert coa.find(Code("3")) is not None  # Equity (default)
    assert coa.find(Code("4")) is not None  # Income (default)
    assert coa.find(Code("5")) is not None  # Expense (default)

def test_coa_constructor_empty_rootspec():
    coa = COA(rootspec={})
    assert len(list(coa)) == 5
    for i, account_type in enumerate(AccountType, start=1):
        code = Code(str(i))
        account = coa.find(code)
        assert account is not None
        assert account.name == account_type.name.capitalize()


# LLM-generated content at query #16
#--------------------------

```python
def test_subaccount_constructor():
    coa = COA(code="COA001", name="Test COA")
    parent_account = Account(code="ACC001", name="Parent Account", coa=coa, type=AccountType.ASSET)
    subaccount = SubAccount(code="SUB001", name="Sub Account", parent=parent_account)

    assert subaccount.code == "SUB001"
    assert subaccount.name == "Sub Account"
    assert subaccount.parent == parent_account
    assert subaccount.type == AccountType.ASSET
    assert subaccount.coa == coa


# LLM-generated content at query #17
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #18
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("1000")
    name = "Cash"
    parent = Account(code=Code("1000"), name="Assets", type=AccountType.ASSET, coa=COA())
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #19
#--------------------------

```python
def test_parentinstance_is_not_none():
    coa = COA()
    parent_code = Code("1")
    code = Code("1.1")
    name = "Test Account"
    coa.add(parent_code, code, name)
    parentinstance = coa._accounts.get(parent_code)
    assert parentinstance is not None


# LLM-generated content at query #20
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 5
    assert all(isinstance(account, RootAccount) for account in coa.accounts)
    assert all(account.parent is None for account in coa.accounts)

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec)
    asset_account = coa.find(Code("1"))
    liability_account = coa.find(Code("2"))
    assert asset_account is not None
    assert liability_account is not None
    assert asset_account.name == "Custom Asset"
    assert liability_account.name == "Custom Liability"


# LLM-generated content at query #21
#--------------------------

```python
def test_nodify_creates_node_with_account_and_empty_children():
    coa = COA()
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert node.account == account
    assert node.children == []

def test_nodify_creates_node_with_account_and_children():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Sub Account 1")
    coa.add(Code("1"), Code("1.2"), "Sub Account 2")
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert node.account == account
    assert len(node.children) == 2
    assert node.children[0].account.code == Code("1.1")
    assert node.children[1].account.code == Code("1.2")

def test_nodify_creates_node_with_nested_children():
    coa = COA()
    coa.add(Code("1"), Code("1.1"), "Sub Account 1")
    coa.add(Code("1.1"), Code("1.1.1"), "Sub Account 1.1")
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert node.account == account
    assert len(node.children) == 1
    assert node.children[0].account.code == Code("1.1")
    assert len(node.children[0].children) == 1
    assert node.children[0].children[0].account.code == Code("1.1.1")


# LLM-generated content at query #22
#--------------------------

```python
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 5
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability"),
        AccountType.EQUITY: (Code("3"), "Custom Equity"),
        AccountType.INCOME: (Code("4"), "Custom Income"),
        AccountType.EXPENSE: (Code("5"), "Custom Expense")
    }
    coa = COA(rootspec)
    assert len(list(coa.accounts)) == 5
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        if account.type == AccountType.ASSET:
            assert account.code == Code("1")
            assert account.name == "Custom Asset"
        elif account.type == AccountType.LIABILITY:
            assert account.code == Code("2")
            assert account.name == "Custom Liability"
        elif account.type == AccountType.EQUITY:
            assert account.code == Code("3")
            assert account.name == "Custom Equity"
        elif account.type == AccountType.INCOME:
            assert account.code == Code("4")
            assert account.name == "Custom Income"
        elif account.type == AccountType.EXPENSE:
            assert account.code == Code("5")
            assert account.name == "Custom Expense"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2"), "Custom Liability")
    }
    coa = COA(rootspec)
    assert len(list(coa.accounts)) == 5
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        if account.type == AccountType.ASSET:
            assert account.code == Code("1")
            assert account.name == "Custom Asset"
        elif account.type == AccountType.LIABILITY:
            assert account.code == Code("2")
            assert account.name == "Custom Liability"
        elif account.type == AccountType.EQUITY:
            assert account.code == Code("3")
            assert account.name == "Equity"
        elif account.type == AccountType.INCOME:
            assert account.code == Code("4")
            assert account.name == "Income"
        elif account.type == AccountType.EXPENSE:
            assert account.code == Code("5")
            assert account.name == "Expense"


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Asset"
    assert coa._accounts[Code("2")].name == "Liability"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Revenue"
    assert coa._accounts[Code("5")].name == "Expense"

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("10"), "Current Assets"),
        AccountType.LIABILITY: (Code("20"), "Current Liabilities")
    }
    coa = COA(rootspec)
    assert coa._accounts[Code("10")].name == "Current Assets"
    assert coa._accounts[Code("20")].name == "Current Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Revenue"
    assert coa._accounts[Code("5")].name == "Expense"

def test_coa_constructor_initializes_empty_subaccounts():
    coa = COA()
    assert len(coa._subaccounts) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_reader = lambda: COA(accounts=[Account(code="1000", name="Cash")])
    assert isinstance(mock_reader(), COA)
    assert len(mock_reader().accounts) == 1
    assert mock_reader().accounts[0].code == "1000"
    assert mock_reader().accounts[0].name == "Cash"


# LLM-generated content at query #26
#--------------------------

```python
def test_subaccount_constructor():
    coa = COA(code="COA001", name="Main COA")
    parent_account = Account(code="ACC001", name="Parent Account", coa=coa, type=AccountType.ASSET)
    subaccount = SubAccount(code="SUB001", name="Sub Account", parent=parent_account)
    assert subaccount.code == "SUB001"
    assert subaccount.name == "Sub Account"
    assert subaccount.parent == parent_account


# LLM-generated content at query #27
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("54321"), name="Parent Account", type=AccountType.ASSET, coa=COA("Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #28
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test Account"
    parent = Account(code=Code("1234"), name="Parent Account", coa=COA("Test COA"), account_type=AccountType.ASSET)
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #29
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #30
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == 5
    assert all(isinstance(acc, RootAccount) for acc in coa.accounts)
    assert coa.find(Code("1")).name == "Asset"
    assert coa.find(Code("2")).name == "Liability"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expense"

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Assets"),
        AccountType.LIABILITY: (Code("2000"), "Liabilities"),
        AccountType.EQUITY: (Code("3000"), "Equities"),
        AccountType.INCOME: (Code("4000"), "Incomes"),
        AccountType.EXPENSE: (Code("5000"), "Expenses")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1000")).name == "Assets"
    assert coa.find(Code("2000")).name == "Liabilities"
    assert coa.find(Code("3000")).name == "Equities"
    assert coa.find(Code("4000")).name == "Incomes"
    assert coa.find(Code("5000")).name == "Expenses"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Assets"),
        AccountType.LIABILITY: (Code("2000"), "Liabilities")
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1000")).name == "Assets"
    assert coa.find(Code("2000")).name == "Liabilities"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expense"


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
def test_coa_constructor_without_rootspec():
    coa = COA()
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1"), "Assets"),
        AccountType.LIABILITY: (Code("2"), "Liabilities"),
        AccountType.EQUITY: (Code("3"), "Equity"),
        AccountType.INCOME: (Code("4"), "Income"),
        AccountType.EXPENSE: (Code("5"), "Expenses")
    }
    coa = COA(rootspec=rootspec)
    assert len(list(coa.accounts)) == len(AccountType)
    for account in coa.accounts:
        assert isinstance(account, RootAccount)
        assert account.parent is None
        if account.type == AccountType.ASSET:
            assert account.code == Code("1")
            assert account.name == "Assets"
        elif account.type == AccountType.LIABILITY:
            assert account.code == Code("2")
            assert account.name == "Liabilities"
        elif account.type == AccountType.EQUITY:
            assert account.code == Code("3")
            assert account.name == "Equity"
        elif account.type == AccountType.INCOME:
            assert account.code == Code("4")
            assert account.name == "Income"
        elif account.type == AccountType.EXPENSE:
            assert account.code == Code("5")
            assert account.name == "Expenses"


# LLM-generated content at query #33
#--------------------------

```python
def test_parent_instance_not_none():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    assert coa._accounts.get(parent_code) is not None


# LLM-generated content at query #34
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("67890"), name="Parent Account", type=AccountType.ASSET, coa=COA("Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #35
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #36
#--------------------------

```python
def test_parent_instance_not_none():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    assert coa._accounts[parent_code] is not None


# LLM-generated content at query #37
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


# LLM-generated content at query #38
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


# LLM-generated content at query #39
#--------------------------

```python
def test_coa_constructor_default():
    coa = COA()
    assert len(coa._accounts) == 5
    assert len(coa._subaccounts) == 0
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

def test_coa_constructor_with_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Assets"),
        AccountType.LIABILITY: (Code("2000"), "Liabilities"),
        AccountType.EQUITY: (Code("3000"), "Equity"),
        AccountType.INCOME: (Code("4000"), "Income"),
        AccountType.EXPENSE: (Code("5000"), "Expenses")
    }
    coa = COA(rootspec)
    assert len(coa._accounts) == 5
    assert len(coa._subaccounts) == 0
    assert coa.find(Code("1000")) is not None
    assert coa.find(Code("2000")) is not None
    assert coa.find(Code("3000")) is not None
    assert coa.find(Code("4000")) is not None
    assert coa.find(Code("5000")) is not None


# LLM-generated content at query #40
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
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Revenue"
    assert coa._accounts[Code("5")].name == "Expense"


# LLM-generated content at query #41
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    mock_read_coa = lambda: COA(accounts=[Account(id="1", name="Cash")])
    result = mock_read_coa()
    assert isinstance(result, COA)
    assert len(result.accounts) == 1
    assert result.accounts[0].id == "1"
    assert result.accounts[0].name == "Cash"


# LLM-generated content at query #42
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa.find(Code("1"))
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #43
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #44
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #45
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(list(coa)) == 5
    assert coa.find(Code("1")) is not None
    assert coa.find(Code("2")) is not None
    assert coa.find(Code("3")) is not None
    assert coa.find(Code("4")) is not None
    assert coa.find(Code("5")) is not None

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Custom Asset"),
        AccountType.LIABILITY: (Code("2000"), "Custom Liability"),
        AccountType.EQUITY: (Code("3000"), "Custom Equity"),
        AccountType.INCOME: (Code("4000"), "Custom Income"),
        AccountType.EXPENSE: (Code("5000"), "Custom Expense"),
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1000")).name == "Custom Asset"
    assert coa.find(Code("2000")).name == "Custom Liability"
    assert coa.find(Code("3000")).name == "Custom Equity"
    assert coa.find(Code("4000")).name == "Custom Income"
    assert coa.find(Code("5000")).name == "Custom Expense"

def test_coa_constructor_with_partial_rootspec():
    rootspec = {
        AccountType.ASSET: (Code("1000"), "Custom Asset"),
    }
    coa = COA(rootspec=rootspec)
    assert coa.find(Code("1000")).name == "Custom Asset"
    assert coa.find(Code("2")).name == "Liability"
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("5")).name == "Expense"


# LLM-generated content at query #46
#--------------------------

```python
def test_parent_instance_not_none():
    coa = COA()
    parent_code = Code("1")
    code = Code("2")
    name = "Test Account"
    coa.add(parent_code, code, name)
    parentinstance = coa._accounts.get(parent_code)
    assert parentinstance is not None


# LLM-generated content at query #47
#--------------------------

```python
def test_subaccount_constructor():
    coa = COA("test_coa")
    parent_account = Account("test_account", "Test Account", AccountType.ASSET, coa)
    code = Code("12345")
    name = "Test SubAccount"

    subaccount = SubAccount(code, name, parent_account)

    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent_account


# LLM-generated content at query #48
#--------------------------

```python
def test_coa_constructor_initializes_root_accounts():
    rootspec = {
        AccountType.ASSETS: (Code("1"), "Custom Assets"),
        AccountType.LIABILITIES: (Code("2"), "Custom Liabilities")
    }
    coa = COA(rootspec=rootspec)

    assert coa.find(Code("1")) is not None
    assert coa.find(Code("1")).name == "Custom Assets"
    assert coa.find(Code("1")).type == AccountType.ASSETS

    assert coa.find(Code("2")) is not None
    assert coa.find(Code("2")).name == "Custom Liabilities"
    assert coa.find(Code("2")).type == AccountType.LIABILITIES

    assert coa.find(Code("3")) is not None
    assert coa.find(Code("3")).name == "Equity"
    assert coa.find(Code("3")).type == AccountType.EQUITY

    assert coa.find(Code("4")) is not None
    assert coa.find(Code("4")).name == "Income"
    assert coa.find(Code("4")).type == AccountType.INCOME

    assert coa.find(Code("5")) is not None
    assert coa.find(Code("5")).name == "Expenses"
    assert coa.find(Code("5")).type == AccountType.EXPENSES


# LLM-generated content at query #49
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("12345")
    name = "Test SubAccount"
    parent = Account(code=Code("67890"), name="Parent Account", coa=COA("Test COA"), type=AccountType.ASSET)
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #50
#--------------------------

```python
def test_coa_constructor_with_default_rootspec():
    coa = COA()
    assert len(coa._accounts) == 5
    assert coa._accounts[Code("1")].name == "Assets"
    assert coa._accounts[Code("2")].name == "Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Revenue"
    assert coa._accounts[Code("5")].name == "Expenses"

def test_coa_constructor_with_custom_rootspec():
    rootspec = {
        AccountType.ASSETS: (Code("1000"), "Current Assets"),
        AccountType.LIABILITIES: (Code("2000"), "Current Liabilities"),
    }
    coa = COA(rootspec)
    assert coa._accounts[Code("1000")].name == "Current Assets"
    assert coa._accounts[Code("2000")].name == "Current Liabilities"
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Revenue"
    assert coa._accounts[Code("5")].name == "Expenses"


# LLM-generated content at query #51
#--------------------------

```python
def test_parent_account_not_defined():
    coa = COA()
    with pytest.raises(ValueError, match="Parent account is not (yet) defined."):
        coa.add(Code("999"), Code("001"), "Test Account")


# LLM-generated content at query #52
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
    assert coa._accounts[Code("3")].name == "Equity"
    assert coa._accounts[Code("4")].name == "Income"
    assert coa._accounts[Code("5")].name == "Expense"


# LLM-generated content at query #53
#--------------------------

```python
def test_subaccount_constructor():
    code = Code("123")
    name = "Test SubAccount"
    parent = Account(code=Code("100"), name="Parent Account", coa=COA(code="COA1", name="Test COA"))
    subaccount = SubAccount(code=code, name=name, parent=parent)
    assert subaccount.code == code
    assert subaccount.name == name
    assert subaccount.parent == parent


# LLM-generated content at query #54
#--------------------------

```python
def test_read_chart_of_accounts_call_returns_coa_instance():
    reader = ReadChartOfAccounts()
    result = reader()
    assert isinstance(result, COA)


# LLM-generated content at query #55
#--------------------------

```python
def test_nodify_returns_node_instance():
    coa = COA()
    account = coa._accounts[Code("1")]
    node = coa.nodify(account)
    assert isinstance(node, COA.Node)


# LLM-generated content at query #56
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


