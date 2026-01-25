####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(1000))),
        account2: Balance(period.since, Quantity(Decimal(0)))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(100)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(100)), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT)
        ]
    )

    # Mock read_initial_balances and read_journal_entries
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return [journal_entry1, journal_entry2]

    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].amount == Amount(Decimal(100))
    assert ledger1.entries[0].balance == Quantity(Decimal(1100))
    assert ledger1.entries[1].amount == Amount(Decimal(50))
    assert ledger1.entries[1].balance == Quantity(Decimal(1150))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal(100))
    assert ledger2.entries[0].balance == Quantity(Decimal(100))
    assert ledger2.entries[1].amount == Amount(Decimal(50))
    assert ledger2.entries[1].balance == Quantity(Decimal(150))


# LLM-generated content at query #2
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the ReadInitialBalances implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("-500.00")))
        }

    # Create an instance of ReadInitialBalances
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Define a test period
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Call the __call__ method
    result = read_initial_balances(test_period)

    # Assert the result is as expected
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Asset:Cash") in result
    assert Account("Liability:Loan") in result
    assert result[Account("Asset:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liability:Loan")].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #3
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock DateRange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Create a mock ReadInitialBalances instance
    read_initial_balances = ReadInitialBalances()

    # Call the __call__ method
    result = read_initial_balances(period)

    # Assert the result is of type InitialBalances (Dict[Account, Balance])
    assert isinstance(result, dict)

    # Assert all keys are Account instances and values are Balance instances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
    expected_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == expected_balances[account1].value
    assert result[account2].value == expected_balances[account2].value


# LLM-generated content at query #5
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("50.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("20.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("20.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("100.00")
    assert len(account1_ledger.entries) == 2

    # Check first entry of account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("120.00")
    assert entry1.amount == Decimal("20.00")
    assert entry1.is_debit
    assert not entry1.is_credit
    assert entry1.debit == Decimal("20.00")
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("90.00")
    assert entry2.amount == Decimal("30.00")
    assert entry2.is_credit
    assert not entry2.is_debit
    assert entry2.credit == Decimal("30.00")
    assert entry2.debit is None

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("50.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("30.00")
    assert entry1.amount == Decimal("20.00")
    assert entry1.is_credit
    assert not entry1.is_debit
    assert entry1.credit == Decimal("20.00")
    assert entry1.debit is None

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("60.00")
    assert entry2.amount == Decimal("30.00")
    assert entry2.is_debit
    assert not entry2.is_credit
    assert entry2.debit == Decimal("30.00")
    assert entry2.credit is None


# LLM-generated content at query #6
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")

    # Mock initial balances
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entries = [journal_entry1]

    # Mock read_initial_balances and read_journal_entries
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return journal_entries

    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal("100.00")
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal("200.00")
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.posting == journal_entry1.postings[1]
    assert entry2.balance == Quantity(Decimal("150.00"))


# LLM-generated content at query #7
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Asset:Cash")
    account2 = Account("Expense:Rent")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Rent payment",
        postings=[
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
        ]
    )

    # Call the function
    result = build_general_ledger(period, [journal_entry], initial_balances)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period

    # Check ledgers
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers

    # Check cash account ledger
    cash_ledger = result.ledgers[account1]
    assert cash_ledger.account == account1
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 1

    cash_entry = cash_ledger.entries[0]
    assert cash_entry.posting == journal_entry.postings[1]
    assert cash_entry.balance == Decimal("500.00")
    assert cash_entry.amount == Decimal("500.00")
    assert cash_entry.is_credit
    assert not cash_entry.is_debit
    assert cash_entry.credit == Decimal("500.00")
    assert cash_entry.debit is None

    # Check rent account ledger
    rent_ledger = result.ledgers[account2]
    assert rent_ledger.account == account2
    assert rent_ledger.initial.value == Decimal("0.00")
    assert len(rent_ledger.entries) == 1

    rent_entry = rent_ledger.entries[0]
    assert rent_entry.posting == journal_entry.postings[0]
    assert rent_entry.balance == Decimal("500.00")
    assert rent_entry.amount == Decimal("500.00")
    assert rent_entry.is_debit
    assert not rent_entry.is_credit
    assert rent_entry.debit == Decimal("500.00")
    assert rent_entry.credit is None


# LLM-generated content at query #8
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("100.00")
    assert len(account1_ledger.entries) == 2

    # First entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("200.00")
    assert len(account2_ledger.entries) == 2

    # First entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("180.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=PostingDirection.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=PostingDirection.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=PostingDirection.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=PostingDirection.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Quantity(Decimal("100.00"))
    assert len(account1_ledger.entries) == 2

    # Check first entry of account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("120.00"))
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("50.00"))
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("20.00"))
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #10
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    journal_entries = [journal_entry1, journal_entry2]

    # Mock read_initial_balances and read_journal_entries
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return journal_entries

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Call the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger1.entries[1].amount == Amount(Decimal("30.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("120.00"))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger2.entries[1].amount == Amount(Decimal("30.00"))
    assert ledger2.entries[1].balance == Quantity(Decimal("180.00"))


# LLM-generated content at query #11
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")

    # Mock initial balances
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
        ]
    )

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return [journal_entry1, journal_entry2]

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Call the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].amount == Amount(Decimal(50))
    assert account1_ledger.entries[0].balance == Quantity(Decimal(150))
    assert account1_ledger.entries[1].amount == Amount(Decimal(30))
    assert account1_ledger.entries[1].balance == Quantity(Decimal(120))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal(50))
    assert account2_ledger.entries[0].balance == Quantity(Decimal(150))
    assert account2_ledger.entries[1].amount == Amount(Decimal(30))
    assert account2_ledger.entries[1].balance == Quantity(Decimal(180))


# LLM-generated content at query #12
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    expected_balances = {
        Account("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    # Test
    result = mock_read_initial_balances(period)

    # Assert
    assert result == expected_balances
    assert isinstance(result, dict)
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == period.since


# LLM-generated content at query #13
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Cash")
    account2 = Account("Revenue")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("1000.00")
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("1500.00")
    assert entry1.debit == Amount(Decimal("500.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("1300.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("200.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("500.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("500.00"))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("300.00")
    assert entry2.debit == Amount(Decimal("200.00"))
    assert entry2.credit is None


# LLM-generated content at query #14
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Exercise
    result = read_initial_balances(period)

    # Verify
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("2000.00"))


# LLM-generated content at query #15
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")

    # Mock initial balances
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(1000))),
        account2: Balance(period.since, Quantity(Decimal(2000)))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(500)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(500)), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(300)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(300)), direction=Direction.DEBIT)
        ]
    )

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return [journal_entry1, journal_entry2]

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Call the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].amount == Amount(Decimal(500))
    assert account1_ledger.entries[0].balance == Quantity(Decimal(1500))
    assert account1_ledger.entries[1].amount == Amount(Decimal(300))
    assert account1_ledger.entries[1].balance == Quantity(Decimal(1200))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal(500))
    assert account2_ledger.entries[0].balance == Quantity(Decimal(1500))
    assert account2_ledger.entries[1].amount == Amount(Decimal(300))
    assert account2_ledger.entries[1].balance == Quantity(Decimal(1800))


# LLM-generated content at query #16
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    account1 = Account("Account1")
    account2 = Account("Account2")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    # Mock functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal("100.00")
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))

    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal("200.00")
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))


# LLM-generated content at query #17
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the call
    result = mock_read_initial_balances(period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("100.00")
    assert result[account2].value == Decimal("200.00")


# LLM-generated content at query #18
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {
            Account("Cash"): Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
            Account("Accounts Receivable"): Balance(datetime.date(2022, 12, 31), Quantity(Decimal("500.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        assert period == test_period
        return [
            JournalEntry(
                date=datetime.date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        account=Account("Cash"),
                        amount=Amount(Decimal("200.00")),
                        direction=Posting.Direction.DEBIT
                    ),
                    Posting(
                        account=Account("Revenue"),
                        amount=Amount(Decimal("200.00")),
                        direction=Posting.Direction.CREDIT
                    )
                ]
            )
        ]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period

    # Check Cash ledger
    cash_ledger = result.ledgers[Account("Cash")]
    assert cash_ledger.account.name == "Cash"
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1200.00"))

    # Check Accounts Receivable ledger
    ar_ledger = result.ledgers[Account("Accounts Receivable")]
    assert ar_ledger.account.name == "Accounts Receivable"
    assert ar_ledger.initial.value == Decimal("500.00")
    assert len(ar_ledger.entries) == 0

    # Check Revenue ledger (should be created automatically)
    revenue_ledger = result.ledgers[Account("Revenue")]
    assert revenue_ledger.account.name == "Revenue"
    assert revenue_ledger.initial.value == Decimal("0.00")
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-200.00"))


# LLM-generated content at query #19
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(period.since, Quantity(Decimal(1000)))
    expected_initial_balances = {mock_account: mock_balance}

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    result = mock_read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account] == mock_balance


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("50.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("25.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("25.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("100.00")
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("125.00")
    assert entry1.debit == Amount(Decimal("25.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("95.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("50.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("75.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("25.00"))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("45.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200))),
    }

    # Create test journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT),
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT),
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Quantity(Decimal(100))
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal(150))  # 100 + 50
    assert entry1.debit == Amount(Decimal(50))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal(120))  # 150 - 30
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal(30))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Quantity(Decimal(200))
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal(150))  # 200 - 50
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal(50))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal(180))  # 150 + 30
    assert entry2.debit == Amount(Decimal(30))
    assert entry2.credit is None


# LLM-generated content at query #3
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock ReadInitialBalances instance
    read_initial_balances = ReadInitialBalances()

    # Define a test period
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Call the __call__ method
    result = read_initial_balances(test_period)

    # Assert that the result is a dictionary
    assert isinstance(result, dict)

    # Assert that the keys are Account instances
    for account in result.keys():
        assert isinstance(account, Account)

    # Assert that the values are Balance instances
    for balance in result.values():
        assert isinstance(balance, Balance)


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert len(result) == 2
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #5
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account("TestAccount")
    test_balance = Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00")))
    test_initial_balances = {test_account: test_balance}

    # Mock the read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return test_initial_balances

    # Mock the read_journal_entries function
    test_journal_entry = JournalEntry(
        datetime.date(2023, 6, 15),
        "Test description",
        [
            Posting(test_account, Amount(Decimal("50.00")), Direction.DEBIT),
            Posting(Account("AnotherAccount"), Amount(Decimal("50.00")), Direction.CREDIT),
        ],
    )
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return [test_journal_entry]

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 2  # TestAccount and AnotherAccount

    # Check the TestAccount ledger
    test_ledger = result.ledgers[test_account]
    assert test_ledger.account == test_account
    assert test_ledger.initial == test_balance
    assert len(test_ledger.entries) == 1

    # Check the ledger entry
    entry = test_ledger.entries[0]
    assert entry.posting == test_journal_entry.postings[0]
    assert entry.balance == Quantity(Decimal("150.00"))  # 100 + 50
    assert entry.date == datetime.date(2023, 6, 15)
    assert entry.description == "Test description"
    assert entry.amount == Amount(Decimal("50.00"))
    assert entry.is_debit
    assert not entry.is_credit
    assert entry.debit == Amount(Decimal("50.00"))
    assert entry.credit is None

    # Check the AnotherAccount ledger
    another_ledger = result.ledgers[Account("AnotherAccount")]
    assert another_ledger.account == Account("AnotherAccount")
    assert another_ledger.initial == Balance(test_period.since, Quantity(Decimal(0)))
    assert len(another_ledger.entries) == 1

    # Check the ledger entry for AnotherAccount
    another_entry = another_ledger.entries[0]
    assert another_entry.posting == test_journal_entry.postings[1]
    assert another_entry.balance == Quantity(Decimal("-50.00"))
    assert another_entry.date == datetime.date(2023, 6, 15)
    assert another_entry.description == "Test description"
    assert another_entry.amount == Amount(Decimal("50.00"))
    assert not another_entry.is_debit
    assert another_entry.is_credit
    assert another_entry.debit is None
    assert another_entry.credit == Amount(Decimal("50.00"))


# LLM-generated content at query #6
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(1000))),
        account2: Balance(period.since, Quantity(Decimal(2000)))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(100)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(100)), direction=Direction.CREDIT)
        ]
    )
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(200)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(200)), direction=Direction.DEBIT)
        ]
    )
    journal_entries = [journal_entry1, journal_entry2]

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        return journal_entries

    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.initial.value == Quantity(Decimal(1000))
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].amount == Amount(Decimal(100))
    assert account1_ledger.entries[0].is_debit
    assert account1_ledger.entries[0].balance == Quantity(Decimal(1100))
    assert account1_ledger.entries[1].amount == Amount(Decimal(200))
    assert account1_ledger.entries[1].is_credit
    assert account1_ledger.entries[1].balance == Quantity(Decimal(900))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.initial.value == Quantity(Decimal(2000))
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal(100))
    assert account2_ledger.entries[0].is_credit
    assert account2_ledger.entries[0].balance == Quantity(Decimal(1900))
    assert account2_ledger.entries[1].amount == Amount(Decimal(200))
    assert account2_ledger.entries[1].is_debit
    assert account2_ledger.entries[1].balance == Quantity(Decimal(2100))


# LLM-generated content at query #7
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
    expected_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("-500.00")


# LLM-generated content at query #8
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Asset", "Cash")
    account2 = Account("Equity", "Capital")
    initial_balance = Balance(period.since, Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance}

    # Mock journal entries
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(500)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(500)), direction=Direction.CREDIT),
        ]
    )

    # Mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        return [journal_entry]

    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balance
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry.postings[0]
    assert entry1.balance == Quantity(Decimal(1500))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.posting == journal_entry.postings[1]
    assert entry2.balance == Quantity(Decimal(-500))


# LLM-generated content at query #9
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    mock_initial_balances = {
        Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("500.00")))
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return mock_initial_balances

    # Act
    result = mock_read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Asset:Cash") in result
    assert Account("Liability:Loan") in result
    assert result[Account("Asset:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liability:Loan")].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("50.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("25.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("25.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("100.00"))
    assert len(ledger1.entries) == 2

    # Check first entry in ledger1
    entry1_ledger1 = ledger1.entries[0]
    assert entry1_ledger1.posting == journal_entry1.postings[0]
    assert entry1_ledger1.balance == Quantity(Decimal("125.00"))
    assert entry1_ledger1.debit == Amount(Decimal("25.00"))
    assert entry1_ledger1.credit is None

    # Check second entry in ledger1
    entry2_ledger1 = ledger1.entries[1]
    assert entry2_ledger1.posting == journal_entry2.postings[0]
    assert entry2_ledger1.balance == Quantity(Decimal("95.00"))
    assert entry2_ledger1.debit is None
    assert entry2_ledger1.credit == Amount(Decimal("30.00"))

    # Check ledger2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("50.00"))
    assert len(ledger2.entries) == 2

    # Check first entry in ledger2
    entry1_ledger2 = ledger2.entries[0]
    assert entry1_ledger2.posting == journal_entry1.postings[1]
    assert entry1_ledger2.balance == Quantity(Decimal("75.00"))
    assert entry1_ledger2.debit is None
    assert entry1_ledger2.credit == Amount(Decimal("25.00"))

    # Check second entry in ledger2
    entry2_ledger2 = ledger2.entries[1]
    assert entry2_ledger2.posting == journal_entry2.postings[1]
    assert entry2_ledger2.balance == Quantity(Decimal("45.00"))
    assert entry2_ledger2.debit == Amount(Decimal("30.00"))
    assert entry2_ledger2.credit is None


# LLM-generated content at query #11
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")

    # Mock initial balances
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
        ]
    )

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return [journal_entry1, journal_entry2]

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].amount == Amount(Decimal(50))
    assert account1_ledger.entries[0].balance == Quantity(Decimal(150))
    assert account1_ledger.entries[1].amount == Amount(Decimal(30))
    assert account1_ledger.entries[1].balance == Quantity(Decimal(120))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal(50))
    assert account2_ledger.entries[0].balance == Quantity(Decimal(150))
    assert account2_ledger.entries[1].amount == Amount(Decimal(30))
    assert account2_ledger.entries[1].balance == Quantity(Decimal(180))


# LLM-generated content at query #12
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    test_period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    test_account1 = Account("TestAccount1")
    test_account2 = Account("TestAccount2")
    test_initial_balances = {
        test_account1: Balance(date=datetime.date(2022, 12, 31), value=Quantity(Decimal("100.00"))),
        test_account2: Balance(date=datetime.date(2022, 12, 31), value=Quantity(Decimal("200.00")))
    }

    # Mock journal entries
    test_journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=test_account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=test_account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return test_initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return [test_journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 2
    assert test_account1 in result.ledgers
    assert test_account2 in result.ledgers

    # Check ledger entries
    ledger1 = result.ledgers[test_account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger1.entries[0].is_debit
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))

    ledger2 = result.ledgers[test_account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger2.entries[0].is_credit
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))


# LLM-generated content at query #13
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return [journal_entry1, journal_entry2]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].amount == Amount(Decimal("50.00"))
    assert account1_ledger.entries[0].balance == Quantity(Decimal("150.00"))
    assert account1_ledger.entries[1].amount == Amount(Decimal("30.00"))
    assert account1_ledger.entries[1].balance == Quantity(Decimal("120.00"))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal("50.00"))
    assert account2_ledger.entries[0].balance == Quantity(Decimal("150.00"))
    assert account2_ledger.entries[1].amount == Amount(Decimal("30.00"))
    assert account2_ledger.entries[1].balance == Quantity(Decimal("180.00"))


# LLM-generated content at query #14
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    mock_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000)))
    mock_initial_balances = {mock_account: mock_balance}

    # Mock journal entries
    mock_journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(
                account=mock_account,
                amount=Amount(Decimal(500)),
                direction=PostingDirection.DEBIT
            )
        ]
    )
    mock_journal_entries = [mock_journal_entry]

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return mock_initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        return mock_journal_entries

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    general_ledger = program(mock_period)

    # Assertions
    assert general_ledger.period == mock_period
    assert len(general_ledger.ledgers) == 1
    assert mock_account in general_ledger.ledgers

    ledger = general_ledger.ledgers[mock_account]
    assert ledger.account == mock_account
    assert ledger.initial == mock_balance
    assert len(ledger.entries) == 1

    entry = ledger.entries[0]
    assert entry.ledger == ledger
    assert entry.posting == mock_journal_entry.postings[0]
    assert entry.balance == Quantity(Decimal(1500))  # Initial balance (1000) + debit (500)
    assert entry.date == datetime.date(2023, 1, 15)
    assert entry.description == "Test transaction"
    assert entry.amount == Amount(Decimal(500))
    assert entry.is_debit is True
    assert entry.is_credit is False
    assert entry.debit == Amount(Decimal(500))
    assert entry.credit is None


# LLM-generated content at query #15
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Asset", "Cash")
    account2 = Account("Liability", "Loans")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("-500.00")
    assert result[account1].date == period.since
    assert result[account2].date == period.since


# LLM-generated content at query #16
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=PostingDirection.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=PostingDirection.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=PostingDirection.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=PostingDirection.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("100.00")
    assert len(account1_ledger.entries) == 2

    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("200.00")
    assert len(account2_ledger.entries) == 2

    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("180.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


