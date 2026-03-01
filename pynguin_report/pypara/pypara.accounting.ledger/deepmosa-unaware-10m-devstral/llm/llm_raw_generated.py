####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("150.00")
    assert entry1.date == datetime.date(2023, 1, 15)
    assert entry1.description == "Test entry 1"
    assert entry1.amount == Amount(Decimal("50.00"))
    assert entry1.is_debit
    assert not entry1.is_credit
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.date == datetime.date(2023, 2, 20)
    assert entry2.description == "Test entry 2"
    assert entry2.amount == Amount(Decimal("30.00"))
    assert not entry2.is_debit
    assert entry2.is_credit
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("50.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("100.00")
    assert entry1.date == datetime.date(2023, 1, 15)
    assert entry1.description == "Test entry 1"
    assert entry1.amount == Amount(Decimal("50.00"))
    assert not entry1.is_debit
    assert entry1.is_credit
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("70.00")
    assert entry2.date == datetime.date(2023, 2, 20)
    assert entry2.description == "Test entry 2"
    assert entry2.amount == Amount(Decimal("30.00"))
    assert entry2.is_debit
    assert not entry2.is_credit
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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
        date=datetime.date(2023, 2, 20),
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

    # Check first entry of account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("200.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("180.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #3
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:AccountsPayable")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test transaction 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("300.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("300.00")), direction=Direction.DEBIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
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
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].amount == Amount(Decimal("200.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("1200.00"))
    assert ledger1.entries[1].amount == Amount(Decimal("300.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("900.00"))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal("200.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("700.00"))
    assert ledger2.entries[1].amount == Amount(Decimal("300.00"))
    assert ledger2.entries[1].balance == Quantity(Decimal("400.00"))


# LLM-generated content at query #4
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

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

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2

    # Check first entry for account1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal(150))  # 100 + 50

    # Check second entry for account1
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal(120))  # 150 - 30

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 2

    # Check first entry for account2
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal(150))  # 200 - 50

    # Check second entry for account2
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal(180))  # 150 + 30


# LLM-generated content at query #5
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Cash")
    account2 = Account("Revenue")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test entry 1",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
            ]
        ),
        JournalEntry(
            date=datetime.date(2023, 2, 20),
            description="Test entry 2",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("300.00")), direction=Direction.CREDIT),
                Posting(account=account2, amount=Amount(Decimal("300.00")), direction=Direction.DEBIT)
            ]
        )
    ]

    # Call the function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2

    # Check Cash account ledger
    cash_ledger = general_ledger.ledgers[account1]
    assert cash_ledger.account == account1
    assert cash_ledger.initial == initial_balances[account1]
    assert len(cash_ledger.entries) == 2

    # First entry
    entry1 = cash_ledger.entries[0]
    assert entry1.posting.amount == Amount(Decimal("500.00"))
    assert entry1.balance == Quantity(Decimal("1500.00"))
    assert entry1.is_debit
    assert entry1.debit == Amount(Decimal("500.00"))
    assert entry1.credit is None

    # Second entry
    entry2 = cash_ledger.entries[1]
    assert entry2.posting.amount == Amount(Decimal("300.00"))
    assert entry2.balance == Quantity(Decimal("1200.00"))
    assert entry2.is_credit
    assert entry2.credit == Amount(Decimal("300.00"))
    assert entry2.debit is None

    # Check Revenue account ledger
    revenue_ledger = general_ledger.ledgers[account2]
    assert revenue_ledger.account == account2
    assert revenue_ledger.initial == initial_balances[account2]
    assert len(revenue_ledger.entries) == 2

    # First entry
    entry1 = revenue_ledger.entries[0]
    assert entry1.posting.amount == Amount(Decimal("500.00"))
    assert entry1.balance == Quantity(Decimal("500.00"))
    assert entry1.is_credit
    assert entry1.credit == Amount(Decimal("500.00"))
    assert entry1.debit is None

    # Second entry
    entry2 = revenue_ledger.entries[1]
    assert entry2.posting.amount == Amount(Decimal("300.00"))
    assert entry2.balance == Quantity(Decimal("200.00"))
    assert entry2.is_debit
    assert entry2.debit == Amount(Decimal("300.00"))
    assert entry2.credit is None


# LLM-generated content at query #6
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period and expected initial balances
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_balances = {
        Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    # Call the function
    result = mock_read_initial_balances(period)

    # Assert the result matches expected balances
    assert result == expected_balances


# LLM-generated content at query #7
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("50.00")))
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
        date=datetime.date(2023, 2, 20),
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
    assert entry1.balance == Decimal("25.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("25.00"))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("55.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #8
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
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #9
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        journal_entry = JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test transaction",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
            ]
        )
        return [journal_entry]

    # Create GeneralLedgerProgram instance
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
    assert len(account1_ledger.entries) == 1
    assert account1_ledger.entries[0].amount == Amount(Decimal("50.00"))
    assert account1_ledger.entries[0].balance == Quantity(Decimal("150.00"))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 1
    assert account2_ledger.entries[0].amount == Amount(Decimal("50.00"))
    assert account2_ledger.entries[0].balance == Quantity(Decimal("150.00"))


# LLM-generated content at query #10
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period and expected initial balances
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_initial_balances = {
        Account("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal("500.00"))),
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the __call__ method
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert the result matches the expected initial balances
    assert result == expected_initial_balances


# LLM-generated content at query #11
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")

    # Mock initial balances
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("200.00")))
    }

    # Mock journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Test entry",
            [
                Posting(account1, Amount(Decimal("50.00")), 1),
                Posting(account2, Amount(Decimal("50.00")), -1)
            ]
        )
    ]

    # Mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
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
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.amount == Amount(Decimal("50.00"))
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.is_debit

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.amount == Amount(Decimal("50.00"))
    assert entry2.balance == Quantity(Decimal("150.00"))
    assert entry2.is_credit


# LLM-generated content at query #12
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account1 = Account("Account1")
    mock_account2 = Account("Account2")
    expected_initial_balances = {
        mock_account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        mock_account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert mock_account1 in result
    assert mock_account2 in result
    assert result[mock_account1].value == Decimal("1000.00")
    assert result[mock_account2].value == Decimal("2000.00")


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_balances = {
        Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("-500.00"))),
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_balances


# LLM-generated content at query #14
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    account1 = Account("Account1")
    account2 = Account("Account2")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
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
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("50.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("20.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #15
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("200.00")))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        datetime.date(2023, 1, 15),
        "Test Entry 1",
        [
            Posting(account1, Amount(Decimal("50.00")), Direction.DEBIT),
            Posting(account2, Amount(Decimal("50.00")), Direction.CREDIT)
        ]
    )
    journal_entries = [journal_entry1]

    # Mock read functions
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return journal_entries

    # Create GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Call the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.posting == journal_entry1.postings[1]
    assert entry2.balance == Quantity(Decimal("150.00"))


# LLM-generated content at query #16
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(period.since, Quantity(Decimal(1000)))
    expected_initial_balances = {mock_account: mock_balance}

    # Mock implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account] == mock_balance


# LLM-generated content at query #17
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Mock accounts and balances
    account1 = Account("Asset:Cash")
    account2 = Account("Liability:Loan")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Create a mock ReadInitialBalances instance
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    read_initial_balances = mock_read_initial_balances

    # Call the method
    result = read_initial_balances(period)

    # Assert the result
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #18
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account1 = Account("TestAccount1")
    test_account2 = Account("TestAccount2")
    test_initial_balance1 = Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00")))
    test_initial_balance2 = Balance(datetime.date(2022, 12, 31), Quantity(Decimal("200.00")))
    test_initial_balances = {test_account1: test_initial_balance1, test_account2: test_initial_balance2}

    # Mock the read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return test_initial_balances

    # Create test journal entries
    test_journal_entry1 = JournalEntry(
        datetime.date(2023, 1, 15),
        "Test entry 1",
        [
            Posting(test_account1, Amount(Decimal("50.00")), Direction.DEBIT),
            Posting(test_account2, Amount(Decimal("50.00")), Direction.CREDIT),
        ]
    )
    test_journal_entry2 = JournalEntry(
        datetime.date(2023, 2, 20),
        "Test entry 2",
        [
            Posting(test_account1, Amount(Decimal("30.00")), Direction.CREDIT),
            Posting(test_account2, Amount(Decimal("30.00")), Direction.DEBIT),
        ]
    )
    test_journal_entries = [test_journal_entry1, test_journal_entry2]

    # Mock the read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return test_journal_entries

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert result.period == test_period
    assert len(result.ledgers) == 2
    assert test_account1 in result.ledgers
    assert test_account2 in result.ledgers

    # Verify ledger entries for test_account1
    ledger1 = result.ledgers[test_account1]
    assert ledger1.account == test_account1
    assert ledger1.initial == test_initial_balance1
    assert len(ledger1.entries) == 2

    # Verify first entry
    entry1 = ledger1.entries[0]
    assert entry1.posting == test_journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))  # 100 + 50

    # Verify second entry
    entry2 = ledger1.entries[1]
    assert entry2.posting == test_journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("120.00"))  # 150 - 30

    # Verify ledger entries for test_account2
    ledger2 = result.ledgers[test_account2]
    assert ledger2.account == test_account2
    assert ledger2.initial == test_initial_balance2
    assert len(ledger2.entries) == 2

    # Verify first entry
    entry1 = ledger2.entries[0]
    assert entry1.posting == test_journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("150.00"))  # 200 - 50

    # Verify second entry
    entry2 = ledger2.entries[1]
    assert entry2.posting == test_journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("180.00"))  # 150 + 30


# LLM-generated content at query #19
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(period.since, Quantity(Decimal(1000)))
    expected_initial_balances = {mock_account: mock_balance}

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account] == mock_balance
    assert result == expected_initial_balances


# LLM-generated content at query #20
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Receivable")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("300.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("300.00")), direction=Direction.DEBIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        return [journal_entry1, journal_entry2]

    # Create program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    account1_ledger = general_ledger.ledgers[account1]
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert account1_ledger.entries[0].balance == Quantity(Decimal("1200.00"))
    assert account1_ledger.entries[1].amount == Amount(Decimal("300.00"))
    assert account1_ledger.entries[1].balance == Quantity(Decimal("900.00"))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert account2_ledger.entries[0].balance == Quantity(Decimal("300.00"))
    assert account2_ledger.entries[1].amount == Amount(Decimal("300.00"))
    assert account2_ledger.entries[1].balance == Quantity(Decimal("600.00"))


# LLM-generated content at query #21
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

    # Create test journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
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
    assert account1_ledger.initial == initial_balances[account1]
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("120.00"))
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial == initial_balances[account2]
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("50.00"))
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("20.00"))
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #22
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
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("20.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("20.00")), direction=Direction.CREDIT)
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
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].amount == Amount(Decimal("20.00"))
    assert account1_ledger.entries[0].balance == Quantity(Decimal("120.00"))
    assert account1_ledger.entries[1].amount == Amount(Decimal("30.00"))
    assert account1_ledger.entries[1].balance == Quantity(Decimal("90.00"))

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal("20.00"))
    assert account2_ledger.entries[0].balance == Quantity(Decimal("30.00"))
    assert account2_ledger.entries[1].amount == Amount(Decimal("30.00"))
    assert account2_ledger.entries[1].balance == Quantity(Decimal("60.00"))


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Mock the initial balances
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(1000))),
        account2: Balance(period.since, Quantity(Decimal(2000)))
    }

    # Create a mock function for ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Test the function
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assertions
    assert isinstance(result, dict)
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal(1000))
    assert result[account2].value == Quantity(Decimal(2000))


# LLM-generated content at query #24
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test transaction",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        return [journal_entry]

    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execution
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))

    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))


# LLM-generated content at query #25
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account("TestAccount")
    test_initial_balance = Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00")))

    # Mock the read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {test_account: test_initial_balance}

    # Mock the read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        assert period == test_period
        return []

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 1
    assert test_account in result.ledgers
    assert result.ledgers[test_account].account == test_account
    assert result.ledgers[test_account].initial == test_initial_balance
    assert len(result.ledgers[test_account].entries) == 0


# LLM-generated content at query #26
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Mock journal entries
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

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
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


# LLM-generated content at query #27
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2

    # Check first entry of ledger1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of ledger1
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("120.00"))
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check ledger2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 2

    # Check first entry of ledger2
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of ledger2
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("180.00"))
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #28
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
    expected_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    # Test
    read_initial_balances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == expected_balances[account1].value
    assert result[account2].value == expected_balances[account2].value


# LLM-generated content at query #29
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

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #30
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(0)))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
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
    assert account1_ledger.initial.value == Decimal(100)
    assert len(account1_ledger.entries) == 2

    # Check first entry of account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)  # 100 + 50
    assert entry1.debit == Amount(Decimal(50))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)  # 150 - 30
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal(30))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal(0)
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(50)  # 0 + 50
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal(50))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(20)  # 50 - 30
    assert entry2.debit == Amount(Decimal(30))
    assert entry2.credit is None


# LLM-generated content at query #31
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_initial_balances = {
        Account("Cash"): Balance(period.since, Quantity(Decimal(1000))),
        Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal(500))),
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances


# LLM-generated content at query #32
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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


# LLM-generated content at query #33
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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


# LLM-generated content at query #34
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Expenses:Rent")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Rent payment",
        postings=[
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
        ]
    )

    # Build general ledger
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)

    # Assertions
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check cash ledger
    cash_ledger = general_ledger.ledgers[account1]
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    assert cash_ledger.entries[0].is_credit
    assert cash_ledger.entries[0].cntraccts == [account2]

    # Check rent ledger
    rent_ledger = general_ledger.ledgers[account2]
    assert len(rent_ledger.entries) == 1
    assert rent_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert rent_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    assert rent_ledger.entries[0].is_debit
    assert rent_ledger.entries[0].cntraccts == [account1]


# LLM-generated content at query #35
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account("TestAccount")
    test_initial_balance = Balance(test_period.since, Quantity(Decimal(1000)))
    test_initial_balances = {test_account: test_initial_balance}

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return test_initial_balances

    # Mock journal entries
    test_journal_entry = JournalEntry(
        date=test_period.since,
        description="Test entry",
        postings=[
            Posting(
                account=test_account,
                amount=Amount(Decimal(500)),
                direction=Posting.Direction.DEBIT,
            )
        ]
    )
    test_journal_entries = [test_journal_entry]

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        assert period == test_period
        return test_journal_entries

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 1
    assert test_account in result.ledgers

    ledger = result.ledgers[test_account]
    assert ledger.account == test_account
    assert ledger.initial == test_initial_balance
    assert len(ledger.entries) == 1

    entry = ledger.entries[0]
    assert entry.ledger == ledger
    assert entry.posting == test_journal_entry.postings[0]
    assert entry.balance == Quantity(Decimal(1500))
    assert entry.date == test_period.since
    assert entry.description == "Test entry"
    assert entry.amount == Amount(Decimal(500))
    assert entry.is_debit
    assert not entry.is_credit
    assert entry.debit == Amount(Decimal(500))
    assert entry.credit is None


# LLM-generated content at query #36
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Asset", "Cash")
    account2 = Account("Liability", "Accounts Payable")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #37
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
        account2: Balance(period.since, Quantity(Decimal(0)))
    }

    # Mock journal entries
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(100)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(100)), direction=Direction.CREDIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return [journal_entry]

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
    assert len(account1_ledger.entries) == 1
    entry = account1_ledger.entries[0]
    assert entry.amount == Amount(Decimal(100))
    assert entry.balance == Quantity(Decimal(1100))
    assert entry.is_debit
    assert not entry.is_credit

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 1
    entry = account2_ledger.entries[0]
    assert entry.amount == Amount(Decimal(100))
    assert entry.balance == Quantity(Decimal(100))
    assert entry.is_credit
    assert not entry.is_debit


# LLM-generated content at query #38
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {
            Account("Assets:Cash"): Balance(test_period.since, Quantity(Decimal("1000.00"))),
            Account("Liabilities:Loans"): Balance(test_period.since, Quantity(Decimal("500.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return [
            JournalEntry(
                date=datetime.date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        account=Account("Assets:Cash"),
                        amount=Amount(Decimal("200.00")),
                        direction=Direction.DEBIT
                    ),
                    Posting(
                        account=Account("Liabilities:Loans"),
                        amount=Amount(Decimal("200.00")),
                        direction=Direction.CREDIT
                    )
                ]
            )
        ]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period

    # Check ledgers
    assert len(result.ledgers) == 2

    # Check Cash ledger
    cash_ledger = result.ledgers[Account("Assets:Cash")]
    assert cash_ledger.account == Account("Assets:Cash")
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1200.00"))
    assert cash_ledger.entries[0].is_debit
    assert not cash_ledger.entries[0].is_credit

    # Check Loans ledger
    loans_ledger = result.ledgers[Account("Liabilities:Loans")]
    assert loans_ledger.account == Account("Liabilities:Loans")
    assert loans_ledger.initial.value == Quantity(Decimal("500.00"))
    assert len(loans_ledger.entries) == 1
    assert loans_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert loans_ledger.entries[0].balance == Quantity(Decimal("300.00"))
    assert loans_ledger.entries[0].is_credit
    assert not loans_ledger.entries[0].is_debit


# LLM-generated content at query #39
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
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test transaction",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
            ]
        )
    ]

    # Call the function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check Cash ledger
    cash_ledger = general_ledger.ledgers[account1]
    assert cash_ledger.account == account1
    assert cash_ledger.initial == initial_balances[account1]
    assert len(cash_ledger.entries) == 1
    entry = cash_ledger.entries[0]
    assert entry.posting == journal_entries[0].postings[0]
    assert entry.balance == Quantity(Decimal("1500.00"))
    assert entry.debit == Amount(Decimal("500.00"))
    assert entry.credit is None

    # Check Revenue ledger
    revenue_ledger = general_ledger.ledgers[account2]
    assert revenue_ledger.account == account2
    assert revenue_ledger.initial == initial_balances[account2]
    assert len(revenue_ledger.entries) == 1
    entry = revenue_ledger.entries[0]
    assert entry.posting == journal_entries[0].postings[1]
    assert entry.balance == Quantity(Decimal("500.00"))
    assert entry.debit is None
    assert entry.credit == Amount(Decimal("500.00"))


# LLM-generated content at query #40
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Create test journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=PostingDirection.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=PostingDirection.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=PostingDirection.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=PostingDirection.DEBIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function under test
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal(100)
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)  # 100 + 50
    assert entry1.debit == Amount(Decimal(50))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)  # 150 - 30
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal(30))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal(200)
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(150)  # 200 - 50
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal(50))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(180)  # 150 + 30
    assert entry2.debit == Amount(Decimal(30))
    assert entry2.credit is None


# LLM-generated content at query #41
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Asset:Cash")
    account2 = Account("Liability:Loans")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert len(result) == 2
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #42
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00"))),
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


# LLM-generated content at query #43
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock ReadInitialBalances instance
    read_initial_balances = ReadInitialBalances()

    # Define a test period
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Call the __call__ method
    result = read_initial_balances(period)

    # Assert that the result is a dictionary
    assert isinstance(result, dict)

    # Assert that the keys are Account instances
    for key in result.keys():
        assert isinstance(key, Account)

    # Assert that the values are Balance instances
    for value in result.values():
        assert isinstance(value, Balance)


# LLM-generated content at query #44
#--------------------------

```python
def test_build_general_ledger():
    # Setup
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    account1 = Account("Asset:Cash")
    account2 = Account("Income:Sales")
    account3 = Account("Expense:Rent")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00"))),
        account3: Balance(period.since, Quantity(Decimal("0.00")))
    }

    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
            ]
        ),
        JournalEntry(
            date=datetime.date(2023, 1, 20),
            description="Rent Payment",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT),
                Posting(account=account3, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT)
            ]
        )
    ]

    # Execute
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert len(general_ledger.ledgers) == 3

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("1000.00")
    assert len(account1_ledger.entries) == 2

    # First entry (sale)
    entry1 = account1_ledger.entries[0]
    assert entry1.posting.amount == Amount(Decimal("500.00"))
    assert entry1.balance == Quantity(Decimal("1500.00"))
    assert entry1.is_debit
    assert entry1.debit == Amount(Decimal("500.00"))
    assert entry1.credit is None

    # Second entry (rent payment)
    entry2 = account1_ledger.entries[1]
    assert entry2.posting.amount == Amount(Decimal("200.00"))
    assert entry2.balance == Quantity(Decimal("1300.00"))
    assert entry2.is_credit
    assert entry2.credit == Amount(Decimal("200.00"))
    assert entry2.debit is None

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 1

    # First entry (sale)
    entry1 = account2_ledger.entries[0]
    assert entry1.posting.amount == Amount(Decimal("500.00"))
    assert entry1.balance == Quantity(Decimal("500.00"))
    assert entry1.is_credit
    assert entry1.credit == Amount(Decimal("500.00"))
    assert entry1.debit is None

    # Check account3 ledger
    account3_ledger = general_ledger.ledgers[account3]
    assert account3_ledger.account == account3
    assert account3_ledger.initial.value == Decimal("0.00")
    assert len(account3_ledger.entries) == 1

    # First entry (rent payment)
    entry1 = account3_ledger.entries[0]
    assert entry1.posting.amount == Amount(Decimal("200.00"))
    assert entry1.balance == Quantity(Decimal("200.00"))
    assert entry1.is_debit
    assert entry1.debit == Amount(Decimal("200.00"))
    assert entry1.credit is None


# LLM-generated content at query #45
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Asset:Cash")
    account2 = Account("Income:Sales")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
        ]
    )

    journal = [journal_entry1]

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
    assert len(account1_ledger.entries) == 1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("1500.00")
    assert entry1.amount == Amount(Decimal("500.00"))
    assert entry1.is_debit
    assert not entry1.is_credit

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 1
    entry2 = account2_ledger.entries[0]
    assert entry2.posting == journal_entry1.postings[1]
    assert entry2.balance == Decimal("500.00")
    assert entry2.amount == Amount(Decimal("500.00"))
    assert entry2.is_credit
    assert not entry2.is_debit

    # Test with no initial balance for an account
    account3 = Account("Expense:Rent")
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Rent payment",
        postings=[
            Posting(account=account3, amount=Amount(Decimal("300.00")), direction=Direction.DEBIT),
            Posting(account=account1, amount=Amount(Decimal("300.00")), direction=Direction.CREDIT)
        ]
    )

    journal_with_new_account = [journal_entry1, journal_entry2]
    general_ledger_new = build_general_ledger(period, journal_with_new_account, initial_balances)

    assert len(general_ledger_new.ledgers) == 3
    assert account3 in general_ledger_new.ledgers
    account3_ledger = general_ledger_new.ledgers[account3]
    assert account3_ledger.initial.value == Decimal("0.00")
    assert len(account3_ledger.entries) == 1
    entry3 = account3_ledger.entries[0]
    assert entry3.balance == Decimal("300.00")
    assert entry3.amount == Amount(Decimal("300.00"))
    assert entry3.is_debit

    # Test with out-of-period journal entry
    journal_entry_out_of_period = JournalEntry(
        date=datetime.date(2023, 2, 1),
        description="Out of period",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("100.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("100.00")), direction=Direction.CREDIT)
        ]
    )

    journal_with_out_of_period = [journal_entry1, journal_entry_out_of_period]
    general_ledger_out_of_period = build_general_ledger(period, journal_with_out_of_period, initial_balances)

    # The out-of-period entry should not be included
    assert len(general_ledger_out_of_period.ledgers[account1].entries) == 1
    assert len(general_ledger_out_of_period.ledgers[account2].entries) == 1


# LLM-generated content at query #46
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Asset:Cash")
    account2 = Account("Expense:Rent")
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Rent payment",
        postings=[
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Posting.Direction.DEBIT),
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Posting.Direction.CREDIT)
        ]
    )

    journal = [journal_entry1]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries for account1
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].amount == Amount(Decimal("500.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("500.00"))
    assert ledger1.entries[0].is_credit
    assert ledger1.entries[0].cntraccts == [account2]

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].amount == Amount(Decimal("500.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("500.00"))
    assert ledger2.entries[0].is_debit
    assert ledger2.entries[0].cntraccts == [account1]


# LLM-generated content at query #47
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

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #48
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("Account1")
        account2 = Account("Account2")
        return {
            account1: Balance(period.since, Quantity(Decimal("100.00"))),
            account2: Balance(period.since, Quantity(Decimal("200.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        account1 = Account("Account1")
        account2 = Account("Account2")
        journal_entry = JournalEntry(
            date=datetime.date(2023, 6, 15),
            description="Test transaction",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Posting.Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Posting.Direction.CREDIT)
            ]
        )
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2

    # Check Account1 ledger
    account1_ledger = general_ledger.ledgers[Account("Account1")]
    assert account1_ledger.account.name == "Account1"
    assert account1_ledger.initial.value == Quantity(Decimal("100.00"))
    assert len(account1_ledger.entries) == 1
    assert account1_ledger.entries[0].amount == Amount(Decimal("50.00"))
    assert account1_ledger.entries[0].balance == Quantity(Decimal("150.00"))
    assert account1_ledger.entries[0].is_debit is True

    # Check Account2 ledger
    account2_ledger = general_ledger.ledgers[Account("Account2")]
    assert account2_ledger.account.name == "Account2"
    assert account2_ledger.initial.value == Quantity(Decimal("200.00"))
    assert len(account2_ledger.entries) == 1
    assert account2_ledger.entries[0].amount == Amount(Decimal("50.00"))
    assert account2_ledger.entries[0].balance == Quantity(Decimal("150.00"))
    assert account2_ledger.entries[0].is_credit is True


# LLM-generated content at query #49
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Asset")
    account2 = Account("Liability")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT)
        ]
    )
    journal_entries = [journal_entry1]

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        return journal_entries

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
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("1200.00"))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.posting == journal_entry1.postings[1]
    assert entry2.balance == Quantity(Decimal("-700.00"))


# LLM-generated content at query #50
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Create test journal entries
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
    assert entry1.balance == Decimal("150.00")
    assert entry1.is_debit
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.is_credit
    assert entry2.credit == Amount(Decimal("30.00"))
    assert entry2.debit is None

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("200.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("150.00")
    assert entry1.is_credit
    assert entry1.credit == Amount(Decimal("50.00"))
    assert entry1.debit is None

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("180.00")
    assert entry2.is_debit
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #51
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account("TestAccount")
    test_initial_balance = Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    test_initial_balances = {test_account: test_initial_balance}

    # Mock the read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return test_initial_balances

    # Mock journal entries
    test_journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test transaction",
        postings=[
            Posting(
                account=test_account,
                amount=Amount(Decimal("500.00")),
                direction=Posting.Direction.DEBIT
            )
        ]
    )

    # Mock the read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        assert period == test_period
        return [test_journal_entry]

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    result = program(test_period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 1
    assert test_account in result.ledgers

    ledger = result.ledgers[test_account]
    assert ledger.account == test_account
    assert ledger.initial == test_initial_balance
    assert len(ledger.entries) == 1

    entry = ledger.entries[0]
    assert entry.posting == test_journal_entry.postings[0]
    assert entry.balance == Quantity(Decimal("1500.00"))
    assert entry.date == datetime.date(2023, 6, 15)
    assert entry.description == "Test transaction"
    assert entry.amount == Amount(Decimal("500.00"))
    assert entry.is_debit
    assert not entry.is_credit
    assert entry.debit == Amount(Decimal("500.00"))
    assert entry.credit is None


# LLM-generated content at query #52
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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
    assert account1_ledger.initial == initial_balances[account1]
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("120.00"))
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial == initial_balances[account2]
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("180.00"))
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #53
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation
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


# LLM-generated content at query #54
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Asset:Cash")
    account2 = Account("Income:Sales")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Initial deposit",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 1),
        description="Sale",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Verify the general ledger structure
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Verify account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("1000.00")
    assert len(account1_ledger.entries) == 2

    # Verify first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("1500.00")
    assert entry1.debit == Amount(Decimal("500.00"))
    assert entry1.credit is None

    # Verify second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("1700.00")
    assert entry2.debit == Amount(Decimal("200.00"))
    assert entry2.credit is None

    # Verify account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 2

    # Verify first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("500.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("500.00"))

    # Verify second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("700.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("200.00"))


# LLM-generated content at query #55
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    account1 = Account("Account1")
    account2 = Account("Account2")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
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
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("50.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("20.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #56
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Receivable")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test transaction 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("300.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("300.00")), direction=Direction.DEBIT)
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
    assert entry1.balance == Decimal("1200.00")
    assert entry1.debit == Amount(Decimal("200.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("900.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("300.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("500.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("300.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("200.00"))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("600.00")
    assert entry2.debit == Amount(Decimal("300.00"))
    assert entry2.credit is None


# LLM-generated content at query #57
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

    # Assert that all keys in the result are Account instances
    for key in result.keys():
        assert isinstance(key, Account)

    # Assert that all values in the result are Balance instances
    for value in result.values():
        assert isinstance(value, Balance)


# LLM-generated content at query #58
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(period.since, Quantity(Decimal(1000)))
    expected_initial_balances = {mock_account: mock_balance}

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert mock_account in result
    assert result[mock_account] == mock_balance


# LLM-generated content at query #59
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

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert len(result) == 2
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #60
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account("TestAccount")
    test_balance = Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    expected_initial_balances = {test_account: test_balance}

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return expected_initial_balances

    # Test the call
    result = mock_read_initial_balances(test_period)

    # Verify the result
    assert result == expected_initial_balances
    assert result[test_account] == test_balance


# LLM-generated content at query #61
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the ReadInitialBalances implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal("500.00")))
        }

    # Create an instance of ReadInitialBalances
    read_initial_balances = mock_read_initial_balances

    # Define a test period
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Call the method
    result = read_initial_balances(test_period)

    # Assert the result
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Cash") in result
    assert Account("Accounts Receivable") in result
    assert result[Account("Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Accounts Receivable")].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #62
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    # Mock implementation of ReadInitialBalances
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
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #63
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
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


# LLM-generated content at query #64
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

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #65
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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
        date=datetime.date(2023, 2, 20),
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

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("100.00")
    assert len(account1_ledger.entries) == 2

    # Check first entry of account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("200.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("180.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #66
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Asset", "Cash")
    account2 = Account("Liability", "Loans")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #67
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
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


# LLM-generated content at query #68
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
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
        date=datetime.date(2023, 1, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
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


# LLM-generated content at query #69
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
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

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        return [journal_entry1, journal_entry2]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("100.00"))
    assert len(ledger1.entries) == 2

    # Check first entry of account1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))

    # Check second entry of account1
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("120.00"))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("200.00"))
    assert len(ledger2.entries) == 2

    # Check first entry of account2
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("150.00"))

    # Check second entry of account2
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("180.00"))


# LLM-generated content at query #70
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00"))),
    }

    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT),
        ],
    )

    # Mock functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("100.00"))
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))

    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("200.00"))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))


# LLM-generated content at query #71
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Mock the initial balances
    mock_initial_balances = {
        Account("Account1"): Balance(period.since, Quantity(Decimal(1000))),
        Account("Account2"): Balance(period.since, Quantity(Decimal(2000)))
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return mock_initial_balances

    # Test the __call__ method
    read_initial_balances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Account1") in result
    assert Account("Account2") in result
    assert result[Account("Account1")].value == Quantity(Decimal(1000))
    assert result[Account("Account2")].value == Quantity(Decimal(2000))


# LLM-generated content at query #72
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(period.since, Quantity(Decimal(1000)))
    expected_initial_balances = {mock_account: mock_balance}

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Act
    result = mock_read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account] == mock_balance


# LLM-generated content at query #73
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
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2

    # Check first entry of ledger1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("125.00"))
    assert entry1.debit == Amount(Decimal("25.00"))
    assert entry1.credit is None

    # Check second entry of ledger1
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("95.00"))
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check ledger2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 2

    # Check first entry of ledger2
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("25.00"))
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("25.00"))

    # Check second entry of ledger2
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("55.00"))
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #74
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {
            Account("Cash"): Balance(test_period.since, Quantity(Decimal("1000.00"))),
            Account("Revenue"): Balance(test_period.since, Quantity(Decimal("0.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return [
            JournalEntry(
                date=datetime.date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        account=Account("Cash"),
                        amount=Amount(Decimal("500.00")),
                        direction=Posting.Direction.DEBIT
                    ),
                    Posting(
                        account=Account("Revenue"),
                        amount=Amount(Decimal("500.00")),
                        direction=Posting.Direction.CREDIT
                    )
                ]
            )
        ]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period

    # Check Cash ledger
    cash_ledger = result.ledgers[Account("Cash")]
    assert cash_ledger.account.name == "Cash"
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))

    # Check Revenue ledger
    revenue_ledger = result.ledgers[Account("Revenue")]
    assert revenue_ledger.account.name == "Revenue"
    assert revenue_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))


# LLM-generated content at query #75
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {
            Account("Assets:Cash"): Balance(test_period.since, Quantity(Decimal("1000.00"))),
            Account("Liabilities:Loans"): Balance(test_period.since, Quantity(Decimal("-500.00")))
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
                        account=Account("Assets:Cash"),
                        amount=Amount(Decimal("200.00")),
                        direction=Posting.Direction.DEBIT
                    ),
                    Posting(
                        account=Account("Income:Sales"),
                        amount=Amount(Decimal("200.00")),
                        direction=Posting.Direction.CREDIT
                    )
                ]
            )
        ]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period

    # Check ledgers
    assert len(result.ledgers) == 3  # 2 from initial + 1 from journal

    # Check Assets:Cash ledger
    cash_ledger = result.ledgers[Account("Assets:Cash")]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1200.00"))

    # Check Liabilities:Loans ledger
    loans_ledger = result.ledgers[Account("Liabilities:Loans")]
    assert loans_ledger.initial.value == Quantity(Decimal("-500.00"))
    assert len(loans_ledger.entries) == 0

    # Check Income:Sales ledger (created from journal)
    sales_ledger = result.ledgers[Account("Income:Sales")]
    assert sales_ledger.initial.value == Quantity(Decimal("0"))
    assert len(sales_ledger.entries) == 1
    assert sales_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert sales_ledger.entries[0].balance == Quantity(Decimal("200.00"))


# LLM-generated content at query #76
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

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #77
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock DateRange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Create a mock ReadInitialBalances instance
    read_initial_balances = ReadInitialBalances()

    # Call the __call__ method
    result = read_initial_balances(period)

    # Assert that the result is of type InitialBalances (Dict[Account, Balance])
    assert isinstance(result, dict)

    # Assert that all keys in the result are Account instances
    for account in result.keys():
        assert isinstance(account, Account)

    # Assert that all values in the result are Balance instances
    for balance in result.values():
        assert isinstance(balance, Balance)


# LLM-generated content at query #78
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Mock read_initial_balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Mock read_journal_entries
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        journal_entry = JournalEntry(
            date=datetime.date(2023, 6, 15),
            description="Test transaction",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
            ]
        )
        return [journal_entry]

    # Create GeneralLedgerProgram
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
    assert account1_ledger.initial.value == Quantity(Decimal("100.00"))
    assert len(account1_ledger.entries) == 1
    entry = account1_ledger.entries[0]
    assert entry.amount == Amount(Decimal("50.00"))
    assert entry.balance == Quantity(Decimal("150.00"))
    assert entry.is_debit is True

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.initial.value == Quantity(Decimal("200.00"))
    assert len(account2_ledger.entries) == 1
    entry = account2_ledger.entries[0]
    assert entry.amount == Amount(Decimal("50.00"))
    assert entry.balance == Quantity(Decimal("150.00"))
    assert entry.is_credit is True


# LLM-generated content at query #79
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Mock the read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Create mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
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

    # Mock the read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
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


# LLM-generated content at query #80
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("200.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Posting.Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Posting.Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Posting.Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Posting.Direction.DEBIT)
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

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal("100.00")
    assert len(ledger1.entries) == 2

    # Check first entry for account1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal("200.00")
    assert len(ledger2.entries) == 2

    # Check first entry for account2
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry for account2
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("180.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Asset:Cash")
    account2 = Account("Expense:Rent")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Rent payment",
        postings=[
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
        ]
    )

    journal = [journal_entry1]

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
    assert account1_ledger.initial == initial_balances[account1]
    assert len(account1_ledger.entries) == 1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("500.00"))
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("500.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial == initial_balances[account2]
    assert len(account2_ledger.entries) == 1
    entry2 = account2_ledger.entries[0]
    assert entry2.posting == journal_entry1.postings[0]
    assert entry2.balance == Quantity(Decimal("500.00"))
    assert entry2.debit == Amount(Decimal("500.00"))
    assert entry2.credit is None


# LLM-generated content at query #2
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period and initial balances
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_initial_balances = {
        Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Create a mock function that implements ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the mock function
    result = mock_read_initial_balances(period)

    # Assert the result matches the expected initial balances
    assert result == expected_initial_balances


# LLM-generated content at query #3
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Create test accounts
    account1 = Account("Account1")
    account2 = Account("Account2")

    # Create initial balances
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("50.00")))
    }

    # Create test journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("25.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("25.00")), direction=Direction.CREDIT)
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


# LLM-generated content at query #4
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Mock initial balances
    account1 = Account("Account1")
    account2 = Account("Account2")
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
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        return [journal_entry1, journal_entry2]

    # Create program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute
    result = program(period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period

    # Check ledgers
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers

    # Check account1 ledger
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal(100)
    assert len(ledger1.entries) == 2

    # First entry
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)  # 100 + 50

    # Second entry
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)  # 150 - 30

    # Check account2 ledger
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal(200)
    assert len(ledger2.entries) == 2

    # First entry
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(150)  # 200 - 50

    # Second entry
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(180)  # 150 + 30


# LLM-generated content at query #5
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period and expected initial balances
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_initial_balances = {
        Account("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal("500.00"))),
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the mock implementation
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert the result matches the expected initial balances
    assert result == expected_initial_balances


# LLM-generated content at query #6
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock period
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Create a mock ReadInitialBalances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal("500.00")))
        }

    # Test the function
    initial_balances = mock_read_initial_balances(period)

    # Assertions
    assert isinstance(initial_balances, dict)
    assert len(initial_balances) == 2
    assert Account("Cash") in initial_balances
    assert Account("Accounts Receivable") in initial_balances
    assert initial_balances[Account("Cash")].value == Quantity(Decimal("1000.00"))
    assert initial_balances[Account("Accounts Receivable")].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #7
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    test_account = Account("TestAccount")
    test_initial_balance = Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000)))

    # Mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {test_account: test_initial_balance}

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        journal_entry = JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test Entry",
            postings=[
                Posting(
                    account=test_account,
                    amount=Amount(Decimal(500)),
                    direction=Posting.Direction.DEBIT,
                ),
                Posting(
                    account=Account("AnotherAccount"),
                    amount=Amount(Decimal(500)),
                    direction=Posting.Direction.CREDIT,
                ),
            ],
        )
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(test_period)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == test_period
    assert len(general_ledger.ledgers) == 2  # TestAccount and AnotherAccount

    # Check TestAccount ledger
    test_ledger = general_ledger.ledgers[test_account]
    assert test_ledger.account == test_account
    assert test_ledger.initial == test_initial_balance
    assert len(test_ledger.entries) == 1

    entry = test_ledger.entries[0]
    assert entry.ledger == test_ledger
    assert entry.posting.account == test_account
    assert entry.amount == Amount(Decimal(500))
    assert entry.balance == Quantity(Decimal(1500))  # 1000 + 500
    assert entry.is_debit
    assert not entry.is_credit
    assert entry.debit == Amount(Decimal(500))
    assert entry.credit is None
    assert entry.date == datetime.date(2023, 1, 15)
    assert entry.description == "Test Entry"
    assert len(entry.cntraccts) == 1
    assert entry.cntraccts[0].name == "AnotherAccount"


# LLM-generated content at query #8
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {
            Account("Cash"): Balance(test_period.since, Quantity(Decimal("1000.00"))),
            Account("Revenue"): Balance(test_period.since, Quantity(Decimal("0.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return [
            JournalEntry(
                date=datetime.date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        account=Account("Cash"),
                        amount=Amount(Decimal("500.00")),
                        direction=Direction.DEBIT,
                        journal=None  # type: ignore
                    ),
                    Posting(
                        account=Account("Revenue"),
                        amount=Amount(Decimal("500.00")),
                        direction=Direction.CREDIT,
                        journal=None  # type: ignore
                    )
                ]
            )
        ]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    result = program(test_period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 2

    # Check Cash ledger
    cash_ledger = result.ledgers[Account("Cash")]
    assert cash_ledger.account.name == "Cash"
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[0].is_debit

    # Check Revenue ledger
    revenue_ledger = result.ledgers[Account("Revenue")]
    assert revenue_ledger.account.name == "Revenue"
    assert revenue_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    assert revenue_ledger.entries[0].is_credit


# LLM-generated content at query #9
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Asset", "Cash")
    account2 = Account("Liability", "Loans")
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


# LLM-generated content at query #10
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
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
    assert result == expected_initial_balances
    assert isinstance(result[account1], Balance)
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Asset:Cash")
    account2 = Account("Income:Sales")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Initial sale",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Another sale",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("300.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("300.00")), direction=Direction.CREDIT)
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.initial.value == Decimal("1000.00")
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].balance == Decimal("1500.00")
    assert account1_ledger.entries[1].balance == Decimal("1800.00")

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].balance == Decimal("500.00")
    assert account2_ledger.entries[1].balance == Decimal("800.00")

    # Check ledger entries properties
    entry = account1_ledger.entries[0]
    assert entry.date == datetime.date(2023, 1, 15)
    assert entry.description == "Initial sale"
    assert entry.amount == Amount(Decimal("500.00"))
    assert entry.is_debit
    assert not entry.is_credit
    assert entry.debit == Amount(Decimal("500.00"))
    assert entry.credit is None
    assert entry.cntraccts == [account2]


# LLM-generated content at query #12
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("200.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("180.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Mock the initial balances
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the __call__ method
    read_initial_balances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert the result matches the expected initial balances
    assert result == expected_initial_balances


# LLM-generated content at query #14
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
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
        date=datetime.date(2023, 1, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return [journal_entry1, journal_entry2]

    # Create GeneralLedgerProgram
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


# LLM-generated content at query #15
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Mock the read_initial_balances function
    def mock_read_initial_balances(period):
        return initial_balances

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
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    # Mock the read_journal_entries function
    def mock_read_journal_entries(period):
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


# LLM-generated content at query #16
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("50.00")))
    }

    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("25.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("25.00")), direction=Direction.CREDIT)
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

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2

    # Check first entry for account1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("125.00"))

    # Check second entry for account1
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("95.00"))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 2

    # Check first entry for account2
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("25.00"))

    # Check second entry for account2
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("55.00"))


# LLM-generated content at query #17
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    mock_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    mock_account1 = Account("Asset", "Cash")
    mock_account2 = Account("Equity", "Capital")

    mock_initial_balances = {
        mock_account1: Balance(mock_period.since, Quantity(Decimal("1000.00"))),
        mock_account2: Balance(mock_period.since, Quantity(Decimal("0.00")))
    }

    mock_journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(mock_account1, Amount(Decimal("500.00")), Direction.DEBIT),
            Posting(mock_account2, Amount(Decimal("500.00")), Direction.CREDIT)
        ]
    )

    # Mock functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return mock_initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return [mock_journal_entry]

    # Create the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )

    # Execute the program
    result = program(mock_period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == mock_period
    assert len(result.ledgers) == 2
    assert mock_account1 in result.ledgers
    assert mock_account2 in result.ledgers

    # Check ledger entries
    cash_ledger = result.ledgers[mock_account1]
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))

    capital_ledger = result.ledgers[mock_account2]
    assert len(capital_ledger.entries) == 1
    assert capital_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert capital_ledger.entries[0].balance == Quantity(Decimal("500.00"))


# LLM-generated content at query #18
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
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")
    assert result[account1].date == period.since
    assert result[account2].date == period.since


# LLM-generated content at query #19
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock DateRange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Create mock accounts and balances
    account1 = Account("Account1")
    account2 = Account("Account2")
    balance1 = Balance(period.since, Quantity(Decimal("1000.00")))
    balance2 = Balance(period.since, Quantity(Decimal("2000.00")))

    # Create expected initial balances
    expected_initial_balances = {
        account1: balance1,
        account2: balance2
    }

    # Create a mock ReadInitialBalances instance
    read_initial_balances = lambda p: expected_initial_balances

    # Call the method
    result = read_initial_balances(period)

    # Assert the result matches the expected initial balances
    assert result == expected_initial_balances


# LLM-generated content at query #20
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=PostingDirection.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=PostingDirection.CREDIT)
        ]
    )
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=PostingDirection.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=PostingDirection.DEBIT)
        ]
    )
    journal_entries = [journal_entry1, journal_entry2]

    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        return journal_entries

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


# LLM-generated content at query #21
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Create test journal entries
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
    assert account1_ledger.initial.value == Decimal(100)
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)  # 100 + 50
    assert entry1.debit == Amount(Decimal(50))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)  # 150 - 30
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal(30))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal(200)
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(150)  # 200 - 50
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal(50))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(180)  # 150 + 30
    assert entry2.debit == Amount(Decimal(30))
    assert entry2.credit is None


# LLM-generated content at query #22
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal("500.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        journal_entry = JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test transaction",
            postings=[
                Posting(
                    account=Account("Cash"),
                    amount=Amount(Decimal("200.00")),
                    direction=Posting.Direction.DEBIT
                ),
                Posting(
                    account=Account("Accounts Receivable"),
                    amount=Amount(Decimal("200.00")),
                    direction=Posting.Direction.CREDIT
                )
            ]
        )
        return [journal_entry]

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2

    # Check Cash ledger
    cash_ledger = general_ledger.ledgers[Account("Cash")]
    assert cash_ledger.account.name == "Cash"
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1200.00"))
    assert cash_ledger.entries[0].is_debit is True

    # Check Accounts Receivable ledger
    ar_ledger = general_ledger.ledgers[Account("Accounts Receivable")]
    assert ar_ledger.account.name == "Accounts Receivable"
    assert ar_ledger.initial.value == Quantity(Decimal("500.00"))
    assert len(ar_ledger.entries) == 1
    assert ar_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert ar_ledger.entries[0].balance == Quantity(Decimal("300.00"))
    assert ar_ledger.entries[0].is_credit is True


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_initial_balances = {
        Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("-500.00")))
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances


# LLM-generated content at query #24
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Assets:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Liabilities:Loans"): Balance(period.since, Quantity(Decimal("-500.00")))
        }

    # Create an instance of the protocol
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Define test period
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Call the method
    result = read_initial_balances(test_period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Assets:Cash") in result
    assert Account("Liabilities:Loans") in result
    assert result[Account("Assets:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liabilities:Loans")].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #25
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Create journal entries
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
    assert account1_ledger.initial.value == Decimal(100)
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)  # 100 + 50

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)  # 150 - 30

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal(200)
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(150)  # 200 - 50

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(180)  # 150 + 30


# LLM-generated content at query #26
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00")))
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
    assert entry1.balance == Decimal("150.00")
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry of account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("120.00")
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 2

    # Check first entry of account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("50.00")
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry of account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("20.00")
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #27
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))

    # Mock initial balances
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

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
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
    assert account1_ledger.entries[0].is_debit
    assert account1_ledger.entries[1].amount == Amount(Decimal("30.00"))
    assert account1_ledger.entries[1].is_credit

    # Check ledger entries for account2
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].amount == Amount(Decimal("50.00"))
    assert account2_ledger.entries[0].is_credit
    assert account2_ledger.entries[1].amount == Amount(Decimal("30.00"))
    assert account2_ledger.entries[1].is_debit


# LLM-generated content at query #28
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("50.00")))
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
            Posting(account=account1, amount=Amount(Decimal("15.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("15.00")), direction=Direction.DEBIT)
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

    # Check account1 entries
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal("125.00")

    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal("110.00")

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("50.00")
    assert len(account2_ledger.entries) == 2

    # Check account2 entries
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal("25.00")

    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal("40.00")


# LLM-generated content at query #29
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00"))),
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Test the call
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("2000.00"))


# LLM-generated content at query #30
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        journal_entry1 = JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test Entry 1",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT)
            ]
        )
        journal_entry2 = JournalEntry(
            date=datetime.date(2023, 2, 20),
            description="Test Entry 2",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("300.00")), direction=Direction.CREDIT),
                Posting(account=account2, amount=Amount(Decimal("300.00")), direction=Direction.DEBIT)
            ]
        )
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
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].amount == Amount(Decimal("500.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("1500.00"))
    assert ledger1.entries[1].amount == Amount(Decimal("300.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("1200.00"))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal("500.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("1500.00"))
    assert ledger2.entries[1].amount == Amount(Decimal("300.00"))
    assert ledger2.entries[1].balance == Quantity(Decimal("1800.00"))


# LLM-generated content at query #31
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    # Mock implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert len(result) == 2
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #32
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account1 = Account("TestAccount1")
    test_account2 = Account("TestAccount2")
    test_initial_balances = {
        test_account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        test_account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("50.00")))
    }

    # Mock journal entries
    test_journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=test_account1, amount=Amount(Decimal("25.00")), direction=Direction.DEBIT),
            Posting(account=test_account2, amount=Amount(Decimal("25.00")), direction=Direction.CREDIT)
        ]
    )

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return test_initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return [test_journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert result.period == test_period
    assert len(result.ledgers) == 2
    assert test_account1 in result.ledgers
    assert test_account2 in result.ledgers

    # Verify ledger entries for test_account1
    account1_ledger = result.ledgers[test_account1]
    assert account1_ledger.account == test_account1
    assert account1_ledger.initial == test_initial_balances[test_account1]
    assert len(account1_ledger.entries) == 1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == test_journal_entry.postings[0]
    assert entry1.balance == Quantity(Decimal("125.00"))

    # Verify ledger entries for test_account2
    account2_ledger = result.ledgers[test_account2]
    assert account2_ledger.account == test_account2
    assert account2_ledger.initial == test_initial_balances[test_account2]
    assert len(account2_ledger.entries) == 1
    entry2 = account2_ledger.entries[0]
    assert entry2.posting == test_journal_entry.postings[1]
    assert entry2.balance == Quantity(Decimal("25.00"))


# LLM-generated content at query #33
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(0)))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 15),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
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
    assert account1_ledger.initial.value == Decimal(100)
    assert len(account1_ledger.entries) == 2

    # Check first entry for account1
    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)  # 100 + 50
    assert entry1.debit == Amount(Decimal(50))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)  # 150 - 30
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal(30))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal(0)
    assert len(account2_ledger.entries) == 2

    # Check first entry for account2
    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(50)  # 0 + 50
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal(50))

    # Check second entry for account2
    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(20)  # 50 - 30
    assert entry2.debit == Amount(Decimal(30))
    assert entry2.credit is None


# LLM-generated content at query #34
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Cash"): Balance(test_period.since, Quantity(Decimal("1000.00"))),
            Account("Accounts Receivable"): Balance(test_period.since, Quantity(Decimal("500.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        journal_entry = JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test transaction",
            postings=[
                Posting(
                    account=Account("Cash"),
                    amount=Amount(Decimal("200.00")),
                    direction=Posting.Direction.DEBIT
                ),
                Posting(
                    account=Account("Accounts Receivable"),
                    amount=Amount(Decimal("200.00")),
                    direction=Posting.Direction.CREDIT
                )
            ]
        )
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    general_ledger = program(test_period)

    # Assertions
    assert general_ledger.period == test_period
    assert len(general_ledger.ledgers) == 2

    # Check Cash ledger
    cash_ledger = general_ledger.ledgers[Account("Cash")]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1200.00"))
    assert cash_ledger.entries[0].is_debit is True

    # Check Accounts Receivable ledger
    ar_ledger = general_ledger.ledgers[Account("Accounts Receivable")]
    assert ar_ledger.initial.value == Quantity(Decimal("500.00"))
    assert len(ar_ledger.entries) == 1
    assert ar_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert ar_ledger.entries[0].balance == Quantity(Decimal("300.00"))
    assert ar_ledger.entries[0].is_credit is True


# LLM-generated content at query #35
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
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
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 15),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    journal_entries = [journal_entry1, journal_entry2]

    # Mock functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return journal_entries

    # Create the program
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
    assert ledger1.initial.value == Decimal("100.00")
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger1.entries[1].amount == Amount(Decimal("30.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("120.00"))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal("200.00")
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger2.entries[1].amount == Amount(Decimal("30.00"))
    assert ledger2.entries[1].balance == Quantity(Decimal("180.00"))


# LLM-generated content at query #36
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
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
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger1 entries
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].posting == journal_entry1.postings[0]
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger1.entries[1].posting == journal_entry2.postings[0]
    assert ledger1.entries[1].balance == Quantity(Decimal("120.00"))

    # Check ledger2 entries
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].posting == journal_entry1.postings[1]
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger2.entries[1].posting == journal_entry2.postings[1]
    assert ledger2.entries[1].balance == Quantity(Decimal("180.00"))


# LLM-generated content at query #37
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Liability:Loans"): Balance(period.since, Quantity(Decimal("500.00")))
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
    assert Account("Liability:Loans") in result
    assert result[Account("Asset:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liability:Loans")].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #38
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Mock initial balances
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
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT)
        ]
    )

    # Mock functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return [journal_entry1, journal_entry2]

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

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal("100.00")
    assert len(ledger1.entries) == 2

    # Check first entry in ledger1
    entry1_1 = ledger1.entries[0]
    assert entry1_1.posting == journal_entry1.postings[0]
    assert entry1_1.balance == Decimal("150.00")
    assert entry1_1.debit == Amount(Decimal("50.00"))
    assert entry1_1.credit is None

    # Check second entry in ledger1
    entry1_2 = ledger1.entries[1]
    assert entry1_2.posting == journal_entry2.postings[0]
    assert entry1_2.balance == Decimal("120.00")
    assert entry1_2.debit is None
    assert entry1_2.credit == Amount(Decimal("30.00"))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal("200.00")
    assert len(ledger2.entries) == 2

    # Check first entry in ledger2
    entry2_1 = ledger2.entries[0]
    assert entry2_1.posting == journal_entry1.postings[1]
    assert entry2_1.balance == Decimal("150.00")
    assert entry2_1.debit is None
    assert entry2_1.credit == Amount(Decimal("50.00"))

    # Check second entry in ledger2
    entry2_2 = ledger2.entries[1]
    assert entry2_2.posting == journal_entry2.postings[1]
    assert entry2_2.balance == Decimal("180.00")
    assert entry2_2.debit == Amount(Decimal("30.00"))
    assert entry2_2.credit is None


# LLM-generated content at query #39
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Cash")
    account2 = Account("Revenue")
    account3 = Account("Expense")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("0.00"))),
    }

    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test transaction 1",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT),
            ],
        ),
        JournalEntry(
            date=datetime.date(2023, 2, 20),
            description="Test transaction 2",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT),
                Posting(account=account3, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT),
            ],
        ),
    ]

    # Call the function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("1000.00")
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].balance == Decimal("1500.00")
    assert account1_ledger.entries[1].balance == Decimal("1300.00")

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("0.00")
    assert len(account2_ledger.entries) == 1
    assert account2_ledger.entries[0].balance == Decimal("500.00")

    # Check account3 ledger
    account3_ledger = general_ledger.ledgers[account3]
    assert account3_ledger.account == account3
    assert account3_ledger.initial.value == Decimal("0.00")
    assert len(account3_ledger.entries) == 1
    assert account3_ledger.entries[0].balance == Decimal("200.00")

    # Check ledger entries properties
    entry = account1_ledger.entries[0]
    assert entry.date == datetime.date(2023, 1, 15)
    assert entry.description == "Test transaction 1"
    assert entry.amount == Amount(Decimal("500.00"))
    assert entry.is_debit
    assert not entry.is_credit
    assert entry.debit == Amount(Decimal("500.00"))
    assert entry.credit is None


# LLM-generated content at query #40
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=PostingDirection.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=PostingDirection.CREDIT)
        ]
    )
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=PostingDirection.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=PostingDirection.DEBIT)
        ]
    )
    journal_entries = [journal_entry1, journal_entry2]

    # Mock read functions
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return journal_entries

    # Create program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal(100)
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].amount == Amount(Decimal(50))
    assert ledger1.entries[0].balance == Quantity(Decimal(150))
    assert ledger1.entries[1].amount == Amount(Decimal(30))
    assert ledger1.entries[1].balance == Quantity(Decimal(120))

    # Check ledger2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal(200)
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal(50))
    assert ledger2.entries[0].balance == Quantity(Decimal(150))
    assert ledger2.entries[1].amount == Amount(Decimal(30))
    assert ledger2.entries[1].balance == Quantity(Decimal(180))


# LLM-generated content at query #41
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00"))),
    }
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test Entry 1",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
                Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT),
            ],
        ),
        JournalEntry(
            date=datetime.date(2023, 2, 20),
            description="Test Entry 2",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
                Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT),
            ],
        ),
    ]

    # Create mock functions for dependencies
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
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


# LLM-generated content at query #42
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Mock journal entries
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT)
        ]
    )

    # Mock read functions
    def mock_read_initial_balances(period):
        return initial_balances

    def mock_read_journal_entries(period):
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger entries
    ledger1 = general_ledger.ledgers[account1]
    ledger2 = general_ledger.ledgers[account2]

    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal(100)
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].amount == Amount(Decimal(50))
    assert ledger1.entries[0].balance == Quantity(Decimal(150))

    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal(200)
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].amount == Amount(Decimal(50))
    assert ledger2.entries[0].balance == Quantity(Decimal(150))


# LLM-generated content at query #43
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period and expected initial balances
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    expected_initial_balances = {
        Account("Cash"): Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        Account("Accounts Receivable"): Balance(datetime.date(2022, 12, 31), Quantity(Decimal("500.00")))
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the __call__ method
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert the result matches the expected initial balances
    assert result == expected_initial_balances


# LLM-generated content at query #44
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    mock_account1 = Account("Account1")
    mock_account2 = Account("Account2")
    mock_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            mock_account1: Balance(mock_period.since, Quantity(Decimal(100))),
            mock_account2: Balance(mock_period.since, Quantity(Decimal(200))),
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        posting1 = Posting(mock_account1, Amount(Decimal(50)), Direction.DEBIT)
        posting2 = Posting(mock_account2, Amount(Decimal(50)), Direction.CREDIT)
        journal_entry = JournalEntry(
            date=datetime.date(2023, 6, 15),
            description="Test transaction",
            postings=[posting1, posting2]
        )
        return [journal_entry]

    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(mock_period)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == mock_period
    assert len(general_ledger.ledgers) == 2
    assert mock_account1 in general_ledger.ledgers
    assert mock_account2 in general_ledger.ledgers

    # Check ledger entries for account1
    ledger1 = general_ledger.ledgers[mock_account1]
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.amount == Amount(Decimal(50))
    assert entry1.balance == Quantity(Decimal(150))
    assert entry1.is_debit

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[mock_account2]
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.amount == Amount(Decimal(50))
    assert entry2.balance == Quantity(Decimal(150))
    assert entry2.is_credit


# LLM-generated content at query #45
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account1 = Account("Account1")
    mock_account2 = Account("Account2")
    expected_balances = {
        mock_account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        mock_account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert mock_account1 in result
    assert mock_account2 in result
    assert result[mock_account1].value == Decimal("1000.00")
    assert result[mock_account2].value == Decimal("2000.00")


# LLM-generated content at query #46
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Create test journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
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
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].balance == Quantity(Decimal("150.00"))
    assert account1_ledger.entries[1].balance == Quantity(Decimal("120.00"))

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].balance == Quantity(Decimal("150.00"))
    assert account2_ledger.entries[1].balance == Quantity(Decimal("180.00"))

    # Test with empty journal
    empty_ledger = build_general_ledger(period, [], initial_balances)
    assert len(empty_ledger.ledgers[account1].entries) == 0
    assert len(empty_ledger.ledgers[account2].entries) == 0

    # Test with new account in journal
    account3 = Account("Account3")
    journal_entry3 = JournalEntry(
        date=datetime.date(2023, 3, 10),
        description="Test Entry 3",
        postings=[
            Posting(account=account3, amount=Amount(Decimal("100.00")), direction=Direction.DEBIT),
            Posting(account=account1, amount=Amount(Decimal("100.00")), direction=Direction.CREDIT)
        ]
    )
    extended_ledger = build_general_ledger(period, [journal_entry3], initial_balances)
    assert account3 in extended_ledger.ledgers
    assert len(extended_ledger.ledgers[account3].entries) == 1
    assert extended_ledger.ledgers[account3].entries[0].balance == Quantity(Decimal("100.00"))


# LLM-generated content at query #47
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Assets", "Cash")
    account2 = Account("Liabilities", "Loans")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    result = mock_read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("-500.00")


# LLM-generated content at query #48
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(period.since, Quantity(Decimal(1000)))
    expected_initial_balances = {mock_account: mock_balance}

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account] == mock_balance


# LLM-generated content at query #49
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

    # Test
    result = mock_read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #50
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    account1 = Account("Account1")
    account2 = Account("Account2")
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balance1 = Balance(period.since, Quantity(Decimal(100)))
    initial_balance2 = Balance(period.since, Quantity(Decimal(200)))
    initial_balances = {account1: initial_balance1, account2: initial_balance2}

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT),
        ]
    )
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT),
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
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balance1
    assert len(ledger1.entries) == 2

    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal(150))

    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal(120))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balance2
    assert len(ledger2.entries) == 2

    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal(150))

    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal(180))


# LLM-generated content at query #51
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    expected_initial_balances = {
        Account("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
        Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal("500.00")))
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances


# LLM-generated content at query #52
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the ReadInitialBalances implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("-500.00")))
        }

    # Create an instance of the protocol
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Define a test period
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Call the method
    result = read_initial_balances(test_period)

    # Assert the result
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Asset:Cash") in result
    assert Account("Liability:Loan") in result
    assert result[Account("Asset:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liability:Loan")].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #53
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(0))),
    }

    # Mock journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT),
        ],
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT),
        ],
    )

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
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
    assert ledger1.entries[0].amount == Amount(Decimal(50))
    assert ledger1.entries[0].balance == Quantity(Decimal(150))
    assert ledger1.entries[1].amount == Amount(Decimal(30))
    assert ledger1.entries[1].balance == Quantity(Decimal(120))

    # Check ledger entries for account2
    ledger2 = general_ledger.ledgers[account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal(50))
    assert ledger2.entries[0].balance == Quantity(Decimal(50))
    assert ledger2.entries[1].amount == Amount(Decimal(30))
    assert ledger2.entries[1].balance == Quantity(Decimal(20))


# LLM-generated content at query #54
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Asset:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
            Account("Liability:Loan"): Balance(period.since, Quantity(Decimal("-500.00")))
        }

    # Create an instance of the mock
    read_initial_balances = mock_read_initial_balances

    # Define test period
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Call the method
    result = read_initial_balances(test_period)

    # Assert the result is as expected
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Asset:Cash") in result
    assert Account("Liability:Loan") in result
    assert result[Account("Asset:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liability:Loan")].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #55
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets", "Cash")
    account2 = Account("Liabilities", "Accounts Payable")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #56
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00")))
    }

    # Create journal entries
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

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

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


# LLM-generated content at query #57
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Arrange
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    read_initial_balances: ReadInitialBalances = mock_read_initial_balances

    # Act
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert len(result) == 2
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #58
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    account3 = Account("Account3")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("100.00"))),
        account2: Balance(period.since, Quantity(Decimal("200.00"))),
    }

    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT),
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT),
            Posting(account=account3, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
        ]
    )

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.account == account1
    assert account1_ledger.initial.value == Decimal("100.00")
    assert len(account1_ledger.entries) == 1
    assert account1_ledger.entries[0].posting == journal_entry1.postings[0]
    assert account1_ledger.entries[0].balance == Decimal("150.00")

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal("200.00")
    assert len(account2_ledger.entries) == 2
    assert account2_ledger.entries[0].posting == journal_entry1.postings[1]
    assert account2_ledger.entries[0].balance == Decimal("150.00")
    assert account2_ledger.entries[1].posting == journal_entry2.postings[0]
    assert account2_ledger.entries[1].balance == Decimal("180.00")

    # Check account3 ledger
    account3_ledger = general_ledger.ledgers[account3]
    assert account3_ledger.account == account3
    assert account3_ledger.initial.value == Decimal("0.00")
    assert len(account3_ledger.entries) == 1
    assert account3_ledger.entries[0].posting == journal_entry2.postings[1]
    assert account3_ledger.entries[0].balance == Decimal("30.00")


# LLM-generated content at query #59
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    # Create journal entries
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

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
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


# LLM-generated content at query #60
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account1 = Account("TestAccount1")
    test_account2 = Account("TestAccount2")
    test_initial_balances = {
        test_account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        test_account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("200.00")))
    }

    # Create mock journal entries
    test_journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test transaction",
        postings=[
            Posting(account=test_account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=test_account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return test_initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        return [test_journal_entry]

    # Compile the program
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


# LLM-generated content at query #61
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {
            Account("Cash"): Balance(test_period.since, Quantity(Decimal("1000.00"))),
            Account("Revenue"): Balance(test_period.since, Quantity(Decimal("0.00")))
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
                        amount=Amount(Decimal("500.00")),
                        direction=Posting.Direction.DEBIT
                    ),
                    Posting(
                        account=Account("Revenue"),
                        amount=Amount(Decimal("500.00")),
                        direction=Posting.Direction.CREDIT
                    )
                ]
            )
        ]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period

    # Check Cash ledger
    cash_ledger = result.ledgers[Account("Cash")]
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[0].is_debit

    # Check Revenue ledger
    revenue_ledger = result.ledgers[Account("Revenue")]
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    assert revenue_ledger.entries[0].is_credit


# LLM-generated content at query #62
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

    # Mock function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    read_initial_balances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert isinstance(result, dict)
    assert all(isinstance(key, Account) for key in result.keys())
    assert all(isinstance(value, Balance) for value in result.values())


# LLM-generated content at query #63
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock the period
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Mock the initial balances
    account1 = Account("Asset", "Cash")
    account2 = Account("Liability", "Accounts Payable")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Test the call
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #64
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account1 = Account("TestAccount1")
    test_account2 = Account("TestAccount2")
    test_balance1 = Balance(test_period.since, Quantity(Decimal("100.00")))
    test_balance2 = Balance(test_period.since, Quantity(Decimal("200.00")))
    test_initial_balances = {test_account1: test_balance1, test_account2: test_balance2}

    # Mock the read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return test_initial_balances

    # Create test journal entries
    test_journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry 1",
        postings=[
            Posting(account=test_account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=test_account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT),
        ]
    )

    test_journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test Entry 2",
        postings=[
            Posting(account=test_account1, amount=Amount(Decimal("30.00")), direction=Direction.CREDIT),
            Posting(account=test_account2, amount=Amount(Decimal("30.00")), direction=Direction.DEBIT),
        ]
    )

    test_journal_entries = [test_journal_entry1, test_journal_entry2]

    # Mock the read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return test_journal_entries

    # Create the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 2
    assert test_account1 in result.ledgers
    assert test_account2 in result.ledgers

    # Verify ledger entries for test_account1
    ledger1 = result.ledgers[test_account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger1.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger1.entries[1].amount == Amount(Decimal("30.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("120.00"))

    # Verify ledger entries for test_account2
    ledger2 = result.ledgers[test_account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].amount == Amount(Decimal("50.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger2.entries[1].amount == Amount(Decimal("30.00"))
    assert ledger2.entries[1].balance == Quantity(Decimal("180.00"))


# LLM-generated content at query #65
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the __call__ method
    read_initial_balances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert the result matches expected initial balances
    assert result == expected_initial_balances
    assert len(result) == 2
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #66
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    account1 = Account("Account1")
    account2 = Account("Account2")

    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(200)))
    }

    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=Direction.DEBIT),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=Direction.CREDIT)
        ]
    )

    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test entry 2",
        postings=[
            Posting(account=account1, amount=Amount(Decimal(30)), direction=Direction.CREDIT),
            Posting(account=account2, amount=Amount(Decimal(30)), direction=Direction.DEBIT)
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
    assert account1_ledger.initial.value == Decimal(100)
    assert len(account1_ledger.entries) == 2

    entry1 = account1_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)  # 100 + 50 (debit)

    entry2 = account1_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)  # 150 - 30 (credit)

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.account == account2
    assert account2_ledger.initial.value == Decimal(200)
    assert len(account2_ledger.entries) == 2

    entry1 = account2_ledger.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(150)  # 200 - 50 (credit)

    entry2 = account2_ledger.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(180)  # 150 + 30 (debit)


# LLM-generated content at query #67
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    expected_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("2000.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_balances

    # Test
    read_initial_balances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #68
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    test_account1 = Account("TestAccount1")
    test_account2 = Account("TestAccount2")
    test_initial_balances = {
        test_account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00"))),
        test_account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("200.00")))
    }

    # Create mock journal entries
    test_journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(account=test_account1, amount=Amount(Decimal("50.00")), direction=Direction.DEBIT),
            Posting(account=test_account2, amount=Amount(Decimal("50.00")), direction=Direction.CREDIT)
        ]
    )

    # Mock read_initial_balances function
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return test_initial_balances

    # Mock read_journal_entries function
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        assert period == test_period
        return [test_journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Verify the result
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 2
    assert test_account1 in result.ledgers
    assert test_account2 in result.ledgers

    # Verify ledger1
    ledger1 = result.ledgers[test_account1]
    assert ledger1.account == test_account1
    assert ledger1.initial == test_initial_balances[test_account1]
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.posting == test_journal_entry.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Verify ledger2
    ledger2 = result.ledgers[test_account2]
    assert ledger2.account == test_account2
    assert ledger2.initial == test_initial_balances[test_account2]
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.posting == test_journal_entry.postings[1]
    assert entry2.balance == Quantity(Decimal("150.00"))
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("50.00"))


# LLM-generated content at query #69
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 31)
    )

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Cash"): Balance(test_period.since, Quantity(Decimal("1000.00"))),
            Account("Accounts Receivable"): Balance(test_period.since, Quantity(Decimal("500.00")))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry[_T]]:
        journal_entry = JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test transaction",
            postings=[
                Posting(
                    account=Account("Cash"),
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.DEBIT
                ),
                Posting(
                    account=Account("Accounts Receivable"),
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.CREDIT
                )
            ]
        )
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program
    general_ledger = program(test_period)

    # Assertions
    assert general_ledger.period == test_period
    assert len(general_ledger.ledgers) == 2

    # Check Cash ledger
    cash_ledger = general_ledger.ledgers[Account("Cash")]
    assert cash_ledger.account.name == "Cash"
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert cash_ledger.entries[0].balance == Decimal("1200.00")
    assert cash_ledger.entries[0].is_debit is True

    # Check Accounts Receivable ledger
    ar_ledger = general_ledger.ledgers[Account("Accounts Receivable")]
    assert ar_ledger.account.name == "Accounts Receivable"
    assert ar_ledger.initial.value == Decimal("500.00")
    assert len(ar_ledger.entries) == 1
    assert ar_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert ar_ledger.entries[0].balance == Decimal("300.00")
    assert ar_ledger.entries[0].is_credit is True


# LLM-generated content at query #70
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Liabilities:Loans")
    expected_initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("-500.00")))
    }

    # Mock implementation
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


# LLM-generated content at query #71
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

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
                    Posting(Account("Cash"), Amount(Decimal("200.00")), Direction.DEBIT),
                    Posting(Account("Revenue"), Amount(Decimal("200.00")), Direction.CREDIT)
                ]
            )
        ]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    result = program(test_period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 3  # Cash, Accounts Receivable, and Revenue

    # Check Cash ledger
    cash_ledger = result.ledgers[Account("Cash")]
    assert cash_ledger.account.name == "Cash"
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1200.00"))

    # Check Accounts Receivable ledger (no transactions)
    ar_ledger = result.ledgers[Account("Accounts Receivable")]
    assert ar_ledger.account.name == "Accounts Receivable"
    assert ar_ledger.initial.value == Quantity(Decimal("500.00"))
    assert len(ar_ledger.entries) == 0

    # Check Revenue ledger (created automatically)
    revenue_ledger = result.ledgers[Account("Revenue")]
    assert revenue_ledger.account.name == "Revenue"
    assert revenue_ledger.initial.value == Quantity(Decimal("0"))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].amount == Amount(Decimal("200.00"))
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("200.00"))


# LLM-generated content at query #72
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Create a mock DateRange
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))

    # Create mock accounts and balances
    account1 = Account("Account1")
    account2 = Account("Account2")
    balance1 = Balance(period.since, Quantity(Decimal("1000.00")))
    balance2 = Balance(period.since, Quantity(Decimal("2000.00")))

    # Create expected initial balances
    expected_initial_balances = {
        account1: balance1,
        account2: balance2
    }

    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test the __call__ method
    result = mock_read_initial_balances(period)

    # Assert the result matches the expected initial balances
    assert result == expected_initial_balances
    assert result[account1].value == Decimal("1000.00")
    assert result[account2].value == Decimal("2000.00")


# LLM-generated content at query #73
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets", "Cash")
    account2 = Account("Liabilities", "Accounts Payable")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000.00"))),
        account2: Balance(period.since, Quantity(Decimal("500.00")))
    }

    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #74
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Mock initial balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Cash"): Balance(period.since, Quantity(Decimal(1000))),
            Account("Accounts Receivable"): Balance(period.since, Quantity(Decimal(500)))
        }

    # Mock journal entries
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[_T]]:
        journal_entry = JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test transaction",
            postings=[
                Posting(Account("Cash"), Amount(Decimal(200)), Direction.DEBIT),
                Posting(Account("Revenue"), Amount(Decimal(200)), Direction.CREDIT)
            ]
        )
        return [journal_entry]

    # Create the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3  # Cash, Accounts Receivable, Revenue

    # Check Cash ledger
    cash_ledger = general_ledger.ledgers[Account("Cash")]
    assert cash_ledger.initial.value == Quantity(Decimal(1000))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].amount == Amount(Decimal(200))
    assert cash_ledger.entries[0].balance == Quantity(Decimal(1200))

    # Check Accounts Receivable ledger
    ar_ledger = general_ledger.ledgers[Account("Accounts Receivable")]
    assert ar_ledger.initial.value == Quantity(Decimal(500))
    assert len(ar_ledger.entries) == 0

    # Check Revenue ledger
    revenue_ledger = general_ledger.ledgers[Account("Revenue")]
    assert revenue_ledger.initial.value == Quantity(Decimal(0))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].amount == Amount(Decimal(200))
    assert revenue_ledger.entries[0].balance == Quantity(Decimal(200))


# LLM-generated content at query #75
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Setup
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    mock_account = Account("TestAccount")
    mock_balance = Balance(period.since, Quantity(Decimal(1000)))
    expected_initial_balances = {mock_account: mock_balance}

    # Mock implementation
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Test
    read_initial_balances: ReadInitialBalances = mock_read_initial_balances
    result = read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert result[mock_account] == mock_balance


# LLM-generated content at query #76
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
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

    journal = [journal_entry1, journal_entry2]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2

    # Check first entry for account1
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit == Amount(Decimal("50.00"))
    assert entry1.credit is None

    # Check second entry for account1
    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Quantity(Decimal("120.00"))
    assert entry2.debit is None
    assert entry2.credit == Amount(Decimal("30.00"))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 2

    # Check first entry for account2
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Quantity(Decimal("150.00"))
    assert entry1.debit is None
    assert entry1.credit == Amount(Decimal("50.00"))

    # Check second entry for account2
    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Quantity(Decimal("180.00"))
    assert entry2.debit == Amount(Decimal("30.00"))
    assert entry2.credit is None


# LLM-generated content at query #77
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


# LLM-generated content at query #78
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(100))),
        account2: Balance(period.since, Quantity(Decimal(0)))
    }

    # Create journal entries
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

    journal = [journal_entry1, journal_entry2]

    # Call the function
    result = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers

    # Check ledger1
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Decimal(100)
    assert len(ledger1.entries) == 2

    # Check ledger1 entries
    entry1 = ledger1.entries[0]
    assert entry1.posting == journal_entry1.postings[0]
    assert entry1.balance == Decimal(150)

    entry2 = ledger1.entries[1]
    assert entry2.posting == journal_entry2.postings[0]
    assert entry2.balance == Decimal(120)

    # Check ledger2
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Decimal(0)
    assert len(ledger2.entries) == 2

    # Check ledger2 entries
    entry1 = ledger2.entries[0]
    assert entry1.posting == journal_entry1.postings[1]
    assert entry1.balance == Decimal(50)

    entry2 = ledger2.entries[1]
    assert entry2.posting == journal_entry2.postings[1]
    assert entry2.balance == Decimal(20)


