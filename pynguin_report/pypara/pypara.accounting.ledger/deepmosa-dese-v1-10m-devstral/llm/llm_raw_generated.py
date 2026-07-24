####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_general_ledger_program_call_returns_general_ledger():
    class MockGeneralLedgerProgram:
        def __call__(self, period):
            return "GeneralLedger"

    program = MockGeneralLedgerProgram()
    result = program(DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31)))
    assert result == "GeneralLedger"


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100), account=Account("Test"), journal=Journal(description="Test Journal"))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        journal=Journal(
            date=datetime.date(2023, 1, 1),
            description="Test transaction",
            postings=[
                Posting(account=Account("Asset"), amount=Amount(100), direction=Direction.DEBIT),
                Posting(account=Account("Equity"), amount=Amount(100), direction=Direction.CREDIT)
            ]
        ),
        amount=Amount(100),
        direction=Direction.DEBIT
    )
    balance = Quantity(100)

    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #6
#--------------------------

```python
def test_read_initial_balances_call():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    read_balances = ReadInitialBalances()
    result = read_balances(period)
    assert isinstance(result, InitialBalances)


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test"), journal=Journal())
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #8
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        date=datetime.date(2023, 1, 1),
        account=Account("Test Account"),
        amount=Amount(100, "USD"),
        direction=Direction.DEBIT,
        journal=Journal(
            date=datetime.date(2023, 1, 1),
            description="Test Description",
            postings=[]
        )
    )
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #9
#--------------------------

```python
def test_general_ledger_program_call_returns_general_ledger():
    program = GeneralLedgerProgram()
    period = DateRange(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    result = program(period)
    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #10
#--------------------------

```python
def test_ledger_constructor_initializes_correctly():
    account = Account("Test Account")
    initial_balance = Balance(Quantity(100))
    ledger = Ledger(account, initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #11
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, Amount(100, "USD"), datetime.date(2023, 1, 1), "Test entry")
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #13
#--------------------------

```python
def test_ledger_constructor():
    account = Account("Test Account")
    initial = Balance(Quantity(100))
    ledger = Ledger(account, initial)
    assert ledger.account == account
    assert ledger.initial == initial
    assert ledger.entries == []


# LLM-generated content at query #14
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Cash"), journal=Journal(description="Test"))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #15
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, "Test", 100)
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #16
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = InitialBalances({})
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    initial = InitialBalances({account: initial_balance})
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert account in result.ledgers
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_with_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    initial = InitialBalances({})
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].posting == journal_entry.postings[0]

def test_build_general_ledger_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    source = object()
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test Entry", source)
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    initial = InitialBalances({})
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert account not in result.ledgers

def test_build_general_ledger_multiple_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    source = object()
    journal_entry1 = JournalEntry(datetime.date(2023, 6, 15), "Test Entry 1", source)
    journal_entry1.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(50)))
    journal_entry2 = JournalEntry(datetime.date(2023, 7, 15), "Test Entry 2", source)
    journal_entry2.post(datetime.date(2023, 7, 15), account, Quantity(Decimal(-30)))
    journal = [journal_entry1, journal_entry2]
    initial = InitialBalances({})
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 2
    assert result.ledgers[account].entries[0].posting == journal_entry1.postings[0]
    assert result.ledgers[account].entries[1].posting == journal_entry2.postings[0]


# LLM-generated content at query #17
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #18
#--------------------------

```python
def test_ledger_constructor_with_default_entries():
    account = Account("Test Account")
    initial = Balance(Quantity(100))
    ledger = Ledger(account, initial)
    assert ledger.account == account
    assert ledger.initial == initial
    assert ledger.entries == []

def test_ledger_constructor_with_non_default_entries():
    account = Account("Test Account")
    initial = Balance(Quantity(100))
    ledger = Ledger(account, initial)
    ledger.entries = [LedgerEntry(ledger, Posting(Quantity(50), Direction.DEBIT), Quantity(150))]
    assert ledger.account == account
    assert ledger.initial == initial
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(150)


# LLM-generated content at query #19
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #20
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test"), date=datetime.date(2023, 1, 1))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #21
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #22
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #23
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("TestAccount")
    initial = InitialBalances({account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))})
    journal_entry_in_period = JournalEntry(
        datetime.date(2023, 6, 1),
        "Test Entry",
        None
    ).post(datetime.date(2023, 6, 1), account, Quantity(Decimal(50)))
    journal_entry_out_of_period = JournalEntry(
        datetime.date(2024, 1, 1),
        "Test Entry",
        None
    ).post(datetime.date(2024, 1, 1), account, Quantity(Decimal(50)))
    journal = [journal_entry_in_period, journal_entry_out_of_period]
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting == journal_entry_in_period.postings[0]


# LLM-generated content at query #24
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #25
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Assets:Cash"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #26
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_with_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 15),
        "Test Entry",
        "Test Source"
    )
    posting = Posting(
        journal_entry,
        datetime.date(2023, 1, 15),
        account,
        Direction.INC,
        Amount(Quantity(Decimal("50.00")))
    )
    journal_entry.postings.append(posting)
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].posting == posting

def test_build_general_ledger_with_multiple_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Test Account 1")
    account2 = Account("Test Account 2")
    initial = {}
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 15),
        "Test Entry",
        "Test Source"
    )
    posting1 = Posting(
        journal_entry,
        datetime.date(2023, 1, 15),
        account1,
        Direction.INC,
        Amount(Quantity(Decimal("50.00")))
    )
    posting2 = Posting(
        journal_entry,
        datetime.date(2023, 1, 15),
        account2,
        Direction.DEC,
        Amount(Quantity(Decimal("30.00")))
    )
    journal_entry.postings.extend([posting1, posting2])
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].posting == posting1
    assert result.ledgers[account2].entries[0].posting == posting2

def test_build_general_ledger_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal_entry = JournalEntry(
        datetime.date(2022, 12, 15),
        "Test Entry",
        "Test Source"
    )
    posting = Posting(
        journal_entry,
        datetime.date(2022, 12, 15),
        account,
        Direction.INC,
        Amount(Quantity(Decimal("50.00")))
    )
    journal_entry.postings.append(posting)
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


# LLM-generated content at query #27
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #28
#--------------------------

```python
def test_read_initial_balances_call_returns_initial_balances():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    read_initial_balances = ReadInitialBalances()
    result = read_initial_balances(period)
    assert isinstance(result, InitialBalances)


# LLM-generated content at query #29
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(Journal(), Account(), Amount(100, "USD"), datetime.date(2023, 1, 1), Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #31
#--------------------------

```python
def test_ledger_constructor_with_default_entries():
    account = Account("Test Account")
    initial = Balance(Quantity(100.0))
    ledger = Ledger(account, initial)
    assert ledger.account == account
    assert ledger.initial == initial
    assert ledger.entries == []


# LLM-generated content at query #32
#--------------------------

```python
def test_build_general_ledger_creates_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal = [
        JournalEntry(
            date=datetime.date(2023, 1, 1),
            description="Test Entry",
            source="Test Source"
        ).post(datetime.date(2023, 1, 1), account, Quantity(Decimal(100)))
    ]

    general_ledger = build_general_ledger(period, journal, initial)

    assert account in general_ledger.ledgers
    assert general_ledger.ledgers[account].account == account
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal(0))


# LLM-generated content at query #33
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        amount=Amount(100, "USD"),
        account=Account("Test Account"),
        direction=Direction.DEBIT,
        journal=Journal(
            date=datetime.date(2023, 1, 1),
            description="Test Description",
            postings=[
                Posting(
                    amount=Amount(100, "USD"),
                    account=Account("Test Account"),
                    direction=Direction.DEBIT,
                ),
                Posting(
                    amount=Amount(100, "USD"),
                    account=Account("Counter Account"),
                    direction=Direction.CREDIT,
                ),
            ],
        ),
    )
    entry = LedgerEntry(ledger, posting, Quantity(100, "USD"))

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == Quantity(100, "USD")


# LLM-generated content at query #34
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = InitialBalances({account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))})
    journal_entry_in_period = JournalEntry(
        datetime.date(2023, 6, 15),
        "Test Entry",
        "Test Source"
    ).post(datetime.date(2023, 6, 15), account, Quantity(Decimal(50)))
    journal_entry_outside_period = JournalEntry(
        datetime.date(2024, 1, 1),
        "Test Entry",
        "Test Source"
    ).post(datetime.date(2024, 1, 1), account, Quantity(Decimal(50)))
    journal = [journal_entry_in_period, journal_entry_outside_period]
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting == journal_entry_in_period.postings[0]


# LLM-generated content at query #35
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #36
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #37
#--------------------------

```python
def test_build_general_ledger_predicate():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = [
        JournalEntry(
            date=datetime.date(2023, 6, 15),
            description="Test entry",
            source="Test source",
        ).post(datetime.date(2023, 6, 15), Account("Test Account"), Quantity(Decimal("100")))
    ]
    initial = {Account("Test Account"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal("50")))}
    result = build_general_ledger(period, journal, initial)
    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #38
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), journal=Journal(description="Test"))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #39
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_with_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("50.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].posting == journal_entry.postings[0]

def test_build_general_ledger_with_postings_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2022, 12, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2022, 12, 15), account, Quantity(Decimal("50.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_multiple_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Test Account 1")
    account2 = Account("Test Account 2")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal("50.00")))
    journal_entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal("-30.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #40
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = Quantity(100)

    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #41
#--------------------------

```python
def test_read_initial_balances_call():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    read_initial_balances = ReadInitialBalances()
    result = read_initial_balances(period)
    assert isinstance(result, InitialBalances)


# LLM-generated content at query #42
#--------------------------

```python
def test_build_general_ledger_with_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 0


# LLM-generated content at query #43
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("TestAccount")
    initial = InitialBalances({account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))})
    journal_entry = JournalEntry(
        datetime.date(2023, 6, 15),
        "Test Entry",
        None
    ).post(datetime.date(2023, 6, 15), account, Quantity(Decimal(50)))

    general_ledger = build_general_ledger(period, [journal_entry], initial)

    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting.amount == Amount(Decimal(50))


# LLM-generated content at query #44
#--------------------------

```python
def test_general_ledger_program_call_returns_general_ledger():
    program = GeneralLedgerProgram()
    period = DateRange(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    result = program(period)
    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #45
#--------------------------

```python
def test_call_returns_initial_balances():
    read_balances = ReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = read_balances(period)
    assert isinstance(result, InitialBalances)


# LLM-generated content at query #46
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account = Account("TestAccount")
    initial = InitialBalances({account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))})
    journal = [
        JournalEntry(datetime.date(2022, 12, 31), "Old Entry", None).post(datetime.date(2022, 12, 31), account, Quantity(Decimal(100))),
        JournalEntry(datetime.date(2023, 1, 15), "Valid Entry", None).post(datetime.date(2023, 1, 15), account, Quantity(Decimal(200))),
        JournalEntry(datetime.date(2023, 2, 1), "Future Entry", None).post(datetime.date(2023, 2, 1), account, Quantity(Decimal(300)))
    ]
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].posting.amount == Amount(Decimal(200))


# LLM-generated content at query #47
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("TestAccount")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_with_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("TestAccount")
    initial = {}
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", "TestSource")
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("50.00")))
    journal = [entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert result.ledgers[account].account == account
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("50.00"))

def test_build_general_ledger_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("TestAccount")
    initial = {}
    entry = JournalEntry(datetime.date(2024, 1, 1), "Test Entry", "TestSource")
    entry.post(datetime.date(2024, 1, 1), account, Quantity(Decimal("50.00")))
    journal = [entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_multiple_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("TestAccount1")
    account2 = Account("TestAccount2")
    initial = {}
    entry1 = JournalEntry(datetime.date(2023, 6, 15), "Test Entry 1", "TestSource1")
    entry1.post(datetime.date(2023, 6, 15), account1, Quantity(Decimal("50.00")))
    entry2 = JournalEntry(datetime.date(2023, 7, 15), "Test Entry 2", "TestSource2")
    entry2.post(datetime.date(2023, 7, 15), account2, Quantity(Decimal("100.00")))
    journal = [entry1, entry2]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account2].account == account2
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("50.00"))
    assert result.ledgers[account2].entries[0].balance == Quantity(Decimal("100.00"))


# LLM-generated content at query #48
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #49
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #50
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(Journal(), Account(), Amount(100, "USD"), datetime.date(2023, 1, 1), Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #51
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        journal=Journal(
            date=datetime.date(2023, 1, 1),
            description="Test",
            postings=[]
        ),
        account=Account("Test"),
        amount=Amount(100, "USD"),
        direction=Direction.DEBIT
    )
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #52
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, Account("Test"), Amount(100, "USD"), datetime.date(2023, 1, 1), Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #53
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #54
#--------------------------

```python
def test_build_general_ledger_with_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 0


# LLM-generated content at query #55
#--------------------------

```python
def test_general_ledger_program_call_returns_general_ledger():
    program = GeneralLedgerProgram()
    period = DateRange(start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))
    result = program(period)
    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #56
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #57
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        journal=Journal(
            date=datetime.date(2023, 1, 1),
            description="Test transaction",
            postings=[
                Posting(account=Account("Test Account"), amount=Amount(100, "USD"), direction=Direction.DEBIT),
                Posting(account=Account("Another Account"), amount=Amount(100, "USD"), direction=Direction.CREDIT)
            ]
        ),
        account=Account("Test Account"),
        amount=Amount(100, "USD"),
        direction=Direction.DEBIT
    )
    balance = Quantity(100, "USD")

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #58
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Assets:Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00")))
    initial = {account: initial_balance}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial_balance
    assert result.ledgers[account].entries == []

def test_build_general_ledger_with_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Assets:Cash")
    account2 = Account("Expenses:Rent")
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Rent payment",
        source=None
    )
    journal_entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal("-500.00")))
    journal_entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal("500.00")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("-500.00"))
    assert result.ledgers[account2].entries[0].balance == Quantity(Decimal("500.00"))

def test_build_general_ledger_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Assets:Cash")
    journal_entry = JournalEntry(
        date=datetime.date(2022, 12, 15),
        description="Out of period",
        source=None
    )
    journal_entry.post(datetime.date(2022, 12, 15), account, Quantity(Decimal("100.00")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_and_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Assets:Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00")))
    initial = {account: initial_balance}
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Deposit",
        source=None
    )
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("500.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("1500.00"))


# LLM-generated content at query #59
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100), date=datetime.date(2023, 1, 1), journal=Journal(description="Test"))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #60
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #61
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #62
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        journal=Journal(
            date=datetime.date(2023, 1, 1),
            description="Test transaction",
            postings=[
                Posting(
                    account=Account("Assets:Cash"),
                    amount=Amount(100, "USD"),
                    direction=Direction.DEBIT
                ),
                Posting(
                    account=Account("Income:Salary"),
                    amount=Amount(100, "USD"),
                    direction=Direction.CREDIT
                )
            ]
        ),
        account=Account("Assets:Cash"),
        amount=Amount(100, "USD"),
        direction=Direction.DEBIT
    )
    balance = Quantity(100, "USD")

    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #63
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #64
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, Amount(100), Account("Test"), datetime.date(2023, 1, 1))
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #65
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), journal=Journal(description="Test"))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #66
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #67
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test Account"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #68
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #69
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #70
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        journal=Journal(
            date=datetime.date(2023, 1, 1),
            description="Test transaction",
            postings=[
                Posting(account=Account("Account1"), amount=Amount(100), direction=Direction.DEBIT),
                Posting(account=Account("Account2"), amount=Amount(100), direction=Direction.CREDIT)
            ]
        ),
        amount=Amount(100),
        direction=Direction.DEBIT
    )
    balance = Quantity(100)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #71
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    account = Account("Test Account")
    posting = Posting(Journal("Test Journal", datetime.date(2023, 1, 1), "Test Description", [Posting(account, Amount(100, "USD"), Direction.DEBIT)]), account, Amount(100, "USD"), Direction.DEBIT)
    balance = Quantity(100, "USD")

    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #72
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100, "USD"), account=Account("Test"), journal=Journal(description="Test"))
    balance = Quantity(100, "USD")

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #73
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), account=Account("Test"), journal=Journal(description="Test"))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #74
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #75
#--------------------------

```python
def test_build_general_ledger_creates_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", None)
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("100.00")))
    initial_balances = {}

    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)

    assert account in general_ledger.ledgers
    assert general_ledger.ledgers[account].account == account
    assert general_ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))


# LLM-generated content at query #76
#--------------------------

```python
def test_general_ledger_program_call_returns_general_ledger():
    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger()

    program = MockGeneralLedgerProgram()
    period = DateRange(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #77
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #78
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #79
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = Quantity(100)

    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #80
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(Journal(), Account(), Amount(100, "USD"), datetime.date(2023, 1, 1), Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #81
#--------------------------

```python
def test_read_initial_balances_call_returns_initial_balances():
    read_initial_balances = ReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = read_initial_balances(period)
    assert isinstance(result, InitialBalances)


# LLM-generated content at query #82
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(account=Account("TestAccount"), amount=Amount(100, "USD"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #83
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("TestAccount")
    initial = InitialBalances({account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))})
    journal_entry_in_period = JournalEntry(
        datetime.date(2023, 6, 15),
        "Test entry in period",
        None
    ).post(datetime.date(2023, 6, 15), account, Quantity(Decimal(100)))
    journal_entry_out_of_period = JournalEntry(
        datetime.date(2024, 1, 1),
        "Test entry out of period",
        None
    ).post(datetime.date(2024, 1, 1), account, Quantity(Decimal(200)))
    journal = [journal_entry_in_period, journal_entry_out_of_period]

    general_ledger = build_general_ledger(period, journal, initial)

    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting.amount == Amount(Decimal(100))


# LLM-generated content at query #84
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(Journal(), Account(), Amount(100, "USD"), datetime.date(2023, 1, 1), Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #85
#--------------------------

```python
def test_ledger_constructor_with_default_entries():
    account = Account("Test Account")
    initial = Balance(Quantity(100))
    ledger = Ledger(account, initial)
    assert ledger.account == account
    assert ledger.initial == initial
    assert ledger.entries == []


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(Account("Test"), Amount(100, "USD"), datetime.date(2023, 1, 1), Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, Amount(100, "USD"), Account("Test"), datetime.date(2023, 1, 1), "Test Description")
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #3
#--------------------------

```python
def test_read_initial_balances_call_returns_initial_balances():
    read_initial_balances = ReadInitialBalances()
    period = DateRange(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    result = read_initial_balances(period)
    assert isinstance(result, InitialBalances)


# LLM-generated content at query #4
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_with_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account 1")
    account2 = Account("Account 2")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", None)
    journal_entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal("50.00")))
    journal_entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal("-50.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("50.00"))
    assert result.ledgers[account2].entries[0].balance == Quantity(Decimal("-50.00"))

def test_build_general_ledger_with_initial_and_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", None)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("50.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert result.ledgers[account].initial == initial_balance
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("150.00"))

def test_build_general_ledger_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test Entry", None)
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal("50.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_build_general_ledger_includes_postings_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    journal_entry = JournalEntry(
        datetime.date(2023, 6, 15),
        "Test Entry",
        None
    ).post(datetime.date(2023, 6, 15), account, Quantity(Decimal(50)))
    initial_balances = {account: initial_balance}
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert len(general_ledger.ledgers[account].entries) == 1


# LLM-generated content at query #6
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    initial = {account: initial_balance}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_with_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].posting == journal_entry.postings[0]
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal(50))

def test_build_general_ledger_with_multiple_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Test Account 1")
    account2 = Account("Test Account 2")
    initial = {}
    journal_entry1 = JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", "Source1")
    journal_entry1.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(50)))
    journal_entry2 = JournalEntry(datetime.date(2023, 1, 20), "Test Entry 2", "Source2")
    journal_entry2.post(datetime.date(2023, 1, 20), account2, Quantity(Decimal(-30)))
    journal = [journal_entry1, journal_entry2]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal(50))
    assert result.ledgers[account2].entries[0].balance == Quantity(Decimal(-30))

def test_build_general_ledger_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Test Account")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2022, 12, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2022, 12, 15), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100, "USD"), account=Account("Test"), direction=Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


