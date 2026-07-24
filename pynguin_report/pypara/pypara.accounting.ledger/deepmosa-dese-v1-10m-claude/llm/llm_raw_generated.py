####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar, Generic
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger(Generic[_T]):
        def __init__(self, entries: list[_T]):
            self.entries = entries
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[str]:
            return GeneralLedger([f"Entry for {period.start} to {period.end}"])
    
    program = ConcreteGeneralLedgerProgram()
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert len(result.entries) == 1
    assert "2023-01-01" in result.entries[0]
    assert "2023-12-31" in result.entries[0]


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2024, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("Cash")
    test_quantity = MockQuantity(500.0)
    test_ledger = MockLedger()
    
    test_journal = MockJournal(
        description="Test transaction",
        postings=[]
    )
    
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        direction: str
        account: MockAccount
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Description", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        account=test_account,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="12345")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_different_values():
    account = Account(name="Savings", number="67890")
    initial_balance = Balance(value=Quantity(5000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.number == "67890"
    assert ledger.initial.value == Quantity(5000)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    mock_account = type('Account', (), {})()
    mock_amount = type('Amount', (), {'__repr__': lambda self: 'Amount(100)'})()
    mock_quantity = type('Quantity', (), {'__repr__': lambda self: 'Quantity(100)'})()
    
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'amount': mock_amount,
        'is_debit': True,
        'is_credit': False,
        'direction': 'debit',
        'account': mock_account,
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': [
                type('Posting', (), {
                    'direction': 'debit',
                    'account': mock_account
                })(),
                type('Posting', (), {
                    'direction': 'credit',
                    'account': type('Account', (), {})()
                })()
            ]
        })()
    })()
    
    mock_ledger = type('Ledger', (), {})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #6
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        account: MockAccount
        amount: MockAmount
        direction: str
        journal: MockJournal
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create test instances
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 15),
        account=test_account,
        amount=test_amount,
        direction="debit",
        journal=test_journal,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="General Ledger")
    test_balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns values
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    ledger = Ledger(name="Test Ledger")
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #8
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    ledger = Ledger()
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2024, 1, 1),
        account=account,
        amount=amount,
        journal=journal,
        direction="debit"
    )
    balance = Quantity(value=500.0)

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert ledger_entry.ledger is ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is balance


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger_with_single_posting():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance}
    
    entry = JournalEntry(date(2024, 1, 15), "Test entry", "source")
    entry.post(date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    entry.post(date(2024, 1, 15), account2, Quantity(Decimal(100)))
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    general_ledger = build_general_ledger(period, [entry], initial_balances)
    
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    
    account1 = Account("1000", "Cash")
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal(5000)))
    initial_balances = {account1: initial_balance}
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0
    assert general_ledger.ledgers[account1].initial == initial_balance


def test_build_general_ledger_multiple_entries():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance}
    
    entry1 = JournalEntry(date(2024, 1, 15), "Entry 1", "source1")
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    entry1.post(date(2024, 1, 15), account2, Quantity(Decimal(100)))
    
    entry2 = JournalEntry(date(2024, 2, 20), "Entry 2", "source2")
    entry2.post(date(2024, 2, 20), account1, Quantity(Decimal(-200)))
    entry2.post(date(2024, 2, 20), account2, Quantity(Decimal(200)))
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    general_ledger = build_general_ledger(period, [entry1, entry2], initial_balances)
    
    assert len(general_ledger.ledgers[account1].entries) == 2
    assert len(general_ledger.ledgers[account2].entries) == 2


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance}
    
    entry_in_period = JournalEntry(date(2024, 2, 15), "In period", "source1")
    entry_in_period.post(date(2024, 2, 15), account1, Quantity(Decimal(-100)))
    entry_in_period.post(date(2024, 2, 15), account2, Quantity(Decimal(100)))
    
    entry_out_period = JournalEntry(date(2024, 12, 25), "Out of period", "source2")
    entry_out_period.post(date(2024, 12, 25), account1, Quantity(Decimal(-200)))
    entry_out_period.post(date(2024, 12, 25), account2, Quantity(Decimal(200)))
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 24))
    
    general_ledger = build_general_ledger(period, [entry_in_period, entry_out_period], initial_balances)
    
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1


def test_build_general_ledger_creates_missing_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    account3 = Account("3000", "Revenue")
    
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance}
    
    entry = JournalEntry(date(2024, 1, 15), "Test entry", "source")
    entry.post(date(2024, 1, 15), account2, Quantity(Decimal(100)))
    entry.post(date(2024, 1, 15), account3, Quantity(Decimal(-100)))
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    general_ledger = build_general_ledger(period, [entry], initial_balances)
    
    assert account2 in general_ledger.ledgers
    assert account3 in general_ledger.ledgers
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal(0


# LLM-generated content at query #10
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger()

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is quantity


# LLM-generated content at query #11
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return MockJournal()
    
    @dataclass
    class MockJournal:
        description: str = "Test description"
        postings: list = None
        
        def __post_init__(self):
            if self.postings is None:
                self.postings = []
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test instances
    mock_ledger = MockLedger()
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=MockAmount(100.0, "USD"),
        account=MockAccount("Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockQuantity(1000.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly initialized all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        account: MockAccount
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2024, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("Cash")
    test_journal = MockJournal("Test transaction", [])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account,
        journal=test_journal
    )
    test_balance = MockQuantity(500.0)
    test_ledger = MockLedger()
    
    # Test constructor
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #13
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.amount import Amount, Quantity
    from pypara.core.period import DateRange
    from pypara.core.account import Account
    
    # Setup test data
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create initial balances
    initial = {account1: Balance(period.since, Quantity(Decimal(1000)))}
    
    # Create a journal entry with postings
    entry = JournalEntry(date(2023, 6, 15), "Test entry", "source_obj")
    entry.post(date(2023, 6, 15), account1, Quantity(Decimal(-500)))
    entry.post(date(2023, 6, 15), account2, Quantity(Decimal(500)))
    
    journal = [entry]
    
    # Build the general ledger
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Assertions - the predicate at line 1 should evaluate to True
    assert general_ledger is not None
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[account1], Ledger)
    assert isinstance(general_ledger.ledgers[account2], Ledger)
    assert general_ledger.ledgers[account1].account == account1
    assert general_ledger.ledgers[account2].account == account2
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1


# LLM-generated content at query #14
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar, Generic
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger(Generic[_T]):
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[dict]:
            return GeneralLedger({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []


# LLM-generated content at query #15
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create instances
    mock_account = MockAccount(name="Cash")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test transaction", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=mock_amount,
        journal=mock_journal,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger()
    mock_quantity = MockQuantity(value=100.0)
    
    # Test constructor
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_quantity)
    
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #16
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}


def test_read_initial_balances_call_with_different_periods():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            if period.start.year == 2023:
                return InitialBalances({"account1": 1000})
            return InitialBalances({"account1": 2000})
    
    reader = ConcreteReadInitialBalances()
    period_2023 = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    period_2024 = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    result_2023 = reader(period_2023)
    result_2024 = reader(period_2024)
    
    assert result_2023.balances == {"account1": 1000}
    assert result_2024.balances == {"account1": 2000}


def test_read_initial_balances_call_empty_balances():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {}
    assert len(result.balances) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    posting_obj = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        journal=Journal(description="Test Journal", postings=[]),
        account=account,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=100.0)
    
    ledger_entry = LedgerEntry(
        ledger=ledger,
        posting=posting_obj,
        balance=balance
    )
    
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting_obj
    assert ledger_entry.balance == balance


# LLM-generated content at query #18
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="asset")
    initial_balance = Balance(value=1000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #19
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects
    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    # Create test data
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    posting1 = Posting(date=date(2024, 1, 1), account=account, amount=amount, direction="debit", journal=None)
    posting2 = Posting(date=date(2024, 1, 1), account=Account(name="Counter"), amount=amount, direction="credit", journal=None)
    journal = Journal(description="Test Journal", postings=[posting1, posting2])
    posting1.journal = journal
    posting2.journal = journal
    ledger = Ledger(name="Test Ledger")

    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting1, balance=quantity)

    # Assert constructor properly assigns attributes
    assert entry.ledger is ledger
    assert entry.posting is posting1
    assert entry.balance is quantity
    assert entry.ledger.name == "Test Ledger"
    assert entry.balance.value == 100.0


# LLM-generated content at query #20
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        account: MockAccount
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2023, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=500.0)
    test_ledger = MockLedger()
    
    mock_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=MockJournal(description="Test transaction", postings=[])
    )
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=mock_posting,
        balance=test_quantity
    )
    
    # Assert constructor properly assigned all fields
    assert entry.ledger is test_ledger
    assert entry.posting is mock_posting
    assert entry.balance is test_quantity
    assert entry.ledger == test_ledger
    assert entry.posting == mock_posting
    assert entry.balance == test_quantity


# LLM-generated content at query #21
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    balance = Quantity(value=100.0)
    ledger = Ledger()
    
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #22
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    test_account = Account(name="Test Account")
    test_amount = Amount(value=100.0, currency="USD")
    test_quantity = Quantity(value=100.0)
    test_journal = Journal(description="Test Journal", postings=[])
    test_posting = Posting(
        date=date(2024, 1, 1),
        amount=test_amount,
        account=test_account,
        direction="debit",
        journal=test_journal
    )
    test_ledger = Ledger(name="Test Ledger")

    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #23
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("TestAccount")
    test_journal = MockJournal("Test transaction", [])
    test_posting = MockPosting(test_date, test_amount, test_account, "debit", True, False, test_journal)
    test_balance = MockQuantity(500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    # Assert constructor properly assigns all attributes
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #24
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Account, Quantity, Amount
    
    # Create a date range
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Create accounts
    account_a = Account("1000", "Cash")
    account_b = Account("2000", "Payable")
    
    # Create initial balances
    initial = {account_a: Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    # Create journal entries - one within period, one outside
    entry_within = JournalEntry(date=date(2023, 6, 15), description="Entry within period", source="source1")
    entry_outside = JournalEntry(date=date(2024, 1, 15), description="Entry outside period", source="source2")
    
    # Add postings to entries
    entry_within.post(date(2023, 6, 15), account_a, Quantity(Decimal(100)))
    entry_outside.post(date(2024, 1, 15), account_b, Quantity(Decimal(50)))
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [entry_within, entry_outside], initial)
    
    # The predicate at line 16 should evaluate to True only for entries within the date range
    # Verify that only the posting from entry_within is included
    assert account_a in general_ledger.ledgers
    assert len(general_ledger.ledgers[account_a].entries) == 1
    
    # Verify that account_b is not in ledgers (posting was outside period)
    assert account_b not in general_ledger.ledgers


# LLM-generated content at query #25
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import Account, Amount, Quantity, DateRange
    
    # Setup test data
    test_account_1 = Account("1000", "Cash")
    test_account_2 = Account("2000", "Accounts Payable")
    test_account_3 = Account("3000", "Revenue")
    
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create initial balances
    initial_balances = {
        test_account_1: Balance(period_start, Quantity(Decimal("1000")))
    }
    
    # Create journal entries with postings
    entry1 = JournalEntry(period_start, "Initial deposit", "source1")
    entry1.post(period_start, test_account_1, Quantity(Decimal("500")))
    entry1.post(period_start, test_account_3, Quantity(Decimal("-500")))
    
    entry2 = JournalEntry(date(2024, 6, 15), "Payment", "source2")
    entry2.post(date(2024, 6, 15), test_account_1, Quantity(Decimal("-300")))
    entry2.post(date(2024, 6, 15), test_account_2, Quantity(Decimal("300")))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert test_account_1 in general_ledger.ledgers
    assert test_account_2 in general_ledger.ledgers
    assert test_account_3 in general_ledger.ledgers
    
    # Check ledger for account 1 (Cash)
    ledger_1 = general_ledger.ledgers[test_account_1]
    assert ledger_1.account == test_account_1
    assert ledger_1.initial.value == Quantity(Decimal("1000"))
    assert len(ledger_1.entries) == 2
    assert ledger_1.entries[0].balance == Quantity(Decimal("1500"))
    assert ledger_1.entries[1].balance == Quantity(Decimal("1200"))
    
    # Check ledger for account 2 (Accounts Payable)
    ledger_2 = general_ledger.ledgers[test_account_2]
    assert ledger_2.account == test_account_2
    assert ledger_2.initial.value == Quantity(Decimal("0"))
    assert len(ledger_2.entries) == 1
    assert ledger_2.entries[0].balance == Quantity(Decimal("300"))
    
    # Check ledger for account 3 (Revenue)
    ledger_3 = general_ledger.ledgers[test_account_3]
    assert ledger_3.account == test_account_3
    assert ledger_3.initial.value == Quantity(Decimal("0"))
    assert len(ledger_3.entries) == 1
    assert ledger_3.entries[0].balance == Quantity(Decimal("-500"))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import Account, Quantity, DateRange
    
    test_account = Account("1000", "Cash")
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    initial_balances = {
        test_account: Balance(period_start, Quantity(Decimal("5000")))
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert test_account in general_ledger.ledgers
    assert len(general_ledger.ledgers[test_account].entries) == 0
    assert general_ledger.ledgers[test_account].initial.value == Quantity(Decimal("5000"))


def test_build_general_ledger_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import Account, Quantity, DateRange
    
    test_account_1 = Account("1000", "Cash")
    test_account_2 = Account("3000", "Revenue")
    
    period_start = date(2024, 6, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    initial_balances = {}
    
    # Create entry before period
    entry_before = JournalEntry(date(2024, 1, 15), "Before period", "source1")
    entry_before.post(date(2024, 1, 15), test_account_1, Quantity(Decimal("100")))
    entry_before.post(date(2024, 1, 15), test_account_2, Quantity(Decimal("-100")))
    
    # Create entry within period
    entry_within = JournalEntry(date(2024, 7, 1), "Within period", "source2")
    entry_within.post(date(2024, 7, 1), test_account_1, Quantity(Decimal("200")))
    entry_within.post(date(2024, 7, 1), test_account_2, Quantity(Decimal("-200")))
    
    # Create entry after period
    entry_after = JournalEntry(date(2025, 1, 15), "After period", "source3")
    entry_after.post(date(2025, 1, 15), test_account_1, Quantity(Decimal("300")))
    entry_after.post(date(2025, 1, 15), test_account_2, Quantity(Decimal("-300")))
    
    journal = [entry_before, entry_within, entry_after]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers[test_account_1].entries) == 1
    assert general_ledger.ledgers[test_account_1].entries[0].balance == Quantity(Decimal("200"))


# LLM-generated content at query #26
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockAmount:
        value: float
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create instances
    mock_journal = MockJournal(description="Test transaction", postings=[])
    mock_amount = MockAmount(value=100.0)
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        journal=mock_journal,
        amount=mock_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger()
    mock_quantity = MockQuantity(value=100.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #27
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    from typing import Protocol
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalances(Protocol):
        def __call__(self, period: DateRange) -> InitialBalances:
            ...
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert result.balances["account1"] == 1000
    assert result.balances["account2"] == 2000


# LLM-generated content at query #28
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_id="ACC001")
    initial_balance = Balance(value=1000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_account_and_balance():
    account = Account(name="Savings", account_id="SAV123")
    balance = Balance(value=5000)
    
    ledger = Ledger(account=account, initial=balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.account_id == "SAV123"
    assert ledger.initial.value == 5000
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


def test_ledger_constructor_entries_initialized_empty():
    account = Account(name="Checking", account_id="CHK456")
    initial = Balance(value=2500)
    
    ledger = Ledger(account=account, initial=initial)
    
    assert hasattr(ledger, 'entries')
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #29
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    # Create test objects
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    
    posting1 = Posting(
        date=date(2024, 1, 1),
        amount=amount,
        account=account,
        journal=Journal(description="Test Journal", postings=[]),
        direction="debit"
    )
    
    ledger = Ledger()
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting1, balance=quantity)
    
    # Assert constructor correctly assigns attributes
    assert ledger_entry.ledger is ledger
    assert ledger_entry.posting is posting1
    assert ledger_entry.balance is quantity
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting1
    assert ledger_entry.balance == quantity


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger(name="Test Ledger")

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == quantity


# LLM-generated content at query #31
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, data):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger({"period": period})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range


# LLM-generated content at query #32
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_account = type('Account', (), {})()
    mock_amount = type('Amount', (), {'value': Decimal('100.00')})()
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'amount': mock_amount,
        'is_debit': True,
        'is_credit': False,
        'direction': 'debit',
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': [
                type('Posting', (), {'account': mock_account, 'direction': 'credit'})()
            ]
        })()
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {'value': Decimal('500.00')})()
    
    # Test constructor
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #33
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    from typing import Protocol
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalances(Protocol):
        def __call__(self, period: DateRange) -> InitialBalances:
            ...
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert result.balances["account1"] == 1000
    assert result.balances["account2"] == 2000


# LLM-generated content at query #34
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for the constructor
    mock_account = type('Account', (), {})()
    mock_amount = type('Amount', (), {'__init__': lambda self: None})()
    mock_posting = type('Posting', (), {
        'date': date(2023, 1, 15),
        'amount': mock_amount,
        'is_debit': True,
        'is_credit': False,
        'direction': 'debit',
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': [
                type('Post', (), {'account': mock_account, 'direction': 'credit'})()
            ]
        })()
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_quantity = type('Quantity', (), {})()

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )

    # Assert the constructor properly assigned all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #35
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="12345")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #36
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger(name="Test Ledger")
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #37
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity
    
    test_ledger = Ledger(name="Test Ledger")
    test_account = Account(name="Cash")
    test_amount = Amount(value=100.0, currency="USD")
    test_journal = Journal(description="Test transaction", postings=[])
    test_posting = Posting(
        date=date(2024, 1, 1),
        amount=test_amount,
        account=test_account,
        journal=test_journal,
        direction="debit"
    )
    test_balance = Quantity(value=500.0)
    
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance
    assert entry.ledger.name == "Test Ledger"
    assert entry.posting.date == date(2024, 1, 1)
    assert entry.balance.value == 500.0


# LLM-generated content at query #38
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Create test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    initial_balances = {}
    
    # Create a journal entry with postings
    entry = JournalEntry(date(2024, 6, 15), "Test entry", "source")
    
    # Create accounts
    account1 = Account("1000", "Assets", "Cash")
    account2 = Account("2000", "Liabilities", "Payables")
    
    # Add postings to the entry
    entry.post(date(2024, 6, 15), account1, Quantity(Decimal(100)))
    entry.post(date(2024, 6, 15), account2, Quantity(Decimal(-100)))
    
    # Build general ledger
    journal = [entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that ledgers were created for both accounts
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers) == 2
    assert isinstance(general_ledger.ledgers[account1], Ledger)
    assert isinstance(general_ledger.ledgers[account2], Ledger)
    assert general_ledger.ledgers[account1].account == account1
    assert general_ledger.ledgers[account2].account == account2


# LLM-generated content at query #39
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit"
    )
    balance = Quantity(100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #40
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_to_false():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    from pypara.accounting.posting import Posting
    
    # Create a journal entry with a date outside the accounting period
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Create a journal entry dated before the period starts
    entry = JournalEntry(date=date(2022, 12, 31), description="Test", source="test_source")
    
    account = Account(code="1000", title="Test Account")
    posting = Posting(entry, date(2022, 12, 31), account, Direction.INC, Amount(Quantity(Decimal(100))))
    entry.postings.append(posting)
    
    journal = [entry]
    initial = {}
    
    # Build the general ledger
    result = build_general_ledger(period, journal, initial)
    
    # The predicate at line 16 should evaluate to False because j.date (2022-12-31) is not within period.since <= j.date <= period.until
    # Therefore, the posting should not be added to any ledger
    assert len(result.ledgers) == 0


# LLM-generated content at query #41
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.commons import DateRange
    from pypara.accounting.postings import Account
    from pypara.core.monetary import Quantity
    
    # Create a date range for the period
    period_start = date(2024, 1, 1)
    period_end = date(2024, 1, 31)
    period = DateRange(period_start, period_end)
    
    # Create accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create journal entries with dates inside and outside the period
    entry_inside = JournalEntry(date(2024, 1, 15), "Entry inside period", "source1")
    entry_outside_before = JournalEntry(date(2023, 12, 31), "Entry before period", "source2")
    entry_outside_after = JournalEntry(date(2024, 2, 1), "Entry after period", "source3")
    
    # Add postings to entries
    entry_inside.post(date(2024, 1, 15), account1, Quantity(Decimal(100)))
    entry_inside.post(date(2024, 1, 15), account2, Quantity(Decimal(-100)))
    
    entry_outside_before.post(date(2023, 12, 31), account1, Quantity(Decimal(50)))
    entry_outside_before.post(date(2023, 12, 31), account2, Quantity(Decimal(-50)))
    
    entry_outside_after.post(date(2024, 2, 1), account1, Quantity(Decimal(75)))
    entry_outside_after.post(date(2024, 2, 1), account2, Quantity(Decimal(-75)))
    
    # Build the general ledger
    initial_balances = {}
    journal_entries = [entry_inside, entry_outside_before, entry_outside_after]
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Verify that only postings from the entry inside the period are included
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # The account1 ledger should have exactly 1 entry (from entry_inside)
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(100))
    
    # The account2 ledger should have exactly 1 entry (from entry_inside)
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(-100))


# LLM-generated content at query #42
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit"
    )
    balance = Quantity(500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.balance.value == 500.0


# LLM-generated content at query #43
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Transaction", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 15),
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="General Ledger")
    test_balance = MockQuantity(value=500.0)
    
    # Test constructor
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assertions
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #44
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalancesImpl:
        def __call__(self, period: DateRange) -> InitialBalances:
            test_balances = {
                "account1": 1000.0,
                "account2": 2000.0
            }
            return InitialBalances(test_balances)
    
    read_balances = ReadInitialBalancesImpl()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = read_balances(date_range)
    
    assert isinstance(result, InitialBalances)
    assert result.balances["account1"] == 1000.0
    assert result.balances["account2"] == 2000.0
    assert result.balances == {"account1": 1000.0, "account2": 2000.0}


# LLM-generated content at query #45
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger(name="General Ledger")
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == quantity


# LLM-generated content at query #46
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_ledger = object()
    
    mock_journal = object()
    mock_journal.description = "Test Description"
    mock_journal.postings = []
    
    mock_amount = object()
    mock_amount.__class__.__name__ = "Amount"
    
    mock_posting = object()
    mock_posting.date = date(2024, 1, 15)
    mock_posting.journal = mock_journal
    mock_posting.amount = mock_amount
    mock_posting.direction = "debit"
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_balance = object()
    mock_balance.__class__.__name__ = "Quantity"
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert that the constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #47
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Test Account")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #48
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        account: MockAccount
        amount: MockAmount
        date: date
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Set up test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_date = date(2023, 1, 1)
    test_posting = MockPosting(
        account=test_account,
        amount=test_amount,
        date=test_date,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=MockJournal(description="Test Journal", postings=[])
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #49
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        account: MockAccount
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create instances of mock objects
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_quantity = MockQuantity(value=100.0)
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=mock_amount,
        journal=MockJournal(description="Test Journal", postings=[mock_posting]),
        direction="debit",
        account=mock_account,
        is_debit=True,
        is_credit=False
    )
    mock_posting.journal.postings = [mock_posting]
    mock_ledger = MockLedger()

    # Test constructor
    from ledger_entry import LedgerEntry
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )

    assert ledger_entry.ledger is mock_ledger
    assert ledger_entry.posting is mock_posting
    assert ledger_entry.balance is mock_quantity


# LLM-generated content at query #50
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[dict]:
            return GeneralLedger[dict]({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []


# LLM-generated content at query #51
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Description", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        journal=journal,
        amount=amount,
        account=account,
        direction="debit"
    )
    balance = Quantity(value=500.0)
    ledger = Ledger()
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #52
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        journal: Journal

        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"

        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=journal
    )
    balance = Quantity(value=500.0)
    ledger = Ledger()

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #53
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.common import DateRange, Account, Quantity, Amount
    
    # Create a date range for the accounting period
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create an account
    account = Account("1000", "Cash")
    
    # Create initial balances
    initial = {account: Balance(period_start, Quantity(Decimal(1000)))}
    
    # Create journal entries with dates inside and outside the period
    entry_inside = JournalEntry(datetime.date(2024, 6, 15), "Inside period", "source1")
    entry_outside_before = JournalEntry(datetime.date(2023, 12, 31), "Before period", "source2")
    entry_outside_after = JournalEntry(datetime.date(2025, 1, 1), "After period", "source3")
    
    # Add postings to entries
    entry_inside.post(datetime.date(2024, 6, 15), account, Quantity(Decimal(100)))
    entry_outside_before.post(datetime.date(2023, 12, 31), account, Quantity(Decimal(50)))
    entry_outside_after.post(datetime.date(2025, 1, 1), account, Quantity(Decimal(75)))
    
    # Build the general ledger
    journal = [entry_inside, entry_outside_before, entry_outside_after]
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Verify that only the posting from the entry inside the period was added
    ledger = general_ledger.ledgers[account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.journal_entry.date == datetime.date(2024, 6, 15)


# LLM-generated content at query #54
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    ledger = Ledger()
    posting = Posting(
        date=date(2024, 1, 15),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit"
    )
    balance = Quantity(500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #55
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockLedger:
        name: str

    # Setup test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(value=100.0, currency="USD")
    test_account = MockAccount(name="Cash")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger(name="General Ledger")

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Verify constructor properly assigns all fields
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #56
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    # Create instances
    amount = MockAmount(value=100.0, currency="USD")
    account = MockAccount(name="TestAccount")
    journal = MockJournal(description="Test Journal Entry", postings=[])
    posting = MockPosting(
        date=date(2024, 1, 1),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    ledger = MockLedger(name="TestLedger")
    balance = MockAmount(value=500.0, currency="USD")
    
    # Test constructor
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #57
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        journal=journal,
        account=account,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #58
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Account, Quantity, Amount
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balance1 = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balance2 = Balance(date(2024, 1, 1), Quantity(Decimal(500)))
    initial_balances = {account1: initial_balance1, account2: initial_balance2}
    
    # Create journal entries
    entry1 = JournalEntry(date(2024, 1, 15), "Test entry 1", "source1")
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    entry1.post(date(2024, 1, 15), account2, Quantity(Decimal(100)))
    
    entry2 = JournalEntry(date(2024, 2, 15), "Test entry 2", "source2")
    entry2.post(date(2024, 2, 15), account1, Quantity(Decimal(50)))
    entry2.post(date(2024, 2, 15), account2, Quantity(Decimal(-50)))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balance1
    assert len(ledger1.entries) == 2
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balance2
    assert len(ledger2.entries) == 2
    
    # Check final balances
    assert ledger1.entries[0].balance == Quantity(Decimal(900))
    assert ledger1.entries[1].balance == Quantity(Decimal(950))
    assert ledger2.entries[0].balance == Quantity(Decimal(600))
    assert ledger2.entries[1].balance == Quantity(Decimal(550))


def test_build_general_ledger_with_account_not_in_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Account, Quantity
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balance1 = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance1}
    
    # Create journal entry with account not in initial balances
    entry1 = JournalEntry(date(2024, 1, 15), "Test entry", "source1")
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    entry1.post(date(2024, 1, 15), account2, Quantity(Decimal(100)))
    
    journal = [entry1]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 2
    assert account2 in general_ledger.ledgers
    
    # Check that account2 was created with zero initial balance
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal(0))
    assert len(ledger2.entries) == 1


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Account, Quantity
    
    # Setup test data with period that excludes some entries
    period = DateRange(date(2024, 2, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    
    initial_balance1 = Balance(date(2024, 2, 1), Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance1}
    
    # Create journal entries - one outside period, one inside
    entry1 = JournalEntry(date(2024, 1, 15), "Entry outside period", "source1")
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    
    entry2 = JournalEntry(date(2024, 3, 15), "Entry inside period", "source2")
    entry2.post(date(2024, 3, 15), account1, Quantity(Decimal(50)))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions - only entry2 should be included
    assert len(general_ledger.ledgers) == 1
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal(1050))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Account, Quantity
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    
    initial_balance1 = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance1}
    
    # Build general ledger with empty journal
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 1
    ledger1 = general_ledger.ledgers[account1]
    assert len(ledger1.entries) == 0
    assert ledger1.initial == initial_balance1


# LLM-generated content at query #59
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')
    
    # Create mock objects
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting(Generic[_T]):
        account: MockAccount
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger(Generic[_T]):
        name: str

    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_date = date(2024, 1, 15)
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Journal Entry", postings=[])
    test_posting = MockPosting(
        account=test_account,
        date=test_date,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="Test Ledger")

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assertions
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity
    assert ledger_entry.ledger.name == "Test Ledger"
    assert ledger_entry.posting.account.name == "Test Account"
    assert ledger_entry.balance.value == 100.0


# LLM-generated content at query #60
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Quantity, Balance
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.core.commons import DateRange
    
    # Setup
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {account1: Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    # Create a journal entry with postings
    source = "test_source"
    entry = JournalEntry(date(2023, 6, 15), "Test entry", source)
    entry.post(date(2023, 6, 15), account1, Quantity(Decimal(-100)))
    entry.post(date(2023, 6, 15), account2, Quantity(Decimal(100)))
    
    journal = [entry]
    
    # Execute
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify the predicate at line 1 (function returns GeneralLedger instance)
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers) == 2


# LLM-generated content at query #61
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for testing
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting(Generic[_T]):
        date: date
        journal: MockJournal
        amount: MockAmount
        direction: str
        account: MockAccount

        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"

        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class MockLedger(Generic[_T]):
        pass

    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_date = date(2023, 1, 15)
    
    counter_posting = MockPosting(
        date=test_date,
        journal=None,
        amount=test_amount,
        direction="credit",
        account=test_account
    )
    
    test_journal = MockJournal(
        description="Test Transaction",
        postings=[counter_posting]
    )
    
    test_posting = MockPosting(
        date=test_date,
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        account=test_account
    )
    
    test_ledger = MockLedger()
    
    # Test constructor
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assertions
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #62
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("Cash")
    test_quantity = MockQuantity(500.0)
    test_ledger = MockLedger()
    
    test_journal = MockJournal(
        description="Test transaction",
        postings=[]
    )
    
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assert constructor initialized all attributes correctly
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #63
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import Quantity, DateRange
    from pypara.accounting.accounts import Account
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    # Create accounts
    account_cash = Account("1000", "Cash")
    account_revenue = Account("4000", "Revenue")
    
    # Create initial balances
    initial_balances = {
        account_cash: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("1000")))
    }
    
    # Create journal entries
    entry1 = JournalEntry(
        date=datetime.date(2024, 1, 15),
        description="Initial deposit",
        source=None
    )
    entry1.post(datetime.date(2024, 1, 15), account_cash, Quantity(Decimal("500")))
    entry1.post(datetime.date(2024, 1, 15), account_revenue, Quantity(Decimal("-500")))
    
    entry2 = JournalEntry(
        date=datetime.date(2024, 2, 10),
        description="Revenue earned",
        source=None
    )
    entry2.post(datetime.date(2024, 2, 10), account_cash, Quantity(Decimal("300")))
    entry2.post(datetime.date(2024, 2, 10), account_revenue, Quantity(Decimal("-300")))
    
    # Build general ledger
    journal_entries = [entry1, entry2]
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account_cash in general_ledger.ledgers
    assert account_revenue in general_ledger.ledgers
    
    # Check cash ledger
    cash_ledger = general_ledger.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial.value == Quantity(Decimal("1000"))
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1800"))
    
    # Check revenue ledger
    revenue_ledger = general_ledger.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert revenue_ledger.initial.value == Quantity(Decimal("0"))
    assert len(revenue_ledger.entries) == 2
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500"))
    assert revenue_ledger.entries[1].balance == Quantity(Decimal("-800"))


def test_build_general_ledger_empty_journal():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import Quantity, DateRange
    from pypara.accounting.accounts import Account
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account_cash = Account("1000", "Cash")
    
    initial_balances = {
        account_cash: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("5000")))
    }
    
    # Build general ledger with empty journal
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account_cash in general_ledger.ledgers
    assert len(general_ledger.ledgers[account_cash].entries) == 0
    assert general_ledger.ledgers[account_cash].initial.value == Quantity(Decimal("5000"))


def test_build_general_ledger_outside_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import Quantity, DateRange
    from pypara.accounting.accounts import Account
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account_cash = Account("1000", "Cash")
    
    initial_balances = {
        account_cash: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("1000")))
    }
    
    # Create journal entry outside period
    entry = JournalEntry(
        date=datetime.date(2023, 12, 15),
        description="Before period",
        source=None
    )
    entry.post(datetime.date(2023, 12, 15), account_cash, Quantity(Decimal("100")))
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [entry], initial_balances)
    
    # Assertions - entry should not be included
    assert len(general_ledger.ledgers[account_cash].entries) == 0


def test_build_general_ledger_creates_new_accounts():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import Quantity, DateRange
    from pypara.accounting.accounts import Account
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account_cash = Account("1000", "Cash")
    account_expense = Account("5000", "Expense")
    
    initial_balances = {
        account_cash: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("1000")))
    }
    
    # Create journal entry with new account
    entry = JournalEntry(
        date=datetime.date(2024, 6, 15),
        description="Expense",
        source=None
    )
    entry.post(datetime.date(2024, 6, 15), account_expense, Quantity(Decimal("200")))
    entry.post(datetime.date(2024, 6, 15), account_cash, Quantity(Decimal("-200")))
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [entry], initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 2
    assert account_expense in general_ledger.ledgers
    assert general_ledger.ledgers[account_expense].initial.value == Quantity(Decimal("0"))
    assert len(general_ledger.ledgers[account_expense].entries) == 1


# LLM-generated content at query #64
#--------------------------

```python
def test_build_general_ledger_posting_account_not_in_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Direction, Posting, Amount
    from pypara.accounting.generic import Account, Balance, Quantity, DateRange
    from pypara.accounting.ledger import build_general_ledger
    
    # Setup
    test_date = date(2024, 1, 15)
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account = Account("1000", "Test Account")
    
    # Create a journal entry with a posting
    journal_entry = JournalEntry(
        date=test_date,
        description="Test entry",
        source="test_source"
    )
    journal_entry.post(test_date, account, Quantity(Decimal("100")))
    
    # Build general ledger with empty initial balances
    initial = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial)
    
    # At line 18, the predicate `posting.account not in ledgers` should evaluate to False
    # because after the first posting is processed, the account should be in ledgers
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1


# LLM-generated content at query #65
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        account: MockAccount
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2023, 1, 15)
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=1000.0)
    test_ledger = MockLedger()
    test_journal = MockJournal(description="Test Entry", postings=[])
    
    test_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity


# LLM-generated content at query #66
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        account: MockAccount
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("Assets")
    test_journal = MockJournal("Test transaction", [])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account,
        journal=test_journal
    )
    test_balance = MockQuantity(500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #67
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test instances
    test_date = date(2024, 1, 15)
    test_account = MockAccount(name="Assets")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockAmount(value=500.0, currency="USD")
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor sets attributes correctly
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #68
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Test Account")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        journal=journal,
        account=account,
        direction="debit"
    )
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #69
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_account = type('Account', (), {})()
    mock_journal = type('Journal', (), {'description': 'Test Journal', 'postings': []})()
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'account': mock_account,
        'journal': mock_journal,
        'amount': 100.0,
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = 500.0
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance == mock_balance
    assert entry.ledger is not None
    assert entry.posting is not None
    assert entry.balance is not None


# LLM-generated content at query #70
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        journal: Journal

        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"

        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    # Create test data
    test_account = Account(name="Test Account")
    test_amount = Amount(value=100.0, currency="USD")
    test_balance = Quantity(value=100.0)
    test_date = date(2023, 1, 1)
    
    counter_posting = Posting(
        date=test_date,
        account=Account(name="Counter Account"),
        amount=test_amount,
        direction="credit",
        journal=None
    )
    
    test_posting = Posting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        direction="debit",
        journal=Journal(description="Test Journal", postings=[counter_posting])
    )
    
    test_ledger = Ledger()

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_balance
    assert ledger_entry.date == test_date
    assert ledger_entry.description == "Test Journal"
    assert ledger_entry.amount == test_amount
    assert ledger_entry.is_debit is True
    assert ledger_entry.is_credit is False
    assert ledger_entry.debit == test_amount
    assert ledger_entry.credit is None


# LLM-generated content at query #71
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, journal=journal, account=account, direction="debit")
    ledger = Ledger()
    balance = Quantity(value=100.0)

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert ledger_entry.ledger is ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is balance
    assert ledger_entry.date == date(2023, 1, 1)
    assert ledger_entry.description == "Test Journal"
    assert ledger_entry.amount == amount


# LLM-generated content at query #72
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create test data
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test Description", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 15),
        amount=mock_amount,
        journal=mock_journal,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger()
    mock_quantity = MockQuantity(value=1000.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_quantity)
    
    # Assert constructor properly assigns attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #73
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_account = type('Account', (), {})()
    mock_amount = type('Amount', (), {'value': Decimal('100.00')})()
    mock_posting = type('Posting', (), {
        'account': mock_account,
        'amount': mock_amount,
        'date': date(2023, 1, 15),
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False,
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': []
        })()
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {'value': Decimal('500.00')})()
    
    # Test constructor
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assertions
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #74
#--------------------------

```python
def test_build_general_ledger_predicate_false():
    from decimal import Decimal
    import datetime
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Account, Quantity, Posting, Amount
    
    # Create a date range
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create a journal entry with a date BEFORE the period
    entry_date = datetime.date(2023, 12, 31)
    account = Account("TestAccount")
    
    entry = JournalEntry(entry_date, "Test Entry", "source")
    entry.post(entry_date, account, Quantity(Decimal(100)))
    
    journal = [entry]
    initial = {}
    
    # Build the general ledger
    result = build_general_ledger(period, journal, initial)
    
    # The predicate "period.since <= j.date <= period.until" should be False
    # because entry_date (2023-12-31) is before period_start (2024-01-01)
    # Therefore, the account should not be in the ledgers
    assert account not in result.ledgers


# LLM-generated content at query #75
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="12345")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_different_values():
    account = Account(name="Savings", number="67890")
    initial_balance = Balance(value=Quantity(5000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.number == "67890"
    assert ledger.initial.value == Quantity(5000)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #76
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2024, 1, 1),
        journal=journal,
        amount=amount,
        account=account,
        direction="debit"
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #77
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        journal=journal,
        account=account,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #78
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Setup test data
    test_date = date(2024, 1, 15)
    period = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account1: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00"))),
        account2: Balance(date(2024, 1, 1), Quantity(Decimal("500.00")))
    }
    
    # Create a journal entry
    entry = JournalEntry(test_date, "Test transaction", "source_object")
    entry.post(test_date, account1, Quantity(Decimal("-100.00")))
    entry.post(test_date, account2, Quantity(Decimal("100.00")))
    
    journal = [entry]
    
    # Build the general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert general_ledger.ledgers[account1].account == account1
    assert general_ledger.ledgers[account2].account == account2
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal("1000.00"))
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal("500.00"))
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1


def test_build_general_ledger_creates_missing_accounts():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    test_date = date(2024, 1, 15)
    period = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account1: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))
    }
    
    entry = JournalEntry(test_date, "Test transaction", "source_object")
    entry.post(test_date, account1, Quantity(Decimal("-100.00")))
    entry.post(test_date, account2, Quantity(Decimal("100.00")))
    
    journal = [entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert account2 in general_ledger.ledgers
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal("0.00"))


def test_build_general_ledger_filters_by_period():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    period = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    account = Account("1000", "Cash")
    
    initial_balances = {
        account: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))
    }
    
    # Entry within period
    entry_in = JournalEntry(date(2024, 1, 15), "In period", "source1")
    entry_in.post(date(2024, 1, 15), account, Quantity(Decimal("100.00")))
    
    # Entry outside period
    entry_out = JournalEntry(date(2024, 2, 15), "Out of period", "source2")
    entry_out.post(date(2024, 2, 15), account, Quantity(Decimal("200.00")))
    
    journal = [entry_in, entry_out]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting.date == date(2024, 1, 15)


def test_build_general_ledger_empty_journal():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    period = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    account = Account("1000", "Cash")
    
    initial_balances = {
        account: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 0
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal("1000.00"))


# LLM-generated content at query #79
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar, Generic
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger(Generic[_T]):
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[str]:
            return GeneralLedger(f"Ledger from {period.start} to {period.end}")
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data == "Ledger from 2024-01-01 to 2024-12-31"


# LLM-generated content at query #80
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    from typing import NamedTuple
    
    class DateRange(NamedTuple):
        start: date
        end: date
    
    class InitialBalances(NamedTuple):
        balances: dict
    
    class ReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balances={"account1": 1000, "account2": 2000})
    
    read_balances = ReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = read_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert period.start == date(2023, 1, 1)
    assert period.end == date(2023, 12, 31)


def test_read_initial_balances_call_empty_balances():
    from datetime import date
    from typing import NamedTuple
    
    class DateRange(NamedTuple):
        start: date
        end: date
    
    class InitialBalances(NamedTuple):
        balances: dict
    
    class ReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balances={})
    
    read_balances = ReadInitialBalances()
    period = DateRange(start=date(2023, 6, 1), end=date(2023, 6, 30))
    result = read_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {}


def test_read_initial_balances_call_multiple_accounts():
    from datetime import date
    from typing import NamedTuple
    
    class DateRange(NamedTuple):
        start: date
        end: date
    
    class InitialBalances(NamedTuple):
        balances: dict
    
    class ReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balances={"acc1": 100, "acc2": 200, "acc3": 300, "acc4": 400})
    
    read_balances = ReadInitialBalances()
    period = DateRange(start=date(2024, 1, 1), end=date(2024, 3, 31))
    result = read_balances(period)
    
    assert len(result.balances) == 4
    assert result.balances["acc1"] == 100
    assert result.balances["acc4"] == 400


# LLM-generated content at query #81
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    # Create mock objects for testing
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting(Generic[_T]):
        date: date
        account: MockAccount
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger(Generic[_T]):
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Description", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_balance


# LLM-generated content at query #82
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return self._journal
        
        def set_journal(self, journal):
            self._journal = journal
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    account = MockAccount(name="Test Account")
    amount = MockAmount(value=100.0, currency="USD")
    posting = MockPosting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    journal = MockJournal(description="Test Description", postings=[posting])
    posting.set_journal(journal)
    
    balance = MockQuantity(value=500.0)
    ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #83
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Cash")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2024, 1, 15),
        amount=Amount(value=100.0, currency="USD"),
        journal=journal,
        account=account,
        direction="debit"
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance
    assert entry.ledger.name == "General Ledger"
    assert entry.posting.date == date(2024, 1, 15)
    assert entry.balance.value == 500.0


# LLM-generated content at query #84
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2023, 1, 1), account=account, amount=amount, direction="debit", journal=journal)
    ledger = Ledger(name="General Ledger")

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == quantity


# LLM-generated content at query #85
#--------------------------

```python
def test_ledger_entry_constructor():
    from dataclasses import dataclass
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: Decimal
        currency: str
    
    @dataclass
    class MockQuantity:
        value: Decimal
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=Decimal("100.00"), currency="USD")
    test_quantity = MockQuantity(value=Decimal("100.00"))
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    
    # Construct LedgerEntry
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assert constructor sets all attributes correctly
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #86
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.accounting.generic import Quantity
    from pypara.core.timeline import DateRange

    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000))),
        account2: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(500))),
    }
    
    # Create journal entries
    source_obj = object()
    entry1 = JournalEntry(datetime.date(2023, 1, 15), "Test entry 1", source_obj)
    entry1.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(-100)))
    entry1.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal(100)))
    
    entry2 = JournalEntry(datetime.date(2023, 2, 20), "Test entry 2", source_obj)
    entry2.post(datetime.date(2023, 2, 20), account1, Quantity(Decimal(50)))
    entry2.post(datetime.date(2023, 2, 20), account2, Quantity(Decimal(-50)))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal(1000))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal(900))
    assert ledger1.entries[1].balance == Quantity(Decimal(950))
    
    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal(500))
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal(600))
    assert ledger2.entries[1].balance == Quantity(Decimal(550))


def test_build_general_ledger_with_new_accounts():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.accounts import Account
    from pypara.core.timeline import DateRange

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    initial_balances = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(5000)))}
    
    source_obj = object()
    entry = JournalEntry(datetime.date(2023, 3, 10), "Revenue entry", source_obj)
    entry.post(datetime.date(2023, 3, 10), account1, Quantity(Decimal(1000)))
    entry.post(datetime.date(2023, 3, 10), account2, Quantity(Decimal(-1000)))
    
    journal = [entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert account2 in general_ledger.ledgers
    
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.initial.date == period.since
    assert ledger2.initial.value == Quantity(Decimal(0))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal(-1000))


def test_build_general_ledger_outside_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.accounts import Account
    from pypara.core.timeline import DateRange

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    
    initial_balances = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    source_obj = object()
    entry_before = JournalEntry(datetime.date(2022, 12, 31), "Before period", source_obj)
    entry_before.post(datetime.date(2022, 12, 31), account1, Quantity(Decimal(100)))
    
    entry_after = JournalEntry(datetime.date(2024, 1, 1), "After period", source_obj)
    entry_after.post(datetime.date(2024, 1, 1), account1, Quantity(Decimal(100)))
    
    journal = [entry_before, entry_after]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[account1].entries) == 0


def test_build_general_ledger_empty_journal():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.accounts import Account
    from pypara.core.timeline import DateRange

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    
    initial_balances = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    journal = []
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[account1].entries) == 0
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal(1000))


# LLM-generated content at query #87
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test entry", postings=[])
    posting = Posting(
        date=date(2023, 1, 15),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #88
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_account = type('Account', (), {})()
    mock_journal = type('Journal', (), {'description': 'Test transaction', 'postings': []})()
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'account': mock_account,
        'journal': mock_journal,
        'amount': type('Amount', (), {})(),
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {})()
    
    # Test constructor
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #89
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_ledger = object()
    mock_account = object()
    mock_journal = object()
    mock_posting = object()
    mock_amount = object()
    mock_balance = object()
    
    # Set up posting mock with necessary attributes
    posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'account': mock_account,
        'journal': mock_journal,
        'amount': mock_amount,
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False
    })()
    
    balance = type('Quantity', (), {})()
    ledger = type('Ledger', (), {})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    # Assert constructor correctly assigns attributes
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #90
#--------------------------

```python
def test_build_general_ledger_posting_account_not_in_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.generic import Balance, Account, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange
    
    # Create test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    # Create a journal entry with postings
    entry = JournalEntry(date(2024, 6, 15), "Test entry", "source")
    entry.post(date(2024, 6, 15), account1, Quantity(Decimal(100)))
    entry.post(date(2024, 6, 15), account2, Quantity(Decimal(-100)))
    
    # Initial balances only for account1
    initial = {account1: Balance(period.since, Quantity(Decimal(50)))}
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [entry], initial)
    
    # At line 18, the predicate "posting.account not in ledgers" should evaluate to False
    # for account1 (which is in initial balances) and True for account2 (which is not)
    # We verify that account2 was added to ledgers (meaning the predicate was True for it)
    assert account2 in general_ledger.ledgers
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers) == 2


# LLM-generated content at query #91
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalancesImpl:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    read_balances = ReadInitialBalancesImpl()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = read_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert period.start == date(2023, 1, 1)
    assert period.end == date(2023, 12, 31)


# LLM-generated content at query #92
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')
    
    # Create mock objects for dependencies
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor sets all fields correctly
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #93
#--------------------------

```python
def test_build_general_ledger_predicate_filters_postings_within_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Account, Quantity, DateRange
    from pypara.accounting.ledger import build_general_ledger
    
    # Create test data
    account1 = Account("1000", "Test Account 1")
    account2 = Account("2000", "Test Account 2")
    
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 1, 31)
    period = DateRange(period_start, period_end)
    
    # Create journal entries with dates inside and outside the period
    entry_inside = JournalEntry(datetime.date(2024, 1, 15), "Inside period", "source1")
    entry_outside_before = JournalEntry(datetime.date(2023, 12, 31), "Before period", "source2")
    entry_outside_after = JournalEntry(datetime.date(2024, 2, 1), "After period", "source3")
    
    # Add postings to entries
    entry_inside.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(100)))
    entry_inside.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal(-100)))
    
    entry_outside_before.post(datetime.date(2023, 12, 31), account1, Quantity(Decimal(50)))
    entry_outside_before.post(datetime.date(2023, 12, 31), account2, Quantity(Decimal(-50)))
    
    entry_outside_after.post(datetime.date(2024, 2, 1), account1, Quantity(Decimal(75)))
    entry_outside_after.post(datetime.date(2024, 2, 1), account2, Quantity(Decimal(-75)))
    
    # Create initial balances
    initial = {
        account1: Balance(period_start, Quantity(Decimal(0))),
        account2: Balance(period_start, Quantity(Decimal(0)))
    }
    
    # Build general ledger with all entries
    journal = [entry_outside_before, entry_inside, entry_outside_after]
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Verify that only postings within the period are included
    # Entry inside period should have 2 postings (1 debit, 1 credit)
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    
    # Verify the postings are from the inside entry only
    assert general_ledger.ledgers[account1].entries[0].posting.journal_entry.date == datetime.date(2024, 1, 15)
    assert general_ledger.ledgers[account2].entries[0].posting.journal_entry.date == datetime.date(2024, 1, 15)


# LLM-generated content at query #94
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    # Create test data
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2024, 1, 1),
        journal=journal,
        amount=amount,
        account=account,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=100.0)
    
    # Test constructor
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #95
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=1000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_different_values():
    account = Account(name="Savings", account_type="LIABILITY")
    initial_balance = Balance(value=5000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.initial.value == 5000
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #96
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar, Generic
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger(Generic[_T]):
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[dict]:
            return GeneralLedger({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []


# LLM-generated content at query #97
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 15),
        amount=Amount(100.0, "USD"),
        journal=Journal("Test transaction", []),
        account=Account("Cash"),
        direction="debit"
    )
    balance = Quantity(500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #98
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        account: MockAccount
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("Cash")
    test_journal = MockJournal("Test transaction", [])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account,
        journal=test_journal
    )
    test_balance = MockQuantity(500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns values
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #99
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_ledger = object()
    
    mock_journal = object()
    mock_journal.description = "Test transaction"
    mock_journal.postings = []
    
    mock_amount = object()
    mock_amount.value = 100
    
    mock_posting = object()
    mock_posting.date = date(2024, 1, 15)
    mock_posting.journal = mock_journal
    mock_posting.amount = mock_amount
    mock_posting.direction = "debit"
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_balance = object()
    mock_balance.value = 1000
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly assigned all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #100
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return self._journal
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        name: str
    
    # Setup test data
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test Description", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=mock_amount,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_posting._journal = mock_journal
    mock_quantity = MockQuantity(value=100.0)
    mock_ledger = MockLedger(name="Test Ledger")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor properly assigns attributes
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_quantity
    assert entry.ledger.name == "Test Ledger"
    assert entry.posting.date == date(2023, 1, 1)
    assert entry.balance.value == 100.0


# LLM-generated content at query #101
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    posting_obj = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        journal=Journal(description="Test Entry", postings=[]),
        direction="debit"
    )
    ledger_obj = Ledger()
    balance = Quantity(value=100.0)

    ledger_entry = LedgerEntry(
        ledger=ledger_obj,
        posting=posting_obj,
        balance=balance
    )

    assert ledger_entry.ledger == ledger_obj
    assert ledger_entry.posting == posting_obj
    assert ledger_entry.balance == balance


# LLM-generated content at query #102
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.core import Account, Quantity, DateRange, Amount
    
    # Setup test data
    test_date = date(2023, 1, 15)
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(1000)))
    }
    
    # Create a journal entry with postings
    entry = JournalEntry(test_date, "Test transaction", "source_obj")
    entry.post(test_date, account1, Quantity(Decimal(-100)))
    entry.post(test_date, account2, Quantity(Decimal(100)))
    
    journal = [entry]
    
    # Build general ledger
    result = build_general_ledger(period, journal, initial_balances)
    
    # Verify the result is a GeneralLedger instance
    assert isinstance(result, GeneralLedger)
    
    # Verify the period is set correctly
    assert result.period == period
    
    # Verify ledgers were created for all accounts
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    
    # Verify ledger entries were added
    assert len(result.ledgers[account1].entries) > 0
    assert len(result.ledgers[account2].entries) > 0
    
    # Verify the initial balance for account1
    assert result.ledgers[account1].initial.value == Quantity(Decimal(1000))
    
    # Verify account2 was created with zero initial balance
    assert result.ledgers[account2].initial.value == Quantity(Decimal(0))


# LLM-generated content at query #103
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    # Create test data
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=journal
    )
    ledger = Ledger(name="Test Ledger")

    # Test constructor
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == quantity
    assert entry.date == date(2023, 1, 1)
    assert entry.amount == amount
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #104
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Test Account")
    posting_obj = Posting(
        date=date(2023, 1, 1),
        journal=Journal(description="Test Journal", postings=[]),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        direction="debit"
    )
    ledger_obj = Ledger()
    balance_qty = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger_obj, posting=posting_obj, balance=balance_qty)

    assert entry.ledger is ledger_obj
    assert entry.posting is posting_obj
    assert entry.balance is balance_qty


# LLM-generated content at query #105
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return MockJournal()
    
    @dataclass
    class MockJournal:
        description: str = "Test Description"
        
        @property
        def postings(self):
            return []
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create instances
    mock_ledger = MockLedger()
    mock_posting = MockPosting(
        date=date(2024, 1, 15),
        amount=MockAmount(100.0, "USD"),
        account=MockAccount("Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockQuantity(500.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor assignments
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #106
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar("_T")

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        account: Account
        date: date
        journal: Journal
        amount: Amount
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    account = Account(name="Test Account")
    journal = Journal(description="Test Description", postings=[])
    posting = Posting(
        account=account,
        date=date(2023, 1, 1),
        journal=journal,
        amount=Amount(value=100.0, currency="USD"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #107
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test data
    test_date = date(2024, 1, 15)
    test_amount = MockAmount(value=100.0, currency="USD")
    test_account = MockAccount(name="Cash")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockAmount(value=500.0, currency="USD")
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns all fields
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_balance


# LLM-generated content at query #108
#--------------------------

```python
def test_read_initial_balances_protocol_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    reader = ConcreteReadInitialBalances()
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert result.balances["account1"] == 1000
    assert result.balances["account2"] == 2000


# LLM-generated content at query #109
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.temporal import DateRange
    
    # Setup
    period = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    account = Account("1000", "Test Account")
    
    # Create journal entries with different dates
    entry_before = JournalEntry(date(2023, 12, 31), "Before period", "source1")
    entry_within = JournalEntry(date(2024, 1, 15), "Within period", "source2")
    entry_after = JournalEntry(date(2024, 2, 1), "After period", "source3")
    
    # Add postings to entries
    entry_before.post(date(2023, 12, 31), account, Quantity(Decimal(100)))
    entry_within.post(date(2024, 1, 15), account, Quantity(Decimal(50)))
    entry_after.post(date(2024, 2, 1), account, Quantity(Decimal(75)))
    
    journal = [entry_before, entry_within, entry_after]
    initial = {account: Balance(period.since, Quantity(Decimal(0)))}
    
    # Execute
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Assert - line 16 predicate: period.since <= j.date <= period.until
    # Only the entry within period should be in the ledger
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting.journal.date == date(2024, 1, 15)


# LLM-generated content at query #110
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Amount, Quantity
    from pypara.accounting.accounts import Account

    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create accounts
    account_a = Account(name="Account A", code="001")
    account_b = Account(name="Account B", code="002")
    
    # Create initial balances
    initial_balances = {
        account_a: Balance(start_date, Quantity(Decimal(1000))),
        account_b: Balance(start_date, Quantity(Decimal(500)))
    }
    
    # Create journal entries with postings
    je1 = JournalEntry(date=datetime.date(2023, 6, 15), description="Test entry 1", source="test_source_1")
    je1.post(datetime.date(2023, 6, 15), account_a, Quantity(Decimal(-100)))
    je1.post(datetime.date(2023, 6, 15), account_b, Quantity(Decimal(100)))
    
    je2 = JournalEntry(date=datetime.date(2023, 7, 20), description="Test entry 2", source="test_source_2")
    je2.post(datetime.date(2023, 7, 20), account_a, Quantity(Decimal(50)))
    je2.post(datetime.date(2023, 7, 20), account_b, Quantity(Decimal(-50)))
    
    journal = [je1, je2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account_a in general_ledger.ledgers
    assert account_b in general_ledger.ledgers
    
    # Check ledger for account_a
    ledger_a = general_ledger.ledgers[account_a]
    assert ledger_a.account == account_a
    assert ledger_a.initial.value == Quantity(Decimal(1000))
    assert len(ledger_a.entries) == 2
    
    # Check ledger for account_b
    ledger_b = general_ledger.ledgers[account_b]
    assert ledger_b.account == account_b
    assert ledger_b.initial.value == Quantity(Decimal(500))
    assert len(ledger_b.entries) == 2
    
    # Check final balances
    assert ledger_a.entries[-1].balance == Quantity(Decimal(950))
    assert ledger_b.entries[-1].balance == Quantity(Decimal(550))


def test_build_general_ledger_with_new_accounts():
    import datetime
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    from pypara.accounting.accounts import Account

    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create accounts
    account_a = Account(name="Account A", code="001")
    account_b = Account(name="Account B", code="002")
    account_c = Account(name="Account C", code="003")
    
    # Create initial balances (only for account_a and account_b)
    initial_balances = {
        account_a: Balance(start_date, Quantity(Decimal(1000))),
        account_b: Balance(start_date, Quantity(Decimal(500)))
    }
    
    # Create journal entries with posting to new account_c
    je = JournalEntry(date=datetime.date(2023, 6, 15), description="Test entry", source="test_source")
    je.post(datetime.date(2023, 6, 15), account_a, Quantity(Decimal(-200)))
    je.post(datetime.date(2023, 6, 15), account_c, Quantity(Decimal(200)))
    
    journal = [je]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 3
    assert account_c in general_ledger.ledgers
    
    # Check that account_c was created with zero initial balance
    ledger_c = general_ledger.ledgers[account_c]
    assert ledger_c.initial.value == Quantity(Decimal(0))
    assert len(ledger_c.entries) == 1
    assert ledger_c.entries[0].balance == Quantity(Decimal(200))


def test_build_general_ledger_empty_journal():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    from pypara.accounting.accounts import Account

    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create accounts
    account_a = Account(name="Account A", code="001")
    
    # Create initial balances
    initial_balances = {
        account_a: Balance(start_date, Quantity(Decimal(1000)))
    }
    
    # Empty journal
    journal = []
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 1
    assert account_a in general_ledger.ledgers
    assert len(general_ledger.ledgers[account_a].entries) == 0
    assert general_ledger.ledgers[account_a].initial.value == Quantity(Decimal(1000))


def test_build_general_ledger_outside_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.commons import DateRange
    from pypara.core.monetary import Quantity
    from pypara.accounting.accounts import Account

    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create accounts
    account_a = Account(name="Account A", code="001")
    account_b = Account(name="Account B", code="002")
    
    # Create initial balances
    initial_balances = {
        account_a: Balance(start_date, Quantity(Decimal(1000))),
        account_b: Balance(start_date, Quantity(Decimal(500)))
    }
    
    # Create journal entry outside the period
    je = JournalEntry(date=datetime.date(2024, 6, 15), description="Outside period", source="test_source")


# LLM-generated content at query #111
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger_instance = Ledger()
    posting_instance = Posting(
        date=date(2023, 1, 1),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance_instance = Quantity(500.0)

    entry = LedgerEntry(
        ledger=ledger_instance,
        posting=posting_instance,
        balance=balance_instance
    )

    assert entry.ledger is ledger_instance
    assert entry.posting is posting_instance
    assert entry.balance is balance_instance
    assert entry.balance.value == 500.0


# LLM-generated content at query #112
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        account: MockAccount
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return MockJournal()
    
    @dataclass
    class MockJournal:
        description: str = "Test Description"
        postings: list = None
        
        def __post_init__(self):
            if self.postings is None:
                self.postings = []
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create instances
    test_ledger = MockLedger()
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        account=MockAccount(name="Cash"),
        amount=MockAmount(value=100, currency="USD"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = MockQuantity(value=1000)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #113
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=1000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #114
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        account: MockAccount
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Journal Entry", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        account=test_account,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor properly assigned all fields
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #115
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    
    # Test constructor with all required parameters
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Verify all attributes are set correctly
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_ledger = object()
    mock_account = object()
    mock_journal = object()
    
    # Create a Posting object
    posting = Posting(
        account=mock_account,
        journal=mock_journal,
        amount=Amount(Decimal("100.00"), "USD"),
        date=date(2024, 1, 15),
        direction=Direction.DEBIT
    )
    
    # Create a Quantity object
    balance = Quantity(Decimal("500.00"), "USD")
    
    # Create a LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=posting,
        balance=balance
    )
    
    # Assert that the constructor properly assigned all attributes
    assert ledger_entry.ledger is mock_ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is balance


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        name: str
    
    # Set up test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(value=100.0, currency="USD")
    test_account = MockAccount(name="Cash")
    test_journal = MockJournal(description="Test Transaction", postings=[])
    
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger(name="General Ledger")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns attributes
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    # Create mock objects for dependencies
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert that all fields are correctly assigned
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    mock_account = type('Account', (), {})()
    mock_journal = type('Journal', (), {'description': 'Test transaction', 'postings': []})()
    mock_amount = type('Amount', (), {})()
    mock_quantity = type('Quantity', (), {})()
    
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'account': mock_account,
        'journal': mock_journal,
        'amount': mock_amount,
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False
    })()
    
    mock_ledger = type('Ledger', (), {})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for the constructor parameters
    mock_ledger = object()
    mock_journal = object()
    mock_account = object()
    
    # Create a Posting object
    posting = Posting(
        account=mock_account,
        journal=mock_journal,
        amount=Amount(100, "USD"),
        direction=Direction.DEBIT,
        date=date(2023, 1, 1)
    )
    
    # Create a Quantity object
    balance = Quantity(500, "USD")
    
    # Create a LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=posting,
        balance=balance
    )
    
    # Assert that the constructor properly assigned the attributes
    assert ledger_entry.ledger is mock_ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is balance


# LLM-generated content at query #6
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        account: MockAccount
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class MockLedger:
        pass

    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        account=test_account
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger()

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance
    assert entry.date == date(2023, 1, 1)
    assert entry.amount == test_amount


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="123456")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #8
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Quantity, Account
    from decimal import Decimal
    import datetime
    
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    journal = []
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances():
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Quantity, Account
    from decimal import Decimal
    import datetime
    
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    journal = []
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Liabilities")
    initial = {
        account1: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("1000"))),
        account2: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("500")))
    }
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].initial.value == Quantity(Decimal("1000"))
    assert result.ledgers[account2].initial.value == Quantity(Decimal("500"))


def test_build_general_ledger_with_postings_in_period():
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Quantity, Account, Direction, Amount
    from pypara.accounting.journaling import Posting
    from decimal import Decimal
    import datetime
    
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    entry = JournalEntry(datetime.date(2024, 6, 15), "Test entry", "source")
    posting1 = Posting(entry, datetime.date(2024, 6, 15), account1, Direction.INC, Amount(Decimal("100")))
    posting2 = Posting(entry, datetime.date(2024, 6, 15), account2, Direction.DEC, Amount(Decimal("100")))
    entry.postings.append(posting1)
    entry.postings.append(posting2)
    
    journal = [entry]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


def test_build_general_ledger_filters_postings_outside_period():
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Quantity, Account, Direction, Amount
    from pypara.accounting.journaling import Posting
    from decimal import Decimal
    import datetime
    
    period = DateRange(datetime.date(2024, 6, 1), datetime.date(2024, 6, 30))
    account1 = Account("1000", "Cash")
    
    entry_before = JournalEntry(datetime.date(2024, 5, 15), "Before period", "source")
    posting_before = Posting(entry_before, datetime.date(2024, 5, 15), account1, Direction.INC, Amount(Decimal("50")))
    entry_before.postings.append(posting_before)
    
    entry_in = JournalEntry(datetime.date(2024, 6, 15), "In period", "source")
    posting_in = Posting(entry_in, datetime.date(2024, 6, 15), account1, Direction.INC, Amount(Decimal("100")))
    entry_in.postings.append(posting_in)
    
    entry_after = JournalEntry(datetime.date(2024, 7, 15), "After period", "source")
    posting_after = Posting(entry_after, datetime.date(2024, 7, 15), account1, Direction.INC, Amount(Decimal("75")))
    entry_after.postings.append(posting_after)
    
    journal = [entry_before, entry_in, entry_after]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert account1 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("100"))


def test_build_general_ledger_creates_ledger_for_new_accounts():
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.core import DateRange, Quantity, Account, Direction, Amount
    from pypara.accounting.journaling import Posting
    from decimal import Decimal
    import datetime
    
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    
    entry = JournalEntry(datetime.date(2024, 6, 15), "Test entry", "source")
    posting = Posting(entry, datetime.date(2024, 6, 15), account1, Direction.INC, Amount(Decimal("100")))
    entry.postings.append(posting)
    
    journal = [entry]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert account1 in result.ledgers
    assert result.ledgers[account1].initial.date == period.since
    assert result.ledgers[account1].initial.value == Quantity(Decimal(0))


def test_build_general_ledger_accumulates_multiple_postings():
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.core import DateRange, Quantity, Account, Direction, Amount
    from pypara.accounting.journaling import Posting
    from decimal import Decimal
    import datetime
    
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    
    entry1 = JournalEntry(datetime.date(2024, 6, 15), "Entry 1", "source1")
    posting1 = Posting(entry1, datetime.date(2024, 6, 15), account1, Direction.INC, Amount(Decimal("100")))
    entry1.postings.append(posting1)
    
    entry2 = JournalEntry(datetime.date(2024, 6, 20), "Entry 2", "source2")
    posting2 = Posting(entry2, datetime.date(2024, 6, 20), account1, Direction.INC, Amount(Decimal("50")))
    entry2.postings.append(posting2)
    
    journal =


# LLM-generated content at query #9
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        account: Account
        amount: Amount
        date: date
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    ledger = Ledger(name="Test Ledger")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        account=Account(name="Cash"),
        amount=Amount(value=100.0, currency="USD"),
        date=date(2024, 1, 1),
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #10
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    posting_obj = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        journal=Journal(description="Test transaction", postings=[]),
        account=account,
        direction="debit"
    )
    balance = Quantity(value=500.0)
    ledger = Ledger()
    
    entry = LedgerEntry(ledger=ledger, posting=posting_obj, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting_obj
    assert entry.balance is balance
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test transaction"
    assert entry.amount is amount
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #11
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        direction: str
        account: MockAccount
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        account=test_account,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns values
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
        account: MockAccount

    @dataclass
    class MockLedger:
        pass

    # Create test instances
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_account = MockAccount(name="Cash")
    test_journal = MockJournal(description="Test Transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account
    )
    test_ledger = MockLedger()

    # Test constructor
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_quantity)

    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity


# LLM-generated content at query #13
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction, Posting, Amount
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.core import DateRange, Quantity
    
    # Create test accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("1000")))
    }
    
    # Create a journal entry with postings
    entry = JournalEntry(
        date=datetime.date(2024, 1, 15),
        description="Test entry",
        source="test_source"
    )
    entry.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal("-100")))
    entry.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal("100")))
    
    # Create accounting period
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [entry], initial_balances)
    
    # Verify the predicate: general_ledger contains both accounts as ledgers
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[account1], Ledger)
    assert isinstance(general_ledger.ledgers[account2], Ledger)
    assert general_ledger.ledgers[account1].account == account1
    assert general_ledger.ledgers[account2].account == account2
    assert general_ledger.period == period


# LLM-generated content at query #14
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return self.journal_ref
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockLedger:
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Set up test data
    account = MockAccount(name="TestAccount")
    amount = MockAmount(value=100.0, currency="USD")
    journal = MockJournal(description="Test Journal", postings=[])
    posting = MockPosting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    posting.journal_ref = journal
    
    ledger = MockLedger()
    balance = MockQuantity(value=100.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    # Assert constructor properly assigned all fields
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #15
#--------------------------

```python
def test_read_initial_balances_protocol_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    reader = ConcreteReadInitialBalances()
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert period.start == start_date
    assert period.end == end_date


# LLM-generated content at query #16
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="123456")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Balance, Amount, Quantity, DateRange
    from pypara.accounting.ledger import build_general_ledger
    
    # Create test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create a journal entry within the period
    entry_within = JournalEntry(date(2024, 6, 15), "Within period", "source1")
    entry_within.postings.append(
        Posting(entry_within, date(2024, 6, 15), account1, Direction.INC, Amount(Quantity(Decimal(100))))
    )
    entry_within.postings.append(
        Posting(entry_within, date(2024, 6, 15), account2, Direction.DEC, Amount(Quantity(Decimal(100))))
    )
    
    # Create a journal entry before the period
    entry_before = JournalEntry(date(2023, 12, 31), "Before period", "source2")
    entry_before.postings.append(
        Posting(entry_before, date(2023, 12, 31), account1, Direction.INC, Amount(Quantity(Decimal(50))))
    )
    entry_before.postings.append(
        Posting(entry_before, date(2023, 12, 31), account2, Direction.DEC, Amount(Quantity(Decimal(50))))
    )
    
    # Create a journal entry after the period
    entry_after = JournalEntry(date(2025, 1, 1), "After period", "source3")
    entry_after.postings.append(
        Posting(entry_after, date(2025, 1, 1), account1, Direction.INC, Amount(Quantity(Decimal(75))))
    )
    entry_after.postings.append(
        Posting(entry_after, date(2025, 1, 1), account2, Direction.DEC, Amount(Quantity(Decimal(75))))
    )
    
    initial_balances = {
        account1: Balance(period_start, Quantity(Decimal(1000))),
        account2: Balance(period_start, Quantity(Decimal(500)))
    }
    
    journal = [entry_before, entry_within, entry_after]
    
    # Build the general ledger
    gl = build_general_ledger(period, journal, initial_balances)
    
    # Verify: The predicate at line 16 filters by date range
    # Only postings from entry_within should be included
    assert len(gl.ledgers[account1].entries) == 1
    assert len(gl.ledgers[account2].entries) == 1
    assert gl.ledgers[account1].entries[0].posting.journal_entry.date == date(2024, 6, 15)
    assert gl.ledgers[account2].entries[0].posting.journal_entry.date == date(2024, 6, 15)


# LLM-generated content at query #18
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    test_account = Account(name="Test Account")
    test_amount = Amount(value=100.0, currency="USD")
    test_quantity = Quantity(value=100.0)
    test_posting = Posting(
        date=date(2023, 1, 1),
        amount=test_amount,
        account=test_account,
        journal=Journal(description="Test Journal", postings=[]),
        direction="debit"
    )
    test_ledger = Ledger(name="Test Ledger")
    
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_quantity)
    
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity


# LLM-generated content at query #19
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Any
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, data: Any = None):
            self.data = data
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(data={"period": period})
    
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    date_range = DateRange(start_date, end_date)
    
    program = ConcreteGeneralLedgerProgram()
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["period"].start == start_date
    assert result.data["period"].end == end_date


# LLM-generated content at query #20
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="123456")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_different_balance():
    account = Account(name="Savings", number="654321")
    initial_balance = Balance(value=Quantity(5000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.number == "654321"
    assert ledger.initial.value == Quantity(5000)
    assert len(ledger.entries) == 0


def test_ledger_constructor_zero_balance():
    account = Account(name="Empty Account", number="000000")
    initial_balance = Balance(value=Quantity(0))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial.value == Quantity(0)
    assert ledger.entries == []


# LLM-generated content at query #21
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalancesImpl:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    read_balances = ReadInitialBalancesImpl()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = read_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}


def test_read_initial_balances_call_with_empty_balances():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalancesImpl:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({})
    
    read_balances = ReadInitialBalancesImpl()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = read_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {}


def test_read_initial_balances_call_returns_correct_type():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalancesImpl:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account": 5000})
    
    read_balances = ReadInitialBalancesImpl()
    period = DateRange(date(2024, 1, 1), date(2024, 6, 30))
    result = read_balances(period)
    
    assert hasattr(result, 'balances')
    assert isinstance(result.balances, dict)


# LLM-generated content at query #22
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 15),
        amount=Amount(value=100.0, currency="USD"),
        journal=Journal(description="Test entry", postings=[]),
        account=Account(name="Cash"),
        direction="debit"
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.balance.value == 500.0


# LLM-generated content at query #23
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert result.balances["account1"] == 1000
    assert result.balances["account2"] == 2000


# LLM-generated content at query #24
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    from typing import Protocol
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ReadInitialBalances(Protocol):
        def __call__(self, period: DateRange) -> InitialBalances:
            ...
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return InitialBalances({"account1": 1000, "account2": 2000})
    
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances(date_range)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert result.balances["account1"] == 1000
    assert result.balances["account2"] == 2000


# LLM-generated content at query #25
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    ledger = Ledger()
    account = Account(name="Cash")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #26
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="12345")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_different_values():
    account = Account(name="Savings", number="67890")
    initial_balance = Balance(value=Quantity(5000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.number == "67890"
    assert ledger.initial.value == Quantity(5000)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #27
#--------------------------

```python
def test_ledger_entry_constructor():
    from dataclasses import dataclass
    from datetime import date
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return self._journal
        
        def set_journal(self, journal):
            self._journal = journal
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        account=Account(name="Cash"),
        amount=Amount(value=100.0, currency="USD"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    journal = Journal(description="Test entry", postings=[posting])
    posting.set_journal(journal)
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #28
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for testing
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        account: MockAccount
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        name: str

    # Create test instances
    test_ledger = MockLedger(name="Test Ledger")
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Transaction", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 1),
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = MockQuantity(value=500.0)

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assert constructor sets attributes correctly
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #29
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for the constructor
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    mock_ledger = MockLedger()
    mock_journal = MockJournal(description="Test transaction", postings=[])
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_posting = MockPosting(
        date=date(2024, 1, 15),
        journal=mock_journal,
        amount=mock_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockAmount(value=500.0, currency="USD")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor properly assigns all attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #30
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    account = Account("1000", "Cash")
    initial_balance = Balance(date(2023, 1, 1), Quantity(Decimal(1000)))
    initial = {account: initial_balance}
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert account in result.ledgers
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial_balance


def test_build_general_ledger_with_postings_within_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.accounting.journaling import Direction
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    initial_balance1 = Balance(date(2023, 1, 1), Quantity(Decimal(1000)))
    initial_balance2 = Balance(date(2023, 1, 1), Quantity(Decimal(500)))
    initial = {account1: initial_balance1, account2: initial_balance2}
    
    entry = JournalEntry(date(2023, 1, 15), "Test entry", None)
    entry.post(date(2023, 1, 15), account1, Quantity(Decimal(-100)))
    entry.post(date(2023, 1, 15), account2, Quantity(Decimal(100)))
    
    journal = [entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


def test_build_general_ledger_with_postings_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial_balance = Balance(date(2023, 1, 1), Quantity(Decimal(1000)))
    initial = {account: initial_balance}
    
    entry = JournalEntry(date(2024, 1, 15), "Test entry", None)
    entry.post(date(2024, 1, 15), account, Quantity(Decimal(-100)))
    
    journal = [entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 0


def test_build_general_ledger_creates_new_ledger_for_new_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    initial_balance1 = Balance(date(2023, 1, 1), Quantity(Decimal(1000)))
    initial = {account1: initial_balance1}
    
    entry = JournalEntry(date(2023, 1, 15), "Test entry", None)
    entry.post(date(2023, 1, 15), account1, Quantity(Decimal(-100)))
    entry.post(date(2023, 1, 15), account2, Quantity(Decimal(100)))
    
    journal = [entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account2].initial == Balance(date(2023, 1, 1), Quantity(Decimal(0)))


def test_build_general_ledger_multiple_entries():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial_balance = Balance(date(2023, 1, 1), Quantity(Decimal(1000)))
    initial = {account: initial_balance}
    
    entry1 = JournalEntry(date(2023, 1, 15), "Entry 1", None)
    entry1.post(date(2023, 1, 15), account, Quantity(Decimal(100)))
    
    entry2 = JournalEntry(date(2023, 2, 15), "Entry 2", None)
    entry2.post(date(2023, 2, 15), account, Quantity(Decimal(-50)))
    
    journal = [entry1, entry2]
    
    result = build_general_ledger(period, journal, initial)
    
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 2


# LLM-generated content at query #31
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.ranges import DateRange
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {account1: Balance(date(2024, 1, 1), Quantity(Decimal(1000)))}
    
    # Create journal entry with postings
    je = JournalEntry(date(2024, 6, 15), "Test entry", "source_obj")
    je.post(date(2024, 6, 15), account1, Quantity(Decimal(-100)))
    je.post(date(2024, 6, 15), account2, Quantity(Decimal(100)))
    
    journal = [je]
    
    # Build general ledger
    gl = build_general_ledger(period, journal, initial_balances)
    
    # Verify the predicate at line 1: function returns a GeneralLedger instance
    assert gl is not None
    assert gl.period == period
    assert len(gl.ledgers) == 2
    assert account1 in gl.ledgers
    assert account2 in gl.ledgers
    assert gl.ledgers[account1].account == account1
    assert gl.ledgers[account2].account == account2


# LLM-generated content at query #32
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        account: Account
        amount: Amount
        date: date
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        account=account,
        amount=amount,
        date=date(2023, 1, 1),
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #33
#--------------------------

```python
def test_build_general_ledger_filters_postings_within_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.temporal import DateRange
    
    # Create test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    period_start = date(2023, 1, 1)
    period_end = date(2023, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create journal entries with dates inside and outside the period
    source = "test_source"
    entry_inside = JournalEntry(date(2023, 6, 15), "Inside period", source)
    entry_before = JournalEntry(date(2022, 12, 31), "Before period", source)
    entry_after = JournalEntry(date(2024, 1, 1), "After period", source)
    
    # Add postings to entries
    entry_inside.post(date(2023, 6, 15), account1, Quantity(Decimal(100)))
    entry_inside.post(date(2023, 6, 15), account2, Quantity(Decimal(-100)))
    
    entry_before.post(date(2022, 12, 31), account1, Quantity(Decimal(50)))
    entry_before.post(date(2022, 12, 31), account2, Quantity(Decimal(-50)))
    
    entry_after.post(date(2024, 1, 1), account1, Quantity(Decimal(75)))
    entry_after.post(date(2024, 1, 1), account2, Quantity(Decimal(-75)))
    
    # Build general ledger
    initial_balances = {
        account1: Balance(period_start, Quantity(Decimal(0))),
        account2: Balance(period_start, Quantity(Decimal(0)))
    }
    
    journal = [entry_before, entry_inside, entry_after]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that only postings from entry_inside are included
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].posting.amount == Quantity(Decimal(100))
    assert general_ledger.ledgers[account2].entries[0].posting.amount == Quantity(Decimal(100))


# LLM-generated content at query #34
#--------------------------

```python
def test_ledger_entry_constructor():
    from dataclasses import dataclass
    from datetime import date
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: 'Journal'
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity
    
    test_account = Account(name="Test Account")
    test_amount = Amount(value=100.0, currency="USD")
    test_journal = Journal(description="Test Journal", postings=[])
    test_posting = Posting(
        date=date(2023, 1, 1),
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    test_ledger = Ledger()
    test_balance = Quantity(value=100.0)
    
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #35
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.accounts import Account, Amount
    from pypara.core.quantity import Quantity
    from pypara.core.daterange import DateRange
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    # Create accounts
    account_cash = Account(name="Cash", code="1000", account_type="Asset")
    account_revenue = Account(name="Revenue", code="4000", account_type="Revenue")
    
    # Create initial balances
    initial_balances = {
        account_cash: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))
    }
    
    # Create a journal entry with postings
    source = "Test Transaction"
    entry = JournalEntry(date=date(2024, 1, 15), description="Test posting", source=source)
    
    # Post amounts (debit cash, credit revenue)
    entry.post(date(2024, 1, 15), account_cash, Quantity(Decimal("500.00")))
    entry.post(date(2024, 1, 15), account_revenue, Quantity(Decimal("-500.00")))
    
    # Build general ledger
    journal = [entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account_cash in general_ledger.ledgers
    assert account_revenue in general_ledger.ledgers
    
    # Check cash ledger
    cash_ledger = general_ledger.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    
    # Check revenue ledger
    revenue_ledger = general_ledger.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert revenue_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500.00"))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity
    from pypara.core.daterange import DateRange
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account = Account(name="Cash", code="1000", account_type="Asset")
    initial_balances = {account: Balance(date(2024, 1, 1), Quantity(Decimal("500.00")))}
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal("500.00"))
    assert len(general_ledger.ledgers[account].entries) == 0


def test_build_general_ledger_multiple_entries():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity
    from pypara.core.daterange import DateRange
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account_cash = Account(name="Cash", code="1000", account_type="Asset")
    account_revenue = Account(name="Revenue", code="4000", account_type="Revenue")
    
    initial_balances = {
        account_cash: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))
    }
    
    # Create multiple journal entries
    entry1 = JournalEntry(date=date(2024, 1, 15), description="Entry 1", source="Source1")
    entry1.post(date(2024, 1, 15), account_cash, Quantity(Decimal("100.00")))
    entry1.post(date(2024, 1, 15), account_revenue, Quantity(Decimal("-100.00")))
    
    entry2 = JournalEntry(date=date(2024, 2, 15), description="Entry 2", source="Source2")
    entry2.post(date(2024, 2, 15), account_cash, Quantity(Decimal("200.00")))
    entry2.post(date(2024, 2, 15), account_revenue, Quantity(Decimal("-200.00")))
    
    journal = [entry1, entry2]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert len(general_ledger.ledgers[account_cash].entries) == 2
    assert len(general_ledger.ledgers[account_revenue].entries) == 2
    assert general_ledger.ledgers[account_cash].entries[0].balance == Quantity(Decimal("1100.00"))
    assert general_ledger.ledgers[account_cash].entries[1].balance == Quantity(Decimal("1300.00"))


def test_build_general_ledger_out_of_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity
    from pypara.core.daterange import DateRange
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account = Account(name="Cash", code="1000", account_type="Asset")
    initial_balances = {account: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))}
    
    # Create entry outside period
    entry = JournalEntry(date=date(2025, 1, 15), description="Out of period", source="Source")
    entry.post(date(2025, 1, 15), account, Quantity(Decimal("100.00")))
    
    journal = [entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Entry outside period should not be added
    assert len(general_ledger.ledgers[account].entries) == 0


# LLM-generated content at query #36
#--------------------------

```python
def test_build_general_ledger_predicate_false():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    from pypara.accounting.journaling import Posting
    
    # Create test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create a journal entry with a date AFTER the period
    entry_date = datetime.date(2024, 1, 15)
    entry = JournalEntry(entry_date, "Test entry", "source")
    
    # Create an account and posting
    account = Account("1000", "Test Account")
    posting = Posting(entry, entry_date, account, Direction.INC, Amount(Quantity(Decimal(100))))
    
    # Manually add posting to entry (simulating post() behavior)
    entry.postings.append(posting)
    
    # Build general ledger with empty initial balances
    journal = [entry]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    # The predicate "period.since <= j.date <= period.until" should be False
    # because entry_date (2024-01-15) is after period.until (2023-12-31)
    # Therefore, the posting should NOT be added to any ledger
    assert len(result.ledgers) == 0


# LLM-generated content at query #37
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_account = type('Account', (), {})()
    mock_amount = type('Amount', (), {'value': Decimal('100.00')})()
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'amount': mock_amount,
        'is_debit': True,
        'is_credit': False,
        'direction': 'debit',
        'account': mock_account,
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': [
                type('Posting', (), {'direction': 'debit', 'account': mock_account})(),
                type('Posting', (), {'direction': 'credit', 'account': mock_account})()
            ]
        })()
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {'value': Decimal('1000.00')})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor correctly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #38
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar("_T")

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        account: Account
        amount: Amount
        date: date
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    posting_date = date(2024, 1, 15)
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(account=account, amount=amount, date=posting_date, journal=journal, direction="debit")
    balance = Quantity(value=500.0)
    ledger = Ledger()

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.date == posting_date
    assert entry.description == "Test transaction"
    assert entry.amount == amount
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #39
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar, Generic
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger(Generic[_T]):
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[dict]:
            return GeneralLedger[dict]({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []


# LLM-generated content at query #40
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    # Create test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("Test Account")
    test_journal = MockJournal("Test Description", [])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger("Test Ledger")
    test_balance = MockAmount(500.0, "USD")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns fields
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #41
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    account = Account(name="Test Account")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        journal=journal,
        direction="debit"
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #42
#--------------------------

```python
def test_build_general_ledger_posting_account_in_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.interval import DateRange
    from pypara.accounting.journaling import Posting
    
    # Create test data
    test_account = Account(name="TestAccount", code="1000")
    test_date = date(2024, 1, 1)
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    
    # Create initial balances with the test account
    initial_balances = {test_account: Balance(test_date, Quantity(Decimal("100")))}
    
    # Create a journal entry
    journal_entry = JournalEntry(date=test_date, description="Test entry", source="test")
    
    # Create a posting and add it to the journal entry
    posting = Posting(journal_entry, test_date, test_account, Direction.INC, Amount(Quantity(Decimal("50"))))
    journal_entry.postings.append(posting)
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    
    # The predicate "posting.account not in ledgers" should evaluate to False
    # because the account was already in initial_balances
    assert test_account in general_ledger.ledgers
    assert len(general_ledger.ledgers) == 1


# LLM-generated content at query #43
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, DateRange
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.accounting.account import Account
    
    # Setup test data
    test_date = date(2023, 1, 15)
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Create accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(1000)))
    }
    
    # Create a journal entry with postings
    entry = JournalEntry(test_date, "Test transaction", "source_data")
    entry.post(test_date, account1, Quantity(Decimal(-100)))
    entry.post(test_date, account2, Quantity(Decimal(100)))
    entry.validate()
    
    # Build general ledger
    journal = [entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify the predicate: both accounts should have ledgers
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[account1], Ledger)
    assert isinstance(general_ledger.ledgers[account2], Ledger)
    assert general_ledger.ledgers[account1].account == account1
    assert general_ledger.ledgers[account2].account == account2


# LLM-generated content at query #44
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.accounts import Account
    from pypara.core import Quantity, DateRange
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.core import Quantity, DateRange
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    acc1 = Account("1000", "Cash")
    acc2 = Account("2000", "Accounts Payable")
    initial = {
        acc1: Balance(date(2023, 1, 1), Quantity(Decimal("1000.00"))),
        acc2: Balance(date(2023, 1, 1), Quantity(Decimal("500.00")))
    }
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert acc1 in result.ledgers
    assert acc2 in result.ledgers
    assert result.ledgers[acc1].initial.value == Quantity(Decimal("1000.00"))
    assert result.ledgers[acc2].initial.value == Quantity(Decimal("500.00"))


def test_build_general_ledger_with_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.core import Quantity, DateRange, Amount
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    acc1 = Account("1000", "Cash")
    acc2 = Account("2000", "Accounts Payable")
    
    entry = JournalEntry(date(2023, 6, 15), "Test transaction", "source")
    entry.post(date(2023, 6, 15), acc1, Quantity(Decimal("-100.00")))
    entry.post(date(2023, 6, 15), acc2, Quantity(Decimal("100.00")))
    
    initial = {
        acc1: Balance(date(2023, 1, 1), Quantity(Decimal("1000.00"))),
        acc2: Balance(date(2023, 1, 1), Quantity(Decimal("0.00")))
    }
    
    result = build_general_ledger(period, [entry], initial)
    
    assert len(result.ledgers) == 2
    assert len(result.ledgers[acc1].entries) == 1
    assert len(result.ledgers[acc2].entries) == 1


def test_build_general_ledger_creates_ledger_for_new_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.core import Quantity, DateRange
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    acc_new = Account("3000", "Revenue")
    
    entry = JournalEntry(date(2023, 6, 15), "Test transaction", "source")
    entry.post(date(2023, 6, 15), acc_new, Quantity(Decimal("500.00")))
    
    initial = {}
    
    result = build_general_ledger(period, [entry], initial)
    
    assert acc_new in result.ledgers
    assert result.ledgers[acc_new].initial.date == date(2023, 1, 1)
    assert result.ledgers[acc_new].initial.value == Quantity(Decimal("0"))


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.core import Quantity, DateRange
    
    period = DateRange(date(2023, 6, 1), date(2023, 12, 31))
    acc = Account("1000", "Cash")
    
    entry_in_period = JournalEntry(date(2023, 6, 15), "In period", "source")
    entry_in_period.post(date(2023, 6, 15), acc, Quantity(Decimal("100.00")))
    
    entry_out_period = JournalEntry(date(2023, 5, 15), "Out of period", "source")
    entry_out_period.post(date(2023, 5, 15), acc, Quantity(Decimal("200.00")))
    
    initial = {acc: Balance(date(2023, 6, 1), Quantity(Decimal("0.00")))}
    
    result = build_general_ledger(period, [entry_in_period, entry_out_period], initial)
    
    assert len(result.ledgers[acc].entries) == 1


def test_build_general_ledger_accumulates_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.core import Quantity, DateRange
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    acc = Account("1000", "Cash")
    
    entry1 = JournalEntry(date(2023, 3, 1), "Transaction 1", "source")
    entry1.post(date(2023, 3, 1), acc, Quantity(Decimal("100.00")))
    
    entry2 = JournalEntry(date(2023, 6, 1), "Transaction 2", "source")
    entry2.post(date(2023, 6, 1), acc, Quantity(Decimal("50.00")))
    
    initial = {acc: Balance(date(2023, 1, 1), Quantity(Decimal("1000.00")))}
    
    result = build_general_ledger(period, [entry1, entry2], initial)
    
    assert len(result.ledgers[acc].entries) == 2
    assert result.ledgers[acc].entries[0].balance == Quantity(Decimal("1100.00"))
    assert result.ledgers[acc].entries[1].balance == Quantity(Decimal("1150.00"))


# LLM-generated content at query #45
#--------------------------

```python
def test_ledger_entry_constructor():
    from dataclasses import dataclass
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_ledger = object()
    
    mock_journal = type('Journal', (), {'description': 'Test transaction', 'postings': []})()
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'journal': mock_journal,
        'amount': Decimal('100.00'),
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False,
        'account': object()
    })()
    
    mock_balance = Decimal('500.00')
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor sets attributes correctly
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #46
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit"
    )
    balance = Quantity(1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #47
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    posting_obj = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=Journal(description="Test transaction", postings=[])
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting_obj, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting_obj
    assert entry.balance == balance


# LLM-generated content at query #48
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Set up test data
    account = MockAccount(name="Test Account")
    amount = MockAmount(value=100.0, currency="USD")
    journal = MockJournal(description="Test Description", postings=[])
    posting = MockPosting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=journal
    )
    balance = MockQuantity(value=500.0)
    ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    # Assertions
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #49
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, data):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger({"period": period})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range


# LLM-generated content at query #50
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Cash")
    posting_obj = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        journal=Journal(description="Test entry", postings=[]),
        direction="debit"
    )
    ledger_obj = Ledger(name="General Ledger")
    balance_qty = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger_obj, posting=posting_obj, balance=balance_qty)

    assert entry.ledger == ledger_obj
    assert entry.posting == posting_obj
    assert entry.balance == balance_qty


# LLM-generated content at query #51
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, data: dict):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger({"period": period})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["period"].start == date(2023, 1, 1)
    assert result.data["period"].end == date(2023, 12, 31)


# LLM-generated content at query #52
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_account = MockAccount("Cash")
    test_quantity = MockQuantity(1000.0)
    test_ledger = MockLedger()
    test_journal = MockJournal("Test transaction", [])
    
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assert constructor properly assigned all attributes
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity


# LLM-generated content at query #53
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    import datetime
    from decimal import Decimal
    from pypara.core.monetary import Quantity, Amount
    from pypara.core.period import DateRange
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import Ledger, build_general_ledger
    from pypara.accounting.accounts import Account
    
    # Create test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal(1000)))
    }
    
    # Create a journal entry with postings
    entry = JournalEntry(date=datetime.date(2024, 6, 15), description="Test entry", source="test")
    entry.post(datetime.date(2024, 6, 15), account1, Quantity(Decimal(-100)))
    entry.post(datetime.date(2024, 6, 15), account2, Quantity(Decimal(100)))
    
    journal = [entry]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that ledgers were created for both accounts
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers) == 2
    
    # Verify ledger properties
    assert general_ledger.ledgers[account1].account == account1
    assert general_ledger.ledgers[account2].account == account2
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal(1000))
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal(0))
    
    # Verify entries were added
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1


# LLM-generated content at query #54
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar("_T")

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger()

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert ledger_entry.ledger is ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is quantity


# LLM-generated content at query #55
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Entry", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger()
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #56
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, direction="debit", journal=journal)
    ledger = Ledger(name="Test Ledger")

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == quantity


# LLM-generated content at query #57
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    account = Account(name="Test Account")
    journal = Journal(description="Test Journal", postings=[])
    amount = Amount(value=100.0, currency="USD")
    posting = Posting(date=date(2023, 1, 1), journal=journal, amount=amount, account=account, direction="debit")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #58
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from decimal import Decimal
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: Decimal
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    @dataclass
    class MockQuantity:
        value: Decimal
    
    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=Decimal("100.00"), currency="USD")
    test_posting = MockPosting(
        date=date(2023, 1, 15),
        amount=test_amount,
        journal=MockJournal(description="Test Entry", postings=[]),
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="Test Ledger")
    test_balance = MockQuantity(value=Decimal("500.00"))
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    # Assert constructor properly assigns attributes
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #59
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger(name="Test Ledger")
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Transaction", postings=[])
    posting = Posting(
        date=date(2023, 1, 15),
        journal=journal,
        amount=amount,
        account=account,
        direction="debit"
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #60
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #61
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    posting = Posting(
        date=date(2024, 1, 15),
        amount=Amount(value=100.0, currency="USD"),
        journal=Journal(description="Test transaction", postings=[]),
        account=Account(name="Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #62
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar("_T")

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        direction: str
        account: Account

        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"

        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        journal=journal,
        amount=Amount(value=100.0, currency="USD"),
        direction="debit",
        account=account
    )
    balance = Quantity(value=100.0)
    ledger = Ledger()

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #63
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=100.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #64
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        journal: Journal

        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"

        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2024, 1, 1), amount=amount, account=account, direction="debit", journal=journal)
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance
    assert entry.ledger.name == "General Ledger"
    assert entry.posting.date == date(2024, 1, 1)
    assert entry.balance.value == 500.0


# LLM-generated content at query #65
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        journal=Journal(description="Test entry", postings=[]),
        amount=Amount(value=100.0, currency="USD"),
        account=Account(name="Cash"),
        direction="debit"
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test entry"
    assert entry.amount == Amount(value=100.0, currency="USD")
    assert entry.is_debit is True
    assert entry.is_credit is False
    assert entry.debit == Amount(value=100.0, currency="USD")
    assert entry.credit is None


# LLM-generated content at query #66
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        journal: object
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockLedger:
        pass

    @dataclass
    class MockQuantity:
        value: float

    # Create instances
    mock_ledger = MockLedger()
    mock_journal = MockJournal(description="Test Entry", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 15),
        amount=MockAmount(value=100.0, currency="USD"),
        account=MockAccount(name="Test Account"),
        direction="debit",
        journal=mock_journal,
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockQuantity(value=500.0)

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )

    # Assert constructor properly assigns attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #67
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 15),
        journal=Journal(description="Test transaction", postings=[]),
        amount=Amount(value=100.0, currency="USD"),
        account=Account(name="Cash"),
        direction="debit"
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #68
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    test_ledger = Ledger(name="Test Ledger")
    test_account = Account(name="Cash")
    test_amount = Amount(value=100.0, currency="USD")
    test_journal = Journal(description="Test transaction", postings=[])
    test_posting = Posting(
        date=date(2023, 1, 1),
        amount=test_amount,
        account=test_account,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = Quantity(value=500.0)

    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #69
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    # Create instances
    mock_ledger = MockLedger(name="Test Ledger")
    mock_journal = MockJournal(
        description="Test Journal Entry",
        postings=[]
    )
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_posting = MockPosting(
        date=date(2023, 1, 15),
        journal=mock_journal,
        amount=mock_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockAmount(value=500.0, currency="USD")
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly assigns attributes
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #70
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    account = Account(name="Test Account")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        direction="debit",
        journal=journal
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #71
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    ledger = Ledger()
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2024, 1, 1), amount=amount, account=account, direction="debit", journal=journal)
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #72
#--------------------------

```python
def test_ledger_entry_constructor():
    from dataclasses import dataclass
    from datetime import date
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return self._journal
        
        def set_journal(self, journal):
            self._journal = journal
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity
    
    # Create test data
    test_ledger = Ledger(name="Test Ledger")
    test_account = Account(name="Test Account")
    test_amount = Amount(value=100.0, currency="USD")
    test_quantity = Quantity(value=100.0)
    
    test_posting = Posting(
        date=date(2024, 1, 1),
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_posting.set_journal(Journal(description="Test Journal", postings=[test_posting]))
    
    # Test constructor
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity
    assert entry.ledger.name == "Test Ledger"
    assert entry.posting.date == date(2024, 1, 1)
    assert entry.balance.value == 100.0


# LLM-generated content at query #73
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from decimal import Decimal
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: Decimal
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=Decimal("100.00"), currency="USD")
    test_journal = MockJournal(description="Test Description", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 1),
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockAmount(value=Decimal("500.00"), currency="USD")
    
    # Test constructor with all required fields
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Verify all attributes are correctly assigned
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_balance


# LLM-generated content at query #74
#--------------------------

```python
def test_build_general_ledger_predicate_filters_postings_within_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.models import Account, Quantity, Amount, DateRange, Posting
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    period = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    initial_balances = {account1: Balance(date(2024, 1, 1), Quantity(Decimal(1000)))}
    
    # Create journal entries with dates inside and outside period
    entry_inside = JournalEntry(date(2024, 1, 15), "Inside period", "source1")
    entry_inside.postings.append(Posting(entry_inside, date(2024, 1, 15), account1, Direction.DEC, Amount(Decimal(100))))
    entry_inside.postings.append(Posting(entry_inside, date(2024, 1, 15), account2, Direction.INC, Amount(Decimal(100))))
    
    entry_outside_before = JournalEntry(date(2023, 12, 31), "Before period", "source2")
    entry_outside_before.postings.append(Posting(entry_outside_before, date(2023, 12, 31), account1, Direction.DEC, Amount(Decimal(50))))
    entry_outside_before.postings.append(Posting(entry_outside_before, date(2023, 12, 31), account2, Direction.INC, Amount(Decimal(50))))
    
    entry_outside_after = JournalEntry(date(2024, 2, 1), "After period", "source3")
    entry_outside_after.postings.append(Posting(entry_outside_after, date(2024, 2, 1), account1, Direction.DEC, Amount(Decimal(75))))
    entry_outside_after.postings.append(Posting(entry_outside_after, date(2024, 2, 1), account2, Direction.INC, Amount(Decimal(75))))
    
    journal = [entry_outside_before, entry_inside, entry_outside_after]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that only postings within period are included
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Account1 ledger should have exactly 1 entry (only from entry_inside)
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].posting.journal_entry.date == date(2024, 1, 15)
    
    # Account2 ledger should have exactly 1 entry (only from entry_inside)
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account2].entries[0].posting.journal_entry.date == date(2024, 1, 15)


# LLM-generated content at query #75
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[dict]:
            return GeneralLedger({"start": period.start, "end": period.end})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["start"] == date(2023, 1, 1)
    assert result.data["end"] == date(2023, 12, 31)


# LLM-generated content at query #76
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        account: Account
        amount: Amount
        date: date
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger(name="Test Ledger")
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Entry", postings=[])
    posting = Posting(
        account=account,
        amount=amount,
        date=date(2024, 1, 1),
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #77
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.accounting.models import Account, Amount
    from pypara.core import DateRange, Quantity
    
    # Create test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    date_since = datetime.date(2024, 1, 1)
    date_until = datetime.date(2024, 12, 31)
    period = DateRange(date_since, date_until)
    
    # Create initial balances
    initial_balances = {
        account1: Balance(date_since, Quantity(Decimal(1000)))
    }
    
    # Create a journal entry with postings
    entry = JournalEntry(date_since, "Test entry", "source")
    entry.post(date_since, account1, Quantity(Decimal(-100)))
    entry.post(date_since, account2, Quantity(Decimal(100)))
    
    journal = [entry]
    
    # Build general ledger
    result = build_general_ledger(period, journal, initial_balances)
    
    # Verify the result is a GeneralLedger instance
    assert isinstance(result, GeneralLedger)
    
    # Verify the period is correct
    assert result.period == period
    
    # Verify both accounts are in the ledgers
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    
    # Verify account1 has initial balance and the posting
    assert isinstance(result.ledgers[account1], Ledger)
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account1].initial.value == Quantity(Decimal(1000))
    assert len(result.ledgers[account1].entries) == 1
    
    # Verify account2 was created with zero initial balance
    assert isinstance(result.ledgers[account2], Ledger)
    assert result.ledgers[account2].account == account2
    assert result.ledgers[account2].initial.value == Quantity(Decimal(0))
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #78
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        journal: Journal
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2024, 1, 15),
        amount=amount,
        journal=journal,
        account=account,
        direction="debit"
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance
    assert entry.ledger.name == "General Ledger"
    assert entry.posting.date == date(2024, 1, 15)
    assert entry.posting.amount.value == 100.0
    assert entry.balance.value == 500.0


# LLM-generated content at query #79
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="12345")
    balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=balance)
    
    assert ledger.account == account
    assert ledger.initial == balance
    assert ledger.entries == []


def test_ledger_constructor_with_different_values():
    account = Account(name="Savings", number="98765")
    balance = Balance(value=Quantity(5000))
    
    ledger = Ledger(account=account, initial=balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.number == "98765"
    assert ledger.initial.value == Quantity(5000)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


def test_ledger_constructor_initializes_empty_entries():
    account = Account(name="Checking", number="11111")
    balance = Balance(value=Quantity(2500))
    
    ledger = Ledger(account=account, initial=balance)
    
    assert hasattr(ledger, 'entries')
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #80
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_to_false():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Account
    from pypara.accounting.journaling import JournalEntry, Direction, Posting, Amount
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Quantity
    
    # Create test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account = Account("TestAccount", "1000")
    initial_balances = {account: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(100)))}
    
    # Create a journal entry with a posting OUTSIDE the period
    entry = JournalEntry(
        date=datetime.date(2023, 12, 31),  # Before period.since
        description="Test Entry",
        source="test_source"
    )
    
    # Add a posting to the entry
    posting = Posting(
        entry=entry,
        date=datetime.date(2023, 12, 31),
        account=account,
        direction=Direction.INC,
        amount=Amount(Quantity(Decimal(50)))
    )
    entry.postings.append(posting)
    
    journal = [entry]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # The predicate at line 16: `period.since <= j.date <= period.until`
    # should evaluate to False for our entry (2023-12-31 is not within 2024-01-01 to 2024-12-31)
    # Therefore, the posting should NOT be added to the ledger
    assert len(general_ledger.ledgers[account].entries) == 0


# LLM-generated content at query #81
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    # Create mock objects
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test data
    account = MockAccount(name="Test Account")
    amount = MockAmount(value=100.0, currency="USD")
    journal = MockJournal(description="Test Journal", postings=[])
    posting = MockPosting(
        date=date(2023, 1, 1),
        amount=amount,
        journal=journal,
        account=account,
        direction="debit"
    )
    ledger = MockLedger()
    balance = MockQuantity(value=1000.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    # Assertions
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger is not None
    assert entry.posting is not None
    assert entry.balance is not None


# LLM-generated content at query #82
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert all fields are correctly assigned
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #83
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        account: Account
        amount: Amount
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger_instance = Ledger()
    posting_instance = Posting(
        date=date(2023, 1, 1),
        account=Account(name="Cash"),
        amount=Amount(value=100.0, currency="USD"),
        journal=Journal(description="Test", postings=[]),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance_instance = Quantity(value=1000.0)

    entry = LedgerEntry(
        ledger=ledger_instance,
        posting=posting_instance,
        balance=balance_instance
    )

    assert entry.ledger == ledger_instance
    assert entry.posting == posting_instance
    assert entry.balance == balance_instance


# LLM-generated content at query #84
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return InitialBalances({"account1": 1000, "account2": 2000})
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}


def test_read_initial_balances_call_empty_balances():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return InitialBalances({})
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {}


def test_read_initial_balances_call_with_different_period():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        if period.start.month == 1:
            return InitialBalances({"account1": 5000})
        return InitialBalances({"account1": 3000})
    
    period = DateRange(date(2023, 1, 15), date(2023, 1, 31))
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 5000}


# LLM-generated content at query #85
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return MockJournal()
    
    @dataclass
    class MockJournal:
        description: str = "Test Journal"
        postings: list = None
        
        def __post_init__(self):
            if self.postings is None:
                self.postings = []
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test instances
    test_ledger = MockLedger()
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=MockAmount(100.0, "USD"),
        account=MockAccount("Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = MockQuantity(1000.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #86
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    test_ledger = Ledger(name="Test Ledger")
    test_account = Account(name="Cash")
    test_amount = Amount(value=100.0, currency="USD")
    test_journal = Journal(description="Test transaction", postings=[])
    test_posting = Posting(
        date=date(2023, 1, 1),
        amount=test_amount,
        account=test_account,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = Quantity(value=500.0)

    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #87
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for testing
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_posting_obj = MockPosting(
        date=date(2024, 1, 1),
        amount=test_amount,
        journal=MockJournal(description="Test Description", postings=[]),
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=500.0)

    # Test constructor
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting_obj,
        balance=test_balance
    )

    # Assert constructor sets attributes correctly
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting_obj
    assert ledger_entry.balance is test_balance


# LLM-generated content at query #88
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting(Generic[_T]):
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger(Generic[_T]):
        pass

    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=1000.0)

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Verify constructor assigned all attributes correctly
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #89
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 1),
        journal=test_journal,
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockAmount(value=1000.0, currency="USD")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    # Assert constructor properly assigned all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #90
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        journal: Journal
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    ledger = Ledger()
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2024, 1, 15),
        amount=Amount(value=100.0, currency="USD"),
        account=Account(name="Cash"),
        direction="debit",
        journal=journal,
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger.ledger is ledger
    assert entry.posting.posting is posting
    assert entry.balance.balance is balance


# LLM-generated content at query #91
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Quantity, Account, Amount
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))
    }
    
    # Create journal entries with postings
    entry1 = JournalEntry(date(2024, 1, 15), "Initial transaction", "source1")
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal("-100.00")))
    entry1.post(date(2024, 1, 15), account2, Quantity(Decimal("100.00")))
    
    entry2 = JournalEntry(date(2024, 2, 20), "Second transaction", "source2")
    entry2.post(date(2024, 2, 20), account1, Quantity(Decimal("50.00")))
    entry2.post(date(2024, 2, 20), account2, Quantity(Decimal("-50.00")))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("1000.00"))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal("900.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("950.00"))
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("0.00"))
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal("100.00"))
    assert ledger2.entries[1].balance == Quantity(Decimal("50.00"))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity, Account
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    
    initial_balances = {
        account1: Balance(date(2024, 1, 1), Quantity(Decimal("5000.00")))
    }
    
    journal = []
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0


def test_build_general_ledger_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity, Account
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account1: Balance(date(2024, 1, 1), Quantity(Decimal("1000.00")))
    }
    
    # Entry outside period
    entry1 = JournalEntry(date(2023, 12, 31), "Before period", "source1")
    entry1.post(date(2023, 12, 31), account1, Quantity(Decimal("-100.00")))
    entry1.post(date(2023, 12, 31), account2, Quantity(Decimal("100.00")))
    
    journal = [entry1]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert account2 not in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0


def test_build_general_ledger_creates_new_ledger_for_uninitialized_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity, Account
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {}
    
    entry1 = JournalEntry(date(2024, 1, 15), "Transaction", "source1")
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal("100.00")))
    entry1.post(date(2024, 1, 15), account2, Quantity(Decimal("-100.00")))
    
    journal = [entry1]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal("0.00"))
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal("0.00"))


# LLM-generated content at query #92
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return MockJournal()
    
    @dataclass
    class MockJournal:
        description: str = "Test Journal"
        postings: list = None
        
        def __post_init__(self):
            if self.postings is None:
                self.postings = []
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create instances
    mock_ledger = MockLedger()
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=MockAmount(100.0, "USD"),
        account=MockAccount("Test Account"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockQuantity(100.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly initialized all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #93
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=1000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #94
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_ledger = object()
    
    mock_journal = object()
    mock_journal.description = "Test transaction"
    mock_journal.postings = []
    
    mock_amount = object()
    mock_amount.value = 100
    
    mock_posting = object()
    mock_posting.date = date(2024, 1, 15)
    mock_posting.journal = mock_journal
    mock_posting.amount = mock_amount
    mock_posting.direction = "debit"
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_balance = object()
    
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert ledger_entry.ledger is mock_ledger
    assert ledger_entry.posting is mock_posting
    assert ledger_entry.balance is mock_balance


# LLM-generated content at query #95
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    ledger = Ledger()
    posting = Posting(
        date=date(2024, 1, 15),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #96
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class Account:
        name: str
    
    @dataclass
    class Amount:
        value: float
        currency: str
    
    @dataclass
    class Quantity:
        value: float
    
    @dataclass
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        account: Account
        amount: Amount
        date: date
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity
    
    test_ledger = Ledger(name="Test Ledger")
    test_account = Account(name="Cash")
    test_amount = Amount(value=100.0, currency="USD")
    test_journal = Journal(description="Test Journal", postings=[])
    test_posting = Posting(
        account=test_account,
        amount=test_amount,
        date=date(2024, 1, 1),
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = Quantity(value=100.0)
    
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance
    assert entry.ledger.name == "Test Ledger"
    assert entry.posting.account.name == "Cash"
    assert entry.balance.value == 100.0


# LLM-generated content at query #97
#--------------------------

```python
def test_read_initial_balances_call():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 1000, "account2": 2000})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}


def test_read_initial_balances_call_different_period():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            if period.start.year == 2024:
                return InitialBalances({"account1": 5000})
            return InitialBalances({"account1": 1000})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 5000}


def test_read_initial_balances_call_empty_balances():
    from datetime import date
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class InitialBalances:
        def __init__(self, balances: dict):
            self.balances = balances
    
    class ConcreteReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {}


# LLM-generated content at query #98
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger = Ledger()
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), journal=journal, amount=amount, account=account, direction="debit")
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is quantity
    assert entry.posting.amount == amount


# LLM-generated content at query #99
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar, Generic
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger(Generic[_T]):
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[dict]:
            return GeneralLedger({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []


# LLM-generated content at query #100
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        account: MockAccount
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Description", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        account=test_account,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor properly assigns all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity


# LLM-generated content at query #101
#--------------------------

```python
def test_build_general_ledger_posting_account_not_in_ledgers():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction, Posting, Amount
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Quantity
    from pypara.accounting.accounts import Account
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    # Create a journal entry with a posting for account2
    entry = JournalEntry(datetime.date(2024, 6, 15), "Test entry", "source_object")
    posting = Posting(entry, datetime.date(2024, 6, 15), account2, Direction.INC, Amount(Quantity(Decimal(100))))
    entry.postings.append(posting)
    
    journal = [entry]
    initial = {account1: Balance(period.since, Quantity(Decimal(1000)))}
    
    # Build general ledger
    gl = build_general_ledger(period, journal, initial)
    
    # The predicate "posting.account not in ledgers" at line 18 should evaluate to False
    # because after the posting is processed, account2 should be in the ledgers
    assert account2 in gl.ledgers
    assert len(gl.ledgers) == 2
    assert account1 in gl.ledgers


# LLM-generated content at query #102
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange
    from pypara.accounting.accounts import Account
    from pypara.accounting.generic import Quantity
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import Balance, Quantity
    from pypara.core import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    account = Account("1000", "Cash", AccountType.ASSET)
    initial = {account: Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    result = build_general_ledger(period, journal, initial)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account in result.ledgers
    assert result.ledgers[account].initial.value == Quantity(Decimal(1000))


def test_build_general_ledger_with_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.core import DateRange
    from pypara.accounting.accounts import Account, AccountType
    from pypara.accounting.generic import Amount
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash", AccountType.ASSET)
    account2 = Account("2000", "Payable", AccountType.LIABILITY)
    
    entry = JournalEntry(date(2023, 6, 15), "Test entry", "source")
    entry.post(date(2023, 6, 15), account1, Quantity(Decimal(500)))
    entry.post(date(2023, 6, 15), account2, Quantity(Decimal(-500)))
    
    journal = [entry]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry
    from pypara.core import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    period = DateRange(date(2023, 6, 1), date(2023, 6, 30))
    account = Account("1000", "Cash", AccountType.ASSET)
    
    entry_in = JournalEntry(date(2023, 6, 15), "In period", "source")
    entry_in.post(date(2023, 6, 15), account, Quantity(Decimal(100)))
    
    entry_out = JournalEntry(date(2023, 7, 15), "Out of period", "source")
    entry_out.post(date(2023, 7, 15), account, Quantity(Decimal(200)))
    
    journal = [entry_in, entry_out]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1


def test_build_general_ledger_multiple_postings_same_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry
    from pypara.core import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("1000", "Cash", AccountType.ASSET)
    
    entry1 = JournalEntry(date(2023, 6, 15), "Entry 1", "source")
    entry1.post(date(2023, 6, 15), account, Quantity(Decimal(100)))
    
    entry2 = JournalEntry(date(2023, 6, 20), "Entry 2", "source")
    entry2.post(date(2023, 6, 20), account, Quantity(Decimal(50)))
    
    journal = [entry1, entry2]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 2
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal(100))
    assert result.ledgers[account].entries[1].balance == Quantity(Decimal(150))


# LLM-generated content at query #103
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Entry", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor properly assigned all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity


# LLM-generated content at query #104
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    posting_list = []
    journal = Journal(description="Test transaction", postings=posting_list)
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger(name="General Ledger")

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == quantity


# LLM-generated content at query #105
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        account: MockAccount
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Entry", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    # Assert constructor properly initialized all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #106
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Set up test data
    account = MockAccount(name="Test Account")
    amount = MockAmount(value=100.0, currency="USD")
    journal = MockJournal(description="Test Description", postings=[])
    posting = MockPosting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=journal
    )
    ledger = MockLedger()
    balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    # Assert constructor properly assigned all fields
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #107
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar("_T")

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger()

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is quantity


# LLM-generated content at query #108
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        account: MockAccount
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: object
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        name: str
    
    # Setup test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Description", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger(name="Test Ledger")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assertions
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #109
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", number="123456")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #110
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    from pypara.accounting.journaling import Posting
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    initial_balance1 = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balance2 = Balance(date(2024, 1, 1), Quantity(Decimal(500)))
    initial_balances = {account1: initial_balance1, account2: initial_balance2}
    
    # Create journal entries
    entry1 = JournalEntry(date(2024, 1, 15), "Test entry 1", "source1")
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal(100)))
    entry1.post(date(2024, 1, 15), account2, Quantity(Decimal(-100)))
    
    entry2 = JournalEntry(date(2024, 2, 20), "Test entry 2", "source2")
    entry2.post(date(2024, 2, 20), account1, Quantity(Decimal(-50)))
    entry2.post(date(2024, 2, 20), account2, Quantity(Decimal(50)))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balance1
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal(1100))
    assert ledger1.entries[1].balance == Quantity(Decimal(1050))
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balance2
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal(400))
    assert ledger2.entries[1].balance == Quantity(Decimal(450))


def test_build_general_ledger_with_new_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    initial_balance1 = Balance(date(2024, 1, 1), Quantity(Decimal(5000)))
    initial_balances = {account1: initial_balance1}
    
    # Create journal entry with a new account not in initial balances
    entry = JournalEntry(date(2024, 3, 10), "Revenue entry", "source")
    entry.post(date(2024, 3, 10), account1, Quantity(Decimal(-200)))
    entry.post(date(2024, 3, 10), account2, Quantity(Decimal(200)))
    
    journal = [entry]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Check that new account has correct initial balance
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.initial.date == date(2024, 1, 1)
    assert ledger2.initial.value == Quantity(Decimal(0))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal(200))


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Setup test data
    account = Account("1000", "Cash")
    
    period = DateRange(date(2024, 1, 1), date(2024, 6, 30))
    
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal(1000)))
    initial_balances = {account: initial_balance}
    
    # Create journal entries inside and outside the period
    entry_inside = JournalEntry(date(2024, 3, 15), "Inside period", "source1")
    entry_inside.post(date(2024, 3, 15), account, Quantity(Decimal(100)))
    
    entry_outside = JournalEntry(date(2024, 9, 15), "Outside period", "source2")
    entry_outside.post(date(2024, 9, 15), account, Quantity(Decimal(200)))
    
    journal = [entry_inside, entry_outside]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions - only the inside entry should be included
    ledger = general_ledger.ledgers[account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal(1100))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Setup test data
    account = Account("1000", "Cash")
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal(5000)))
    initial_balances = {account: initial_balance}
    
    # Build general ledger with empty journal
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 1
    assert account in general_ledger.ledgers
    ledger = general_ledger.ledgers[account]


# LLM-generated content at query #111
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Cash")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2024, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #112
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Protocol, TypeVar
    
    _T = TypeVar('_T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, data: dict):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger({"period": (period.start, period.end)})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == (date(2023, 1, 1), date(2023, 12, 31))


# LLM-generated content at query #113
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    mock_account = type('Account', (), {})()
    mock_amount = type('Amount', (), {'__repr__': lambda self: 'Amount(100)'})()
    mock_quantity = type('Quantity', (), {'__repr__': lambda self: 'Quantity(100)'})()
    
    mock_posting = type('Posting', (), {
        'date': date(2023, 1, 15),
        'amount': mock_amount,
        'is_debit': True,
        'is_credit': False,
        'direction': 'debit',
        'account': mock_account,
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': [
                type('PostingItem', (), {'direction': 'debit', 'account': mock_account})(),
                type('PostingItem', (), {'direction': 'credit', 'account': mock_account})()
            ]
        })()
    })()
    
    mock_ledger = type('Ledger', (), {})()

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )

    # Verify constructor properly assigned all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #114
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        direction: str
        account: MockAccount
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test instances
    test_ledger = MockLedger()
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 15),
        journal=test_journal,
        amount=MockAmount(100.0, "USD"),
        direction="debit",
        account=MockAccount("Test Account"),
        is_debit=True,
        is_credit=False
    )
    test_balance = MockQuantity(500.0)

    # Test constructor
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #115
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_ledger = object()
    mock_amount = object()
    mock_journal = object()
    mock_account = object()
    
    # Create a mock Posting object
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'amount': mock_amount,
        'journal': mock_journal,
        'account': mock_account,
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False
    })()
    
    mock_balance = object()
    
    # Test constructor
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Verify all attributes are set correctly
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #116
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockAmount:
        value: float
        currency: str

    @dataclass
    class MockQuantity:
        value: float

    @dataclass
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        account: MockAccount
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class MockLedger:
        pass

    # Create test instances
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_posting = MockPosting(
        date=date(2023, 1, 15),
        amount=test_amount,
        journal=MockJournal(description="Test transaction", postings=[]),
        direction="debit",
        account=test_account
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=500.0)

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assert constructor properly assigned all fields
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_balance


# LLM-generated content at query #117
#--------------------------

```python
def test_build_general_ledger_predicate_filters_postings_within_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.monetary import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account, AccountType

    # Create test data
    account1 = Account("1000", "Cash", AccountType.ASSET)
    account2 = Account("2000", "Payable", AccountType.LIABILITY)
    
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 1, 31)
    period = DateRange(period_start, period_end)
    
    # Create journal entries with postings inside and outside the period
    entry_inside = JournalEntry(datetime.date(2024, 1, 15), "Inside period", "source1")
    entry_inside.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(100)))
    entry_inside.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal(-100)))
    
    entry_before = JournalEntry(datetime.date(2023, 12, 31), "Before period", "source2")
    entry_before.post(datetime.date(2023, 12, 31), account1, Quantity(Decimal(50)))
    entry_before.post(datetime.date(2023, 12, 31), account2, Quantity(Decimal(-50)))
    
    entry_after = JournalEntry(datetime.date(2024, 2, 1), "After period", "source3")
    entry_after.post(datetime.date(2024, 2, 1), account1, Quantity(Decimal(75)))
    entry_after.post(datetime.date(2024, 2, 1), account2, Quantity(Decimal(-75)))
    
    journal = [entry_inside, entry_before, entry_after]
    initial_balances = {
        account1: Balance(period_start, Quantity(Decimal(0))),
        account2: Balance(period_start, Quantity(Decimal(0)))
    }
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that only postings within the period are included
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].posting.journal_entry.date == datetime.date(2024, 1, 15)
    assert general_ledger.ledgers[account2].entries[0].posting.journal_entry.date == datetime.date(2024, 1, 15)


# LLM-generated content at query #118
#--------------------------

```python
def test_read_initial_balances_protocol_call():
    from datetime import date
    from typing import NamedTuple
    
    class DateRange(NamedTuple):
        start: date
        end: date
    
    class InitialBalances(NamedTuple):
        account_id: str
        balance: float
    
    class ReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(account_id="ACC001", balance=1000.0)
    
    reader = ReadInitialBalances()
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))
    result = reader(date_range)
    
    assert result.account_id == "ACC001"
    assert result.balance == 1000.0
    assert isinstance(result, InitialBalances)


def test_read_initial_balances_protocol_call_different_period():
    from datetime import date
    from typing import NamedTuple
    
    class DateRange(NamedTuple):
        start: date
        end: date
    
    class InitialBalances(NamedTuple):
        account_id: str
        balance: float
    
    class ReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            if period.start.month == 2:
                return InitialBalances(account_id="ACC002", balance=5000.0)
            return InitialBalances(account_id="ACC001", balance=1000.0)
    
    reader = ReadInitialBalances()
    date_range = DateRange(start=date(2024, 2, 1), end=date(2024, 2, 29))
    result = reader(date_range)
    
    assert result.account_id == "ACC002"
    assert result.balance == 5000.0


def test_read_initial_balances_protocol_call_zero_balance():
    from datetime import date
    from typing import NamedTuple
    
    class DateRange(NamedTuple):
        start: date
        end: date
    
    class InitialBalances(NamedTuple):
        account_id: str
        balance: float
    
    class ReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(account_id="ACC003", balance=0.0)
    
    reader = ReadInitialBalances()
    date_range = DateRange(start=date(2024, 3, 1), end=date(2024, 3, 31))
    result = reader(date_range)
    
    assert result.account_id == "ACC003"
    assert result.balance == 0.0


# LLM-generated content at query #119
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    mock_account = type('Account', (), {})()
    mock_journal = type('Journal', (), {
        'description': 'Test Description',
        'postings': []
    })()
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'journal': mock_journal,
        'amount': type('Amount', (), {})(),
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False,
        'account': mock_account
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly assigns all attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #120
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        direction: str
        journal: Journal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Test Account")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        direction="debit",
        journal=journal
    )
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #121
#--------------------------

```python
def test_build_general_ledger_posting_account_not_in_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.accounting.ledger import Ledger, build_general_ledger
    from pypara.core.commons import DateRange
    
    # Create test data
    test_date = date(2024, 1, 1)
    period = DateRange(test_date, date(2024, 12, 31))
    
    # Create accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    
    # Create initial balances (only for account1)
    initial_balances = {account1: Balance(test_date, Quantity(Decimal(1000)))}
    
    # Create a journal entry with a posting to account2 (not in initial balances)
    journal_entry = JournalEntry(test_date, "Test entry", "source_obj")
    journal_entry.post(test_date, account1, Quantity(Decimal(-100)))
    journal_entry.post(test_date, account2, Quantity(Decimal(100)))
    
    journal = [journal_entry]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # The predicate at line 18: `if posting.account not in ledgers`
    # should evaluate to False for account1 (it's in initial_balances)
    # and True for account2 (it's not in initial_balances)
    # We test that account2 was added to the ledgers by the condition
    assert account2 in general_ledger.ledgers
    assert general_ledger.ledgers[account2].account == account2


# LLM-generated content at query #122
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockPosting:
        date: date
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
        
        @property
        def journal(self):
            return MockJournal()
    
    @dataclass
    class MockJournal:
        description: str = "Test Description"
        postings: list = None
        
        def __post_init__(self):
            if self.postings is None:
                self.postings = []
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create instances
    mock_ledger = MockLedger()
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=MockAmount(100.0, "USD"),
        account=MockAccount("Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockQuantity(100.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #123
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float
        currency: str

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Journal:
        description: str
        postings: list

    @dataclass
    class Posting(Generic[_T]):
        date: date
        amount: Amount
        account: Account
        journal: Journal
        direction: str
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class Ledger(Generic[_T]):
        pass

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger()

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is quantity
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test transaction"
    assert entry.amount is amount


# LLM-generated content at query #124
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    @dataclass
    class MockAccount:
        name: str
    
    @dataclass
    class MockAmount:
        value: float
        currency: str
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: MockAmount
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    mock_ledger = MockLedger()
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test Journal", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        journal=mock_journal,
        amount=mock_amount,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockQuantity(value=500.0)
    
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert ledger_entry.ledger is mock_ledger
    assert ledger_entry.posting is mock_posting
    assert ledger_entry.balance is mock_balance


# LLM-generated content at query #125
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    mock_account = type('Account', (), {})()
    mock_direction = "debit"
    mock_amount = type('Amount', (), {'value': 100})()
    
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'amount': mock_amount,
        'direction': mock_direction,
        'is_debit': True,
        'is_credit': False,
        'account': mock_account,
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': [
                type('PostingItem', (), {'account': mock_account, 'direction': 'debit'})(),
                type('PostingItem', (), {'account': mock_account, 'direction': 'credit'})()
            ]
        })()
    })()
    
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {'value': 500})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor properly assigns attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


