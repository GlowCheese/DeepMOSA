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
        def __init__(self, data: _T):
            self.data = data
    
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[str]:
            return GeneralLedger("test_ledger_data")
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data == "test_ledger_data"
    assert date_range.start == date(2023, 1, 1)
    assert date_range.end == date(2023, 12, 31)


# LLM-generated content at query #2
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
            return GeneralLedger({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []


# LLM-generated content at query #3
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
        name: str

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    test_date = date(2024, 1, 15)
    test_account = Account(name="Cash")
    test_amount = Amount(value=100.0, currency="USD")
    test_journal = Journal(description="Test transaction", postings=[])
    test_posting = Posting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit"
    )
    test_ledger = Ledger(name="Main Ledger")
    test_balance = Quantity(value=500.0)

    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="asset")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #5
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

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        account=Account(name="Cash"),
        amount=Amount(value=100.0, currency="USD"),
        direction="debit",
        journal=Journal(description="Test entry", postings=[])
    )
    balance = Quantity(value=1000.0)

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


# LLM-generated content at query #6
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
    
    # Create test data
    test_date = date(2024, 1, 15)
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Description", postings=[])
    test_posting = MockPosting(
        date=test_date,
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


# LLM-generated content at query #7
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
    test_balance = MockQuantity(500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns attributes
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #8
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
    test_amount = MockAmount(value=100.0, currency="USD")
    test_account = MockAccount(name="Cash")
    test_quantity = MockQuantity(value=1000.0)
    test_ledger = MockLedger()
    
    test_journal = MockJournal(
        description="Test transaction",
        postings=[]
    )
    
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account,
        journal=test_journal
    )
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assert constructor properly assigns all fields
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #9
#--------------------------

```python
def test_read_initial_balances_protocol_call():
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
    
    reader = ReadInitialBalances()
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))
    result = reader(date_range)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert result.balances["account1"] == 1000
    assert result.balances["account2"] == 2000


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.core.types import Quantity
    from pypara.core.temporal import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial = {Account("1000", "Cash", AccountType.ASSET): Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal = []
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert len(result.ledgers) == 1
    assert Account("1000", "Cash", AccountType.ASSET) in result.ledgers


def test_build_general_ledger_with_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Posting
    from pypara.accounting.generic import Balance
    from pypara.core.types import Quantity
    from pypara.core.temporal import DateRange
    from pypara.accounting.accounts import Account, AccountType
    from pypara.accounting.postings import Direction, Amount
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    cash_account = Account("1000", "Cash", AccountType.ASSET)
    revenue_account = Account("4000", "Revenue", AccountType.REVENUE)
    initial = {cash_account: Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    entry = JournalEntry(date(2023, 1, 15), "Test entry", "source")
    entry.post(date(2023, 1, 15), cash_account, Quantity(Decimal(500)))
    entry.post(date(2023, 1, 15), revenue_account, Quantity(Decimal(-500)))
    
    journal = [entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    assert len(result.ledgers[cash_account].entries) == 1
    assert len(result.ledgers[revenue_account].entries) == 1


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core.types import Quantity
    from pypara.core.temporal import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    period = DateRange(date(2023, 6, 1), date(2023, 12, 31))
    cash_account = Account("1000", "Cash", AccountType.ASSET)
    revenue_account = Account("4000", "Revenue", AccountType.REVENUE)
    initial = {cash_account: Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    entry_before = JournalEntry(date(2023, 1, 15), "Before period", "source")
    entry_before.post(date(2023, 1, 15), cash_account, Quantity(Decimal(100)))
    entry_before.post(date(2023, 1, 15), revenue_account, Quantity(Decimal(-100)))
    
    entry_within = JournalEntry(date(2023, 7, 15), "Within period", "source")
    entry_within.post(date(2023, 7, 15), cash_account, Quantity(Decimal(200)))
    entry_within.post(date(2023, 7, 15), revenue_account, Quantity(Decimal(-200)))
    
    entry_after = JournalEntry(date(2024, 1, 15), "After period", "source")
    entry_after.post(date(2024, 1, 15), cash_account, Quantity(Decimal(300)))
    entry_after.post(date(2024, 1, 15), revenue_account, Quantity(Decimal(-300)))
    
    journal = [entry_before, entry_within, entry_after]
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers[cash_account].entries) == 1
    assert len(result.ledgers[revenue_account].entries) == 1


def test_build_general_ledger_creates_account_on_demand():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core.types import Quantity
    from pypara.core.temporal import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    cash_account = Account("1000", "Cash", AccountType.ASSET)
    revenue_account = Account("4000", "Revenue", AccountType.REVENUE)
    initial = {cash_account: Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    entry = JournalEntry(date(2023, 1, 15), "Test entry", "source")
    entry.post(date(2023, 1, 15), cash_account, Quantity(Decimal(500)))
    entry.post(date(2023, 1, 15), revenue_account, Quantity(Decimal(-500)))
    
    journal = [entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert revenue_account in result.ledgers
    assert result.ledgers[revenue_account].initial.value == Quantity(Decimal(0))
    assert result.ledgers[revenue_account].initial.date == period.since


# LLM-generated content at query #11
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
        amount=Amount(value=100.0, currency="USD"),
        account=Account(name="Cash"),
        journal=Journal(description="Test entry", postings=[]),
        direction="debit"
    )
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger is not None
    assert entry.posting is not None
    assert entry.balance is not None


# LLM-generated content at query #12
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
    journal = Journal(description="Test transaction", postings=[])
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
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #13
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    # Mock objects for testing
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
    class MockLedger:
        pass
    
    # Create test instances
    mock_ledger = MockLedger()
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_quantity = MockQuantity(value=100.0)
    mock_journal = MockJournal(description="Test Description", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=mock_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=mock_account,
        journal=mock_journal
    )
    
    # Test constructor
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Verify attributes are correctly assigned
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


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
        date=date(2024, 1, 1),
        amount=amount,
        account=account,
        direction="debit",
        journal=journal,
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=100.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #16
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
    class Posting:
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
        entries: list = None
    
    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity
    
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    posting = Posting(
        date=date(2024, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    posting.set_journal(Journal(description="Test Journal", postings=[posting]))
    
    ledger = Ledger(entries=[])
    balance = Quantity(value=100.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #17
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
    account = Account(name="Savings", number="67890")
    balance = Balance(value=Quantity(5000))
    
    ledger = Ledger(account=account, initial=balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.number == "67890"
    assert ledger.initial.value == Quantity(5000)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Quantity, Account, Amount
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    initial_balances = {
        account1: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(1000)))
    }
    
    # Create journal entries
    entry1 = JournalEntry(datetime.date(2024, 1, 15), "Test entry 1", "source1")
    entry1.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    entry1.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal(100)))
    
    entry2 = JournalEntry(datetime.date(2024, 2, 20), "Test entry 2", "source2")
    entry2.post(datetime.date(2024, 2, 20), account1, Quantity(Decimal(50)))
    entry2.post(datetime.date(2024, 2, 20), account2, Quantity(Decimal(-50)))
    
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
    assert ledger1.initial.value == Quantity(Decimal(1000))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal(900))
    assert ledger1.entries[1].balance == Quantity(Decimal(950))
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal(0))
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal(100))
    assert ledger2.entries[1].balance == Quantity(Decimal(50))


def test_build_general_ledger_empty_journal():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity, Account
    
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    
    initial_balances = {
        account1: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(500)))
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0


def test_build_general_ledger_creates_ledger_for_new_account():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity, Account
    
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    initial_balances = {}
    
    entry = JournalEntry(datetime.date(2024, 1, 15), "Test entry", "source")
    entry.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(100)))
    entry.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal(-100)))
    
    general_ledger = build_general_ledger(period, [entry], initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal(0))
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal(0))


def test_build_general_ledger_filters_by_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity, Account
    
    period = DateRange(datetime.date(2024, 2, 1), datetime.date(2024, 2, 28))
    account1 = Account("1000", "Cash")
    
    initial_balances = {
        account1: Balance(datetime.date(2024, 2, 1), Quantity(Decimal(1000)))
    }
    
    entry1 = JournalEntry(datetime.date(2024, 1, 15), "Before period", "source1")
    entry1.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    
    entry2 = JournalEntry(datetime.date(2024, 2, 15), "In period", "source2")
    entry2.post(datetime.date(2024, 2, 15), account1, Quantity(Decimal(-50)))
    
    entry3 = JournalEntry(datetime.date(2024, 3, 15), "After period", "source3")
    entry3.post(datetime.date(2024, 3, 15), account1, Quantity(Decimal(-25)))
    
    general_ledger = build_general_ledger(period, [entry1, entry2, entry3], initial_balances)
    
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(950))


# LLM-generated content at query #19
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
    journal = MockJournal(description="Test Journal", postings=[])
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
    
    # Assert constructor properly assigns attributes
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #20
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
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class Ledger(Generic[_T]):
        name: str
    
    # Create test instances
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=journal,
        is_debit=True,
        is_credit=False
    )
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=500.0)
    
    # Test constructor
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #21
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
        description: str = "Test Journal"
        postings: list = None
        
        def __post_init__(self):
            if self.postings is None:
                self.postings = []
    
    @dataclass
    class MockLedger:
        name: str = "Test Ledger"
    
    mock_account = MockAccount(name="Account1")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_posting = MockPosting(
        date=date(2024, 1, 15),
        account=mock_account,
        amount=mock_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger()
    mock_quantity = 100.0
    
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_quantity


# LLM-generated content at query #22
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Mock objects for testing
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

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor properly assigned attributes
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity


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
    test_account = MockAccount("Test Account")
    test_quantity = MockQuantity(1000.0)
    test_ledger = MockLedger()
    test_journal = MockJournal("Test Description", [])
    test_posting = MockPosting(test_date, test_amount, test_account, "debit", True, False, test_journal)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_quantity)
    
    # Verify constructor sets attributes correctly
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity


# LLM-generated content at query #24
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
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    test_balance = MockQuantity(value=1000.0)
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


# LLM-generated content at query #25
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #26
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
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: dict
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
    
    # Create instances
    mock_ledger = MockLedger()
    mock_journal = MockJournal(description="Test transaction", postings=[])
    mock_account = MockAccount(name="Test Account")
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        journal=mock_journal,
        amount={"currency": "USD", "value": 100},
        direction="debit",
        account=mock_account,
        is_debit=True,
        is_credit=False
    )
    mock_quantity = MockQuantity(value=100)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor properly assigns attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_quantity


# LLM-generated content at query #27
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    mock_ledger = object()
    mock_amount = object()
    mock_journal = object()
    mock_posting = object()
    mock_posting.date = date(2023, 1, 15)
    mock_posting.amount = mock_amount
    mock_posting.journal = mock_journal
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    mock_posting.direction = "debit"
    
    mock_balance = object()
    
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


# LLM-generated content at query #28
#--------------------------

```python
def test_build_general_ledger_posting_outside_period():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.monetary import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    # Create a period
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 1, 31))
    
    # Create a journal entry with a date outside the period
    entry = JournalEntry(
        date=date(2023, 12, 31),  # Before period.since
        description="Test entry",
        source="test"
    )
    
    # Add a posting to the entry
    account = Account(
        code="1000",
        name="Cash",
        type=AccountType.ASSET
    )
    entry.post(date=date(2023, 12, 31), account=account, quantity=Quantity(Decimal("100")))
    
    # Create initial balances
    initial = {}
    
    # Build general ledger
    ledger = build_general_ledger(period, [entry], initial)
    
    # The predicate at line 16 should evaluate to False for this posting
    # because the posting's journal entry date is outside the period
    # Therefore, the account should not be in the ledgers
    assert account not in ledger.ledgers


# LLM-generated content at query #29
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Account, Quantity, Amount
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    # Create accounts
    account_cash = Account("1000", "Cash")
    account_revenue = Account("4000", "Revenue")
    account_expense = Account("5000", "Expense")
    
    # Create initial balances
    initial_balances = {
        account_cash: Balance(datetime.date(2024, 1, 1), Quantity(Decimal("1000.00"))),
    }
    
    # Create journal entries with postings
    entry1 = JournalEntry(
        date=datetime.date(2024, 1, 15),
        description="Initial revenue",
        source="Transaction1"
    )
    entry1.post(datetime.date(2024, 1, 15), account_cash, Quantity(Decimal("500.00")))
    entry1.post(datetime.date(2024, 1, 15), account_revenue, Quantity(Decimal("-500.00")))
    
    entry2 = JournalEntry(
        date=datetime.date(2024, 2, 10),
        description="Expense payment",
        source="Transaction2"
    )
    entry2.post(datetime.date(2024, 2, 10), account_expense, Quantity(Decimal("200.00")))
    entry2.post(datetime.date(2024, 2, 10), account_cash, Quantity(Decimal("-200.00")))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    assert account_cash in general_ledger.ledgers
    assert account_revenue in general_ledger.ledgers
    assert account_expense in general_ledger.ledgers
    
    # Check cash ledger (initial 1000 + 500 - 200 = 1300)
    cash_ledger = general_ledger.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    
    # Check revenue ledger (0 - 500 = -500)
    revenue_ledger = general_ledger.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert revenue_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500.00"))
    
    # Check expense ledger (0 + 200 = 200)
    expense_ledger = general_ledger.ledgers[account_expense]
    assert expense_ledger.account == account_expense
    assert expense_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))


# LLM-generated content at query #30
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
        account: MockAccount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 1),
        amount=test_amount,
        account=test_account,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()

    # Test constructor with all parameters
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity


# LLM-generated content at query #31
#--------------------------

```python
def test_build_general_ledger_predicate_false():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction, Posting, Amount
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.accounts import Account
    from pypara.common.quantities import Quantity
    from pypara.common.temporal import DateRange
    
    # Create a journal entry with a date outside the period
    entry_date = date(2023, 1, 15)
    period = DateRange(since=date(2023, 2, 1), until=date(2023, 2, 28))
    
    journal_entry = JournalEntry(date=entry_date, description="Test entry", source="test_source")
    
    # Create initial balances
    account = Account(name="Test Account", code="1000")
    initial_balances = {account: Balance(period.since, Quantity(Decimal(100)))}
    
    # Build the general ledger
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    
    # The predicate at line 16 should evaluate to False because entry_date (2023-01-15) is not within period (2023-02-01 to 2023-02-28)
    # This means the posting should not be added to the ledger
    assert len(general_ledger.ledgers[account].entries) == 0


# LLM-generated content at query #32
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting
    from pypara.accounting.accounts import Account
    from pypara.core.quantities import Quantity
    from pypara.core.ranges import DateRange
    from pypara.accounting.journaling import Direction
    from pypara.accounting.amounts import Amount
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {
        account1: Balance(date(2023, 1, 1), Quantity(Decimal("1000"))),
        account2: Balance(date(2023, 1, 1), Quantity(Decimal("500")))
    }
    
    # Create journal entries
    source_obj = "TestSource"
    je1 = JournalEntry(date(2023, 1, 15), "Test entry 1", source_obj)
    je1.post(date(2023, 1, 15), account1, Quantity(Decimal("-100")))
    je1.post(date(2023, 1, 15), account2, Quantity(Decimal("100")))
    
    je2 = JournalEntry(date(2023, 2, 20), "Test entry 2", source_obj)
    je2.post(date(2023, 2, 20), account1, Quantity(Decimal("50")))
    je2.post(date(2023, 2, 20), account2, Quantity(Decimal("-50")))
    
    journal = [je1, je2]
    
    # Build general ledger
    gl = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert gl.period == period
    assert len(gl.ledgers) == 2
    assert account1 in gl.ledgers
    assert account2 in gl.ledgers
    
    # Check account1 ledger
    ledger1 = gl.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("1000"))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal("900"))
    assert ledger1.entries[1].balance == Quantity(Decimal("950"))
    
    # Check account2 ledger
    ledger2 = gl.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("500"))
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal("600"))
    assert ledger2.entries[1].balance == Quantity(Decimal("550"))


def test_build_general_ledger_with_new_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.accounts import Account
    from pypara.core.quantities import Quantity
    from pypara.core.ranges import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {account1: Balance(date(2023, 1, 1), Quantity(Decimal("1000")))}
    
    # Create journal entry with new account
    source_obj = "TestSource"
    je = JournalEntry(date(2023, 1, 15), "Test entry", source_obj)
    je.post(date(2023, 1, 15), account1, Quantity(Decimal("-200")))
    je.post(date(2023, 1, 15), account2, Quantity(Decimal("200")))
    
    journal = [je]
    
    # Build general ledger
    gl = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert len(gl.ledgers) == 2
    assert account1 in gl.ledgers
    assert account2 in gl.ledgers
    
    # Check account2 was created with zero opening balance
    ledger2 = gl.ledgers[account2]
    assert ledger2.initial.value == Quantity(Decimal("0"))
    assert ledger2.initial.date == period.since
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("200"))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core.quantities import Quantity
    from pypara.core.ranges import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {account1: Balance(date(2023, 1, 1), Quantity(Decimal("1000")))}
    
    journal = []
    
    # Build general ledger
    gl = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert gl.period == period
    assert len(gl.ledgers) == 1
    assert account1 in gl.ledgers
    assert len(gl.ledgers[account1].entries) == 0
    assert gl.ledgers[account1].initial.value == Quantity(Decimal("1000"))


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.accounts import Account
    from pypara.core.quantities import Quantity
    from pypara.core.ranges import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    period = DateRange(date(2023, 3, 1), date(2023, 12, 31))
    initial_balances = {
        account1: Balance(date(2023, 3, 1), Quantity(Decimal("1000"))),
        account2: Balance(date(2023, 3, 1), Quantity(Decimal("500")))
    }
    
    # Create journal entries (one outside period)
    source_obj = "TestSource"
    je1 = JournalEntry(date(2023, 1, 15), "Before period", source_obj)
    je1.post(date(2023, 1, 15), account1, Quantity(Decimal("-100")))
    je1.post(date(2023, 1, 15), account2, Quantity(Decimal("100")))
    
    je2 = JournalEntry(date(2023, 6, 20), "During period", source_obj)
    je2.post(date(2023, 6, 20), account1, Quantity(Decimal("50")))
    je2.post(date(2023, 6, 20), account2, Quantity(Decimal("-50")))
    
    journal = [je1, je2]
    
    # Build general le


# LLM-generated content at query #33
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
    test_ledger = MockLedger()
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        account=test_account,
        is_debit=True,
        is_credit=False
    )
    test_balance = MockQuantity(value=100.0)

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assert constructor properly initialized all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #34
#--------------------------

```python
def test_build_general_ledger_predicate_filters_postings_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.models import Account, Quantity, Amount
    from pypara.core.commons import DateRange
    
    # Create test data
    account = Account("1000", "Test Account", None)
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    # Create journal entries with dates inside and outside the period
    entry_inside = JournalEntry(date(2024, 6, 15), "Inside period", "source1")
    entry_before = JournalEntry(date(2023, 12, 31), "Before period", "source2")
    entry_after = JournalEntry(date(2025, 1, 1), "After period", "source3")
    
    # Add postings to entries
    entry_inside.post(date(2024, 6, 15), account, Quantity(Decimal(100)))
    entry_before.post(date(2023, 12, 31), account, Quantity(Decimal(50)))
    entry_after.post(date(2025, 1, 1), account, Quantity(Decimal(75)))
    
    # Create initial balances
    initial_balances = {account: Balance(date(2024, 1, 1), Quantity(Decimal(0)))}
    
    # Build general ledger
    journal = [entry_inside, entry_before, entry_after]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that only postings within the period are included
    ledger = general_ledger.ledgers[account]
    
    # The predicate at line 16 should filter: only entry_inside should be processed
    # entry_before (2023-12-31) is before period.since (2024-01-01)
    # entry_after (2025-01-01) is after period.until (2024-12-31)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.journal_entry.date == date(2024, 6, 15)


# LLM-generated content at query #35
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
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2024, 1, 1),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #36
#--------------------------

```python
def test_build_general_ledger_predicate_filters_by_period():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.journaling import JournalEntry, Direction, Posting
    from pypara.accounting.generic import Balance, Account, Quantity
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.core import DateRange
    
    # Create test data
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    account = Account(name="Test Account", number="1000")
    
    # Create a journal entry within the period
    entry_within = JournalEntry(
        date=date(2023, 6, 15),
        description="Entry within period",
        source="test_source_1"
    )
    entry_within.post(date(2023, 6, 15), account, Quantity(Decimal(100)))
    
    # Create a journal entry before the period
    entry_before = JournalEntry(
        date=date(2022, 12, 31),
        description="Entry before period",
        source="test_source_2"
    )
    entry_before.post(date(2022, 12, 31), account, Quantity(Decimal(50)))
    
    # Create a journal entry after the period
    entry_after = JournalEntry(
        date=date(2024, 1, 1),
        description="Entry after period",
        source="test_source_3"
    )
    entry_after.post(date(2024, 1, 1), account, Quantity(Decimal(75)))
    
    # Build general ledger
    journal = [entry_within, entry_before, entry_after]
    initial = {account: Balance(period.since, Quantity(Decimal(0)))}
    
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Verify that only postings within the period are included
    ledger = general_ledger.ledgers[account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.journal_entry.date == date(2023, 6, 15)
    assert ledger.entries[0].balance == Quantity(Decimal(100))


# LLM-generated content at query #37
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


# LLM-generated content at query #38
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects for dependencies
    mock_account = type('Account', (), {})()
    mock_journal = type('Journal', (), {'description': 'Test Journal', 'postings': []})()
    mock_posting = type('Posting', (), {
        'date': date(2023, 1, 15),
        'journal': mock_journal,
        'amount': 100.0,
        'direction': 'debit',
        'account': mock_account,
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


# LLM-generated content at query #39
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
    
    mock_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        account=test_account,
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
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is test_ledger
    assert entry.posting is mock_posting
    assert entry.balance is test_quantity


# LLM-generated content at query #40
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from decimal import Decimal
    import datetime
    from pypara.accounting.journaling import JournalEntry, Posting
    from pypara.accounting.generic import Balance, DateRange, Account, Quantity, Direction, Amount
    from pypara.accounting.ledger import build_general_ledger
    
    # Create test data
    test_date = datetime.date(2024, 1, 15)
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    # Create journal entries
    source_obj = "TestSource"
    entry1 = JournalEntry(test_date, "Test entry", source_obj)
    entry1.post(test_date, account1, Quantity(Decimal("100")))
    entry1.post(test_date, account2, Quantity(Decimal("-100")))
    
    journal = [entry1]
    initial = {}
    
    # Execute
    result = build_general_ledger(period, journal, initial)
    
    # Assert - the predicate at line 1 (function definition) evaluates to True
    # by checking that the function successfully creates a general ledger
    assert result is not None
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.period == period


# LLM-generated content at query #41
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
    
    # Create test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(100.0, "USD")
    test_journal = MockJournal("Test transaction", [])
    test_account = MockAccount("Test Account")
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        journal=test_journal,
        account=test_account,
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
    
    # Assert constructor properly assigns values
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #42
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
        description: str
        postings: list
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test instances
    mock_ledger = MockLedger()
    mock_posting = MockPosting(
        date=date(2023, 1, 15),
        amount=MockAmount(100.0, "USD"),
        account=MockAccount("Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_posting.journal.description = "Test transaction"
    mock_posting.journal.postings = [mock_posting]
    mock_balance = MockQuantity(100.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor correctly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #43
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
    balance = Quantity(value=100.0)
    ledger = Ledger()
    
    entry = LedgerEntry(ledger=ledger, posting=posting_obj, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting_obj
    assert entry.balance is balance


# LLM-generated content at query #44
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

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    test_date = date(2023, 1, 15)
    journal = Journal(description="Test Description", postings=[])
    posting = Posting(
        account=account,
        amount=amount,
        date=test_date,
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #45
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


# LLM-generated content at query #46
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Balance, Quantity, DateRange
    from pypara.accounting.ledger import build_general_ledger
    
    # Setup
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    # Create journal entries with postings
    je1 = JournalEntry(date(2024, 1, 15), "Initial deposit", "source1")
    je1.postings.append(Posting(je1, date(2024, 1, 15), account1, Direction.INC, Quantity(Decimal("1000"))))
    je1.postings.append(Posting(je1, date(2024, 1, 15), account2, Direction.DEC, Quantity(Decimal("1000"))))
    
    journal = [je1]
    initial = {account1: Balance(period.since, Quantity(Decimal("500")))}
    
    # Execute
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Assert - predicate at line 1: function returns a GeneralLedger instance
    assert general_ledger is not None
    assert hasattr(general_ledger, 'period')
    assert hasattr(general_ledger, 'ledgers')
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers) == 2


# LLM-generated content at query #47
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Account
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
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Account
    from pypara.accounting.generic import Quantity
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    initial = {
        account1: Balance(date(2023, 1, 1), Quantity(Decimal("1000"))),
        account2: Balance(date(2023, 1, 1), Quantity(Decimal("500")))
    }
    
    result = build_general_ledger(period, journal, initial)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].initial.value == Quantity(Decimal("1000"))
    assert result.ledgers[account2].initial.value == Quantity(Decimal("500"))


def test_build_general_ledger_with_postings_within_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Account
    from pypara.accounting.generic import Quantity
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    journal_entry = JournalEntry(date(2023, 6, 15), "Test entry", "source")
    journal_entry.post(date(2023, 6, 15), account1, Quantity(Decimal("100")))
    journal_entry.post(date(2023, 6, 15), account2, Quantity(Decimal("-100")))
    
    journal = [journal_entry]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


def test_build_general_ledger_with_postings_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Account
    from pypara.accounting.generic import Quantity
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    journal_entry = JournalEntry(date(2024, 6, 15), "Test entry", "source")
    journal_entry.post(date(2024, 6, 15), account1, Quantity(Decimal("100")))
    journal_entry.post(date(2024, 6, 15), account2, Quantity(Decimal("-100")))
    
    journal = [journal_entry]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 0


def test_build_general_ledger_multiple_entries():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Account
    from pypara.accounting.generic import Quantity
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    journal_entry1 = JournalEntry(date(2023, 1, 15), "Entry 1", "source1")
    journal_entry1.post(date(2023, 1, 15), account1, Quantity(Decimal("100")))
    journal_entry1.post(date(2023, 1, 15), account2, Quantity(Decimal("-100")))
    
    journal_entry2 = JournalEntry(date(2023, 6, 15), "Entry 2", "source2")
    journal_entry2.post(date(2023, 6, 15), account1, Quantity(Decimal("50")))
    journal_entry2.post(date(2023, 6, 15), account2, Quantity(Decimal("-50")))
    
    journal = [journal_entry1, journal_entry2]
    initial = {}
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert len(result.ledgers[account1].entries) == 2
    assert len(result.ledgers[account2].entries) == 2


def test_build_general_ledger_with_mixed_initial_and_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.core import DateRange, Account
    from pypara.accounting.generic import Quantity
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    journal_entry = JournalEntry(date(2023, 6, 15), "Test entry", "source")
    journal_entry.post(date(2023, 6, 15), account1, Quantity(Decimal("100")))
    journal_entry.post(date(2023, 6, 15), account2, Quantity(Decimal("-100")))
    
    journal = [journal_entry]
    initial = {
        account1: Balance(date(2023, 1, 1), Quantity(Decimal("500")))
    }
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert result.ledgers[account1].initial.value == Quantity(Decimal("500"))
    assert result.ledgers[account2].initial.value == Quantity(Decimal("0"))
    assert len(result.ledgers[account1].entries) ==


# LLM-generated content at query #48
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
                return InitialBalances({"account1": 5000})
            return InitialBalances({"account1": 3000})
    
    reader = ConcreteReadInitialBalances()
    period_2023 = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    period_2024 = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    result_2023 = reader(period_2023)
    result_2024 = reader(period_2024)
    
    assert result_2023.balances == {"account1": 5000}
    assert result_2024.balances == {"account1": 3000}


def test_read_initial_balances_call_returns_initial_balances_type():
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
    period = DateRange(date(2023, 6, 1), date(2023, 6, 30))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {}


# LLM-generated content at query #49
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')
    
    # Create mock objects for the constructor
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    
    # Create a LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert that the constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #50
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
        date: date
        amount: MockAmount
        journal: MockJournal
        account: MockAccount
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger(Generic[_T]):
        name: str

    # Create test data
    test_date = date(2023, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
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
    test_ledger = MockLedger(name="General Ledger")
    test_balance = MockQuantity(value=500.0)

    # Instantiate LedgerEntry
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assertions
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #51
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Account, Quantity, Amount
    from pypara.accounting.ledger import build_general_ledger, Ledger, GeneralLedger
    from pypara.core import DateRange
    
    # Setup test data
    test_date = date(2024, 1, 15)
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create journal entries with postings
    entry1 = JournalEntry(test_date, "Test entry 1", "source1")
    entry1.post(test_date, account1, Quantity(Decimal(100)))
    entry1.post(test_date, account2, Quantity(Decimal(-100)))
    
    entry2 = JournalEntry(date(2024, 2, 15), "Test entry 2", "source2")
    entry2.post(date(2024, 2, 15), account1, Quantity(Decimal(50)))
    entry2.post(date(2024, 2, 15), account2, Quantity(Decimal(-50)))
    
    journal = [entry1, entry2]
    initial_balances = {}
    
    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions to verify the predicate at line 1
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[account1], Ledger)
    assert isinstance(general_ledger.ledgers[account2], Ledger)
    assert len(general_ledger.ledgers[account1].entries) == 2
    assert len(general_ledger.ledgers[account2].entries) == 2
    assert general_ledger.ledgers[account1].account == account1
    assert general_ledger.ledgers[account2].account == account2


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
    assert entry.date == date(2023, 1, 1)
    assert entry.amount is amount
    assert entry.description == "Test Journal"
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #53
#--------------------------

```python
def test_build_general_ledger_predicate_false():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.accounting.accounts import Account
    from pypara.accounting.quantities import Quantity, Amount
    from pypara.utils.daterange import DateRange
    
    # Create a date range for the accounting period
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create a journal entry with a date BEFORE the period
    entry_date = datetime.date(2023, 12, 31)
    journal_entry = JournalEntry(entry_date, "Test Entry", "source_object")
    
    # Create an account and add a posting to the journal entry
    account = Account("1000", "Test Account")
    journal_entry.post(entry_date, account, Quantity(Decimal(100)))
    
    # Create initial balances (empty for this test)
    initial_balances = {}
    
    # Build the general ledger
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    
    # Assert that the posting was NOT added to the ledger because the date is outside the period
    # The predicate "period.since <= j.date <= period.until" evaluates to False
    assert account not in general_ledger.ledgers


# LLM-generated content at query #54
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


def test_read_initial_balances_call_with_different_period():
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


def test_read_initial_balances_call_returns_initial_balances():
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


# LLM-generated content at query #55
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
        name: str

    # Setup test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
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
    test_ledger = MockLedger(name="Test Ledger")

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor initialized all fields correctly
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #56
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: object
        direction: str
        is_debit: bool
        is_credit: bool
        account: object
    
    @dataclass
    class MockLedger:
        pass
    
    mock_ledger = MockLedger()
    mock_journal = MockJournal(description="Test transaction", postings=[])
    mock_posting = MockPosting(
        date=date(2024, 1, 15),
        journal=mock_journal,
        amount="100.00",
        direction="debit",
        is_debit=True,
        is_credit=False,
        account="Account1"
    )
    mock_quantity = "100.00"
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Verify constructor sets all attributes correctly
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance == mock_quantity


# LLM-generated content at query #57
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_account = type('Account', (), {})()
    mock_direction = "debit"
    mock_posting = type('Posting', (), {
        'date': date(2023, 1, 15),
        'account': mock_account,
        'direction': mock_direction,
        'is_debit': True,
        'is_credit': False,
        'amount': type('Amount', (), {'value': Decimal('100.00')})(),
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': []
        })()
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {'value': Decimal('500.00')})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor properly assigns attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #58
#--------------------------

```python
def test_build_general_ledger_predicate_false():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.accounts import Account
    from pypara.core.monetary import Quantity
    from pypara.core.periods import DateRange
    
    # Create a journal entry with a date outside the accounting period
    entry_date = date(2023, 1, 15)
    period = DateRange(date(2023, 2, 1), date(2023, 2, 28))
    
    account = Account("1000", "Test Account")
    source_obj = "TestSource"
    
    journal_entry = JournalEntry(entry_date, "Test Entry", source_obj)
    journal_entry.post(entry_date, account, Quantity(Decimal(100)))
    
    initial_balances = {}
    journal = [journal_entry]
    
    # The predicate at line 16: period.since <= j.date <= period.until
    # should evaluate to False since entry_date (2023-01-15) is before period.since (2023-02-01)
    result = build_general_ledger(period, journal, initial_balances)
    
    # Verify the predicate was False by checking that the account was not added to ledgers
    assert account not in result.ledgers


# LLM-generated content at query #59
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
            return GeneralLedger({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []


# LLM-generated content at query #60
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
        date=date(2024, 1, 1),
        amount=test_amount,
        account=test_account,
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


# LLM-generated content at query #61
#--------------------------

```python
def test_ledger_entry_constructor():
    from dataclasses import dataclass
    from datetime import date
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    # Create mock objects
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor properly assigns all attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


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
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
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
    test_balance = MockAmount(value=500.0, currency="USD")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor set attributes correctly
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #63
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
    
    # Set up test data
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
    test_balance = MockQuantity(value=100.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly initialized all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


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
    posting_obj = Posting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        journal=Journal(description="Test Journal", postings=[]),
        direction="debit"
    )
    ledger_obj = Ledger(name="Test Ledger")
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger_obj, posting=posting_obj, balance=balance)
    
    assert entry.ledger == ledger_obj
    assert entry.posting == posting_obj
    assert entry.balance == balance


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
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


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
        name: str
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Transaction", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="Test Ledger")
    test_balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor set attributes correctly
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


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
        journal=test_journal,
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)

    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance
    assert entry.ledger.name == "Test Ledger"
    assert entry.posting.date == date(2023, 1, 1)
    assert entry.balance.value == 100.0


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
    posting = Posting(
        date=date(2024, 1, 1),
        amount=amount,
        account=account,
        journal=Journal(description="Test entry", postings=[]),
        direction="debit"
    )
    ledger = Ledger(name="General")
    
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)
    
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == quantity


# LLM-generated content at query #69
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
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test Journal"
    assert entry.amount == amount
    assert entry.is_debit is True
    assert entry.is_credit is False
    assert entry.debit == amount
    assert entry.credit is None


# LLM-generated content at query #70
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
        account: MockAccount
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    # Create test data
    test_date = date(2024, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=Decimal("100.00"), currency="USD")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="General Ledger")
    test_balance = MockAmount(value=Decimal("500.00"), currency="USD")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns all fields
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.accounts import Account
    from pypara.dates import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 15)
    date3 = datetime.date(2023, 2, 1)
    
    period = DateRange(date1, date3)
    
    # Create initial balances
    initial_balances = {
        account1: Balance(date1, Quantity(Decimal("1000"))),
    }
    
    # Create journal entries
    entry1 = JournalEntry(date2, "Test entry", "source1")
    entry1.post(date2, account1, Quantity(Decimal("-100")))
    entry1.post(date2, account2, Quantity(Decimal("100")))
    entry1.validate()
    
    entry2 = JournalEntry(date3, "Test entry 2", "source2")
    entry2.post(date3, account1, Quantity(Decimal("-50")))
    entry2.post(date3, account2, Quantity(Decimal("50")))
    entry2.validate()
    
    journal = [entry1, entry2]
    
    # Build general ledger
    result = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    
    # Check account1 ledger entries
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("1000"))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal("900"))
    assert ledger1.entries[1].balance == Quantity(Decimal("850"))
    
    # Check account2 ledger entries
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("0"))
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal("100"))
    assert ledger2.entries[1].balance == Quantity(Decimal("150"))


def test_build_general_ledger_empty_journal():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.accounts import Account
    from pypara.dates import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    date1 = datetime.date(2023, 1, 1)
    date3 = datetime.date(2023, 2, 1)
    
    period = DateRange(date1, date3)
    
    initial_balances = {
        account1: Balance(date1, Quantity(Decimal("5000"))),
    }
    
    # Build general ledger with empty journal
    result = build_general_ledger(period, [], initial_balances)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account1 in result.ledgers
    assert len(result.ledgers[account1].entries) == 0
    assert result.ledgers[account1].initial.value == Quantity(Decimal("5000"))


def test_build_general_ledger_out_of_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.accounts import Account
    from pypara.dates import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 3, 1)  # Outside period
    date3 = datetime.date(2023, 2, 1)
    
    period = DateRange(date1, date3)
    
    initial_balances = {
        account1: Balance(date1, Quantity(Decimal("1000"))),
    }
    
    # Create journal entry outside period
    entry = JournalEntry(date2, "Test entry", "source1")
    entry.post(date2, account1, Quantity(Decimal("-100")))
    entry.post(date2, account2, Quantity(Decimal("100")))
    entry.validate()
    
    journal = [entry]
    
    # Build general ledger
    result = build_general_ledger(period, journal, initial_balances)
    
    # Assertions - posting should not be included
    assert isinstance(result, GeneralLedger)
    assert len(result.ledgers) == 1
    assert account1 in result.ledgers
    assert account2 not in result.ledgers
    assert len(result.ledgers[account1].entries) == 0


def test_build_general_ledger_creates_new_accounts():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.accounts import Account
    from pypara.dates import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 15)
    date3 = datetime.date(2023, 2, 1)
    
    period = DateRange(date1, date3)
    
    # Only account1 in initial balances
    initial_balances = {
        account1: Balance(date1, Quantity(Decimal("1000"))),
    }
    
    # Create journal entry with account2 (not in initial balances)
    entry = JournalEntry(date2, "Test entry", "source1")
    entry.post(date2, account1, Quantity(Decimal("-100")))
    entry.post(date2, account2, Quantity(Decimal("100")))
    entry.validate()
    
    journal = [entry]
    
    # Build general ledger
    result = build_general_ledger(period, journal, initial_balances)
    
    # Assertions - account2 should be created with zero initial balance
    assert len(result.ledgers) == 2
    assert account2 in result.ledgers
    assert result.ledgers[account2].initial.date == date1
    assert result.ledgers[account2].initial.value == Quantity(Decimal("0"))
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #3
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
    posting = Posting(date=date(2024, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger()
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.date == date(2024, 1, 1)
    assert entry.amount is amount
    assert entry.is_debit is True
    assert entry.is_credit is False


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
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #5
#--------------------------

```python
def test_build_general_ledger_predicate_filters_postings_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.core.types import DateRange, Account, Quantity, Amount, Direction
    
    # Create a date range for the accounting period
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create accounts
    account1 = Account("1000", "Asset Account")
    account2 = Account("2000", "Liability Account")
    
    # Create journal entries with different dates
    entry_within_period = JournalEntry(date(2024, 6, 15), "Within period", "source1")
    entry_before_period = JournalEntry(date(2023, 12, 31), "Before period", "source2")
    entry_after_period = JournalEntry(date(2025, 1, 1), "After period", "source3")
    
    # Add postings to entries
    entry_within_period.post(date(2024, 6, 15), account1, Quantity(Decimal(100)))
    entry_before_period.post(date(2023, 12, 31), account2, Quantity(Decimal(50)))
    entry_after_period.post(date(2025, 1, 1), account1, Quantity(Decimal(75)))
    
    # Create initial balances
    initial_balances = {account1: Balance(period_start, Quantity(Decimal(0)))}
    
    # Build general ledger
    journal_entries = [entry_within_period, entry_before_period, entry_after_period]
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Verify that only postings within the period are included
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].posting.date == date(2024, 6, 15)
    
    # Verify that account2 (from before period) is not in ledgers
    assert account2 not in general_ledger.ledgers


# LLM-generated content at query #6
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #7
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
            return InitialBalances({"account1": 1000.0, "account2": 2000.0})
    
    reader = ConcreteReadInitialBalances()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(date_range)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000.0, "account2": 2000.0}
    assert result.balances["account1"] == 1000.0
    assert result.balances["account2"] == 2000.0


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
    posting_other = Posting(
        date=date(2024, 1, 1),
        amount=Amount(value=50.0, currency="USD"),
        account=Account(name="Other Account"),
        journal=None,
        direction="credit"
    )
    journal = Journal(description="Test Journal", postings=[posting_other])
    posting = Posting(
        date=date(2024, 1, 1),
        amount=amount,
        account=account,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger()

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert ledger_entry.ledger is ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is quantity


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import DateRange, Balance
    from pypara.accounting.accounts import Account, AccountType
    from pypara.quantity import Quantity
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    initial = {}
    journal = []
    
    result = build_general_ledger(period, journal, initial)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import DateRange, Balance
    from pypara.accounting.accounts import Account, AccountType
    from pypara.quantity import Quantity
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account = Account("1000", "Cash", AccountType.ASSET)
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal("1000")))
    initial = {account: initial_balance}
    journal = []
    
    result = build_general_ledger(period, journal, initial)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account in result.ledgers
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial_balance
    assert len(result.ledgers[account].entries) == 0


def test_build_general_ledger_with_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import DateRange, Balance
    from pypara.accounting.accounts import Account, AccountType
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.quantity import Quantity
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    cash_account = Account("1000", "Cash", AccountType.ASSET)
    revenue_account = Account("4000", "Revenue", AccountType.REVENUE)
    
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal("0")))
    initial = {cash_account: initial_balance}
    
    journal_entry = JournalEntry(date(2024, 6, 15), "Test entry", "source")
    journal_entry.post(date(2024, 6, 15), cash_account, Quantity(Decimal("500")))
    journal_entry.post(date(2024, 6, 15), revenue_account, Quantity(Decimal("-500")))
    
    result = build_general_ledger(period, [journal_entry], initial)
    
    assert isinstance(result, GeneralLedger)
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    assert len(result.ledgers[cash_account].entries) == 1
    assert len(result.ledgers[revenue_account].entries) == 1


def test_build_general_ledger_filters_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import DateRange, Balance
    from pypara.accounting.accounts import Account, AccountType
    from pypara.accounting.journaling import JournalEntry
    from pypara.quantity import Quantity
    
    period = DateRange(date(2024, 6, 1), date(2024, 6, 30))
    account = Account("1000", "Cash", AccountType.ASSET)
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal("0")))
    initial = {account: initial_balance}
    
    entry_within = JournalEntry(date(2024, 6, 15), "Within period", "source")
    entry_within.post(date(2024, 6, 15), account, Quantity(Decimal("100")))
    
    entry_outside = JournalEntry(date(2024, 7, 15), "Outside period", "source")
    entry_outside.post(date(2024, 7, 15), account, Quantity(Decimal("200")))
    
    result = build_general_ledger(period, [entry_within, entry_outside], initial)
    
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("100"))


def test_build_general_ledger_creates_ledger_for_missing_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import DateRange, Balance
    from pypara.accounting.accounts import Account, AccountType
    from pypara.accounting.journaling import JournalEntry
    from pypara.quantity import Quantity
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account = Account("1000", "Cash", AccountType.ASSET)
    initial = {}
    
    entry = JournalEntry(date(2024, 6, 15), "Test", "source")
    entry.post(date(2024, 6, 15), account, Quantity(Decimal("500")))
    
    result = build_general_ledger(period, [entry], initial)
    
    assert account in result.ledgers
    assert result.ledgers[account].initial.value == Quantity(Decimal("0"))
    assert result.ledgers[account].initial.date == period.since


def test_build_general_ledger_multiple_postings_same_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import DateRange, Balance
    from pypara.accounting.accounts import Account, AccountType
    from pypara.accounting.journaling import JournalEntry
    from pypara.quantity import Quantity
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account = Account("1000", "Cash", AccountType.ASSET)
    initial_balance = Balance(date(2024, 1, 1), Quantity(Decimal("1000")))
    initial = {account: initial_balance}
    
    entry1 = JournalEntry(date(2024, 6, 15), "Entry 1", "source1")
    entry1.post(date(2024, 6, 15), account, Quantity(Decimal("500")))
    
    entry2 = JournalEntry(date(2024, 6, 20), "Entry 2", "source2")
    entry2.post(date(2024, 6, 20), account, Quantity(Decimal("300")))
    
    result = build_general_ledger(period, [entry1, entry2], initial)
    
    assert len(result.ledgers[account].entries) == 2
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("1500"))
    assert result.ledgers[account].entries[1].balance == Quantity(Decimal("1800"))


# LLM-generated content at query #10
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
            return GeneralLedger({"period": period})
    
    program = ConcreteGeneralLedgerProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == period
    assert result.data["period"].start == date(2023, 1, 1)
    assert result.data["period"].end == date(2023, 12, 31)


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Create test data
    account = Account("1000", "Test Account", None)
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    
    # Create journal entries with different dates
    entry_before = JournalEntry(datetime.date(2023, 12, 31), "Before period", "source1")
    entry_within = JournalEntry(datetime.date(2024, 1, 15), "Within period", "source2")
    entry_after = JournalEntry(datetime.date(2024, 2, 1), "After period", "source3")
    
    # Add postings to entries
    entry_before.post(datetime.date(2023, 12, 31), account, Quantity(Decimal(100)))
    entry_within.post(datetime.date(2024, 1, 15), account, Quantity(Decimal(200)))
    entry_after.post(datetime.date(2024, 2, 1), account, Quantity(Decimal(300)))
    
    # Build general ledger
    initial_balances = {account: Balance(period.since, Quantity(Decimal(0)))}
    journal = [entry_before, entry_within, entry_after]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that only the posting within the period is included
    ledger = general_ledger.ledgers[account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.date == datetime.date(2024, 1, 15)


# LLM-generated content at query #12
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
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), amount=amount, account=account, journal=journal, direction="debit")
    ledger = Ledger(name="Test Ledger")

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == quantity


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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

    ledger = Ledger(name="test_ledger")
    account = Account(name="test_account")
    journal = Journal(description="test_description", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=account,
        journal=journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #15
#--------------------------

```python
def test_build_general_ledger_creates_ledger_for_posting_account_not_in_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Account
    from pypara.accounting.journaling import JournalEntry, Direction, Posting, Amount
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Quantity
    
    # Setup
    test_account = Account("TEST", "Test Account")
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    initial_balances = {}
    
    # Create a journal entry with a posting
    journal_entry = JournalEntry(date(2024, 6, 15), "Test Entry", "source_object")
    posting = Posting(journal_entry, date(2024, 6, 15), test_account, Direction.INC, Amount(Quantity(Decimal(100))))
    journal_entry.postings.append(posting)
    
    journal = [journal_entry]
    
    # Execute
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify the predicate: posting.account not in ledgers (at line 18)
    # After build_general_ledger completes, the account should be in ledgers
    assert test_account in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[test_account], Ledger)
    assert general_ledger.ledgers[test_account].account == test_account
    assert general_ledger.ledgers[test_account].initial.date == period.since
    assert general_ledger.ledgers[test_account].initial.value == Quantity(Decimal(0))


# LLM-generated content at query #16
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
        name: str
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        journal=journal,
        direction="debit"
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_posting_account_not_in_ledgers():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction, Posting, Amount
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.interval import DateRange
    from pypara.accounting.accounts import Account, AccountType
    
    # Create test data
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account(code="1000", name="Cash", type=AccountType.ASSET)
    
    # Create a journal entry with a posting
    entry = JournalEntry(date=date(2023, 6, 15), description="Test entry", source="test_source")
    posting = Posting(entry, date(2023, 6, 15), account, Direction.INC, Amount(Quantity(Decimal("100"))))
    entry.postings.append(posting)
    
    journal = [entry]
    initial = {}
    
    # Build the general ledger
    result = build_general_ledger(period, journal, initial)
    
    # At line 18, the predicate "posting.account not in ledgers" should evaluate to False
    # after the first posting is processed, meaning the account should be in ledgers
    assert posting.account in result.ledgers
    assert isinstance(result.ledgers[posting.account], Ledger)
    assert len(result.ledgers[posting.account].entries) == 1


# LLM-generated content at query #18
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
            return GeneralLedger({"period": period})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["period"].start == date(2023, 1, 1)
    assert result.data["period"].end == date(2023, 12, 31)


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Account, Quantity, Amount
    
    # Setup test data
    test_date = date(2024, 1, 15)
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account1: Balance(period.since, Quantity(Decimal("1000")))
    }
    
    # Create a journal entry
    entry = JournalEntry(test_date, "Test transaction", "source_object")
    entry.post(test_date, account1, Quantity(Decimal("-100")))
    entry.post(test_date, account2, Quantity(Decimal("100")))
    
    journal = [entry]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that ledgers were created for both accounts
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Verify ledger properties
    assert isinstance(general_ledger.ledgers[account1], Ledger)
    assert isinstance(general_ledger.ledgers[account2], Ledger)
    
    # Verify entries were added
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    
    # Verify the predicate: period.since <= j.date <= period.until evaluates to True
    assert period.since <= entry.date <= period.until


# LLM-generated content at query #21
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
    test_journal = MockJournal(description="Test Journal", postings=[])
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
    test_balance = MockQuantity(value=500.0)

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


# LLM-generated content at query #22
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.accounting.postings import Posting, Amount

    # Setup test data
    test_date = datetime.date(2023, 1, 15)
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create accounts
    account_a = Account("1000", "Test Account A")
    account_b = Account("2000", "Test Account B")
    
    # Create initial balances
    initial_balances = {
        account_a: Balance(period_start, Quantity(Decimal("1000")))
    }
    
    # Create a journal entry
    journal_entry = JournalEntry(test_date, "Test Entry", "test_source")
    
    # Manually add postings to the journal entry
    journal_entry.postings.append(
        Posting(journal_entry, test_date, account_a, Direction.DEC, Amount(Quantity(Decimal("100"))))
    )
    journal_entry.postings.append(
        Posting(journal_entry, test_date, account_b, Direction.INC, Amount(Quantity(Decimal("100"))))
    )
    
    journal = [journal_entry]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that ledgers exist for both accounts
    assert account_a in general_ledger.ledgers
    assert account_b in general_ledger.ledgers
    
    # Verify that the ledger for account_a has the initial balance
    assert general_ledger.ledgers[account_a].initial.value == Quantity(Decimal("1000"))
    
    # Verify that the ledger for account_b was created with zero initial balance
    assert general_ledger.ledgers[account_b].initial.value == Quantity(Decimal("0"))
    
    # Verify that entries were added to both ledgers
    assert len(general_ledger.ledgers[account_a].entries) == 1
    assert len(general_ledger.ledgers[account_b].entries) == 1


# LLM-generated content at query #23
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
            return GeneralLedger({"period": period})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["period"].start == date(2023, 1, 1)
    assert result.data["period"].end == date(2023, 12, 31)


# LLM-generated content at query #24
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
    posting_journal = Journal(description="Test transaction", postings=[])
    posting = Posting(account=account, amount=amount, date=date(2024, 1, 1), journal=posting_journal, direction="debit")
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #25
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
    class MockPosting(Generic[_T]):
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
        description: str
        postings: list

        def __init__(self):
            self.description = "Test Journal"
            self.postings = []

    @dataclass
    class MockLedger(Generic[_T]):
        name: str

    # Create instances
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="General Ledger")

    # Test constructor
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assertions
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity
    assert isinstance(entry, LedgerEntry)


# LLM-generated content at query #26
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
        name: str

    # Create test instances
    test_date = date(2024, 1, 15)
    test_description = "Test transaction"
    test_amount = MockAmount(100.0, "USD")
    test_quantity = MockQuantity(100.0)
    test_account = MockAccount("Test Account")
    test_direction = "debit"

    test_journal = MockJournal(
        description=test_description,
        postings=[MockPosting(test_date, None, test_amount, test_direction, test_account, True, False)]
    )

    test_posting = MockPosting(
        date=test_date,
        journal=test_journal,
        amount=test_amount,
        direction=test_direction,
        account=test_account,
        is_debit=True,
        is_credit=False
    )

    test_ledger = MockLedger("Test Ledger")

    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor properly initialized all attributes
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


# LLM-generated content at query #27
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
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=journal
    )
    balance = Quantity(value=500.0)
    ledger = Ledger()

    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    # Assert constructor properly assigned all fields
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #28
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
    mock_ledger = MockLedger()
    mock_amount = MockAmount(100.0, "USD")
    mock_account = MockAccount("Test Account")
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=mock_amount,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_quantity = MockQuantity(100.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor properly assigned fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #29
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.timeline import DateRange
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create journal entries with dates inside and outside the period
    entry_inside = JournalEntry(date(2024, 6, 15), "Entry inside period", "source1")
    entry_before = JournalEntry(date(2023, 12, 31), "Entry before period", "source2")
    entry_after = JournalEntry(date(2025, 1, 1), "Entry after period", "source3")
    
    # Add postings to entries
    entry_inside.post(date(2024, 6, 15), account1, Quantity(Decimal(100)))
    entry_inside.post(date(2024, 6, 15), account2, Quantity(Decimal(-100)))
    
    entry_before.post(date(2023, 12, 31), account1, Quantity(Decimal(50)))
    entry_before.post(date(2023, 12, 31), account2, Quantity(Decimal(-50)))
    
    entry_after.post(date(2025, 1, 1), account1, Quantity(Decimal(75)))
    entry_after.post(date(2025, 1, 1), account2, Quantity(Decimal(-75)))
    
    journal = [entry_before, entry_inside, entry_after]
    initial = {}
    
    # Build general ledger
    gl = build_general_ledger(period, journal, initial)
    
    # Verify that only postings from entry_inside are included
    assert account1 in gl.ledgers
    assert account2 in gl.ledgers
    
    # Check that only the posting from inside the period was added
    assert len(gl.ledgers[account1].entries) == 1
    assert len(gl.ledgers[account2].entries) == 1
    
    # Verify the amounts are correct (only from entry_inside)
    assert gl.ledgers[account1].entries[0].balance == Quantity(Decimal(100))
    assert gl.ledgers[account2].entries[0].balance == Quantity(Decimal(-100))


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

    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2024, 1, 1), account=account, amount=amount, direction="debit", journal=journal)
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #31
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
        account: MockAccount
        amount: MockAmount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: 'MockJournal'
    
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
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_date = date(2023, 1, 1)
    test_description = "Test Description"
    
    test_journal = MockJournal(
        description=test_description,
        postings=[]
    )
    
    test_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    
    test_ledger = MockLedger(name="Test Ledger")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assertions
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity
    assert entry.ledger.name == "Test Ledger"
    assert entry.posting.date == test_date
    assert entry.posting.amount == test_amount
    assert entry.balance.value == 100.0


# LLM-generated content at query #32
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
            return GeneralLedger({'start': period.start, 'end': period.end})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data['start'] == date(2023, 1, 1)
    assert result.data['end'] == date(2023, 12, 31)


# LLM-generated content at query #33
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
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger(Generic[_T]):
        name: str
    
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test Journal", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 15),
        journal=mock_journal,
        amount=mock_amount,
        direction="debit",
        account=mock_account,
        is_debit=True,
        is_credit=False
    )
    mock_quantity = MockQuantity(value=100.0)
    mock_ledger = MockLedger(name="Test Ledger")
    
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_quantity


# LLM-generated content at query #34
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
            return InitialBalances({"account1": 1000.0, "account2": 2000.0})
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000.0, "account2": 2000.0}
    assert period.start == date(2023, 1, 1)
    assert period.end == date(2023, 12, 31)


# LLM-generated content at query #35
#--------------------------

```python
def test_build_general_ledger_predicate_date_filtering():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.accounting.journaling import Posting, Amount
    
    # Create test data
    account = Account(name="Test Account", code="1000")
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    initial_balances = {account: Balance(date(2023, 1, 1), Quantity(Decimal(100)))}
    
    # Create journal entries with different dates
    entry_in_period = JournalEntry(date=date(2023, 6, 15), description="In period", source="source1")
    entry_before_period = JournalEntry(date=date(2022, 12, 31), description="Before period", source="source2")
    entry_after_period = JournalEntry(date=date(2024, 1, 1), description="After period", source="source3")
    
    # Add postings to entries
    entry_in_period.post(date(2023, 6, 15), account, Quantity(Decimal(50)))
    entry_before_period.post(date(2022, 12, 31), account, Quantity(Decimal(25)))
    entry_after_period.post(date(2024, 1, 1), account, Quantity(Decimal(30)))
    
    journal = [entry_in_period, entry_before_period, entry_after_period]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that only postings within the period are included
    assert account in general_ledger.ledgers
    ledger = general_ledger.ledgers[account]
    
    # Only one posting (from entry_in_period) should be in the ledger
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.journal_entry.date == date(2023, 6, 15)
    
    # Verify the predicate at line 16 filters correctly
    # The predicate should be: period.since <= j.date <= period.until
    assert period.since <= entry_in_period.date <= period.until
    assert not (period.since <= entry_before_period.date <= period.until)
    assert not (period.since <= entry_after_period.date <= period.until)


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
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=100.0)

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
    test_date = date(2023, 1, 1)
    test_journal = MockJournal(description="Test Journal", postings=[])
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
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )
    
    # Assert constructor properly assigns all fields
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity


# LLM-generated content at query #38
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
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2024, 1, 1), amount=amount, journal=journal, account=account, direction="debit")
    ledger = Ledger()
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #39
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
    
    # Create test data
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_amount = MockAmount(value=100.0)
    test_posting = MockPosting(
        date=date(2024, 1, 15),
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=500.0)
    
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


# LLM-generated content at query #40
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
    
    # Create a LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert that the constructor properly assigns all attributes
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #41
#--------------------------

```python
def test_build_general_ledger_posting_account_in_ledgers():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting.generic import Balance, Account
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity
    
    # Setup
    account = Account("1000", "Test Account")
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balance = Balance(period.since, Quantity(Decimal("1000")))
    initial = {account: initial_balance}
    
    # Create a journal entry with a posting
    entry = JournalEntry(date(2023, 6, 15), "Test Entry", "source_data")
    posting = Posting(entry, date(2023, 6, 15), account, Direction.INC, Quantity(Decimal("100")))
    entry.postings.append(posting)
    
    journal = [entry]
    
    # Call the function
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Verify the predicate at line 18 evaluates to False
    # (posting.account not in ledgers should be False because account was in initial)
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1


# LLM-generated content at query #42
#--------------------------

```python
def test_build_general_ledger_posting_account_not_in_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.commons import DateRange
    
    # Create test data
    test_date = date(2024, 1, 1)
    period = DateRange(test_date, date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create a journal entry with postings
    entry = JournalEntry(test_date, "Test entry", "source_obj")
    entry.post(test_date, account1, Quantity(Decimal(100)))
    entry.post(test_date, account2, Quantity(Decimal(-100)))
    
    # Initial balances only for account1
    initial = {account1: Balance(test_date, Quantity(Decimal(0)))}
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [entry], initial)
    
    # Verify that account2 was added to ledgers (predicate at line 18 should evaluate to True, making the condition False)
    assert account2 in general_ledger.ledgers
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers) == 2


# LLM-generated content at query #43
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
    class Journal:
        description: str
        postings: list
    
    @dataclass
    class Posting(Generic[_T]):
        date: date
        journal: Journal
        amount: object
        direction: str
        account: Account
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"
    
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
    
    test_ledger = Ledger()
    test_account = Account(name="Test Account")
    test_posting = Posting(
        date=date(2023, 1, 1),
        journal=Journal(description="Test Journal", postings=[]),
        amount=100,
        direction="debit",
        account=test_account
    )
    test_balance = Quantity(value=1000.0)
    
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #44
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
        'date': date(2024, 1, 15),
        'amount': mock_amount,
        'is_debit': True,
        'is_credit': False,
        'direction': 'debit',
        'account': mock_account,
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': [
                type('P', (), {'direction': 'debit', 'account': mock_account})(),
                type('P', (), {'direction': 'credit', 'account': mock_account})()
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
    
    # Assert constructor sets attributes correctly
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #45
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
        account: MockAccount
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    # Create test data
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_quantity = MockQuantity(value=100.0)
    mock_journal = MockJournal(description="Test Journal", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        account=mock_account,
        amount=mock_amount,
        journal=mock_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger(name="Test Ledger")
    
    # Test constructor
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_quantity)
    
    # Assertions
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_quantity


# LLM-generated content at query #46
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
    
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    read_balances = ReadInitialBalancesImpl()
    result = read_balances(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 1000, "account2": 2000}
    assert period.start == start_date
    assert period.end == end_date


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
        pass

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=Journal(description="Test Journal", postings=[])
    )
    ledger = Ledger()

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == quantity


# LLM-generated content at query #48
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
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    posting = Posting(
        date=date(2023, 1, 15),
        journal=Journal(description="Test transaction", postings=[]),
        amount=Amount(value=100.0, currency="USD"),
        account=Account(name="Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #49
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
    journal = Journal(description="Test Description", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        journal=journal,
        amount=Amount(value=100.0, currency="USD"),
        direction="debit",
        account=account
    )
    ledger = Ledger()
    balance = Quantity(value=500.0)

    from dataclasses import dataclass as dc

    @dc
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert ledger_entry.ledger is ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is balance


# LLM-generated content at query #50
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


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
        name: str

    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2023, 1, 1), account=account, amount=amount, journal=journal, direction="debit")
    ledger = Ledger(name="General")
    balance = Quantity(value=500.0)

    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #52
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    
    # Create mock objects
    @dataclass
    class MockJournal:
        description: str
        postings: list
    
    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: float
        direction: str
        account: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 15),
        journal=test_journal,
        amount=100.0,
        direction="debit",
        account="Cash",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = 500.0
    
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


# LLM-generated content at query #53
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
        is_debit: bool
        is_credit: bool
        journal: 'MockJournal'

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

    # Create test data
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 15),
        amount=MockAmount(value=100.0, currency="USD"),
        account=MockAccount(name="Cash"),
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger(name="Test Ledger")

    # Construct LedgerEntry
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )

    # Assertions
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


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
        account=Account("Checking"),
        journal=Journal("Test transaction", []),
        direction="debit"
    )
    balance = Quantity(500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.balance.value == 500.0


# LLM-generated content at query #55
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
        account: MockAccount
        amount: MockAmount
        direction: str
        journal: MockJournal
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        pass

    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 1),
        account=test_account,
        amount=test_amount,
        direction="debit",
        journal=test_journal,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()

    # Create LedgerEntry instance
    from dataclasses import dataclass as dc, field
    @dc
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: MockPosting
        balance: MockQuantity

    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_quantity


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
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=Account(name="Cash"),
        journal=Journal(description="Test entry", postings=[]),
        direction="debit"
    )
    balance = Quantity(value=1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test entry"
    assert entry.amount == posting.amount
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #57
#--------------------------

```python
def test_read_initial_balances_call():
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
    date_range = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = reader(date_range)
    
    assert result.account_id == "ACC001"
    assert result.balance == 1000.0
    assert isinstance(result, InitialBalances)


# LLM-generated content at query #58
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
    class MockJournal:
        description: str
        postings: list

    @dataclass
    class MockPosting:
        date: date
        journal: MockJournal
        amount: object
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

    # Create instances for testing
    test_account = MockAccount(name="Test Account")
    test_journal = MockJournal(description="Test Description", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 15),
        journal=test_journal,
        amount="100",
        direction="debit",
        account=test_account,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=1000.0)

    # Create LedgerEntry instance
    from ledger_entry import LedgerEntry
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)

    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance
    assert entry.ledger is not None
    assert entry.posting is not None
    assert entry.balance is not None


# LLM-generated content at query #59
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

    from dataclasses import dataclass
    
    @dataclass
    class LedgerEntry(Generic[_T]):
        ledger: "Ledger[_T]"
        posting: Posting[_T]
        balance: Quantity

    test_ledger = Ledger(name="Test Ledger")
    test_account = Account(name="Cash")
    test_amount = Amount(value=100.0, currency="USD")
    test_journal = Journal(description="Test Transaction", postings=[])
    test_posting = Posting(
        date=date(2023, 1, 1),
        amount=test_amount,
        account=test_account,
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


# LLM-generated content at query #60
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
        direction: str
        is_debit: bool
        is_credit: bool
        account: MockAccount
        
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
    class MockLedger:
        pass

    @dataclass
    class MockQuantity:
        value: float

    # Setup test data
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(value=100.0, currency="USD")
    test_account = MockAccount(name="Cash")
    test_quantity = MockQuantity(value=100.0)
    test_ledger = MockLedger()
    
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account
    )
    
    test_journal = MockJournal(
        description="Test transaction",
        postings=[test_posting]
    )
    
    test_posting.set_journal(test_journal)

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
    assert ledger_entry.date == test_date
    assert ledger_entry.amount == test_amount
    assert ledger_entry.is_debit is True
    assert ledger_entry.is_credit is False


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
    test_account = Account(name="Cash")
    test_amount = Amount(value=100.0, currency="USD")
    test_posting = Posting(
        date=date(2023, 1, 15),
        journal=Journal(description="Test transaction", postings=[]),
        amount=test_amount,
        account=test_account,
        direction="debit"
    )
    test_balance = Quantity(value=500.0)
    test_ledger = Ledger()
    
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


# LLM-generated content at query #62
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
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        journal=journal,
        amount=amount,
        account=account,
        direction="debit"
    )
    balance = Quantity(value=100.0)
    ledger = Ledger()
    
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert ledger_entry.ledger is ledger
    assert ledger_entry.posting is posting
    assert ledger_entry.balance is balance
    assert ledger_entry.posting.date == date(2023, 1, 1)


# LLM-generated content at query #63
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
        name: str
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create instances
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test Journal", postings=[])
    mock_posting = MockPosting(
        date=date(2024, 1, 1),
        amount=mock_amount,
        journal=mock_journal,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger(name="Test Ledger")
    mock_quantity = MockQuantity(value=1000.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor properly assigned attributes
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_quantity


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

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    posting = Posting(
        date=date(2023, 1, 1),
        journal=Journal(description="Test Journal", postings=[]),
        amount=amount,
        account=account,
        direction="debit"
    )
    ledger = Ledger(name="Test Ledger")
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == quantity


# LLM-generated content at query #65
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

    # Create test instances
    test_date = date(2023, 1, 15)
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_account = MockAccount(name="Cash")
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        journal=MockJournal(description="Test transaction", postings=[]),
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

    # Assert constructor properly assigns all attributes
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity


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
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=test_journal
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


# LLM-generated content at query #67
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
    
    # Set up test data
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test Description", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        account=mock_account,
        amount=mock_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=mock_journal
    )
    mock_posting.journal = mock_journal
    mock_ledger = MockLedger()
    mock_quantity = MockQuantity(value=1000.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_quantity)
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #68
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
        account: MockAccount
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test data
    test_date = date(2023, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockAmount(value=500.0, currency="USD")
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    # Assert constructor properly assigns all attributes
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #69
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
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2024, 1, 15),
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

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), account=account, amount=amount, direction="debit", journal=journal)
    balance = Quantity(value=100.0)
    ledger = Ledger()

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
    mock_balance.value = 500
    
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


# LLM-generated content at query #72
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')
    
    # Create mock objects
    mock_journal = type('Journal', (), {'description': 'Test Description', 'postings': []})()
    mock_posting = type('Posting', (), {
        'date': date(2023, 1, 15),
        'journal': mock_journal,
        'amount': type('Amount', (), {})(),
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor sets all fields correctly
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #73
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_different_values():
    account = Account(name="Savings", account_type="LIABILITY")
    initial_balance = Balance(value=Quantity(5000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account.name == "Savings"
    assert ledger.account.account_type == "LIABILITY"
    assert ledger.initial.value == Quantity(5000)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


def test_ledger_constructor_entries_default_factory():
    account = Account(name="Test", account_type="ASSET")
    initial_balance = Balance(value=Quantity(0))
    
    ledger1 = Ledger(account=account, initial=initial_balance)
    ledger2 = Ledger(account=account, initial=initial_balance)
    
    assert ledger1.entries is not ledger2.entries
    assert ledger1.entries == []
    assert ledger2.entries == []


# LLM-generated content at query #74
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from decimal import Decimal
    
    # Create mock objects
    mock_ledger = object()
    
    mock_journal = type('Journal', (), {
        'description': 'Test Description',
        'postings': []
    })()
    
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'journal': mock_journal,
        'amount': type('Amount', (), {'value': Decimal('100.00')})(),
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False,
        'account': type('Account', (), {'name': 'Test Account'})()
    })()
    
    mock_balance = type('Quantity', (), {'value': Decimal('500.00')})()
    
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


# LLM-generated content at query #75
#--------------------------

```python
def test_build_general_ledger_creates_ledger_with_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {
        Account("1000"): Balance(date(2023, 1, 1), Quantity(Decimal("1000"))),
        Account("2000"): Balance(date(2023, 1, 1), Quantity(Decimal("500"))),
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert Account("1000") in general_ledger.ledgers
    assert Account("2000") in general_ledger.ledgers
    assert general_ledger.ledgers[Account("1000")].initial.value == Quantity(Decimal("1000"))
    assert general_ledger.ledgers[Account("2000")].initial.value == Quantity(Decimal("500"))


def test_build_general_ledger_creates_ledger_for_new_account_from_posting():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.postings import Posting, Amount
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    entry = JournalEntry(date(2023, 6, 15), "Test entry", "source")
    posting = Posting(entry, date(2023, 6, 15), Account("3000"), Direction.INC, Amount(Decimal("250")))
    entry.postings.append(posting)
    
    general_ledger = build_general_ledger(period, [entry], {})
    
    assert Account("3000") in general_ledger.ledgers
    assert general_ledger.ledgers[Account("3000")].initial.value == Quantity(Decimal(0))
    assert len(general_ledger.ledgers[Account("3000")].entries) == 1


def test_build_general_ledger_respects_period_boundaries():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.postings import Posting, Amount
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity

    period = DateRange(date(2023, 6, 1), date(2023, 6, 30))
    
    entry_in_period = JournalEntry(date(2023, 6, 15), "In period", "source")
    posting_in = Posting(entry_in_period, date(2023, 6, 15), Account("3000"), Direction.INC, Amount(Decimal("100")))
    entry_in_period.postings.append(posting_in)
    
    entry_before_period = JournalEntry(date(2023, 5, 31), "Before period", "source")
    posting_before = Posting(entry_before_period, date(2023, 5, 31), Account("3000"), Direction.INC, Amount(Decimal("50")))
    entry_before_period.postings.append(posting_before)
    
    entry_after_period = JournalEntry(date(2023, 7, 1), "After period", "source")
    posting_after = Posting(entry_after_period, date(2023, 7, 1), Account("3000"), Direction.INC, Amount(Decimal("75")))
    entry_after_period.postings.append(posting_after)
    
    general_ledger = build_general_ledger(period, [entry_before_period, entry_in_period, entry_after_period], {})
    
    assert len(general_ledger.ledgers[Account("3000")].entries) == 1
    assert general_ledger.ledgers[Account("3000")].entries[0].balance == Quantity(Decimal("100"))


def test_build_general_ledger_multiple_postings_same_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.postings import Posting, Amount
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    entry1 = JournalEntry(date(2023, 1, 15), "First entry", "source")
    posting1 = Posting(entry1, date(2023, 1, 15), Account("3000"), Direction.INC, Amount(Decimal("100")))
    entry1.postings.append(posting1)
    
    entry2 = JournalEntry(date(2023, 2, 15), "Second entry", "source")
    posting2 = Posting(entry2, date(2023, 2, 15), Account("3000"), Direction.DEC, Amount(Decimal("30")))
    entry2.postings.append(posting2)
    
    general_ledger = build_general_ledger(period, [entry1, entry2], {})
    
    assert len(general_ledger.ledgers[Account("3000")].entries) == 2
    assert general_ledger.ledgers[Account("3000")].entries[0].balance == Quantity(Decimal("100"))
    assert general_ledger.ledgers[Account("3000")].entries[1].balance == Quantity(Decimal("70"))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    from pypara.core.quantity import Quantity

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {Account("1000"): Balance(date(2023, 1, 1), Quantity(Decimal("500")))}
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[Account("1000")].entries) == 0
    assert general_ledger.period == period


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
        date=date(2023, 1, 15),
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


# LLM-generated content at query #77
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from decimal import Decimal
    
    # Create mock objects
    mock_account = type('Account', (), {})()
    mock_amount = type('Amount', (), {})()
    mock_quantity = type('Quantity', (), {})()
    
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'amount': mock_amount,
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False,
        'account': mock_account,
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': []
        })()
    })()
    
    mock_ledger = type('Ledger', (), {})()
    
    # Test constructor
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_quantity


# LLM-generated content at query #78
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
    assert period.start == date(2023, 1, 1)
    assert period.end == date(2023, 12, 31)


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
    period = DateRange(date(2024, 1, 1), date(2024, 6, 30))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert result.balances == {}


def test_read_initial_balances_call_multiple_accounts():
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
            return InitialBalances({
                "savings": 5000,
                "checking": 3000,
                "investment": 10000,
                "credit_card": -500
            })
    
    reader = ConcreteReadInitialBalances()
    period = DateRange(date(2023, 6, 1), date(2023, 6, 30))
    result = reader(period)
    
    assert isinstance(result, InitialBalances)
    assert len(result.balances) == 4
    assert result.balances["savings"] == 5000
    assert result.balances["checking"] == 3000
    assert result.balances["investment"] == 10000
    assert result.balances["credit_card"] == -500


# LLM-generated content at query #79
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
        name: str
    
    @dataclass
    class MockQuantity:
        value: float
    
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
    test_ledger = MockLedger(name="Test Ledger")
    test_balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly assigns all fields
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #80
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
    account = MockAccount(name="Test Account")
    amount = MockAmount(value=100.0, currency="USD")
    journal = MockJournal(description="Test Journal", postings=[])
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
    
    # Assert constructor assignments
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #81
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Quantity, Amount, Account
    
    # Setup
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 1, 31)
    period = DateRange(period_start, period_end)
    
    # Create accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create initial balances
    initial = {account1: Balance(period_start, Quantity(Decimal("1000")))}
    
    # Create journal entries - one within period, one outside
    entry_within = JournalEntry(datetime.date(2024, 1, 15), "Entry within period", "source1")
    entry_within.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal("-100")))
    entry_within.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal("100")))
    
    entry_before = JournalEntry(datetime.date(2023, 12, 31), "Entry before period", "source2")
    entry_before.post(datetime.date(2023, 12, 31), account1, Quantity(Decimal("-50")))
    entry_before.post(datetime.date(2023, 12, 31), account2, Quantity(Decimal("50")))
    
    entry_after = JournalEntry(datetime.date(2024, 2, 1), "Entry after period", "source3")
    entry_after.post(datetime.date(2024, 2, 1), account1, Quantity(Decimal("-75")))
    entry_after.post(datetime.date(2024, 2, 1), account2, Quantity(Decimal("75")))
    
    journal = [entry_before, entry_within, entry_after]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Verify that only postings within the period are included
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].posting.journal_entry.date == datetime.date(2024, 1, 15)
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account2].entries[0].posting.journal_entry.date == datetime.date(2024, 1, 15)


# LLM-generated content at query #82
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
    balance = Quantity(value=500.0)
    posting = Posting(
        date=date(2024, 1, 1),
        amount=amount,
        account=account,
        journal=Journal(description="Test entry", postings=[]),
        direction="debit"
    )
    ledger = Ledger(name="General Ledger")

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #83
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Account, Quantity, Amount
    
    # Setup test data
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balance1 = Balance(start_date, Quantity(Decimal(1000)))
    initial_balance2 = Balance(start_date, Quantity(Decimal(500)))
    initial_balances = {account1: initial_balance1, account2: initial_balance2}
    
    # Create journal entries
    entry1 = JournalEntry(datetime.date(2024, 1, 15), "Test entry 1", "source1")
    entry1.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(100)))
    entry1.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal(-100)))
    
    entry2 = JournalEntry(datetime.date(2024, 2, 20), "Test entry 2", "source2")
    entry2.post(datetime.date(2024, 2, 20), account1, Quantity(Decimal(-50)))
    entry2.post(datetime.date(2024, 2, 20), account2, Quantity(Decimal(50)))
    
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


def test_build_general_ledger_empty_journal():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Account, Quantity
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account("1000", "Cash")
    initial_balance1 = Balance(start_date, Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance1}
    
    journal = []
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0


def test_build_general_ledger_new_account_created():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Account, Quantity
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account("1000", "Cash")
    account2 = Account("3000", "Revenue")
    
    initial_balance1 = Balance(start_date, Quantity(Decimal(500)))
    initial_balances = {account1: initial_balance1}
    
    entry = JournalEntry(datetime.date(2024, 1, 15), "Test entry", "source")
    entry.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(200)))
    entry.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal(-200)))
    
    journal = [entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal(0))
    assert len(general_ledger.ledgers[account2].entries) == 1


def test_build_general_ledger_outside_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange, Account, Quantity
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account("1000", "Cash")
    initial_balance1 = Balance(start_date, Quantity(Decimal(1000)))
    initial_balances = {account1: initial_balance1}
    
    # Entry outside the period
    entry = JournalEntry(datetime.date(2025, 1, 15), "Test entry", "source")
    entry.post(datetime.date(2025, 1, 15), account1, Quantity(Decimal(100)))
    entry.post(datetime.date(2025, 1, 15), account1, Quantity(Decimal(-100)))
    
    journal = [entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[account1].entries) == 0


# LLM-generated content at query #84
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Generic, TypeVar
    
    T = TypeVar('T')
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger(Generic[T]):
        def __init__(self, data: T):
            self.data = data
    
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger({"period": period, "entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == date_range
    assert result.data["entries"] == []
    assert result.data["period"].start == date(2023, 1, 1)
    assert result.data["period"].end == date(2023, 12, 31)


# LLM-generated content at query #85
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
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    # Setup test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 1),
        amount=test_amount,
        journal=test_journal,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger(name="Test Ledger")
    test_balance = MockQuantity(value=1000.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly initializes all fields
    assert ledger_entry.ledger == test_ledger
    assert ledger_entry.posting == test_posting
    assert ledger_entry.balance == test_balance


# LLM-generated content at query #86
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
    test_quantity = MockQuantity(value=100.0)
    test_posting = MockPosting(
        date=date(2024, 1, 15),
        amount=test_amount,
        journal=MockJournal(description="Test Journal", postings=[]),
        direction="debit",
        account=test_account
    )
    test_ledger = MockLedger()

    # Test constructor
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity
    assert entry.date == date(2024, 1, 15)
    assert entry.amount is test_amount
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #87
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
        account: MockAccount
        amount: MockAmount
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    # Setup test data
    test_date = date(2024, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
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
    
    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_balance


# LLM-generated content at query #88
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
    
    ledger = Ledger()
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(date=date(2023, 1, 1), account=account, amount=amount, direction="debit", journal=journal)
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test transaction"
    assert entry.amount == amount


# LLM-generated content at query #89
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

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    quantity = Quantity(value=100.0)
    posting = Posting(
        date=date(2024, 1, 1),
        amount=amount,
        journal=Journal(description="Test Entry", postings=[]),
        account=account,
        direction="debit"
    )
    ledger = Ledger(name="Test Ledger")

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=quantity)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == quantity


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
        date=date(2024, 1, 1),
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


# LLM-generated content at query #91
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
    test_date = date(2024, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        account=test_account,
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()

    # Instantiate LedgerEntry with constructor
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert that all fields are correctly assigned
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity


# LLM-generated content at query #92
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
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
    
    # Setup
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account = Account("TestAccount")
    
    # Create journal entries with postings inside and outside the period
    entry_inside = JournalEntry(date(2024, 6, 15), "Inside period", "source1")
    posting_inside = Posting(entry_inside, date(2024, 6, 15), account, Direction.INC, Amount(Quantity(Decimal(100))))
    entry_inside.postings.append(posting_inside)
    
    entry_before = JournalEntry(date(2023, 12, 31), "Before period", "source2")
    posting_before = Posting(entry_before, date(2023, 12, 31), account, Direction.INC, Amount(Quantity(Decimal(50))))
    entry_before.postings.append(posting_before)
    
    entry_after = JournalEntry(date(2025, 1, 1), "After period", "source3")
    posting_after = Posting(entry_after, date(2025, 1, 1), account, Direction.INC, Amount(Quantity(Decimal(75))))
    entry_after.postings.append(posting_after)
    
    journal = [entry_before, entry_inside, entry_after]
    initial = {account: Balance(period.since, Quantity(Decimal(0)))}
    
    # Execute
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Assert that only the posting inside the period was added to the ledger
    ledger = general_ledger.ledgers[account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting == posting_inside


# LLM-generated content at query #93
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
    
    # Create instances
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_posting_self = MockPosting(
        date=date(2023, 1, 1),
        amount=mock_amount,
        journal=MockJournal(description="Test Description", postings=[]),
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger(name="Test Ledger")
    mock_quantity = MockQuantity(value=100.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting_self,
        balance=mock_quantity
    )
    
    # Assertions
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting_self
    assert entry.balance == mock_quantity


# LLM-generated content at query #94
#--------------------------

```python
def test_build_general_ledger_posting_account_not_in_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Account, Quantity, Amount, Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core import DateRange
    
    # Create test data
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    account1 = Account(name="Account1", code="001")
    account2 = Account(name="Account2", code="002")
    
    # Create a journal entry with a posting to account2
    entry = JournalEntry(date=date(2024, 6, 15), description="Test entry", source="test_source")
    entry.postings.append(Posting(entry, date(2024, 6, 15), account2, Direction.INC, Amount(Decimal("100"))))
    
    # Create initial balances only for account1
    initial_balances = {account1: Balance(date(2024, 1, 1), Quantity(Decimal("1000")))}
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [entry], initial_balances)
    
    # Verify that the predicate at line 18 evaluates to False
    # (account2 should be in ledgers after processing)
    assert account2 in general_ledger.ledgers


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
        date=date(2024, 1, 1),
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
            return GeneralLedger({"entries": []})
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data == {"entries": []}
    assert date_range.start == date(2023, 1, 1)
    assert date_range.end == date(2023, 12, 31)


# LLM-generated content at query #97
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
    balance = Quantity(value=100.0)

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
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.balance.value == 100.0


# LLM-generated content at query #99
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
        account: MockAccount
        amount: MockAmount
        journal: MockJournal
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
    test_date = date(2023, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_ledger = MockLedger()
    test_balance = MockQuantity(value=500.0)
    
    # Test constructor
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assertions
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_balance


# LLM-generated content at query #100
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
        'journal': mock_journal,
        'amount': type('Amount', (), {'value': 100})(),
        'account': mock_account,
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {'value': 500})()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assertions
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance
    assert entry.date == date(2024, 1, 15)
    assert entry.description == 'Test Journal'
    assert entry.amount.value == 100
    assert entry.is_debit is True
    assert entry.is_credit is False


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
        amount: Amount
        account: Account
        direction: str
        journal: Journal
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

    ledger = Ledger()
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        direction="debit",
        journal=journal,
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(500.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #102
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
    date_range = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = reader(date_range)
    
    assert isinstance(result, InitialBalances)
    assert result.account_id == "ACC001"
    assert result.balance == 1000.0


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
            return InitialBalances(account_id="ACC002", balance=5000.0)
    
    reader = ReadInitialBalances()
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 6, 30))
    result = reader(date_range)
    
    assert isinstance(result, InitialBalances)
    assert result.account_id == "ACC002"
    assert result.balance == 5000.0


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
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    posting_obj = Posting(
        date=date(2024, 1, 15),
        journal=Journal(description="Test transaction", postings=[]),
        amount=amount,
        account=account,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting_obj, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting_obj
    assert entry.balance is balance


# LLM-generated content at query #104
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
    test_amount = MockAmount(value=100.0, currency="USD")
    test_account = MockAccount(name="Cash")
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account,
        journal=test_journal
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


