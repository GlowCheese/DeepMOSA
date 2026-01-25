####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test", date=datetime.date(2023, 1, 1)), amount=Amount(100), direction=Direction.DEBIT, account=Account("Test Account"))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test", date=datetime.date(2023, 1, 1)), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #6
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #8
#--------------------------

```python
def test_build_general_ledger_with_valid_input():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 1, 15), Account("Cash"), Quantity(Decimal(500)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[Account("Cash")].entries[-1].balance == Quantity(Decimal(1500))

def test_build_general_ledger_with_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 1, 15), Account("Revenue"), Quantity(Decimal(500)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    assert len(general_ledger.ledgers) == 2
    assert general_ledger.ledgers[Account("Revenue")].entries[-1].balance == Quantity(Decimal(500))

def test_build_general_ledger_with_out_of_period_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2022, 12, 31), Account("Cash"), Quantity(Decimal(500)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[Account("Cash")].entries) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test", date=datetime.date(2023, 1, 1)), amount=Amount(100), account=Account("Account"), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", source)
    journal_entry.post(datetime.date(2023, 1, 1), Account("Test Account"), Quantity(Decimal(100)))
    initial_balances = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[Account("Test Account")].account == Account("Test Account")
    assert general_ledger.ledgers[Account("Test Account")].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(general_ledger.ledgers[Account("Test Account")].entries) == 1
    assert general_ledger.ledgers[Account("Test Account")].entries[0].balance == Quantity(Decimal(100))


# LLM-generated content at query #11
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #12
#--------------------------

```python
def test___call___returns_general_ledger():
    mock_ledger = object()
    mock_program = lambda period: mock_ledger
    date_range = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = mock_program(date_range)
    assert result is mock_ledger

def test___call___receives_correct_date_range():
    received_period = None
    def mock_program(period):
        nonlocal received_period
        received_period = period
        return object()
    date_range = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    mock_program(date_range)
    assert received_period is date_range


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadInitialBalances_call():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class InitialBalances(NamedTuple):
        balance: float

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        if period.start == date(2023, 1, 1) and period.end == date(2023, 1, 31):
            return InitialBalances(balance=1000.0)
        raise ValueError("Invalid date range")

    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = mock_read_initial_balances(period)
    assert result.balance == 1000.0


# LLM-generated content at query #14
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #15
#--------------------------

```python
def test_general_ledger_program_call():
    # Mock DateRange and GeneralLedger classes for testing
    class MockDateRange:
        pass

    class MockGeneralLedger:
        pass

    # Create a concrete implementation of GeneralLedgerProgram
    class TestGeneralLedgerProgram:
        def __call__(self, period: MockDateRange) -> MockGeneralLedger:
            return MockGeneralLedger()

    # Create instance and test
    program = TestGeneralLedgerProgram()
    period = MockDateRange()
    result = program(period)
    assert isinstance(result, MockGeneralLedger)


# LLM-generated content at query #16
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_create_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(date=datetime.date(2023, 5, 15), description="Test Entry", source=None)
    journal_entry.post(datetime.date(2023, 5, 15), Account("Test Account"), Quantity(Decimal(100)))
    initial = {}
    
    general_ledger = build_general_ledger(period, [journal_entry], initial)
    
    assert "Test Account" in general_ledger.ledgers
    assert general_ledger.ledgers["Test Account"].initial == Balance(period.since, Quantity(Decimal(0)))


# LLM-generated content at query #18
#--------------------------

```python
def test_build_general_ledger_creates_new_ledger_for_missing_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {}
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 1), Account("Test Account"), Quantity(Decimal(100)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    assert Account("Test Account") in general_ledger.ledgers


# LLM-generated content at query #19
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.ledgers == {}


# LLM-generated content at query #20
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #21
#--------------------------

```python
def test_build_general_ledger_with_empty_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_non_empty_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account = Account("1", "Cash")
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == initial[account]

def test_build_general_ledger_with_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("1", "Cash")
    account2 = Account("2", "Revenue")
    initial = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Sale", None)
    journal_entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(-50)))
    journal_entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal(50)))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #22
#--------------------------

Here are the test cases for the `build_general_ledger` function:


# LLM-generated content at query #23
#--------------------------

```python
def test_build_general_ledger_with_valid_journal_entries():
    from datetime import date
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.accounting.accounts import Account
    from pypara.commons.numbers import Decimal

    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial = {account1: Balance(date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry(date(2023, 1, 15), "Test Entry", "Source", postings=[Posting(None, date(2023, 1, 15), account2, Direction.INC, Quantity(Decimal(50)))])
    general_ledger = build_general_ledger(period, [journal_entry], initial)
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal(100))
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal(0))


# LLM-generated content at query #24
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="test_account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #25
#--------------------------

```
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #26
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #27
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(Account("Test Account"), Amount(100), datetime.date(2023, 1, 1), Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #28
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #29
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #30
#--------------------------

```python
def test_build_general_ledger():
    # Define test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(50)))
    journal_entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal(30)))
    journal_entries = [journal_entry]

    # Build the general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(150))
    assert general_ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(30))


# LLM-generated content at query #31
#--------------------------

```python
from datetime import date
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

_T = TypeVar("_T")

@dataclass
class Quantity:
    value: float
    unit: str

@dataclass
class Amount(Quantity):
    pass

@dataclass
class Account:
    name: str

@dataclass
class Journal:
    date: date
    description: str
    postings: List["Posting"]

@dataclass
class Posting(Generic[_T]):
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

    @property
    def date(self) -> date:
        return self.posting.date

    @property
    def description(self) -> str:
        return self.posting.journal.description

    @property
    def amount(self) -> Amount:
        return self.posting.amount

    @property
    def cntraccts(self) -> List[Account]:
        return [p.account for p in self.posting.journal.postings if p.direction != self.posting.direction]

    @property
    def is_debit(self) -> bool:
        return self.posting.is_debit

    @property
    def is_credit(self) -> bool:
        return self.posting.is_credit

    @property
    def debit(self) -> Optional[Amount]:
        return self.amount if self.is_debit else None

    @property
    def credit(self) -> Optional[Amount]:
        return self.amount if self.is_credit else None

def test_ledger_entry_constructor():
    account = Account(name="Test Account")
    amount = Amount(value=100.0, unit="USD")
    journal = Journal(date=date(2023, 1, 1), description="Test Journal", postings=[])
    posting = Posting(account=account, amount=amount, direction="debit", journal=journal)
    ledger = Ledger(name="Test Ledger")
    balance = Quantity(value=100.0, unit="USD")
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #32
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("TestLedger")
    posting = Posting(datetime.date(2023, 10, 1), Amount(100), Journal("TestJournal"), Account("TestAccount"), Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #33
#--------------------------

```python
def test___call___returns_initial_balances():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    mock_initial_balances = InitialBalances(assets=1000, liabilities=500, equity=500)
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return mock_initial_balances
    
    read_initial_balances = ReadInitialBalances(mock_read_initial_balances)
    result = read_initial_balances(mock_period)
    
    assert result == mock_initial_balances


# LLM-generated content at query #34
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #35
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(value=100), journal=Journal(description="Test"), date=datetime.date(2023, 1, 1), direction=Direction.DEBIT)
    balance = Quantity(value=100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert result.ledgers[Account("A1")].account == Account("A1")
    assert result.ledgers[Account("A1")].initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    assert len(result.ledgers[Account("A1")].entries) == 0

def test_build_general_ledger_with_journal_entries_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    entry.post(datetime.date(2023, 6, 15), Account("A1"), Quantity(Decimal(50)))
    journal = [entry]
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    assert len(result.ledgers[Account("A1")].entries) == 1
    assert result.ledgers[Account("A1")].entries[0].balance == Quantity(Decimal(150))

def test_build_general_ledger_with_journal_entries_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2022, 12, 31), "Test", source)
    entry.post(datetime.date(2022, 12, 31), Account("A1"), Quantity(Decimal(50)))
    journal = [entry]
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    assert len(result.ledgers[Account("A1")].entries) == 0

def test_build_general_ledger_with_multiple_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    entry.post(datetime.date(2023, 6, 15), Account("A1"), Quantity(Decimal(50)))
    entry.post(datetime.date(2023, 6, 15), Account("A2"), Quantity(Decimal(-50)))
    journal = [entry]
    initial = {
        Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100))),
        Account("A2"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(200))),
    }
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 2
    assert result.ledgers[Account("A1")].entries[0].balance == Quantity(Decimal(150))
    assert result.ledgers[Account("A2")].entries[0].balance == Quantity(Decimal(150))

def test_build_general_ledger_with_new_account_not_in_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    entry.post(datetime.date(2023, 6, 15), Account("A1"), Quantity(Decimal(50)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    assert result.ledgers[Account("A1")].initial == Balance(period.since, Quantity(Decimal(0)))
    assert result.ledgers[Account("A1")].entries[0].balance == Quantity(Decimal(50))


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert ledger.ledgers == {}

def test_build_general_ledger_with_journal_entries_and_no_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", "source")
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert account in ledger.ledgers
    assert ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(100))

def test_build_general_ledger_with_journal_entries_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", "source")
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {account: Balance(period.since, Quantity(Decimal(50)))}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert account in ledger.ledgers
    assert ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(50)))
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))

def test_build_general_ledger_ignores_out_of_period_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    journal_entry_before = JournalEntry(datetime.date(2022, 12, 31), "Before", "source")
    journal_entry_before.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(100)))
    journal_entry_during = JournalEntry(datetime.date(2023, 1, 15), "During", "source")
    journal_entry_during.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(200)))
    journal_entry_after = JournalEntry(datetime.date(2024, 1, 1), "After", "source")
    journal_entry_after.post(datetime.date(2024, 1, 1), account, Quantity(Decimal(300)))
    journal = [journal_entry_before, journal_entry_during, journal_entry_after]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert account in ledger.ledgers
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(200))


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 10, 1), journal=Journal(description="Test Journal"), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #6
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(datetime.date(2023, 10, 1), Journal("Test Journal", [], []), Amount(100), Direction.DEBIT, Account("Test Account"))
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #7
#--------------------------

```python
def test___call___returns_initial_balances():
    from datetime import date
    from typing import Protocol
    from dataclasses import dataclass

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class InitialBalances:
        balances: dict

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return InitialBalances({"account1": 100.0, "account2": 200.0})

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 100.0, "account2": 200.0}

def test___call___handles_empty_balances():
    from datetime import date
    from typing import Protocol
    from dataclasses import dataclass

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class InitialBalances:
        balances: dict

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return InitialBalances({})

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {}

def test___call___uses_provided_period():
    from datetime import date
    from typing import Protocol
    from dataclasses import dataclass

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class InitialBalances:
        balances: dict

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period.start == date(2023, 1, 1)
        assert period.end == date(2023, 12, 31)
        return InitialBalances({"account1": 100.0})

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 100.0}


# LLM-generated content at query #8
#--------------------------

```python
def test_ledger_constructor_sets_account_and_initial_balance():
    account = Account("12345")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []

def test_ledger_constructor_initializes_empty_entries_list():
    account = Account("12345")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_to_true():
    from datetime import date
    from pypara.accounting import JournalEntry, Posting, Direction, Amount, Quantity, Balance, Ledger, GeneralLedger
    from pypara.commons import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = [
        JournalEntry(date(2023, 5, 15), "Test Entry", source=None).post(date(2023, 5, 15), account=None, quantity=Quantity(Decimal(100))),
        JournalEntry(date(2022, 12, 31), "Old Entry", source=None).post(date(2022, 12, 31), account=None, quantity=Quantity(Decimal(100))),
        JournalEntry(date(2024, 1, 1), "Future Entry", source=None).post(date(2024, 1, 1), account=None, quantity=Quantity(Decimal(100)))
    ]
    initial = {}

    general_ledger = build_general_ledger(period, journal, initial)

    assert len(general_ledger.ledgers) == 1
    assert len(list(general_ledger.ledgers.values())[0].entries) == 1


# LLM-generated content at query #10
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #11
#--------------------------

```python
def test___call__():
    from datetime import date
    from typing import Protocol, TypeVar
    from dataclasses import dataclass

    _T = TypeVar('_T')

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        period: DateRange
        entries: list[_T]

    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...

    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger(period=period, entries=[])

    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    program = MockGeneralLedgerProgram()
    result = program(period)
    assert result.period == period
    assert result.entries == []


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #13
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #14
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_true():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2023, 5, 15), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 5, 15), Account("Test Account"), Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = InitialBalances({Account("Test Account"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))})
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period


# LLM-generated content at query #15
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period):
        return {"account1": 100, "account2": 200}

    # Test case
    period = ("2023-01-01", "2023-01-31")
    result = mock_read_initial_balances(period)
    assert result == {"account1": 100, "account2": 200}


# LLM-generated content at query #16
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #17
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #18
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #19
#--------------------------

```python
def test_build_general_ledger_with_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.ledgers == {}


# LLM-generated content at query #20
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #21
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #22
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test Journal"), amount=Amount(value=100), date=datetime.date(2023, 10, 1), direction=Direction.DEBIT)
    balance = Quantity(value=100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #23
#--------------------------

```python
def test___call___returns_general_ledger():
    from datetime import date
    from typing import Protocol, TypeVar
    from dataclasses import dataclass

    _T = TypeVar('_T')

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        period: DateRange
        entries: list[_T]

    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger(period=period, entries=[])

    program = MockGeneralLedgerProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(period)
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.entries == []


# LLM-generated content at query #24
#--------------------------

```python
def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert ledger.ledgers == {}

def test_build_general_ledger_with_journal_entries_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash")
    quantity = Quantity(Decimal(100))
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test", None)
    journal_entry.post(datetime.date(2023, 6, 15), account, quantity)
    initial = {}
    ledger = build_general_ledger(period, [journal_entry], initial)
    assert account in ledger.ledgers
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == quantity

def test_build_general_ledger_with_journal_entries_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash")
    quantity = Quantity(Decimal(100))
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test", None)
    journal_entry.post(datetime.date(2022, 12, 31), account, quantity)
    initial = {}
    ledger = build_general_ledger(period, [journal_entry], initial)
    assert ledger.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    initial = {account: initial_balance}
    ledger = build_general_ledger(period, [], initial)
    assert account in ledger.ledgers
    assert ledger.ledgers[account].initial == initial_balance
    assert len(ledger.ledgers[account].entries) == 0

def test_build_general_ledger_with_multiple_postings_to_same_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash")
    quantity1 = Quantity(Decimal(100))
    quantity2 = Quantity(Decimal(-50))
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test", None)
    journal_entry.post(datetime.date(2023, 6, 15), account, quantity1)
    journal_entry.post(datetime.date(2023, 6, 15), account, quantity2)
    initial = {}
    ledger = build_general_ledger(period, [journal_entry], initial)
    assert len(ledger.ledgers[account].entries) == 2
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(100))
    assert ledger.ledgers[account].entries[1].balance == Quantity(Decimal(50))


# LLM-generated content at query #25
#--------------------------

```python
def test_ledger_constructor():
    account = Account("acc1")
    initial = Balance(100)
    ledger = Ledger(account, initial)
    assert ledger.account == account
    assert ledger.initial == initial
    assert ledger.entries == []


# LLM-generated content at query #26
#--------------------------

```python
def test_build_general_ledger():
    from pypara.accounting import build_general_ledger, DateRange, JournalEntry, Posting, Ledger, GeneralLedger, Balance, Quantity, Account, Direction, Amount
    from datetime import date
    from decimal import Decimal

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    account = Account("cash")
    initial = {account: Balance(date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry(date(2023, 1, 15), "test", "source")
    posting = Posting(journal_entry, date(2023, 1, 15), account, Direction.INC, Amount(Quantity(Decimal(50))))
    journal_entry.postings.append(posting)
    journal = [journal_entry]

    general_ledger = build_general_ledger(period, journal, initial)

    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))


# LLM-generated content at query #27
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #28
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert ledger.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    account = Account("1", "Cash")
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert ledger.ledgers[account].account == account
    assert ledger.ledgers[account].initial == initial[account]
    assert ledger.ledgers[account].entries == []

def test_build_general_ledger_with_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1", "Cash")
    account2 = Account("2", "Revenue")
    entry = JournalEntry(datetime.date(2023, 1, 15), "Sale", None)
    entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(500)))
    entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal(-500)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 2
    assert ledger.ledgers[account1].account == account1
    assert ledger.ledgers[account1].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.ledgers[account1].entries) == 1
    assert ledger.ledgers[account2].account == account2
    assert ledger.ledgers[account2].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.ledgers[account2].entries) == 1

def test_build_general_ledger_with_out_of_period_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash")
    entry = JournalEntry(datetime.date(2022, 12, 31), "Old Sale", None)
    entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(1000)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert ledger.ledgers == {}

def test_build_general_ledger_with_mixed_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1", "Cash")
    account2 = Account("2", "Revenue")
    initial = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    entry1 = JournalEntry(datetime.date(2023, 1, 15), "Sale", None)
    entry1.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(500)))
    entry1.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal(-500)))
    
    entry2 = JournalEntry(datetime.date(2022, 12, 31), "Old Sale", None)
    entry2.post(datetime.date(2022, 12, 31), account1, Quantity(Decimal(1000)))
    
    journal = [entry1, entry2]
    ledger = build_general_ledger(period, journal, initial)
    
    assert ledger.period == period
    assert len(ledger.ledgers) == 2
    assert ledger.ledgers[account1].initial == initial[account1]
    assert len(ledger.ledgers[account1].entries) == 1
    assert ledger.ledgers[account2].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.ledgers[account2].entries) == 1


# LLM-generated content at query #29
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    ledger_entry = LedgerEntry(ledger, posting, balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


