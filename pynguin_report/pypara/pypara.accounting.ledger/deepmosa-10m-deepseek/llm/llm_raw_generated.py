####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
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
    entries: list[_T]

class GeneralLedgerProgram(Protocol[_T]):
    def __call__(self, period: DateRange) -> GeneralLedger[_T]:
        ...

def test_general_ledger_program_call():
    def mock_general_ledger_program(period: DateRange) -> GeneralLedger[int]:
        return GeneralLedger([1, 2, 3])

    program: GeneralLedgerProgram[int] = mock_general_ledger_program
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.entries == [1, 2, 3]


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_constructor():
    account = Account("1234")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[Any]()
    posting = Posting[Any]()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(100.0)
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="test", date=datetime.date(2023, 1, 1)), account=Account("cash"), direction=Direction.DEBIT, amount=Amount(100), balance=Quantity(100))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #6
#--------------------------

```python
def test_ReadInitialBalances___call___returns_InitialBalances():
    period = DateRange(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 12, 31))
    initial_balances = InitialBalances()
    mock_read_initial_balances = lambda p: initial_balances
    result = mock_read_initial_balances(period)
    assert result == initial_balances


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(date=datetime.date(2023, 10, 1), description="Test Journal", postings=[]), account=Account(name="Test Account"), amount=Amount(value=100), direction=Direction.DEBIT)
    balance = Quantity(value=200)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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


# LLM-generated content at query #10
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


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test", date=datetime.date(2023, 1, 1)), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance
    assert entry.date == datetime.date(2023, 1, 1)
    assert entry.description == "Test"
    assert entry.amount == Amount(100)
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == Amount(100)
    assert entry.credit == None


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"cash": 1000, "debt": 500}

    mock_reader = MockReadInitialBalances()
    period = ("2023-01-01", "2023-01-31")
    result = mock_reader(period)
    
    assert result == {"cash": 1000, "debt": 500}


# LLM-generated content at query #14
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
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", source)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert account in ledger.ledgers
    assert ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(100))


def test_build_general_ledger_with_journal_entries_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", source)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {account: Balance(period.since, Quantity(Decimal(50)))}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert account in ledger.ledgers
    assert ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(50)))
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))


def test_build_general_ledger_ignores_out_of_period_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    source = object()
    journal_entry1 = JournalEntry(datetime.date(2022, 12, 31), "Out of period", source)
    journal_entry1.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(100)))
    journal_entry2 = JournalEntry(datetime.date(2023, 1, 15), "In period", source)
    journal_entry2.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(200)))
    journal = [journal_entry1, journal_entry2]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert account in ledger.ledgers
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(200))


# LLM-generated content at query #15
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(Account("Test Account"), Amount(100), Direction.DEBIT, datetime.date(2023, 1, 1))
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #16
#--------------------------

```python
def test_build_general_ledger_with_valid_journal_entry_and_initial_balances():
    from pypara.accounting import JournalEntry, Posting, Direction, Amount, Account, Quantity, Balance, DateRange, GeneralLedger
    import datetime
    from decimal import Decimal

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("123", "Test Account")
    initial_balances = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Source", [Posting(None, datetime.date(2023, 1, 15), account, Direction.INC, Amount(Decimal(50)))])
    journal = [journal_entry]

    general_ledger = build_general_ledger(period, journal, initial_balances)
    assert general_ledger.ledgers[account].entries[-1].balance == Quantity(Decimal(150))


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_with_account_not_in_ledgers():
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = [JournalEntry(date=date(2023, 1, 1), description="test", source=object())]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 0


# LLM-generated content at query #18
#--------------------------

def test_build_general_ledger_with_empty_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.ledgers == {}


# LLM-generated content at query #19
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
    account = Account("1", "Cash", AccountType.ASSETS)
    initial = {account: Balance(period.since, Quantity(Decimal(100)))}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert ledger.ledgers[account].account == account
    assert ledger.ledgers[account].initial == initial[account]
    assert ledger.ledgers[account].entries == []


def test_build_general_ledger_with_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    source = object()
    entry = JournalEntry(datetime.date(2023, 1, 15), "Test", source)
    entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert ledger.ledgers[account].account == account
    assert ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(50))


def test_build_general_ledger_filters_out_of_period_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    source = object()
    entry1 = JournalEntry(datetime.date(2022, 12, 31), "Before", source)
    entry1.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(100)))
    entry2 = JournalEntry(datetime.date(2023, 1, 1), "Within", source)
    entry2.post(datetime.date(2023, 1, 1), account, Quantity(Decimal(50)))
    entry3 = JournalEntry(datetime.date(2024, 1, 1), "After", source)
    entry3.post(datetime.date(2024, 1, 1), account, Quantity(Decimal(25)))
    journal = [entry1, entry2, entry3]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(50))


def test_build_general_ledger_combines_initial_and_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    source = object()
    entry = JournalEntry(datetime.date(2023, 1, 15), "Test", source)
    entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [entry]
    initial = {account: Balance(period.since, Quantity(Decimal(100)))}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.ledgers[account].initial == initial[account]
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))


# LLM-generated content at query #20
#--------------------------

```python
def test___call___returns_general_ledger():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    program = GeneralLedgerProgram()
    result = program(period)
    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting import Account, Balance, Direction, JournalEntry, Posting, Quantity
    from pypara.commons import DateRange, makeguid

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("A1")
    account2 = Account("A2")
    initial = {account1: Balance(date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry(date(2023, 1, 2), "Test Entry", makeguid())
    journal_entry.post(date(2023, 1, 2), account1, Quantity(Decimal(50)))
    journal_entry.post(date(2023, 1, 2), account2, Quantity(Decimal(-30)))

    general_ledger = build_general_ledger(period, [journal_entry], initial)

    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(150))
    assert general_ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(-30))


# LLM-generated content at query #23
#--------------------------

```python
def test_Ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #24
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", postings=[]), amount=Amount(value=100, currency="USD"), direction=Direction.DEBIT)
    balance = Quantity(value=100, unit="USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #25
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


# LLM-generated content at query #26
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
def test_ledger_constructor():
    account = Account("12345")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


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
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal[int](description="test", postings=[]), account=Account(name="test"), amount=Amount(value=100), direction=Direction.DEBIT, date=datetime.date(2023, 1, 1))
    balance = Quantity(value=100)
    entry = LedgerEntry[int](ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #31
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_new_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 1, 15), Account(), Quantity(Decimal(100)))
    initial_balances = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert len(general_ledger.ledgers) == 1


# LLM-generated content at query #32
#--------------------------

```python
def test_build_general_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Cash", AccountType.ASSET)
    initial_balances = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 2), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 2), account, Quantity(Decimal(500)))
    journal = [journal_entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(1500))


# LLM-generated content at query #33
#--------------------------

```python
def test___call___method():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    mock_ledger = GeneralLedger()
    mock_program = lambda period: mock_ledger
    result = mock_program(mock_period)
    assert result == mock_ledger


# LLM-generated content at query #34
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[_T]()
    posting = Posting[_T]()
    balance = Quantity()
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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


# LLM-generated content at query #37
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test", postings=[]), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #38
#--------------------------

```python
def test_build_general_ledger_filters_journal_entries_by_period():
    from datetime import date
    from pypara.accounting import JournalEntry, Posting, Direction, Amount, Account, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.commons.numbers import Decimal
    from pypara.commons.zeitgeist import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("1", "Test Account")
    source = object()
    quantity = Quantity(Decimal("100"))

    # Journal entry within period
    entry_in_period = JournalEntry(date(2023, 6, 1), "Test", source)
    entry_in_period.post(date(2023, 6, 1), account, quantity)

    # Journal entry before period
    entry_before = JournalEntry(date(2022, 12, 31), "Test", source)
    entry_before.post(date(2022, 12, 31), account, quantity)

    # Journal entry after period
    entry_after = JournalEntry(date(2024, 1, 1), "Test", source)
    entry_after.post(date(2024, 1, 1), account, quantity)

    journal = [entry_in_period, entry_before, entry_after]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)

    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting == entry_in_period.postings[0]


# LLM-generated content at query #39
#--------------------------

```python
def test_LedgerEntry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #40
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start_date: date
        end_date: date

    class InitialBalances(NamedTuple):
        balances: dict[str, float]

    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return InitialBalances({"account1": 100.0, "account2": 200.0})

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 100.0, "account2": 200.0}


# LLM-generated content at query #41
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


# LLM-generated content at query #42
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


# LLM-generated content at query #43
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #44
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int]()
    balance = Quantity()
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #45
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[_T]()
    posting = Posting[_T]()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #46
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


# LLM-generated content at query #47
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(journal=Journal(description="Test Journal", postings=[]), amount=Amount(value=100, currency="USD"), direction="debit", date=datetime.date(2023, 10, 1))
    mock_balance = Quantity(value=100, currency="USD")
    ledger_entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #48
#--------------------------

def test_build_general_ledger_creates_ledger_for_new_account():
    from datetime import date
    from pypara.accounting import Account, Balance, JournalEntry, Posting, Direction, Amount, Quantity
    from pypara.commons.zeitgeist import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("1234")
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    journal_entry.post(date(2023, 1, 1), account, Quantity(Decimal(100)))
    initial_balances = {}

    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert account in general_ledger.ledgers
    assert general_ledger.ledgers[account].account == account
    assert general_ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(general_ledger.ledgers[account].entries) == 1


# LLM-generated content at query #49
#--------------------------

```python
def test_general_ledger_program_call():
    period = DateRange(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 12, 31))
    program = GeneralLedgerProgram()
    general_ledger = program(period)
    assert isinstance(general_ledger, GeneralLedger)


# LLM-generated content at query #50
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_true():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 6, 15), Account("Test Account"), Quantity(Decimal(100)))
    initial = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial)
    assert general_ledger.ledgers[Account("Test Account")].entries[-1].balance == Quantity(Decimal(100))


# LLM-generated content at query #51
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


# LLM-generated content at query #52
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


# LLM-generated content at query #53
#--------------------------

```python
def test_build_general_ledger_with_empty_period_and_journal():
    period = DateRange(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 1, 31))
    journal = []
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.ledgers == {}


# LLM-generated content at query #54
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("test_ledger")
    posting = Posting(journal=Journal(date=datetime.date(2023, 1, 1), description="test", postings=[]), account=Account("test_account"), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #55
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(Journal(datetime.date(2023, 10, 1), "Test Journal"), Account("Test Account"), Amount(100), Direction.DEBIT)
    balance = Quantity(100)
    ledger_entry = LedgerEntry(ledger, posting, balance)
    
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #56
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


# LLM-generated content at query #57
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


# LLM-generated content at query #58
#--------------------------

def test_build_general_ledger_creates_new_ledger_for_new_account():
    from datetime import date
    from pypara.accounting import Account, Amount, Direction, JournalEntry, Posting, Quantity
    from pypara.accounting.generic import Balance, DateRange
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from decimal import Decimal

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("1234", "Test Account")
    journal_entry = JournalEntry(date(2023, 1, 15), "Test", None)
    journal_entry.post(date(2023, 1, 15), account, Quantity(Decimal("100")))
    initial_balances = {}

    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)

    assert account in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[account], Ledger)
    assert general_ledger.ledgers[account].account == account
    assert general_ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(general_ledger.ledgers[account].entries) == 1


# LLM-generated content at query #59
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(journal=Journal(description="Test", date=datetime.date(2023, 10, 1)), amount=Amount(100), direction=Direction.DEBIT, account=Account("Test Account"))
    mock_balance = Quantity(100)
    ledger_entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #60
#--------------------------

```python
def test___call___returns_initial_balances():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    expected_balances = InitialBalances()
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        return expected_balances
    read_initial_balances = ReadInitialBalances(mock_read_initial_balances)
    actual_balances = read_initial_balances(period)
    assert actual_balances == expected_balances

def test___call___passes_correct_period():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    received_period = None
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        nonlocal received_period
        received_period = p
        return InitialBalances()
    read_initial_balances = ReadInitialBalances(mock_read_initial_balances)
    read_initial_balances(period)
    assert received_period == period


# LLM-generated content at query #61
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


# LLM-generated content at query #62
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


# LLM-generated content at query #63
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[float]()
    posting = Posting[float](account=Account("cash"), amount=Amount(100.0), direction=PostingDirection.DEBIT, journal=Journal(date=datetime.date(2023, 1, 1), description="Test Journal", postings=[]))
    balance = Quantity(100.0)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #64
#--------------------------

```python
def test_build_general_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entries = [JournalEntry(datetime.date(2023, 1, 2), "Test Entry", "Source", [])]
    journal_entries[0].post(datetime.date(2023, 1, 2), Account("Cash"), Quantity(Decimal(100)))
    
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    assert general_ledger.period == period
    assert Account("Cash") in general_ledger.ledgers
    assert general_ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(1100))


# LLM-generated content at query #65
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test", date=datetime.date(2023, 1, 1), postings=[]), amount=Amount(100), account=Account("Account1"), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #66
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


# LLM-generated content at query #67
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


# LLM-generated content at query #68
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


# LLM-generated content at query #69
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", date=datetime.date(2023, 1, 1)), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #70
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #71
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="test", date=datetime.date(2023, 1, 1), postings=[]), account=Account("test"), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry[int](ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #72
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("test_ledger")
    posting = Posting("test_posting", datetime.date(2023, 10, 1), Amount(100), Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #73
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


# LLM-generated content at query #74
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


# LLM-generated content at query #75
#--------------------------

```python
def test_build_general_ledger_with_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_with_single_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1234")
    quantity = Quantity(Decimal("100.00"))
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", None)
    journal_entry.post(datetime.date(2023, 1, 15), account, quantity)
    initial = {}
    result = build_general_ledger(period, [journal_entry], initial)
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == quantity

def test_build_general_ledger_with_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1234")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("500.00")))
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", None)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    result = build_general_ledger(period, [journal_entry], initial)
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("600.00"))

def test_build_general_ledger_ignores_out_of_period_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1234")
    journal_entry_before = JournalEntry(datetime.date(2022, 12, 31), "Before", None)
    journal_entry_before.post(datetime.date(2022, 12, 31), account, Quantity(Decimal("100.00")))
    journal_entry_during = JournalEntry(datetime.date(2023, 1, 15), "During", None)
    journal_entry_during.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("200.00")))
    journal_entry_after = JournalEntry(datetime.date(2024, 1, 1), "After", None)
    journal_entry_after.post(datetime.date(2024, 1, 1), account, Quantity(Decimal("300.00")))
    initial = {}
    result = build_general_ledger(period, [journal_entry_before, journal_entry_during, journal_entry_after], initial)
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("200.00"))

def test_build_general_ledger_with_multiple_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1234")
    account2 = Account("5678")
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", None)
    journal_entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal("100.00")))
    journal_entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal("-100.00")))
    initial = {}
    result = build_general_ledger(period, [journal_entry], initial)
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("100.00"))
    assert result.ledgers[account2].entries[0].balance == Quantity(Decimal("-100.00"))


# LLM-generated content at query #76
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #77
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


# LLM-generated content at query #78
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


# LLM-generated content at query #79
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


# LLM-generated content at query #80
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    ledger_entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #81
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_constructor_sets_ledger_posting_and_balance():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting import Account, Quantity, Balance, JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.commons import DateRange, makeguid

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    account = Account("1234")
    initial_balance = Balance(date(2023, 1, 1), Quantity(Decimal(100)))
    initial = {account: initial_balance}
    journal_entry = JournalEntry(date(2023, 1, 15), "Test Entry", "Test Source")
    journal_entry.post(date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial)
    ledger = general_ledger.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal(150))


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", date=datetime.date(2023, 10, 1)), amount=Amount(100), direction=Direction.DEBIT)
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
def test_predicate_evaluates_to_true():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2023, 6, 1), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 6, 1), Account("Test Account"), Quantity(Decimal(100)))
    initial = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial)
    assert journal_entry.date >= period.since and journal_entry.date <= period.until


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"asset1": 100, "asset2": 200}

    mock_read_initial_balances = MockReadInitialBalances()
    period = ("2023-01-01", "2023-01-31")
    result = mock_read_initial_balances(period)
    assert result == {"asset1": 100, "asset2": 200}


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert ledger.ledgers == {}

def test_build_general_ledger_with_single_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("123", "Test Account")
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", source)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 1
    assert account in ledger.ledgers
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(100))

def test_build_general_ledger_with_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("123", "Test Account")
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(50)))}
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", source)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 1
    assert account in ledger.ledgers
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))

def test_build_general_ledger_with_out_of_period_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("123", "Test Account")
    source = object()
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test Entry", source)
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.ledgers == {}

def test_build_general_ledger_with_multiple_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("123", "Test Account 1")
    account2 = Account("456", "Test Account 2")
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", source)
    journal_entry.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(100)))
    journal_entry.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal(-50)))
    journal = [journal_entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 2
    assert account1 in ledger.ledgers
    assert account2 in ledger.ledgers
    assert len(ledger.ledgers[account1].entries) == 1
    assert len(ledger.ledgers[account2].entries) == 1
    assert ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(100))
    assert ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(-50))


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_true():
    from datetime import date
    from pypara.accounting import JournalEntry, Posting, Direction, Amount, Quantity
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import Ledger, GeneralLedger, build_general_ledger
    from pypara.commons.numbers import Decimal
    from pypara.commons.zeitgeist import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal_entry_date = date(2023, 6, 1)
    journal_entry = JournalEntry(journal_entry_date, "Test Entry", "Source")
    posting = Posting(journal_entry, journal_entry_date, "Account", Direction.INC, Amount(Quantity(Decimal(100))))
    journal_entry.postings.append(posting)
    initial_balances = {}

    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)

    assert general_ledger.ledgers["Account"].entries[-1].balance == Quantity(Decimal(100))


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Sale",
            "Sale1",
        ).post(datetime.date(2023, 1, 15), Account("Cash"), Quantity(Decimal(500)))
    ]
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[Account("Cash")].initial.value == Quantity(Decimal(1000))
    assert len(general_ledger.ledgers[Account("Cash")].entries) == 1
    assert general_ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(1500))


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(Journal(datetime.date(2023, 10, 1), "Test Journal"), Account("Test Account"), Amount(100), Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #13
#--------------------------

```python
def test___call__():
    from datetime import date
    from typing import Protocol, Generic, TypeVar
    from dataclasses import dataclass

    _T = TypeVar('_T')

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger(Generic[_T]):
        entries: list[_T]

    class GeneralLedgerProgramImpl(GeneralLedgerProgram[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger([])

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    program = GeneralLedgerProgramImpl()
    result = program(period)
    assert isinstance(result, GeneralLedger)
    assert result.entries == []


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(date=datetime.date(2023, 1, 1), description="Test", postings=[]), amount=Amount(value=100), direction=Direction.DEBIT, account=Account(name="Test Account"))
    balance = Quantity(value=100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #16
#--------------------------

```python
def test_ReadInitialBalances___call__():
    period = DateRange(start_date="2023-01-01", end_date="2023-01-31")
    initial_balances = InitialBalances(balances={"account1": 1000, "account2": 2000})
    mock_read_initial_balances = lambda p: initial_balances if p == period else InitialBalances(balances={})
    assert mock_read_initial_balances(period) == initial_balances


# LLM-generated content at query #17
#--------------------------

```python
def test_Ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #18
#--------------------------

```python
def test_ReadInitialBalances_call():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"balance": 1000}

    mock_instance = MockReadInitialBalances()
    period = ("2023-01-01", "2023-01-31")
    result = mock_instance(period)
    assert result == {"balance": 1000}


# LLM-generated content at query #19
#--------------------------

Here are the test cases:


# LLM-generated content at query #20
#--------------------------

```python
def test___call___returns_initial_balances():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    mock_initial_balances = InitialBalances()
    mock_reader = lambda period: mock_initial_balances
    result = mock_reader(mock_period)
    assert result == mock_initial_balances

def test___call___receives_correct_period():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    received_period = None
    def mock_reader(period):
        nonlocal received_period
        received_period = period
        return InitialBalances()
    mock_reader(mock_period)
    assert received_period == mock_period


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_ledger_constructor():
    account = Account("12345", "Checking")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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


# LLM-generated content at query #25
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
def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert general_ledger.ledgers == {Account("A1"): Ledger(Account("A1"), Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100))))}


def test_build_general_ledger_with_journal_entries_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("A1")
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry[_T](datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))


def test_build_general_ledger_with_journal_entries_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("A1")
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry = JournalEntry[_T](datetime.date(2022, 12, 31), "Test Entry", "Source")
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers[account].entries) == 0
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal(100))


def test_build_general_ledger_with_multiple_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("A1")
    account2 = Account("A2")
    initial = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    journal_entry1 = JournalEntry[_T](datetime.date(2023, 1, 15), "Test Entry 1", "Source")
    journal_entry1.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(50)))
    journal_entry2 = JournalEntry[_T](datetime.date(2023, 1, 16), "Test Entry 2", "Source")
    journal_entry2.post(datetime.date(2023, 1, 16), account2, Quantity(Decimal(25)))
    journal = [journal_entry1, journal_entry2]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(150))
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(25))


def test_build_general_ledger_with_zero_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("A1")
    initial = {}
    journal_entry = JournalEntry[_T](datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(50))
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal(0))


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    ledger_entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(100.0)
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #31
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(date=datetime.date(2023, 10, 1), amount=Amount(100), journal=Journal(description="Test Journal", postings=[]), account="Test Account", direction="debit")
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #32
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    ledger_entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #33
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(datetime.date(2023, 10, 1), Journal("Test Journal", []), Amount(100), "debit", Account("Test Account"))
    balance = Quantity(100)
    ledger_entry = LedgerEntry(ledger, posting, balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #34
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(datetime.date(2023, 1, 1), Amount(100), Direction.DEBIT, Journal("Test Journal", []))
    balance = Quantity(100)
    ledger_entry = LedgerEntry(ledger, posting, balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #35
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int]()
    balance = Quantity()
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #36
#--------------------------

```python
def test_build_general_ledger_creates_new_ledger_for_uninitialized_account():
    from pypara.accounting import Account, Balance, Quantity, JournalEntry, Posting, Direction, Amount, DateRange
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger

    period = DateRange(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 12, 31))
    initial_balances = {}
    account = Account("Test Account")
    journal_entry = JournalEntry(date=datetime.date(2023, 1, 15), description="Test", source="Test Source")
    journal_entry.postings.append(Posting(journal_entry, datetime.date(2023, 1, 15), account, Direction.INC, Amount(Quantity(100))))
    journal = [journal_entry]

    general_ledger = build_general_ledger(period, journal, initial_balances)

    assert account in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[account], Ledger)
    assert general_ledger.ledgers[account].account == account
    assert general_ledger.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(100)


# LLM-generated content at query #37
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from pypara.accounting import Account, JournalEntry, Posting, Direction, Amount, Quantity
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.commons.zeitgeist import DateRange

    # Create test data
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    initial = {account1: Balance(date(2023, 1, 1), Quantity(Decimal(100)))}

    # Create journal entries with postings to both accounts
    journal_entry = JournalEntry(date(2023, 1, 2), "Test", "Source")
    journal_entry.post(date(2023, 1, 2), account1, Quantity(Decimal(50)))
    journal_entry.post(date(2023, 1, 2), account2, Quantity(Decimal(-50)))

    # Build general ledger
    general_ledger = build_general_ledger(period, [journal_entry], initial)

    # Assert that ledgers were created for both accounts
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert isinstance(general_ledger.ledgers[account1], Ledger)
    assert isinstance(general_ledger.ledgers[account2], Ledger)


# LLM-generated content at query #38
#--------------------------

```python
def test_build_general_ledger_includes_only_postings_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry_within_period = JournalEntry(datetime.date(2023, 6, 15), "Within period", "source")
    journal_entry_within_period.post(datetime.date(2023, 6, 15), Account("cash"), Quantity(Decimal(100)))
    journal_entry_outside_period = JournalEntry(datetime.date(2022, 12, 31), "Outside period", "source")
    journal_entry_outside_period.post(datetime.date(2022, 12, 31), Account("cash"), Quantity(Decimal(200)))
    initial_balances = {}
    general_ledger = build_general_ledger(period, [journal_entry_within_period, journal_entry_outside_period], initial_balances)
    assert len(general_ledger.ledgers[Account("cash")].entries) == 1


# LLM-generated content at query #39
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


# LLM-generated content at query #40
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


# LLM-generated content at query #41
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(Journal(datetime.date(2023, 10, 1), "Test Journal"), Account("Test Account"), Amount(100), Direction.DEBIT)
    balance = Quantity(100, "USD")
    entry = LedgerEntry(ledger, posting, balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #42
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


# LLM-generated content at query #43
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


# LLM-generated content at query #44
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


# LLM-generated content at query #45
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


# LLM-generated content at query #46
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(
        journal=Journal(description="Test Journal", postings=[]),
        amount=Amount(100),
        direction=Direction.DEBIT,
        account=Account("Test Account"),
        date=datetime.date(2023, 10, 1)
    )
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #47
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


# LLM-generated content at query #48
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = type('Posting', (), {
        'date': datetime.date(2023, 1, 1),
        'journal': type('Journal', (), {'description': 'Test Description'}),
        'amount': 100.0,
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False,
        'account': 'Test Account'
    })()
    balance = 500.0
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == balance
    assert entry.date == datetime.date(2023, 1, 1)
    assert entry.description == 'Test Description'
    assert entry.amount == 100.0
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == 100.0
    assert entry.credit == None


# LLM-generated content at query #49
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(100.0)
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #50
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial = {}
    journal_entry1 = JournalEntry(datetime.date(2023, 2, 1), "Test Entry 1", "Source1")
    journal_entry1.post(datetime.date(2023, 2, 1), Account("A1"), Quantity(Decimal(100)))
    journal_entry2 = JournalEntry(datetime.date(2022, 12, 31), "Test Entry 2", "Source2")
    journal_entry2.post(datetime.date(2022, 12, 31), Account("A2"), Quantity(Decimal(200)))
    journal = [journal_entry1, journal_entry2]
    general_ledger = build_general_ledger(period, journal, initial)
    assert Account("A1") in general_ledger.ledgers
    assert Account("A2") not in general_ledger.ledgers


# LLM-generated content at query #51
#--------------------------

```python
def test_build_general_ledger_initializes_ledgers_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {Account("cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal = []
    general_ledger = build_general_ledger(period, journal, initial_balances)
    assert general_ledger.ledgers[Account("cash")].initial.value == Quantity(Decimal(1000))


# LLM-generated content at query #52
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Mock()
    posting = Mock()
    balance = Mock()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #53
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


# LLM-generated content at query #54
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


# LLM-generated content at query #55
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


# LLM-generated content at query #56
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    ledger_entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #57
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


# LLM-generated content at query #58
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", date=datetime.date(2023, 10, 1)), account=Account(name="Test Account"), amount=Amount(value=100), direction=Direction.DEBIT)
    balance = Quantity(value=100)
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #59
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


# LLM-generated content at query #60
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Mock()
    posting = Mock()
    balance = Mock()
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #61
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 10, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test Journal"))
    balance = Quantity(100)
    entry = LedgerEntry[int](ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #62
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test", date=datetime.date(2023, 1, 1)), amount=Amount(value=100, currency="USD"), account=Account(name="Test Account"), direction=Direction.DEBIT)
    balance = Quantity(value=100, currency="USD")
    entry = LedgerEntry[int](ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #63
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


# LLM-generated content at query #64
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](account=Account("cash"), amount=Amount(100), direction=Direction.DEBIT, date=datetime.date(2023, 1, 1), journal=Journal(description="test", postings=[]))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #65
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


# LLM-generated content at query #66
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 10, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test Journal"), account=Account("Account1"))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #67
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


# LLM-generated content at query #68
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


# LLM-generated content at query #69
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test Journal", postings=[]))
    balance = Quantity(100)
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #70
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test", date=datetime.date(2023, 10, 1), postings=[]), account=Account("Test Account"), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #71
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #72
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


# LLM-generated content at query #73
#--------------------------

```python
def test_build_general_ledger_includes_postings_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 6, 15), Account("Test Account"), Quantity(Decimal(100)))
    initial_balances = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert len(general_ledger.ledgers[Account("Test Account")].entries) == 1

def test_build_general_ledger_excludes_postings_before_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test Entry", "Source")
    journal_entry.post(datetime.date(2022, 12, 31), Account("Test Account"), Quantity(Decimal(100)))
    initial_balances = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert Account("Test Account") not in general_ledger.ledgers

def test_build_general_ledger_excludes_postings_after_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2024, 1, 1), "Test Entry", "Source")
    journal_entry.post(datetime.date(2024, 1, 1), Account("Test Account"), Quantity(Decimal(100)))
    initial_balances = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert Account("Test Account") not in general_ledger.ledgers


# LLM-generated content at query #74
#--------------------------

```python
def test___call__():
    period = DateRange(start_date="2023-01-01", end_date="2023-01-31")
    program = GeneralLedgerProgram()
    result = program(period)
    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #75
#--------------------------

```python
def test_ledger_constructor_with_valid_account_and_initial_balance():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []

def test_ledger_constructor_with_empty_account():
    account = Account("")
    initial_balance = Balance(Quantity(50.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []

def test_ledger_constructor_with_zero_initial_balance():
    account = Account("test_account")
    initial_balance = Balance(Quantity(0.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []

def test_ledger_constructor_with_negative_initial_balance():
    account = Account("test_account")
    initial_balance = Balance(Quantity(-100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #76
#--------------------------

```python
def test_build_general_ledger_with_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert ledger.ledgers == {}

def test_build_general_ledger_with_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Cash")
    initial = {account: Balance(period.since, Quantity(Decimal(100)))}
    entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", None)
    entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [entry]
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert account in ledger.ledgers
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))

def test_build_general_ledger_with_journal_and_no_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("Cash")
    initial = {}
    entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", None)
    entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(50)))
    journal = [entry]
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert account in ledger.ledgers
    assert ledger.ledgers[account].initial.value == Quantity(Decimal(0))
    assert len(ledger.ledgers[account].entries) == 1
    assert ledger.ledgers[account].entries[0].balance == Quantity(Decimal(50))

def test_build_general_ledger_with_multiple_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Cash")
    account2 = Account("Revenue")
    initial = {account1: Balance(period.since, Quantity(Decimal(100)))}
    entry1 = JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", None)
    entry1.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(50)))
    entry1.post(datetime.date(2023, 1, 15), account2, Quantity(Decimal(-50)))
    journal = [entry1]
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert account1 in ledger.ledgers
    assert account2 in ledger.ledgers
    assert ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(150))
    assert ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(-50))


# LLM-generated content at query #77
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(datetime.date(2023, 1, 1), Amount(100), Direction.DEBIT, Journal("Test Journal"), Account("Test Account"))
    balance = Quantity(100)
    ledger_entry = LedgerEntry(ledger, posting, balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #78
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 1, 1), amount=Amount(100), journal=Journal(description="Test Journal"), account=Account("Test Account"), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #79
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int]()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(value=100), direction=Direction.DEBIT, journal=Journal(description="Test", date=datetime.date(2023, 1, 1)), account=Account(name="Test Account"))
    balance = Quantity(value=100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #4
#--------------------------

```python
def test_general_ledger_program_call():
    program = GeneralLedgerProgram()
    period = DateRange(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 12, 31))
    ledger = program(period)
    assert isinstance(ledger, GeneralLedger)


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
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 10, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="test", postings=[]))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #7
#--------------------------

```python
def test___call__():
    from datetime import date
    from typing import Protocol, TypeVar

    _T = TypeVar('_T')
    class GeneralLedger:
        pass

    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger()

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    program = MockGeneralLedgerProgram()
    result = program(period)
    assert isinstance(result, GeneralLedger)


# LLM-generated content at query #8
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
    account = Account("1234")
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert ledger.ledgers[account].account == account
    assert ledger.ledgers[account].initial == initial[account]
    assert ledger.ledgers[account].entries == []

def test_build_general_ledger_with_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1234")
    account2 = Account("5678")
    journal = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test entry",
            source=None,
            postings=[
                Posting(None, datetime.date(2023, 1, 15), account1, Direction.INC, Amount(Quantity(Decimal(100)))),
                Posting(None, datetime.date(2023, 1, 15), account2, Direction.DEC, Amount(Quantity(Decimal(100))))
            ]
        )
    ]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 2
    assert ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(100))
    assert ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(-100))

def test_build_general_ledger_with_out_of_period_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1234")
    journal = [
        JournalEntry(
            date=datetime.date(2022, 12, 31),
            description="Out of period",
            source=None,
            postings=[
                Posting(None, datetime.date(2022, 12, 31), account, Direction.INC, Amount(Quantity(Decimal(100))))
            ]
        )
    ]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert ledger.ledgers == {}

def test_build_general_ledger_with_mixed_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1234")
    account2 = Account("5678")
    initial = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(50)))}
    journal = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test entry",
            source=None,
            postings=[
                Posting(None, datetime.date(2023, 1, 15), account1, Direction.INC, Amount(Quantity(Decimal(100)))),
                Posting(None, datetime.date(2023, 1, 15), account2, Direction.DEC, Amount(Quantity(Decimal(100))))
            ]
        )
    ]
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 2
    assert ledger.ledgers[account1].initial == initial[account1]
    assert ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(150))
    assert ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(-100))


# LLM-generated content at query #9
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 10, 1), amount=Amount(value=100), direction=Direction.DEBIT, journal=Journal[int](description="Test Journal", postings=[]))
    balance = Quantity(value=100)
    entry = LedgerEntry[int](ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #10
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


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #13
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", date=datetime.date(2023, 10, 1)), amount=Amount(100), direction="debit")
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #14
#--------------------------

```python
def test_build_general_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), Account("Cash"), Quantity(Decimal(500)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    assert general_ledger.ledgers[Account("Cash")].entries[-1].balance == Quantity(Decimal(1500))


# LLM-generated content at query #15
#--------------------------

```python
def test_build_general_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test Entry",
            source="Test Source",
        ).post(datetime.date(2023, 1, 15), Account("Cash"), Quantity(Decimal(500)))
    ]
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[Account("Cash")].entries) == 1
    assert general_ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(1500))


# LLM-generated content at query #16
#--------------------------

```python
def test_build_general_ledger_creates_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = [JournalEntry(datetime.date(2023, 1, 1), "Test Entry", object(), [])]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.ledgers == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test", postings=[]), amount=Amount(100), direction=Direction.DEBIT, date=datetime.date(2023, 10, 1))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #18
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import build_general_ledger, Ledger, GeneralLedger
    from pypara.accounting.accounts import Account
    from pypara.commons.numbers import Quantity
    from pypara.commons.zeitgeist import DateRange
    import datetime
    from decimal import Decimal

    account = Account("1", "Test Account")
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))}

    # Create journal entries with different dates
    entry_in_period = JournalEntry(datetime.date(2023, 6, 1), "Test", None)
    entry_in_period.post(datetime.date(2023, 6, 1), account, Quantity(Decimal(100)))

    entry_before_period = JournalEntry(datetime.date(2022, 12, 31), "Test", None)
    entry_before_period.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(200)))

    entry_after_period = JournalEntry(datetime.date(2024, 1, 1), "Test", None)
    entry_after_period.post(datetime.date(2024, 1, 1), account, Quantity(Decimal(300)))

    journal = [entry_in_period, entry_before_period, entry_after_period]

    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial)

    # Only the posting within the period should be included
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting.amount == Amount(Quantity(Decimal(100)))


# LLM-generated content at query #19
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #20
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(amount=Amount(100), date=datetime.date(2023, 1, 1), journal=Journal(description="Test"), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #21
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


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_ReadInitialBalances_call():
    # Mock implementation of ReadInitialBalances
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"account1": 100.0, "account2": 200.0}

    # Create instance and test call
    reader = MockReadInitialBalances()
    test_period = ("2023-01-01", "2023-12-31")
    result = reader(test_period)
    
    # Assertions
    assert isinstance(result, dict)
    assert "account1" in result
    assert "account2" in result
    assert result["account1"] == 100.0
    assert result["account2"] == 200.0


# LLM-generated content at query #25
#--------------------------

```
def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #26
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #27
#--------------------------

```python
def test_general_ledger_program_call():
    # Mock DateRange
    class MockDateRange:
        pass
    
    # Mock GeneralLedger
    class MockGeneralLedger:
        pass
    
    # Create a concrete implementation of GeneralLedgerProgram
    class ConcreteGeneralLedgerProgram:
        def __call__(self, period):
            return MockGeneralLedger()
    
    program = ConcreteGeneralLedgerProgram()
    period = MockDateRange()
    result = program(period)
    
    assert isinstance(result, MockGeneralLedger)


# LLM-generated content at query #28
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, journal=Journal(description="Test Journal", postings=[]), amount=Amount(100), date=datetime.date.today(), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #29
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(account=Account(), amount=Amount(value=100, currency="USD"), direction="debit", date=datetime.date(2023, 1, 1))
    balance = Quantity(value=100, currency="USD")
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #31
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger("Test Ledger")
    posting = Posting(datetime.date(2023, 1, 1), "Test Description", Amount(100.0), Direction.DEBIT, Account("Test Account"))
    balance = Quantity(100.0)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #32
#--------------------------

```python
def test_build_general_ledger_with_empty_journal():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    journal = []
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert general_ledger.ledgers == {}


def test_build_general_ledger_with_single_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account = Account("Cash")
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", None)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(100))


def test_build_general_ledger_with_multiple_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Cash")
    account2 = Account("Revenue")
    journal_entry1 = JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", None)
    journal_entry1.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(100)))
    journal_entry2 = JournalEntry(datetime.date(2023, 1, 16), "Test Entry 2", None)
    journal_entry2.post(datetime.date(2023, 1, 16), account2, Quantity(Decimal(200)))
    journal = [journal_entry1, journal_entry2]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(100))
    assert general_ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(200))


def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account = Account("Cash")
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(50)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", None)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(150))


def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account = Account("Cash")
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test Entry", None)
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(100)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert account not in general_ledger.ledgers


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test_build_general_ledger_includes_journal_entries_within_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting import Account, Quantity, JournalEntry, Posting, Direction, DateRange
    from pypara.accounting.ledger import build_general_ledger

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("1234")
    initial_balances = {}

    # Journal entry within period
    je_within = JournalEntry(date(2023, 6, 15), "Test", None)
    je_within.postings.append(Posting(je_within, date(2023, 6, 15), account, Direction.INC, Quantity(Decimal("100"))))

    # Journal entry before period
    je_before = JournalEntry(date(2022, 12, 31), "Test", None)
    je_before.postings.append(Posting(je_before, date(2022, 12, 31), account, Direction.INC, Quantity(Decimal("50"))))

    # Journal entry after period
    je_after = JournalEntry(date(2024, 1, 1), "Test", None)
    je_after.postings.append(Posting(je_after, date(2024, 1, 1), account, Direction.INC, Quantity(Decimal("200"))))

    journal = [je_within, je_before, je_after]
    gl = build_general_ledger(period, journal, initial_balances)

    assert len(gl.ledgers[account].entries) == 1
    assert gl.ledgers[account].entries[0].posting.amount == Quantity(Decimal("100"))


# LLM-generated content at query #35
#--------------------------

def test_build_general_ledger_creates_ledger_for_new_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting import Account, Direction, JournalEntry, Posting, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.commons.zeitgeist import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account = Account("123", "Test Account")
    journal_entry = JournalEntry(date(2023, 1, 15), "Test", None)
    journal_entry.post(date(2023, 1, 15), account, Quantity(Decimal("100")))
    initial_balances = {}

    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)

    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal("100"))


# LLM-generated content at query #36
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


# LLM-generated content at query #37
#--------------------------

```
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", postings=[]), amount=Amount(value=100), direction="debit")
    balance = Quantity(value=200)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #38
#--------------------------

```python
def test_ledger_constructor():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100))
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #39
#--------------------------

```python
def test_build_general_ledger_with_no_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = [
        JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", source=None).post(
            datetime.date(2023, 1, 15), Account("A1"), Quantity(Decimal(100))
        )
    ]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[Account("A1")].initial.value == Decimal(0)
    assert len(general_ledger.ledgers[Account("A1")].entries) == 1
    assert general_ledger.ledgers[Account("A1")].entries[0].balance == Decimal(100)

def test_build_general_ledger_with_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = [
        JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", source=None).post(
            datetime.date(2023, 1, 15), Account("A1"), Quantity(Decimal(100))
        )
    ]
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(50)))}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[Account("A1")].initial.value == Decimal(50)
    assert len(general_ledger.ledgers[Account("A1")].entries) == 1
    assert general_ledger.ledgers[Account("A1")].entries[0].balance == Decimal(150)

def test_build_general_ledger_with_multiple_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = [
        JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", source=None).post(
            datetime.date(2023, 1, 15), Account("A1"), Quantity(Decimal(100))
        ),
        JournalEntry(datetime.date(2023, 1, 20), "Test Entry 2", source=None).post(
            datetime.date(2023, 1, 20), Account("A1"), Quantity(Decimal(-50))
        )
    ]
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(50)))}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[Account("A1")].initial.value == Decimal(50)
    assert len(general_ledger.ledgers[Account("A1")].entries) == 2
    assert general_ledger.ledgers[Account("A1")].entries[0].balance == Decimal(150)
    assert general_ledger.ledgers[Account("A1")].entries[1].balance == Decimal(100)

def test_build_general_ledger_with_multiple_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = [
        JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", source=None).post(
            datetime.date(2023, 1, 15), Account("A1"), Quantity(Decimal(100))
        ),
        JournalEntry(datetime.date(2023, 1, 20), "Test Entry 2", source=None).post(
            datetime.date(2023, 1, 20), Account("A2"), Quantity(Decimal(200))
        )
    ]
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(50)))}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 2
    assert general_ledger.ledgers[Account("A1")].initial.value == Decimal(50)
    assert general_ledger.ledgers[Account("A2")].initial.value == Decimal(0)
    assert len(general_ledger.ledgers[Account("A1")].entries) == 1
    assert len(general_ledger.ledgers[Account("A2")].entries) == 1
    assert general_ledger.ledgers[Account("A1")].entries[0].balance == Decimal(150)
    assert general_ledger.ledgers[Account("A2")].entries[0].balance == Decimal(200)


# LLM-generated content at query #40
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

    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({"account1": 100.0, "account2": 200.0})

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    reader = MockReadInitialBalances()
    result = reader(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 100.0, "account2": 200.0}

def test___call___handles_empty_period():
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

    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances({})

    period = DateRange(date(2023, 1, 1), date(2023, 1, 1))
    reader = MockReadInitialBalances()
    result = reader(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {}


# LLM-generated content at query #41
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", postings=[]), amount=Amount(100), direction=Direction.DEBIT, date=datetime.date(2023, 1, 1))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #42
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


# LLM-generated content at query #43
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


# LLM-generated content at query #44
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from pypara.accounting import Account, Balance, JournalEntry, Posting, Direction, Amount, Quantity, DateRange
    from pypara.accounting.ledger import build_general_ledger
    from datetime import date
    from decimal import Decimal

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    initial = {account1: Balance(date(2023, 1, 1), Quantity(Decimal(100)))}
    journal = [
        JournalEntry(
            date=date(2023, 1, 15),
            description="Test entry",
            source="test",
            postings=[
                Posting(None, date(2023, 1, 15), account1, Direction.INC, Amount(Quantity(Decimal(50)))),
                Posting(None, date(2023, 1, 15), account2, Direction.DEC, Amount(Quantity(Decimal(50)))),
            ],
        )
    ]

    general_ledger = build_general_ledger(period, journal, initial)
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers


# LLM-generated content at query #45
#--------------------------

```python
def test___call___returns_general_ledger():
    mock_ledger = object()
    mock_period = object()
    program = lambda period: mock_ledger
    result = program(mock_period)
    assert result is mock_ledger

def test___call___receives_correct_period():
    received_period = None
    def program(period):
        nonlocal received_period
        received_period = period
        return object()
    mock_period = object()
    program(mock_period)
    assert received_period is mock_period


# LLM-generated content at query #46
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 10, 1), amount=Amount(100), journal=Journal(description="Test", postings=[]), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #47
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


# LLM-generated content at query #48
#--------------------------

```python
def test_ReadInitialBalances_call():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"account1": 100, "account2": 200}

    mock_instance = MockReadInitialBalances()
    period = ("2023-01-01", "2023-01-31")
    result = mock_instance(period)
    assert result == {"account1": 100, "account2": 200}


# LLM-generated content at query #49
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_to_true():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 6, 15), Account("Test Account"), Quantity(Decimal(100)))
    initial_balances = {}
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert general_ledger.ledgers[Account("Test Account")].entries[0].posting.date == datetime.date(2023, 6, 15)


# LLM-generated content at query #50
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from datetime import date
    from typing import Dict, List

    class MockReadInitialBalances:
        def __call__(self, period):
            return {"account1": 100, "account2": 200}

    mock_read_initial_balances = MockReadInitialBalances()
    period = (date(2023, 1, 1), date(2023, 1, 31))
    result = mock_read_initial_balances(period)
    
    assert result == {"account1": 100, "account2": 200}


# LLM-generated content at query #51
#--------------------------

```
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](journal=Journal(description="Test", date="2023-01-01", postings=[]), account=Account(name="Test Account"), amount=Amount(value=100), direction=Direction.DEBIT)
    balance = Quantity(value=100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #52
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int]()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #53
#--------------------------

def test_build_general_ledger_with_empty_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting import Account, Balance, JournalEntry, Posting, Direction, Amount, Quantity
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.commons.zeitgeist import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.ledgers == {}


# LLM-generated content at query #54
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(account=Account(name="Cash"), amount=Amount(value=100), direction=Direction.DEBIT, date=datetime.date(2023, 1, 1), journal=Journal(description="Test Journal"))
    balance = Quantity(value=100)
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #55
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, Account("account"), Amount(100), Direction.DEBIT, datetime.date(2023, 10, 1))
    balance = Quantity(100)
    ledger_entry = LedgerEntry(ledger, posting, balance)
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == balance


# LLM-generated content at query #56
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


# LLM-generated content at query #57
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


# LLM-generated content at query #58
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal", date=datetime.date(2023, 10, 1)), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #59
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


# LLM-generated content at query #60
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


# LLM-generated content at query #61
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


# LLM-generated content at query #62
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


# LLM-generated content at query #63
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #64
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 10, 1), journal=Journal(), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #65
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(journal=Journal(description="Test Journal"), amount=Amount(100), date=datetime.date(2023, 10, 1))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #66
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


# LLM-generated content at query #67
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(date=datetime.date(2023, 1, 1), journal=Journal(description="Test Journal"), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #68
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger=ledger, amount=Amount(Decimal("100.00")), date=datetime.date(2023, 1, 1), direction=Direction.DEBIT)
    balance = Quantity(Decimal("100.00"))
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #69
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 10, 1), amount=Amount(100), direction=Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry[int](ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #70
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


# LLM-generated content at query #71
#--------------------------

```
def test_LedgerEntry_constructor():
    ledger = Ledger()
    posting = Posting()
    balance = Quantity()
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #72
#--------------------------

```python
def test_LedgerEntry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #73
#--------------------------

```python
def test_ledger_constructor():
    account = Account("Test Account")
    balance = Balance(Quantity(100.0))
    ledger = Ledger(account, balance)
    assert ledger.account == account
    assert ledger.initial == balance
    assert ledger.entries == []


# LLM-generated content at query #74
#--------------------------

```python
def test_build_general_ledger_with_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert general_ledger.ledgers == {}

def test_build_general_ledger_with_single_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(200)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(1200))

def test_build_general_ledger_with_multiple_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1", "Cash", AccountType.ASSETS)
    account2 = Account("2", "Revenue", AccountType.REVENUE)
    initial = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry1 = JournalEntry(datetime.date(2023, 1, 15), "Test Entry 1", "Source")
    journal_entry1.post(datetime.date(2023, 1, 15), account1, Quantity(Decimal(200)))
    journal_entry2 = JournalEntry(datetime.date(2023, 1, 20), "Test Entry 2", "Source")
    journal_entry2.post(datetime.date(2023, 1, 20), account2, Quantity(Decimal(500)))
    journal = [journal_entry1, journal_entry2]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal(1200))
    assert general_ledger.ledgers[account2].entries[0].balance == Quantity(Decimal(500))

def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1", "Cash", AccountType.ASSETS)
    initial = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test Entry", "Source")
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(200)))
    journal = [journal_entry]
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 0
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal(1000))


# LLM-generated content at query #75
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from pypara.accounting import Account, Balance, JournalEntry, Posting, Direction, Amount, Quantity
    from pypara.commons import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(date(2023, 1, 1), Quantity(Decimal(1000)))}

    journal = [
        JournalEntry(date(2023, 1, 15), "Sale", "Customer", postings=[
            Posting(None, date(2023, 1, 15), Account("Revenue"), Direction.INC, Amount(Decimal(500)))
        ]),
        JournalEntry(date(2023, 1, 16), "Purchase", "Supplier", postings=[
            Posting(None, date(2023, 1, 16), Account("Expense"), Direction.DEC, Amount(Decimal(200)))
        ])
    ]

    ledger = build_general_ledger(period, journal, initial_balances)

    assert ledger.period == period
    assert len(ledger.ledgers) == 3
    assert ledger.ledgers[Account("Cash")].initial.value == Quantity(Decimal(1000))
    assert ledger.ledgers[Account("Revenue")].entries[-1].balance == Quantity(Decimal(500))
    assert ledger.ledgers[Account("Expense")].entries[-1].balance == Quantity(Decimal(200))


# LLM-generated content at query #76
#--------------------------

```python
def test_ReadInitialBalances_call():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"account1": 1000, "account2": 2000}

    mock_instance = MockReadInitialBalances()
    period = ("2023-01-01", "2023-01-31")
    result = mock_instance(period)
    assert result == {"account1": 1000, "account2": 2000}


# LLM-generated content at query #77
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, datetime.date(2023, 10, 1), "Test Description", Amount(100), Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #78
#--------------------------

```python
def test_LedgerEntry_constructor():
    ledger = Ledger[int]()
    posting = Posting[int](date=datetime.date(2023, 10, 1), amount=Amount(value=100), direction=Direction.DEBIT)
    balance = Quantity(value=200)
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #79
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


# LLM-generated content at query #80
#--------------------------

```python
def test_ledger_entry_constructor():
    ledger = Ledger()
    posting = Posting(ledger, datetime.date(2023, 10, 1), Amount(100), Direction.DEBIT)
    balance = Quantity(100)
    entry = LedgerEntry(ledger, posting, balance)
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


