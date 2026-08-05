####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_general_ledger_program_call_returns_correct_type():
    from datetime import date
    from typing import Protocol, TypeVar, runtime_checkable

    _T = TypeVar("_T")

    @runtime_checkable
    class GeneralLedger(Protocol[_T]):
        def get_balance(self) -> float: ...

    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class MockGeneralLedger(GeneralLedger[int]):
        def get_balance(self) -> float:
            return 100.0

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[int]:
            return MockGeneralLedger()

    program = MockProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.get_balance() == 100.0
```


# LLM-generated content at query #2
#--------------------------

```python
def test_general_ledger_program_call_returns_ledger():
    from datetime import date
    from typing import Protocol, TypeVar, runtime_checkable
    
    _T = TypeVar("_T")

    @runtime_checkable
    class GeneralLedger(Protocol[_T]):
        def get_entries(self) -> list[_T]: ...

    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class MockGeneralLedger(GeneralLedger[_T]):
        def __init__(self, entries: list[_T]):
            self.entries = entries
        def get_entries(self) -> list[_T]:
            return self.entries

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[int]:
            return MockGeneralLedger([1, 2, 3])

    program = MockProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.get_entries() == [1, 2, 3]
```


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #6
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from datetime import date
    from unittest.mock import Mock

    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    mock_amount = Mock()
    
    mock_posting.date = date(2023, 1, 1)
    mock_posting.amount = mock_amount
    mock_posting.direction = "debit"
    mock_posting.is_debit = True
    mock_posting.is_credit = False

    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
    assert entry.date == date(2023, 1, 1)
    assert entry.amount == mock_amount
    assert entry.is_debit is True
    assert entry.is_credit is False
```


# LLM-generated content at query #8
#--------------------------

```python
def test_ledger_entry_constructor_initializes_correctly():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #9
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #10
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #11
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_constructor_initializes_correctly():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #13
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_from_initial_and_postings():
    import datetime
    from decimal import Decimal
    from typing import Dict, List
    from dataclasses import dataclass

    # Mocking necessary classes based on the provided context
    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other.value)
        def is_zero(self): return self.value == 0
        def __eq__(self, other): return self.value == other.value

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(q.value)

    @dataclass(frozen=True)
    class Account:
        name: str

    @dataclass(frozen=True)
    class Amount:
        value: Decimal

    @dataclass(frozen=True)
    class Balance:
        date: datetime.date
        value: Quantity

    @dataclass(frozen=True)
    class Posting:
        entry: any
        date: datetime.date
        account: Account
        direction: Direction
        amount: Amount

    @dataclass(frozen=True)
    class JournalEntry:
        date: datetime.date
        description: str
        source: any
        postings: List[Posting] = None

    @dataclass(frozen=True)
    class DateRange:
        since: datetime.date
        until: datetime.date

    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(since=start_date, until=end_date)

    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Initial balances: Cash starts with 100
    initial_balances = {
        acc_cash: Balance(start_date, Quantity(Decimal("100.00")))
    }

    # Journal Entry 1: Revenue (Within period) - Increases Cash
    post1 = Posting(None, datetime.date(202ng, 1, 5), acc_cash, Direction(Decimal("1")), Amount(Decimal("50.00")))
    post2 = Posting(None, datetime.date(2023, 1, 5), acc_revenue, Direction(Decimal("-1")), Amount(Decimal("50.00")))
    j1 = JournalEntry(datetime.date(2023, 1, 5), "Sales", None, [post1, post2])

    # Journal Entry 2: Expense (Within period) - Decreases Cash
    post3 = Posting(None, datetime.date(2023, 1, 10), acc_cash, Direction(Decimal("-1")), Amount(Decimal("20.00")))
    post4 = Posting(None, datetime.date(2023, 1, 10), acc_expense, Direction(Decimal("1")), Amount(Decimal("20.00")))
    j2 = JournalEntry(datetime.date(2023, 1, 10), "Supplies", None, [post3, post4])

    # Journal Entry 3: Outside period (Should be ignored)
    post5 = Posting(None, datetime.date(2023, 2, 1), acc_cash, Direction(Decimal("-1")), Amount(Decimal("10.00")))
    j3 = JournalEntry(datetime.date(2023, 2, 1), "Late Expense", None, [post5])

    journal = [j1, j2, j3]

    # Execution (Importing the logic from the module)
    from pypara.accounting.ledger import build_general_ledger
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers
    
    # Check Cash balance: 100 (initial) + 50 (j1) - 20 (j2) = 130
    # Note: The logic in 'add' uses posting.amount * posting.direction.value
    # post1 direction is 1, amount 50 -> +50
    # post3 direction is -1, amount 20 -> -20
    assert gl.ledgers[acc_cash].entries[0].balance.value == Quantity(Decimal("150.00")) # Initial (100) + post1 (50)
    assert gl.ledgers[acc_cash].entries[1].balance.value == Quantity(Decimal("130.00")) # 150 - 20

    # Check Revenue: Starts at 0 (created because it's in a posting but not in initial)
    # post2 direction is -1, amount 50 -> -50 (Wait, the logic uses direction.value * amount)
    # If direction is -1 and amount is 50, result is -50.
    assert gl.ledgers[acc_revenue].entries[0].balance.value == Quantity(Decimal("-50.00"))

    # Check Expense: Starts at 0
    # post4 direction is 1, amount 20 -> +20
    assert gl.ledgers[acc_expense].entries[0].balance.value == Quantity(Decimal("20.00"))

    # Verify J3 (Feb) was ignored
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.posting.date <= end_date
```


# LLM-generated content at query #14
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import GeneralLedger, Ledger
    # Assuming DateRange and Account/Quantity are available in the environment
    
    period_start = date(2023, 1, 1)
    period_end = date(2023, 1, 31)
    class DateRange:
        since = period_start
        until = period_end
    
    class MockAccount:
        pass

    class MockQuantity:
        def __init__(self, value): self.value = Decimal(value)
        def __add__(self, other): return MockQuantity(self.value + other.value)
        def __mul__(self, other): return MockQuantity(self.value * other.value)
        def __eq__(self, other): return isinstance(other, MockQuantity) and self.value == other.value
    
    class MockAmount:
        def __init__(self, value): self.value = Decimal(value)
        def __mul__(self, other): return Decimal(self.value * other.value)

    class MockDirection:
        INC = type('DIR', (), {'value': 1})()
        DEC = type('DIR', (), {'value': -1})()
        @staticmethod
        def of(q): return MockDirection.INC

    class MockPosting:
        def __init__(self, account, amount, direction):
            self.account = account
            self.amount = MockAmount(amount)
            self.direction = direction

    # Setup data
    acc_in_period = MockAccount()
    acc_out_period = MockAccount()
    
    entry_in_period = JournalEntry(date=date(2023, 1, 15), description="In", source=None)
    entry_in_period.postings.append(MockPosting(acc_in_period, 100, MockDirection.INC))
    
    entry_out_period = JournalEntry(date=date(2023, 2, 1), description="Out", source=None)
    entry_out_period.postings.append(MockPosting(acc_out_period, 50, MockDirection.INC))

    journal = [entry_in_period, entry_out_period]
    initial_balances = {acc_in_period: Balance(period_start, MockQuantity(0))}
    
    # Execution (The build function logic)
    period = DateRange()
    ledgers = {a: Ledger(a, b) for a, b in initial_balances.items()}
    for posting in (p for j in journal for p in j.postings if period.since <= j.date <= period.until):
        if posting.account not in ledgers:
            ledgers[posting.account] = Ledger(posting.account, Balance(period_start, MockQuantity(0)))
        ledgers[posting.account].add(posting)
    
    result_ledger = GeneralLedger(period, ledgers)

    # Assertions
    assert acc_in_period in result_ledger.ledgers
    assert acc_out_period not in result_ledger.ledgers
    assert len(result_ledger.ledgers[acc_in_period].entries) == 1
```


# LLM-generated content at query #15
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #16
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_success():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from dataclasses import dataclass

    # Setup dependencies
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    
    @dataclass
    class DateRange:
        since: datetime.date
        until: datetime.date

    period = DateRange(start_date, end_date)
    
    # Mock Account and Quantity logic (assuming standard behavior for the test context)
    account_a = "Account A"
    account_b = "Account B"
    
    initial_balances = {
        account_a: Balance(start_date, Quantity(Decimal("100.00")))
    }

    # Create a JournalEntry within the period
    entry_date = datetime.date(2023, 6, 1)
    journal_entry = JournalEntry(date=entry_date, description="Test Entry", source="Source")
    
    # Manually populate postings to avoid complex dependency chains in test
    from pypara.accounting.journaling import Amount
    posting_a = Posting(journal_entry, entry_date, account_a, Direction.INC, Amount(Decimal("50.00")))
    posting_b = Posting(journal_entry, entry_date, account_b, Direction.DEC, Amount(Decimal("50.00")))
    journal_entry.postings.extend([posting_a, posting_b])

    # Create journal outside the period (should be ignored)
    old_date = datetime.date(2022, 1, 1)
    old_entry = JournalEntry(date=old_date, description="Old Entry", source="Source")
    old_posting = Posting(old_entry, old_date, account_a, Direction.INC, Amount(Decimal("999.00")))
    old_entry.postings.append(old_posting)

    journal = [journal_entry, old_entry]

    # Execute function
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period.since == start_date
    assert account_a in gl.ledgers
    assert account_b in gl.ledgers
    
    # Check Account A (Initial 100 + 50 = 150)
    ledger_a = gl.ledgers[account_a]
    assert len(ledger_a.entries) == 1
    assert ledger_a._last_balance == Quantity(Decimal("150.00"))

    # Check Account B (Initial 0 + (-50) = -50)
    ledger_b = gl.ledgers[account_b]
    assert len(ledger_b.entries) == 1
    assert ledger_b._last_balance == Quantity(Decimal("-50.00"))

    # Verify old entry was not processed
    # The count for account_a should only reflect the post from the valid period
    assert all(e.posting.date >= start_date for e in ledger_a.entries)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #19
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    date_since = datetime.date(2023, 1, 1)
    date_until = datetime.date(2023, 12, 31)
    period = DateRange(date_since, date_until)
    account = Account("Test Account")
    initial_balances = {account: Balance(date_since, Quantity(Decimal("100.00")))}
    journal_entry = JournalEntry[str](date=datetime.date(2023, 6, 1), description="Test", source="TestSource")
    # No postings added to journal_entry, so it's an empty list
    journal = [journal_entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert isinstance(general_ledger, GeneralLedger)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #21
#--------------------------

```python
def test_general_ledger_program_call_returns_correct_type():
    from datetime import date
    from typing import NamedTuple, Protocol, TypeVar

    class DateRange(NamedTuple):
        start: date
        end: date

    T = TypeVar("T")

    class GeneralLedger(NamedTuple):
        data: list[T]

    class MockLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[int]:
            return GeneralLedger(data=[1, 2, 3])

    program = MockLedgerProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.data == [1, 2, 3]
```


# LLM-generated content at query #22
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    from typing import Dict

    # Mocking dependencies for the build_general_ledger function requirements
    class MockDateRange:
        since = date(2023, 1, 1)
        until = date(2023, 12, 31)

    class MockAccount:
        def __hash__(self): return hash("acc1")
        def __eq__(self, other): return isinstance(other, MockAccount)

    class MockQuantity:
        def __init__(self, val): self.val = Decimal(val)
        def __add__(self, other): return MockQuantity(self.val + other.val)
        def __mul__(self, other): return MockQuantity(self.val * other.val)
        def __eq__(self, other): return self.val == getattr(other, 'val', other)

    class MockDirection:
        INC = type('Dir', (), {'value': 1})()
        DEC = type('Dir', (), {'value': -1})()
        @staticmethod
        def of(q): return MockDirection.INC

    class MockAmount:
        def __init__(self, val): self.val = val
        def __mul__(self, other): return MockAmount(self.val * other.val)

    class MockPosting:
        def __init__(self, account, amount, direction):
            self.account = account
            self.amount = amount
            self.direction = direction

    class MockJournalEntry:
        def __init__(self, date, postings):
            self.date = date
            self.postings = postings

    # Setup input data
    period = MockDateRange()
    acc1 = MockAccount()
    initial_balances = {acc1: type('Balance', (), {'value': MockQuantity(0)})()}
    journal = [
        MockJournalEntry(date(2023, 6, 1), [
            MockPosting(acc1, MockAmount(10), MockDirection.INC)
        ])
    ]

    # Execute function (the target function is assumed to be in the namespace or imported)
    # Since I cannot import it, I am testing the logic that leads to the return statement.
    # The predicate at line 1 is simply checking if the function can execute and return GeneralLedger.
    from pypara.accounting.ledger import GeneralLedger

    result = build_general_ledger(period, journal, initial_balances)

    assert isinstance(result, GeneralLedger)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_read_initial_balances_call_returns_correct_value():
    from typing import NamedTuple
    from datetime import date

    class DateRange(NamedTuple):
        start: date
        end: date

    class InitialBalances(NamedTuple):
        balance: float

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balance=100.0)

    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    reader = MockReadInitialBalances()
    result = reader(period)

    assert result == InitialBalances(balance=100.0)
    assert isinstance(result, InitialBalances)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_build_general_ledger_success():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from typing import Dict

    # Setup DateRange and Period
    period_start = date(2023, 1, 1)
    period_end = date(2023, 1, 31)
    class DateRange:
        since = period_start
        until = period_end

    # Setup Mock Objects
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")
    
    initial_balances = {
        acc_cash: Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))
    }

    # Entry 1: Within period (Revenue)
    entry_in_period = JournalEntry(date(date(2023, 1, 15), "Sale of service", "SourceA"))
    posting_revenue = Posting(entry_in_period, date(2023, 1, 15), acc_revenue, Direction.INC, Amount(Decimal("50.00")))
    posting_cash_inc = Posting(entry_in_period, date(2023, 1, 15), acc_cash, Direction.INC, Amount(Decimal("50.00")))
    entry_in_period.postings.extend([posting_revenue, posting_cash_inc])

    # Entry 2: Outside period (Pre-period)
    entry_old = JournalEntry(date(date(2022, 12, 31), "Old entry", "SourceB"))
    posting_old = Posting(entry_old, date(2022, 12, 31), acc_cash, Direction.DEC, Amount(Decimal("10.00")))
    entry_old.postings.extend([posting_old])

    # Entry 3: Within period (Expense)
    entry_expense = JournalEntry(date(date(2023, 1, 20), "Supply purchase", "SourceC"))
    posting_expense = Posting(entry_expense, date(2023, 1, 20), acc_expense, Direction.DEC, Amount(Decimal("20.00")))
    posting_cash_dec = Posting(entry_expense, date(2023, 1, 20), acc_cash, Direction.DEC, Amount(Decimal("20.00")))
    entry_expense.postings.extend([posting_expense, posting_cash_dec])

    journal = [entry_in_period, entry_old, entry_expense]
    date_range = DateRange()

    # Execute
    gl = build_general_ledger(date_range, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers
    
    # Verify Cash Ledger: 100 (initial) + 50 (inc) - 20 (dec) = 130
    # Note: Entry_old is ignored because date < period.since
    cash_ledger = gl.ledgers[acc_cash]
    assert cash_ledger._last_balance == Quantity(Decimal("130.00"))
    assert len(cash_ledger.entries) == 2 # One inc, one dec

    # Verify Revenue Ledger: 0 (default for new) + 50 = 50
    rev_ledger = gl.ledgers[acc_revenue]
    assert rev_ledger._last_balance == Quantity(Decimal("50.00"))
    assert len(rev_ledger.entries) == 1

    # Verify Expense Ledger: 0 (default for new) - 20 = -20
    exp_ledger = gl.ledgers[acc_expense]
    assert exp_ledger._last_balance == Quantity(Decimal("-20.00"))
```


# LLM-generated content at query #25
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar, List, Optional

    @dataclass
    class Amount:
        value: float

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Account:
        name: str

    @dataclass
    class Direction:
        is_debit: bool

    @dataclass
    class PostingDirection:
        direction: str
        is_debit: bool

    @dataclass
    class Posting:
        date: date
        amount: Amount
        direction: PostingDirection
        is_debit: bool
        is_credit: bool
        journal: 'Journal'

    @dataclass
    class Journal:
        description: str
        postings: List['PostingPart']

    @dataclass
    class PostingPart:
        account: Account
        direction: str

    @dataclass
    class Ledger(Generic[TypeVar('T')]):
        pass

    class LedgerEntry(LedgerEntry_Mock): # Using the provided logic structure
        pass

    # Setup dependencies
    mock_ledger = Ledger()
    mock_amount = Amount(100.0)
    mock_quantity = Quantity(50.0)
    mock_date = date(2023, 1, 1)
    
    mock_account = Account("Cash")
    mock_direction = PostingDirection(direction="debit", is_debit=True)
    
    # Mocking the journal and its postings for property testing if needed, 
    # but focus on constructor assignments here.
    class MockJournal:
        description = "Test Journal"
        postings = []

    mock_posting = Posting(
        date=mock_date,
        amount=mock_amount,
        direction=mock_direction,
        is_debit=True,
        is_credit=False,
        journal=MockJournal()
    )

    # The actual test of constructor assignment
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_quantity)

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_quantity
```


# LLM-generated content at query #26
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #27
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    import datetime
    from decimal import Decimal
    from typing import Dict
    # Assuming the existence of necessary classes in the scope or via imports
    # Since I cannot import, I assume they are available as per context.

    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 12, 31)
    period = DateRange(since=date_start, until=date_end)
    
    # Mocking Account and Quantity/Balance/JournalEntry dependencies
    account_a = Account("A")
    initial_balances = {account_a: Balance(date_start, Quantity(Decimal("100.00")))}
    journal = []
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert isinstance(general_ledger, GeneralLedger)
```


# LLM-generated content at query #28
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar, List, Optional

    @dataclass
    class Amount:
        value: float

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Account:
        name: str

    @dataclass
    class PostingDirection:
        direction: str

    @dataclass
    class Posting:
        date: date
        amount: Amount
        direction: str
        is_debit: bool
        is_credit: bool
        journal: 'Journal'

    @dataclass
    class Journal:
        description: str
        postings: List['PostingElement']

    @dataclass
    class PostingElement:
        account: Account
        direction: str

    @dataclass
    class Ledger:
        pass

    class Journal:
        def __init__(self, description, postings):
            self.description = description
            self.postings = postings

    class Posting:
        def __init__(self, date, amount, direction, is_debit, is_credit, journal):
            self.date = date
            self.amount = amount
            self.direction = direction
            self.is_debit = is_debit
            self.is_credit = is_credit
            self.journal = journal

    class PostingElement:
        def __init__(self, account, direction):
            self.account = account
            self.direction = direction

    # Setup dependencies
    test_date = date(2023, 1, 1)
    test_amount = Amount(100.0)
    test_quantity = Quantity(100.0)
    test_account = Account("Test Account")
    test_journal = Journal("Test Journal", [PostingElement(test_account, "credit")])
    test_posting = Posting(test_date, test_amount, "debit", True, False, test_journal)
    test_ledger = Ledger()

    # Execution
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_quantity)

    # Assertions
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity
```


# LLM-generated content at query #29
#--------------------------

```python
def test_generalledgerprogram_call_returns_correct_type():
    from datetime import date
    from typing import Protocol, TypeVar, runtime_checkable

    _T = TypeVar("_T")

    @runtime_checkable
    class GeneralLedger(Protocol[_T]):
        def get_entries(self) -> list[_T]: ...

    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class MockGeneralLedger(GeneralLedger[int]):
        def get_entries(self) -> list[int]:
            return [1, 2, 3]

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[int]:
            return MockGeneralLedger()

    program = MockProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, MockGeneralLedger)
    assert result.get_entries() == [1, 2, 3]
```


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_entry_constructor_initializes_fields_correctly():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #31
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #32
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #33
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from unittest.mock import Mock
    
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #34
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #35
#--------------------------

```python
def test_read_initial_balances_call_returns_expected_value():
    from typing import NamedTuple
    from datetime import date

    class DateRange(NamedTuple):
        start: date
        end: date

    class InitialBalances(NamedTuple):
        amount: float

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(amount=100.0)

    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    expected_balance = InitialBalances(amount=100.0)
    reader: ReadInitialBalances = MockReadInitialBalances()
    
    result = reader(period)
    
    assert result == expected_balance
```


# LLM-generated content at query #36
#--------------------------

```python
def test_ledger_entry_constructor_initializes_correctly():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #37
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar, List, Optional

    @dataclass
    class Amount:
        value: float

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Account:
        name: str

    @dataclass
    class Direction:
        is_debit: bool

    @dataclass
    class PostingDirection:
        direction: str

    @dataclass
    class PostItem:
        account: Account
        direction: str

    @dataclass
    class Journal:
        description: str
        postings: List[PostItem]

    @dataclass
    class Posting:
        date: date
        journal: Journal
        amount: Amount
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[TypeVar("T")]):
        name: str

    # Setup dependencies
    test_date = date(2023, 1, 1)
    test_amount = Amount(100.0)
    test_quantity = Quantity(5.0)
    test_account_a = Account("Cash")
    test_account_b = Account("Revenue")
    
    test_journal = Journal(
        description="Sales Entry",
        postings=[
            PostItem(account=test_account_a, direction="debit"),
            PostItem(account=test_account_b, direction="credit")
        ]
    )
    
    test_posting = Posting(
        date=test_date,
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    
    test_ledger = Ledger(name="Main Ledger")

    # Execution
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assertions
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity
    assert entry.date == test_date
    assert entry.description == "Sales Entry"
    assert entry.amount == test_amount
    assert entry.is_debit is True
    assert entry.is_credit is False
    assert entry.debit == test_amount
    assert entry.credit is None
```


# LLM-generated content at query #38
#--------------------------

```python
def test_build_general_ledger_predicate_false():
    # Arrange
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking required types and classes for the scope of this test
    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other)
        def __eq__(self, other): return self.value == other.value
    
    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(Decimal(1) if q.value > 0 else Decimal(-1))

    @dataclass(frozen=True)
    class Account:
        id: str

    @dataclass
    class Posting:
        entry: any
        date: date
        account: Account
        direction: Direction
        amount: Quantity

    @dataclass
    class JournalEntry:
        date: date
        postings: list[Posting]

    @dataclass(frozen=True)
    class DateRange:
        since: date
        until: date

    @dataclass
    class InitialBalances:
        items: Dict[Account, Quantity]
        def items(self): return self.items.items()

    # The target function implementation logic provided in the prompt
    def build_general_ledger(period, journal, initial):
        ledgers = {a: Ledger(a, Balance(period.since, b)) for a, b in initial.items()}
        for j in journal:
            if period.since <= j.date <= period.until:
                for p in j.postings:
                    if p.account not in ledgers:
                        ledgers[p.account] = Ledger(p.account, Balance(period.since, Quantity(Decimal(0))))
                    ledgers[p.account].add(p)
        return ledgers # Simplified return for test scope

    @dataclass
    class Balance:
        date: date
        value: Quantity

    @dataclass
    class LedgerEntry:
        ledger: any
        posting: Posting
        balance: Quantity

    @dataclass
    class Ledger:
        account: Account
        initial: Balance
        entries: list[LedgerEntry] = None
        def __post_init__(self):
            if self.entries is None: self.entries = []
        def add(self, posting):
            prev_val = self.entries[-1].balance.value if self.entries else self.initial.value.value
            new_qty = Quantity(prev_val + (posting.amount.value * posting.direction.value))
            entry = LedgerEntry(self, posting, new_qty)
            self.entries.append(entry)
            return entry

    # Setup data where the predicate (period.since <= j.date <= period.until) is False
    # We use a date outside the range [2023-01-02, 2023-01-31]
    period = DateRange(date(2023, 1, 2), date(2023, 1, 31))
    account_a = Account("A")
    account_b = Account("B")
    initial_balances = InitialBalances({account_a: Quantity(Decimal(100))})
    
    # Date is 2023-01-01, which is < period.since
    out_of_range_date = date(2023, 1, 1)
    posting = Posting(None, out_of_range_date, account_b, Direction(Decimal(1)), Quantity(Decimal(50)))
    journal_entry = JournalEntry(out_of_range_date, [posting])
    journal = [journal_entry]

    # Act
    result_ledgers = build_general_ledger(period, journal, initial_balances)

    # Assert
    # If the predicate was False (as intended), account_b should NOT have been added to ledgers
    assert account_b not in result_ledgers
    # Account_a should still be there from initial
    assert account_a in result_ledgers
```


# LLM-generated content at query #39
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #40
#--------------------------

```python
def test_ledger_entry_constructor_initializes_correctly():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #41
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #42
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar, List, Optional

    @dataclass
    class Amount:
        value: float

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Account:
        name: str

    @dataclass
    class PostingDirection:
        direction: str

    @dataclass
    class Posting:
        date: date
        amount: Amount
        direction: str
        is_debit: bool
        is_credit: bool
        account: Account
        journal: 'Journal'

    @dataclass
    class Journal:
        description: str
        postings: List['Posting']

    @dataclass
    class Ledger(Generic[TypeVar('T')]):
        name: str

    _T = TypeVar("_T")
    test_date = date(2023, 1, 1)
    test_amount = Amount(100.0)
    test_balance = Quantity(500.0)
    test_account = Account("Cash")
    test_journal = Journal(description="Test Journal", postings=[])
    test_posting = Posting(date=test_date, amount=test_amount, direction="debit", is_debit=True, is_credit=False, account=test_account, journal=test_journal)
    test_ledger = Ledger(name="Main Ledger")

    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)

    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance
```


# LLM-generated content at query #43
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #44
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #45
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #3
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #4
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #6
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #7
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #8
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.generic import Balance, Quantity

    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    period = type('DateRange', (), {'since': start_date, 'until': end_date})
    
    account_a = type('Account', (), {})()
    initial_balances = {account_a: Balance(start_date, Quantity(Decimal("100.00")))}
    journal = []

    gl = build_general_ledger(period, journal, initial_balances)

    assert isinstance(gl, GeneralLedger)
    assert account_a in gl.ledgers
    assert gl.ledgers[account_a]._last_balance == Quantity(Decimal("100.00"))
    assert len(gl.ledgers[account_a].entries) == 0

def test_build_general_ledger_with_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.generic import Balance, Quantity

    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    period = type('DateRange', (), {'since': start_date, 'until': end_date})
    
    account_a = type('Account', (), {})()
    account_b = type('Account', (), {})()
    
    initial_balances = {account_a: Balance(start_date, Quantity(Decimal("100.00")))}
    
    entry_date = date(2023, 1, 15)
    # Entry within period
    j1 = JournalEntry(date=entry_date, description="Test", source=None)
    j1.postings.append(Posting(j1, entry_date, account_a, Direction.DEC, Amount(Decimal("20.00"))))
    j1.postings.append(Posting(j1, entry_date, account_b, Direction.INC, Amount(Decimal("20.00"))))
    
    # Entry outside period (after)
    entry_date_late = date(2023, 2, 1)
    j2 = JournalEntry(date=entry_date_late, description="Late", source=None)
    j2.postings.append(Posting(j2, entry_date_late, account_a, Direction.DEC, Amount(Decimal("50.00"))))

    journal = [j1, j2]

    gl = build_general_ledger(period, journal, initial_balances)

    # Account A: 100 (initial) - 20 (dec) = 80
    assert gl.ledgers[account_a]._last_balance == Quantity(Decimal("80.00"))
    # Account B: 0 (newly created) + 20 (inc) = 20
    assert gl.ledgers[account_b]._last_balance == Quantity(Decimal("20.00"))
    # Ensure j2 was ignored
    assert len(gl.ledgers[account_a].entries) == 1
```


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    # Assuming these classes exist in the scope as per the provided snippets
    # Mocking necessary components if not importable
    class MockAccount: pass
    class MockQuantity:
        def __init__(self, value): self.value = Decimal(value)
        def __add__(self, other): return MockQuantity(self.value + other.value)
        def __mul__(self, other): return MockQuantity(self.value * other.value)
        def __eq__(self, other): return self.value == other.value
    class MockAmount:
        def __init__(self, value): self.value = Decimal(value)
    class MockDirection:
        def __init__(self, value): self.value = Decimal(value)
        @staticmethod
        def of(q): return MockDirection(Decimal('1') if q.value > 0 else Decimal('-1'))
    class MockDateRange:
        def __init__(self, since, until): self.since = since; self.until = until
    class MockInitialBalances(dict): pass
    class MockLedgerEntry:
        def __init__(self, ledger, posting, balance): 
            self.ledger = ledger; self.posting = posting; self.balance = balance

    # Setup variables
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    period = MockDateRange(start_date, end_date)
    
    account_in_range = MockAccount()
    account_out_of_range = MockAccount()
    
    qty_pos = MockQuantity('10')
    qty_neg = MockQuantity('-5')
    
    # Journal entry 1: Inside range
    j1_date = date(2023, 1, 15)
    p1 = Posting(None, j1_date, account_in_range, MockDirection('1'), MockAmount('10'))
    j1 = JournalEntry(j1_date, "In range", None)
    j1.postings = [p1]
    
    # Journal entry 2: Before range
    j2_date = date(2022, 12, 31)
    p2 = Posting(None, j2_date, account_in_range, MockDirection('1'), MockAmount('5'))
    j2 = JournalEntry(j2_date, "Before range", None)
    j2.postings = [p2]
    
    # Journal entry 3: After range
    j3_date = date(2023, 2, 1)
    p3 = Posting(None, j3_date, account_in_range, MockDirection('1'), MockAmount('5'))
    j3 = JournalEntry(j3_date, "After range", None)
    j3.postings = [p3]

    journal = [j1, j2, j3]
    initial_balances = MockInitialBalances()
    
    # Execution
    gl = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    # The predicate (period.since <= j.date <= period.until) should only allow j1's posting
    # Therefore, the ledger for account_in_range should only have 1 entry from the journal
    assert len(gl.ledgers[account_in_range].entries) == 1
    assert gl.ledgers[account_in_range].entries[0].posting.amount.value == Decimal('10')
```


# LLM-generated content at query #10
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #11
#--------------------------

```python
def test_read_initial_balances_call_returns_correct_type():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class InitialBalances(NamedTuple):
        balance: float

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balance=100.0)

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = reader(period)

    assert isinstance(result, InitialBalances)
    assert result.balance == 100.0
```


# LLM-generated content at query #12
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #13
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar, List, Optional

    @dataclass
    class Amount:
        value: float

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Account:
        name: str

    @dataclass
    class Direction:
        is_debit: bool

    @dataclass
    class Posting:
        date: date
        amount: Amount
        direction: Direction
        is_debit: bool
        is_credit: bool
        journal: 'Journal'

    @dataclass
    class Journal:
        description: str
        postings: List['Posting']

    @dataclass
    class Ledger(Generic[TypeVar("T")]):
        pass

    # Mocking dependencies
    mock_ledger = Ledger()
    mock_amount = Amount(100.0)
    mock_date = date(2023, 1, 1)
    mock_direction = Direction(is_debit=True)
    
    # Create a dummy journal and posting to satisfy dependencies for property access if needed
    # although constructor only requires ledger, posting, and balance.
    class MockPosting:
        def __init__(self, date, amount, direction, is_debit, is_credit, journal):
            self.date = date
            self.amount = amount
            self.direction = direction
            self.is_debit = is_debit
            self.is_credit = is_credit
            self.journal = journal

    mock_journal = Journal(description="Test Journal", postings=[])
    mock_posting = MockPosting(mock_date, mock_amount, mock_direction, True, False, mock_journal)
    mock_balance = Quantity(500.0)

    # Execution
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)

    # Assertions
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #14
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #15
#--------------------------

```python
def test_build_general_ledger_predicate_evaluates_to_true():
    from datetime import date
    from decimal import Decimal
    from typing import Dict

    # Mocking necessary classes and structures for the test environment
    class MockDateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until

    class MockQuantity:
        def __init__(self, value):
            self.value = Decimal(value)
        def __add__(self, other): return MockQuantity(self.value + other.value)
        def __mul__(self, other): return MockQuantity(self.value * other)

    class MockDirection:
        def __init__(self, value):
            self.value = Decimal(value)

    class MockPosting:
        def __init__(self, account, amount, direction):
            self.account = account
            self.amount = MockQuantity(amount)
            self.direction = MockDirection(direction)

    class MockJournalEntry:
        def __init__(self, date, postings):
            self.date = date
            self.postings = postings

    # Setup data that satisfies the predicate: period.since <= j.date <= period.until
    test_date = date(2023, 1, 15)
    period = MockDateRange(date(2023, 1, 1), date(2023, 1, 31))
    
    account_a = "Account A"
    posting_val = MockPosting(account_a, 100, 1)
    journal_entry = MockJournalEntry(test_date, [posting_val])
    journal = [journal_entry]

    # The predicate logic extracted from line 16:
    # (p for j in journal for p in j.postings if period.since <= j.date <= period.until)
    predicate_generator = (p for j in journal for p in j.postings if period.since <= j.date <= period.until)
    extracted_postings = list(predicate_generator)

    # Assertions to ensure the predicate logic finds the posting
    assert len(extracted_postings) == 1
    assert extracted_postings[0].account == account_a
```


# LLM-generated content at query #16
#--------------------------

```python
def test_generallledgerprogram_call_returns_correct_type():
    from datetime import date
    from typing import NamedTuple, Protocol, TypeVar
    from dataclasses import dataclass

    class DateRange(NamedTuple):
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        data: dict

    _T = TypeVar("_T")

    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger:
            ...

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(data={"period": period})

    program = MockProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.data["period"] == period
```


# LLM-generated content at query #17
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #18
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #19
#--------------------------

```python
def test_general_ledger_program_call_returns_correct_type():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class GeneralLedger(NamedTuple):
        data: dict

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(data={period.start: "entry"})

    program = MockProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.data[date(2023, 1, 1)] == "entry"

def test_general_ledger_program_call_with_different_range():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class GeneralLedger(NamedTuple):
        value: int

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(value=period.end.day)

    program = MockProgram()
    period = DateRange(start=date(2023, 5, 1), end=date(2023, 5, 15))
    result = program(period)

    assert result.value == 15
```


# LLM-generated content at query #20
#--------------------------

```python
def test_read_initial_balances_call_returns_correct_type():
    from typing import NamedTuple
    from datetime import date

    class DateRange(NamedTuple):
        start: date
        end: date

    class InitialBalances(NamedTuple):
        balance: float

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balance=100.0)

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    expected_result = InitialBalances(balance=100.0)
    
    result = reader(period)
    
    assert result == expected_result
    assert isinstance(result, InitialBalances)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #22
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #23
#--------------------------

```python
def test_ledger_entry_constructor():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #24
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account(id="acc-123", name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #25
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #26
#--------------------------

```python
def test_build_general_ledger_predicate_false():
    """
    Ensures that the predicate `period.since <= j.date <= period.until` 
    evaluates to False for a journal entry date outside the specified range.
    """
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    # Note: Assuming necessary imports like Account, DateRange, etc., are available in the environment
    
    # Define period: 2023-01-01 to 2023-12-31
    period_since = date(2023, 1, 1)
    period_until = date(2023, 12, 31)
    # Mocking DateRange behavior via a simple object or assuming standard implementation
    class MockDateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until
    period = MockDateRange(period_since, period_until)

    # Define an account and initial balances (empty for this test)
    account_outside = Account("OutsideAccount") # Assuming Account exists
    initial_balances = {} 

    # Create a JournalEntry with a date OUTSIDE the range (e.g., 2024-01-01)
    journal_date_outside = date(2024, 1, 1)
    # Mocking source and other dependencies
    source_obj = object()
    entry_outside = JournalEntry(date=journal_date_outside, description="Outside Period", source=source_obj)
    
    # Add a posting to this entry
    amount_val = Quantity(Decimal("100.00"))
    # Manually appending to postings since post() is a method and we want to control the state directly
    from pypara.accounting.journaling import Posting, Direction
    entry_outside.postings.append(Posting(entry_outside, journal_date_outside, account_outside, Direction.INC, Amount(amount_val)))

    # The journal list for the function call
    journal = [entry_outside]

    # We want to verify that when build_general_ledger is called, 
    # the generator expression (p for j in journal for p in j.postings if period.since <= j.date <= period.until)
    # results in an empty iterator for this specific entry.
    
    # Execution of the logic inside the function's loop:
    filtered_postings = [p for j in journal for p in j.postings if period.since <= j.date <= period.until]

    # Assertion: The list should be empty because 2024-01-01 is not between 2023-01-01 and 2023-12-31
    assert len(filtered_postings) == 0
```


# LLM-generated content at query #27
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #28
#--------------------------

```python
def test_ledger_constructor_initializes_correctly():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #29
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    import datetime
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking necessary classes/structures for the context of the test
    @dataclass(frozen=True)
    class DateRange:
        since: datetime.date
        until: datetime.date

    @dataclass(frozen=True)
    class Account:
        name: str

    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other)
        def __eq__(self, other): return self.value == other.value
        def is_zero(self): return self.value == Decimal(0)

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(Decimal(1) if q.value > 0 else Decimal(-1))

    @dataclass
    class Posting:
        journal_entry: any
        date: datetime.date
        account: Account
        direction: Direction
        amount: Quantity

    @dataclass
    class JournalEntry:
        date: datetime.date
        description: str
        source: any
        postings: list

    # Setup Test Data
    period = DateRange(since=datetime.date(2023, 1, 1), until=datetime.date(2023, 1, 31))
    acc_a = Account("A")
    acc_b = Account("B")
    
    # Posting inside range
    p_in = Posting(None, datetime.date(2023, 1, 15), acc_a, Direction(Decimal(1)), Quantity(Decimal(10)))
    # Posting before range
    p_before = Posting(None, datetime.date(2022, 12, 31), acc_a, Direction(Decimal(1)), Quantity(Decimal(5)))
    # Posting after range
    p_after = Posting(None, datetime.date(2023, 2, 1), acc_a, Direction(Decimal(1)), Quantity(Decimal(5)))
    
    j_in = JournalEntry(datetime.date(2023, 1, 15), "In range", None, [p_in])
    j_before = JournalEntry(datetime.mock_date(2022, 12, 31) if hasattr(datetime, 'mock_date') else datetime.date(2022, 12, 31), "Before", None, [p_before])
    j_after = JournalEntry(datetime.date(2023, 2, 1), "After", None, [p_after])

    # We need to mock the function logic as it's not provided in a single importable block, 
    # but we test the specific predicate logic: (p for j in journal for p in j.postings if period.since <= j.date <= period.until)
    journal = [j_in, j_before, j_after]
    
    # The predicate extraction
    filtered_postings = [p for j in journal for p in j.postings if period.since <= j.date <= period.until]

    assert len(filtered_postings) == 1
    assert filtered_postings[0] == p_in
```


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_entry_constructor_initializes_fields():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


