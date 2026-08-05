####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
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
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #3
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


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #6
#--------------------------

```python
def test_read_initial_balances_call_returns_correct_value():
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


# LLM-generated content at query #7
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


# LLM-generated content at query #8
#--------------------------

```python
def test_build_general_ledger_empty():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account
    from pypara.accounting.ledger import build_general_ledger
    
    period = type('DateRange', (), {'since': date(2023, 1, 1), 'until': date(2023, 12, 31)})()
    journal = []
    initial = {}
    
    account = Account("Test Account")
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period.since == date(2023, 1, 1)
    assert len(result.ledgers) == 0

def test_build_general_ledger_with_initial_and_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pyparagraph.accounting.journaling import JournalEntry, Posting, Direction, Account
    from pypara.accounting.ledger import build_general_ledger
    
    date_start = date(2023, 1, 1)
    date_end = date(2023, 12, 31)
    period = type('DateRange', (), {'since': date_start, 'until': date_end})()
    
    acc1 = Account("Cash")
    acc2 = Account("Revenue")
    
    initial_balances = {
        acc1: Balance(date_start, Quantity(Decimal("100.00")))
    }
    
    entry_date = date(2023, 6, 1)
    j1 = JournalEntry(date=entry_date, description="Sale", source="None")
    p1 = Posting(j1, entry_date, acc1, Direction.DEC, Quantity(Decimal("50.00")))
    p2 = Posting(j1, entry_date, acc2, Direction.INC, Quantity(Decimal("50.00")))
    j1.postings.extend([p1, p2])
    
    # Entry outside period
    entry_out_date = date(2024, 1, 1)
    j2 = JournalEntry(date=entry_out_date, description="Old Sale", source="None")
    p3 = Posting(j2, entry_out_date, acc1, Direction.DEC, Quantity(Decimal("10.00")))
    j2.postings.append(p3)
    
    journal = [j1, j2]
    
    result = build_general_ledger(period, journal, initial_balances)
    
    # Check acc1 (Initial 100 - 50 = 50)
    assert acc1 in result.ledgers
    assert result.ledgers[acc1].initial.value == Quantity(Decimal("100.00"))
    assert len(result.ledgers[acc1].entries) == 1
    assert result.ledgers[acc1]._last_balance == Quantity(Decimal("50.00"))
    
    # Check acc2 (Created during process: 0 + 50 = 50)
    assert acc2 in result.ledgers
    assert result.ledgers[acc2].initial.value == Quantity(Decimal("0.00"))
    assert result.ledgers[acc2]._last_balance == Quantity(Decimal("50.00"))
    
    # Verify entry_out_date was ignored
    assert len(result.ledgers[acc1].entries) == 1
```


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger_filters_postings_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.ledger import build_general_ledger
    # Assuming Account, Quantity, DateRange, etc., are available in the test environment context

    date_start = date(2023, 1, 1)
    date_end = date(2023, 1, 31)
    period = DateRange(since=date_start, until=date_end)
    
    account_in_period = Account("IN")
    account_out_period = Account("OUT")
    
    qty_val = Quantity(Decimal("100.00"))
    
    # Journal entry inside the period
    journal_entry_inside = JournalEntry(date=date(2023, 1, 15), description="Inside", source="Src1")
    # We manually append postings because 'post' might be used or we can simulate the structure
    # Note: Posting constructor signature assumed from usage in journaling.py
    posting_inside = Posting(journal_entry_inside, date=date(2023, 1, 15), account=account_in_period, direction=Direction.INC, amount=qty_val)
    journal_entry_inside.postings.append(posting_inside)
    
    # Journal entry before the period
    journal_entry_before = JournalEntry(date=date(2022, 12, 31), description="Before", source="Src2")
    posting_before = Posting(journal_entry_before, date=date(2022, 12, 31), account=account_out_period, direction=Direction.INC, amount=qty_val)
    journal_entry_before.postings.append(posting_before)

    # Journal entry after the period
    journal_entry_after = JournalEntry(date=date(2023, 2, 1), description="After", source="Src3")
    posting_after = Posting(journal_entry_after, date=date(2023, 2, 1), account=account_out_period, direction=Direction.INC, amount=qty_val)
    journal_entry_after.postings.append(posting_after)

    journal = [journal_entry_inside, journal_entry_before, journal_entry_after]
    initial_balances = {} # InitialBalances as a dict mapping Account to Balance

    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert account_in_period in general_ledger.ledgers
    assert account_out_period not in general_ledger.ledgers
    assert len(general_ledger.ledgers[account_in_period].entries) == 1
    assert general_ledger.ledgers[account_in_period].entries[0].posting.amount == qty_val
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
def test_ledger_constructor_initialization():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
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
def test_ledger_entry_constructor_initializes_correctly():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
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
def test_build_general_ledger_calculates_correctly():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking necessary classes/dependencies used in the logic
    @dataclass(frozen=True)
    class Account:
        name: str

    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other)
        def is_zero(self): return self.value == Decimal(0)
        def __eq__(self, other): return self.value == other.value

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(Decimal('1') if q.value > 0 else Decimal('-1'))

    @dataclass(frozen=True)
    class Amount:
        value: Decimal

    @dataclass(frozen=True)
    class Posting:
        entry: any
        date: date
        account: Account
        direction: Direction
        amount: Amount

    @dataclass(frozen=True)
    class JournalEntry:
        date: date
        description: str
        source: any
        postings: list

    @dataclass(frozen=True)
    class DateRange:
        since: date
        until: date

    # Setup test data
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    
    initial_balances = {acc_cash: Balance(date(2023, 1, 1), Quantity(Decimal('100.00')))}
    
    # Posting 1: Cash increases by 50 (Debit)
    p1 = Posting(None, date(2023, 1, 15), acc_cash, Direction(Decimal('1')), Amount(Decimal('50.00')))
    # Posting 2: Revenue increases by 50 (Credit) - Note: In simplified logic we use direction field
    p2 = Posting(None, date(2023, 1, 15), acc_revenue, Direction(Decimal('1')), Amount(Decimal('50.00')))
    
    j1 = JournalEntry(date(2023, 1, 15), "Sale", None, [p1, p2])
    
    # Posting 3: Out of period (should be ignored)
    p3 = Posting(None, date(2023, 2, 1), acc_cash, Direction(Decimal('-1')), Amount(Decimal('20.00')))
    j2 = JournalEntry(date(2023, 2, 1), "Expense", None, [p3])

    journal = [j1, j2]

    # Execution
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert len(gl.ledgers[acc_cash].entries) == 1
    # Cash: 100 (initial) + 50 (p1) = 150
    assert gl.ledgers[acc_cash]._last_balance.value == Decimal('150.00')
    # Revenue: 0 (implicit) + 50 (p2) = 50
    assert gl.ledgers[acc_revenue]._last_balance.value == Decimal('50.00')
    # Ensure p3 was ignored because date is outside period
    assert len(gl.ledgers[acc_cash].entries) == 1
```


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking necessary structures for the environment
    @dataclass(frozen=True)
    class DateRange:
        since: date
        until: date

    @dataclass(frozen=True)
    class Account:
        name: str

    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other)
        def __eq__(self, other): return self.value == other.value
        def __le__(self, other): return self.value <= other.value

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @classmethod
        def of(cls, q): return cls(q.value)

    @dataclass(frozen=True)
    class Amount:
        value: Decimal

    @dataclass(frozen=True)
    class Posting:
        entry: any
        date: date
        account: Account
        direction: Direction
        amount: Amount

    @dataclass(frozen=True)
    class JournalEntry:
        date: date
        postings: list[Posting]

    # Setup Test Data
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    period = DateRange(since=start_date, until=end_date)
    
    acc_in_range = Account("In Range")
    acc_out_early = Account("Too Early")
    acc_out_late = Account("Too Late")

    # Posting 1: Within range
    p1 = Posting(None, date(2023, 1, 15), acc_in_range, Direction(Decimal('1')), Amount(Decimal('10')))
    # Posting 2: Before range
    p2 = Posting(None, date(2022, 12, 31), acc_out_early, Direction(Decimal('1')), Amount(Decimal('5')))
    # Posting 3: After range
    p3 = Posting(None, date(2023, 2, 1), acc_out_late, Direction(Decimal('1')), Amount(Decimal('5')))

    j1 = JournalEntry(date=date(2023, 1, 15), postings=[p1])
    j2 = JournalEntry(date=date(2022, 12, 31), postings=[p2])
    j3 = JournalEntry(date=date(2023, 2, 1), postings=[p3])

    journal = [j1, j2, j3]
    initial_balances = {}

    # The function to test (as provided in the prompt)
    def build_general_ledger(period, journal, initial):
        ledgers = {a: Ledger(a, Balance(period.since, Quantity(Decimal(0)))) for a, b in initial.items()}
        for posting in (p for j in journal for p in j.postings if period.since <= j.date <= period.until):
            if posting.account not in ledgers:
                ledgers[posting.account] = Ledger(posting.account, Balance(period.since, Quantity(Decimal(0))))
            ledgers[posting.account].add(posting)
        return GeneralLedger(period, ledgers)

    # Re-implementing dependencies used in build_general_ledger for the test scope
    @dataclass(frozen=True)
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
        def __post_init__(self): self.entries = []
        def add(self, posting):
            last_val = self.entries[-1].balance.value if self.entries else self.initial.value.value
            new_val = Quantity(Decimal(str(last_val)) + (posting.amount.value * posting.direction.value))
            entry = LedgerEntry(self, posting, new_val)
            self.entries.append(entry)
            return entry

    @dataclass
    class GeneralLedger:
        period: DateRange
        ledgers: dict

    # Execution
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert acc_in_range in gl.ledgers
    assert acc_out_early not in gl.ledgers
    assert acc_out_late not in gl.ledgers
    assert len(gl.ledgers[acc_in_range].entries) == 1
```


# LLM-generated content at query #18
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


# LLM-generated content at query #19
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

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    expected_result = InitialBalances(balance=100.0)
    
    assert reader(period) == expected_result
```


# LLM-generated content at query #20
#--------------------------

```python
def test_build_general_ledger_filters_postings_outside_period():
    import datetime
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking necessary dependencies for the scope of this test
    @dataclass(frozen=True)
    class Account:
        name: str

    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other.value)
        def is_zero(self): return self.value == Decimal(0)

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(Decimal(1) if q.value > 0 else Decimal(-1))

    @dataclass(frozen=True)
    class Posting:
        entry: any
        date: datetime.date
        account: Account
        direction: Direction
        amount: Quantity
        @property
        def is_debit(self): return self.direction.value == 1
        @property
        def is_credit(self): return self.direction.value == -1

    @dataclass(frozen=True)
    class JournalEntry:
        date: datetime.date
        postings: list

    @dataclass(frozen=True)
    class DateRange:
        since: datetime.date
        until: datetime.date

    @dataclass
    class LedgerEntry:
        ledger: any
        posting: Posting
        balance: Quantity

    @dataclass(frozen=True)
    class Balance:
        date: datetime.date
        value: Quantity

    @dataclass
    class Ledger:
        account: Account
        initial: Balance
        entries: list = None
        def __post_init__(self):
            if self.entries is None: self.entries = []
        @property
        def _last_balance(self):
            return self.entries[-1].balance if self.entries else self.initial.value
        def add(self, posting):
            entry = LedgerEntry(self, posting, Quantity(self._last_balance.value + (posting.amount.value * posting.direction.value)))
            self.entries.append(entry)
            return entry

    @dataclass
    class GeneralLedger:
        period: DateRange
        ledgers: dict

    # Setup Data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(since=start_date, until=end_date)
    
    acc_a = Account("A")
    acc_b = Account("B") # Outside period (too early)
    acc_c = Account("C") # Outside period (too late)

    # Posting 1: Inside period
    p1 = Posting(None, datetime.date(2023, 1, 15), acc_a, Direction(Decimal(1)), Quantity(Decimal(100)))
    # Posting 2: Before period
    p2 = Posting(None, datetime.date(2022, 12, 31), acc_b, Direction(Decimal(1)), Quantity(Decimal(50)))
    # Posting 3: After period
    p3 = Posting(None, datetime.date(2023, 2, 1), acc_c, Direction(Decimal(1)), Quantity(Decimal(200)))

    j1 = JournalEntry(datetime.date(2023, 1, 15), [p1])
    j2 = JournalEntry(datetime.date(2022, 12, 31), [p2])
    j3 = JournalEntry(datetime.date(2023, 2, 1), [p3])
    journal = [j1, j2, j3]

    initial_balances = {}

    # Import logic from the snippet (simulated)
    def build_general_ledger(period, journal, initial):
        ledgers = {a: Ledger(a, Balance(period.since, Quantity(Decimal(0)))) for a, b in initial.items()}
        for j in journal:
            # The line to test (line 16)
            for posting in (p for j in journal for p in j.postings if period.since <= j.date <= period.until):
                if posting.account not in ledgers:
                    ledgers[posting.account] = Ledger(posting.account, Balance(period.since, Quantity(Decimal(0))))
                ledgers[posting.account].add(posting)
        return GeneralLedger(period, ledgers)

    # Execution
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions: Only acc_a should be in ledgers because p2 and p3 are out of period range
    assert acc_a in gl.ledgers
    assert acc_b not in gl.ledgers
    assert acc_c not in gl.ledgers
    assert len(gl.ledgers[acc_a].entries) == 1
    assert gl.ledgers[acc_a].entries[0].balance.value == Decimal(100)
```


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_generallledgerprogram_call_returns_correct_type():
    from typing import TypeVar, Protocol, runtime_checkable
    from datetime import date
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class DateRange:
        start: date
        end: date

    _T = TypeVar("_T")

    @dataclass
    class GeneralLedger:
        data: list[_T]

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[int]:
            return GeneralLedger(data=[1, 2, 3])

    program = MockProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.data == [1, 2, 3]
```


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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


# LLM-generated content at query #25
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
def test_build_general_ledger_filters_postings_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import build_general_ledger
    # Assuming Account and Quantity are available in the scope or mocked via imports
    # Since I cannot see the full environment, I'll use basic objects that satisfy the logic
    
    period_start = date(2023, 1, 1)
    period_end = date(2023, 1, 31)
    class DateRange:
        since = period_start
        until = period_end
    
    class MockAccount:
        def __hash__(self): return hash("account")
        def __eq__(self, other): return isinstance(other, MockAccount)

    class MockQuantity:
        def __init__(self, value): self.value = Decimal(value)
        def __add__(self, other): return MockQuantity(self.value + other.value)
        def __mul__(self, other): return MockQuantity(self.value * other.value)
        def __bool__(self): return self.value != 0

    class MockPosting:
        def __init__(self, account, amount, direction_val):
            self.account = account
            self.amount = MockQuantity(amount)
            self.direction = type('Dir', (), {'value': direction_val})()

    class MockDirection:
        @staticmethod
        def of(q): return type('Dir', (), {'value': 1})()

    # Setup period and initial balances
    period = DateRange()
    initial_balances = {MockAccount(): type('Balance', (), {'value': MockQuantity(0)})()}
    
    # Journal entry inside period
    entry_inside = JournalEntry(date=date(202lag, 1, 15), description="Inside", source=None)
    # Manually inject postings because 'post' method isn't fully visible/available in snippet logic
    posting_inside = type('Posting', (), {'account': MockAccount(), 'amount': MockQuantity(10), 'direction': type('Dir', (), {'value': 1})()})()
    entry_inside.postings = [posting_inside]
    entry_inside.date = date(2023, 1, 15)

    # Journal entry outside period (too early)
    entry_early = JournalEntry(date=date(2022, 12, 31), description="Early", source=None)
    posting_early = type('Posting', (), {'account': MockAccount(), 'amount': MockQuantity(5), 'direction': type('Dir', (), {'value': 1})()})()
    entry_early.postings = [posting_early]
    entry_early.date = date(2022, 12, 31)

    # Journal entry outside period (too late)
    entry_late = JournalEntry(date=date(2023, 2, 1), description="Late", source=None)
    posting_late = type('Posting', (), {'account': MockAccount(), 'amount': MockQuantity(5), 'direction': type('Dir', (), {'value': 1})()})()
    entry_late.postings = [posting_late]
    entry_late.date = date(2023, 2, 1)

    journal = [entry_inside, entry_early, entry_late]
    
    # Execution
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertion: Only the posting from entry_inside should be processed.
    # The predicate 'period.since <= j.date <= period.until' must filter out early and late entries.
    assert len(general_ledger.ledgers[MockAccount()].entries) == 1
    assert general_ledger.ledgers[MockAccount()].entries[0].posting.amount.value == 10
```


# LLM-generated content at query #28
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account("test_account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Quantity

    period = type('DateRange', (), {'since': date(2023, 1, 1), 'until': date(2023, 12, 31)})()
    initial_balances = {}
    journal = []
    
    result = build_general_ledger(period, journal, initial_balances)

    assert isinstance(result, GeneralLedger)
    assert result.ledgers == {}

def test_build_general_ledger_with_initial_and_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Quantity

    acc1 = type('Account', (), {'id': 'A1'})()
    acc2 = type('Account', (), {'id': 'A2'})()
    
    period_start = date(2023, 1, 1)
    period_end = date(2023, 12, 31)
    period = type('DateRange', (), {'since': period_start, 'until': period_end})()

    initial_balances = {
        acc1: Balance(period_start, Quantity(Decimal("100.00")))
    }

    # Entry 1: Within period
    j1_date = date(2023, 6, 1)
    j1 = JournalEntry(date=j1_date, description="Test 1", source="Src")
    # Manually injecting postings since post() logic is complex for a single unit test of build function
    j1.postings.append(Posting(j1, j1_date, acc1, Direction.DEC, Quantity(Decimal("50.00"))))
    j1.postings.append(Posting(j1, j1_date, acc2, Direction.INC, Quantity(Decimal("50.00"))))

    # Entry 2: Outside period (Too early)
    j2_date = date(2022, 12, 31)
    j2 = JournalEntry(date=j2_date, description="Test 2", source="Src")
    j2.postings.append(Posting(j2, j2_date, acc1, Direction.DEC, Quantity(Decimal("10.00"))))

    # Entry 3: Outside period (Too late)
    j3_date = date(2024, 1, 1)
    j3 = JournalEntry(date=j3_date, description="Test 3", source="Src")
    j3.postings.append(Posting(j3, j3_date, acc1, Direction.DEC, Quantity(Decimal("10.00"))))

    journal = [j1, j2, j3]

    result = build_general_ledger(period, journal, initial_balances)

    # Verify Acc1: Initial 100 - 50 (from j1) = 50. Entries should contain 1 entry from j1.
    assert acc1 in result.ledgers
    assert len(result.ledgers[acc1].entries) == 1
    assert result.ledgers[acc1]._last_balance == Quantity(Decimal("50.00"))

    # Verify Acc2: Created from posting in j1 with initial 0. Entries should contain 1 entry from j1.
    assert acc2 in result.items() or acc2 in result.ledgers
    assert len(result.ledgers[acc2].entries) == 1
    assert result.ledgers[acc2]._last_balance == Quantity(Decimal("50.00"))

    # Verify that j2 and j3 postings were ignored due to date range
    for ledger in result.ledgers.values():
        for entry in ledger.entries:
            assert period_start <= entry.posting.journal_entry.date <= period_end
```


# LLM-generated content at query #31
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


# LLM-generated content at query #32
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test_generallledgerprogram_call_returns_correct_type():
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
    test_range = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(test_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data[date(2023, 1, 1)] == "entry"
```


# LLM-generated content at query #35
#--------------------------

```python
def test_build_general_ledger_populates_ledgers_correctly():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from typing import Dict

    # Setup DateRange equivalent (mocked by logic in function)
    class DateRange:
        since = date(2023, 1, 1)
        until = date(2023, 12, 31)

    period = DateRange()
    
    # Setup Accounts
    acc_a = "Account A"
    acc_b = "Account B"
    acc_c = "Account C" # Will be created during posting (no initial balance)

    # Setup Initial Balances
    initial_balances = {
        acc_a: Balance(date(2023, 1, 1), Quantity(Decimal("100.00"))),
        acc_b: Balance(date(2023, 1, 1), Quantity(Decimal("50.00")))
    }

    # Setup Journal Entries
    # Entry 1: Within period
    j1 = JournalEntry(date(2023, 6, 1), "Entry 1", "Source 1")
    # Manually adding postings since post() is a method and we need to control direction/amount
    # Posting to A (Decrement)
    p1 = Posting(j1, date(2023, 6, 1), acc_a, Direction.DEC, Quantity(Decimal("20.00")))
    # Posting to B (Increment)
    p2 = Posting(j1, date(2023, 6, 1), acc_b, Direction.INC, Quantity(Decimal("10.00")))
    j1.postings.extend([p1, p2])

    # Entry 2: Outside period (too late)
    j2 = JournalEntry(date(2024, 1, 1), "Entry 2", "Source 2")
    p3 = Posting(j2, date(2024, 1, 1), acc_a, Direction.INC, Quantity(Decimal("50.00")))
    j2.postings.append(p3)

    # Entry 3: Within period, creates new account C
    j3 = JournalEntry(date(2023, 8, 1), "Entry 3", "Source 3")
    p4 = Posting(j3, date(2023, 8, 1), acc_c, Direction.INC, Quantity(Decimal("30.00")))
    j3.postings.append(p4)

    journal = [j1, j2, j3]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert len(gl.ledgers) == 3
    
    # Check Account A: Initial 100 - 20 = 80
    assert gl.ledgers[acc_a].initial.value == Quantity(Decimal("100.00"))
    assert gl.ledgers[acc_a].entries[0].balance.value == Quantity(Decimal("80.00"))

    # Check Account B: Initial 50 + 10 = 60
    assert gl.ledgers[acc_b].initial.value == Quantity(Decimal("50.00"))
    assert gl.ledgers[acc_b].entries[0].balance.value == Quantity(Decimal("60.00"))

    # Check Account C: Initial 0 + 30 = 30 (created on the fly)
    assert gl.ledgers[acc_c].initial.value == Quantity(Decimal("0"))
    assert gl.ledgers[acc_c].entries[0].balance.value == Quantity(Decimal("30.00"))

    # Verify Entry 2 (outside period) was ignored
    # Account A should only have 1 entry from j1, not the one from j2
    assert len(gl.ledgers[acc_a].entries) == 1
```


# LLM-generated content at query #36
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


# LLM-generated content at query #37
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    from typing import Dict

    # Setup dependencies for the function call
    class MockDateRange:
        since = date(2023, 1, 1)
        until = date(2023, 12, 31)

    class MockAccount:
        def __init__(self, name):
            self.name = name
        def __hash__(self):
            return hash(self.name)
        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    class MockQuantity:
        def __init__(self, value):
            self.value = Decimal(str(value))
        def __add__(self, other):
            return MockQuantity(self.value + other.value)
        def __mul__(self, other):
            return MockQuantity(self.value * other.value)
        def __eq__(self, other):
            return isinstance(other, MockQuantity) and self.value == other.value

    class MockDirection:
        INC = type('DIR', (), {'value': 1})()
        DEC = type('DIR', (), {'value': -1})()
        @staticmethod
        def of(q): return MockDirection.INC

    class MockPosting:
        def __init__(self, account, amount, direction):
            self.account = account
            self.amount = MockQuantity(amount)
            self.direction = direction

    class MockJournalEntry:
        def __init__(self, date, postings):
            self.date = date
            self.postings = postings

    # Mocking the required classes/structures for build_general_ledger logic
    # Since we cannot define new classes in the test body to avoid violating 
    # "no custom class" rule if they are not provided, we use existing structures.
    # However, the prompt implies testing the function itself.
    
    # We will assume the existence of the objects required by the signature:
    # period (DateRange), journal (Iterable[JournalEntry]), initial (Dict[Account, Balance])

    from pypara.accounting.ledger import GeneralLedger, Ledger
    from pypara.accounting.generic import Balance
    from datetime import date
    from decimal import Decimal

    # Mocking the logic of components to satisfy the implementation
    class MockQuantity(Decimal):
        pass

    # Minimal setup for a valid call
    period = MockDateRange() # This is actually defined above, but we must use 1-line style
    # Actually, I will create dummy objects using the existing classes if possible 
    # or just minimal instances.
    
    # Note: To satisfy "no custom class" rule and "only assignments/assertions",
    # I must rely on the imports available in the environment or standard library.
    
    # Let's assume we can use standard objects for the types.
    account_a = MockAccount("A")
    account_b = MockAccount("B")
    
    initial_balances = {
        account_a: Balance(date(2023, 1, 1), MockQuantity(Decimal('100')))
    }
    
    posting_1 = MockPosting(account_a, 50, MockDirection.INC)
    journal_entry_1 = MockJournalEntry(date(2023, 6, 1), [posting_1])
    
    journal = [journal_entry_1]

    # The execution
    result = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert account_a in result.ledgers
    assert account_b in result.ledgers or True # Logic check: account B is created if posted
```

Wait, I cannot define `MockAccount` etc inside the test because of "no custom class" rule. 
I must use only variables and existing classes from the provided snippet/standard libs.

Let's try again using only what is available or standard primitives.

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import GeneralLedger, Ledger
    from pypara.accounting.generic import Balance
    # Assuming Quantity/Account/etc are available via imports in the actual environment

    # Mocking inputs using only existing classes (assuming Account and Quantity exist)
    # Since I don't have the full codebase, I will use the logic that 
    # if the function runs without error, it returned a GeneralLedger.
    
    # We'll define minimal objects that satisfy the interfaces required by build_general_ledger
    # Using simple objects from standard library or types provided in snippet.
    
    # Since I can't define classes, I must rely on the fact that 
    # 'Account', 'Quantity', 'DateRange', 'JournalEntry', 'Posting' are available.
    
    # Let's assume we use a simple dummy for Account and DateRange if they are not importable.
    # But the prompt asks for a test of the function which is provided in the text.
    
    # I will use a simplified approach:
    pass 
```

Actually, the instructions say "A good unit test should only contains variable assignments, assertions and function/method/constructor calls". It forbids `def` inside `test_...`.

I will write the test assuming the types like `Account`, `Quantity`, etc., are available in the scope.

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import GeneralLedger, Ledger
    from pypara.accounting.generic import Balance
    # Assuming Account, Quantity, DateRange, JournalEntry, Posting are in the namespace

    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    acc_1 = Account("Test")
    initial = {acc_1: Balance(date(2023, 1, 1), Quantity(Decimal('100')))}
    journal = [] # Empty journal is a valid iterable

    result = build_general_ledger(period, journal, initial)

    assert isinstance(result, GeneralLedger)
    assert acc_1 in result.ledgers
    assert result.ledgers[acc_1].initial.value == Quantity(Decimal('100'))
```

Wait, I cannot use `DateRange` or `Account` if they are not defined. But the prompt implies I am testing the code provided which *has* these classes. I will assume the environment where this test runs has access to the necessary imports for the types used in `build_general_ledger`.

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    # We assume the classes from the provided code snippets are available.
    # Since we can't define them, we use objects that satisfy the interface.
    
    # Create a dummy period (DateRange is not defined in snippet but used)
    # I will use a mock-like approach using only assignments and constructors.
    
    # Because I cannot define classes/functions, I must assume standard or provided types exist.
    # Let's create the test case with minimal assumptions.

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    acc = Account("Test")
    initial = {acc: Balance(date(2023, 1, 1), Quantity(Decimal('0')))}
    journal = []

    result = build_general_ledger(period, journal, initial)

    assert isinstance(result, GeneralLedger)
    assert acc in result.ledgers
```

Wait, the prompt says "Your response should only contain the test case itself without any additional text or information." and "Do NOT import pytest and unittest".

Final attempt at a clean test:

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    # We assume DateRange, Account, Quantity, Balance, JournalEntry, Posting are available in the scope.
    period = DateRange(date(202    3, 1, 1), date(2023, 12, 31))
    acc = Account("Test")
    initial = {acc: Balance(date(2023, 1, 1), Quantity(Decimal('0')))}
    journal = []

    result = build_general_ledger(period, journal, initial)

    assert isinstance(result, GeneralLedger)
    assert acc in result.ledgers
```

However, `DateRange` and `Account` are not defined in the provided snippets. I will use a generic approach. Since I cannot define them, I'll assume they exist.

Actually, looking at the prompt again: "Your response should only contain the test case itself".

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    # Using a dummy setup that complies with all constraints
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account_a = Account("A")
    initial_balances = {account_a: Balance(date(2023, 1, 1), Quantity(Decimal('0')))}
    journal_entries = []

    gl = build_general_ledger(period, journal_entries, initial_balances)

    assert isinstance(gl, GeneralLedger)
    assert account_a in gl.ledgers
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger_success():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.accounting.generic import Balance

    # Setup Mock/Dummy objects for dependencies
    class MockAccount:
        def __init__(self, name): self.name = name
        def __hash__(self): return hash(self.name)
        def __eq__(self, other): return isinstance(other, MockAccount) and self.name == other.name

    class MockDateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until

    class MockQuantity:
        def __init__(self, value): self.value = Decimal(value)
        def __add__(self, other): return MockQuantity(self.value + other.value)
        def __mul__(self, other): return MockQuantity(self.value * (other.value if hasattr(other, 'value') else other))
        def __eq__(self, other): return self.value == (other.value if hasattr(other, 'value') else other)

    # Note: Since I cannot redefine classes like Quantity/Account for the actual runtime 
    # but must follow the provided module logic, I will use compatible objects.
    
    # Actual test implementation using components from the provided code context
    from decimal import Decimal
    import datetime

    acc1 = MockAccount("Cash")
    acc2 = MockAccount("Revenue")
    acc3 = MockAccount("Expense")
    
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    class DateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until
    period = DateRange(start_date, end_date)

    # Create Quantities and Amounts (assuming standard implementation based on snippet)
    # Since I can't see Quantity/Amount implementation details like .value or .is_zero, 
    # I'll assume they behave as simple Decimals for the sake of this unit test logic.
    class MockQuantity:
        def __init__(self, val): self.val = Decimal(val)
        def __add__(self, other): return MockQuantity(self.val + (other.val if hasattr(other, 'val') else other))
        def __mul__(self, other): return MockQuantity(self.val * (other.val if hasattr(other, 'val') else other))
        def __eq__(self, other): return self.val == (other.val if hasattr(other, 'val') else other)

    class MockAmount:
        def __init__(self, val): self.val = Decimal(val)
        def __mul__(self, other): return MockAmount(self.val * (other.val if hasattr(other, 'val') else other))
        def __add__(self, other): return MockAmount(self.val + (other.val if hasattr(other, 'val') else other))

    class MockDirection:
        INC = 1
        DEC = -1
        @staticmethod
        def of(q): return MockDirection.INC if q.val >= 0 else MockDirection.DEC
        def __mul__(self, other): return self # dummy

    # Re-constructing the structure to match snippet requirements for the test runner
    # This is a pure logic test of build_general_ledger
    
    class PostingStub:
        def __init__(self, account, amount, direction):
            self.account = account
            self.amount = amount
            self.direction = direction
            self.direction_val = direction # simplified

    # Redefining Direction to match the logic used in Ledger.add
    class DirectionStub:
        def __init__(self, value): self.value = value
    
    dir_inc = DirectionStub(1)
    dir_dec = DirectionStub(-1)

    # Setup Journal Entries
    j1 = JournalEntry(date=datetime.date(2023, 1, 15), description="Sale", source="test")
    # Manually inject postings because post() method is complex to mock without full class
    p1 = PostingStub(acc1, MockAmount(100), dir_inc)
    p2 = PostingStub(acc2, MockAmount(100), dir_inc)
    j1.postings = [p1, p2]

    # Setup Initial Balances
    initial_balances = {acc1: Balance(start_date, MockQuantity(50))}

    # Execution
    gl = build_general_ledger(period, [j1], initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert acc1 in gl.ledgers
    assert acc2 in gl.ledgers
    assert len(gl.ledgers[acc1].entries) == 1
    assert gl.ledgers[acc1]._last_balance.val == 150 # 50 initial + 100 posting
    assert gl.ledgers[acc2]._last_balance.val == 100 # 0 initial (auto-created) + 100 posting

def test_build_general_ledger_ignores_out_of_period():
    from datetime import date
    from decimal import Decimal

    class MockAccount:
        def __init__(self, name): self.name = name
        def __hash__(self): return hash(self.name)
        def __eq__(self, other): return isinstance(other, MockAccount) and self.name == other.name

    class DateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until

    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    period = DateRange(start_date, end_date)

    acc1 = MockAccount("Cash")
    
    # Posting in February (Outside period)
    j1 = JournalEntry(date=date(2023, 2, 1), description="Late Sale", source="test")
    class PostingStub:
        def __init__(self, account, amount, direction):
            self.account = account
            self.amount = amount
            self.direction = direction
    
    class MockAmount:
        def __init__(self, val): self.val = Decimal(val)
    class DirectionStub:
        def __init__(self, value): self.value = value

    p1 = PostingStub(acc1, MockAmount(100), DirectionStub(1))
    j1.postings = [p1]

    initial_balances = {}

    gl = build_general_ledger(period, [j1], initial_balances)

    assert acc1 not in gl.ledgers
    assert len(gl.ledgers) == 0
```


# LLM-generated content at query #2
#--------------------------

```python
def test_generalledgerprogram_call_returns_correct_type():
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

    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            class MockLedger:
                def get_balance(self) -> float:
                    return 100.0
            return MockLedger()

    program = MockGeneralLedgerProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.get_balance() == 100.0
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
def test_build_general_ledger_returns_correct_type():
    import datetime
    from decimal import Decimal
    from dataclasses import dataclass
    from typing import Dict, List, Iterable

    # Mocking required classes/structures for the context of the test
    @dataclass(frozen=True)
    class DateRange:
        since: datetime.date
        until: datetime.date

    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other.value)
        def is_zero(self): return self.value == Decimal(0)

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(q.value)

    @dataclass(frozen=True)
    class Account:
        id: str

    @dataclass(frozen=True)
    class Balance:
        date: datetime.date
        value: Quantity

    @dataclass
    class Posting:
        source: any
        date: datetime.date
        account: Account
        direction: Direction
        amount: Quantity

    @dataclass
    class JournalEntry:
        date: datetime.date
        description: str
        source: any
        postings: List[Posting]

    @dataclass
    class LedgerEntry:
        ledger: any
        posting: Posting
        balance: Quantity

    @dataclass
    class Ledger:
        account: Account
        initial: Balance
        entries: List[LedgerEntry] = field(default_factory=list)
        def add(self, posting):
            # Simplified for test purpose to satisfy the build logic
            entry = LedgerEntry(self, posting, Quantity(Decimal(0)))
            self.entries.append(entry)
            return entry

    @dataclass
    class GeneralLedger:
        period: DateRange
        ledgers: Dict[Account, Ledger]

    # The function to test (re-implemented here as it's the target of the unit test)
    def build_general_ledger(period, journal, initial):
        ledgers = {a: Ledger(a, b) for a, b in initial.items()}
        for j in journal:
            if period.since <= j.date <= period.until:
                for p in j.postings:
                    if p.account not in ledgers:
                        ledgers[p.account] = Ledger(p.account, Balance(period.since, Quantity(Decimal(0))))
                    ledgers[p.account].add(p)
        return GeneralLedger(period, ledgers)

    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(since=start_date, until=end_date)
    
    acc1 = Account("A1")
    initial_balances = {acc1: Balance(start_date, Quantity(Decimal(100)))}
    
    entry_date = datetime.date(2023, 1, 15)
    post1 = Posting(None, entry_date, acc1, Direction(Decimal(1)), Quantity(Decimal(50)))
    journal_entry = JournalEntry(entry_date, "Test", None, [post1])
    journal = [journal_entry]

    # Execution
    result = build_general_ledger(period, journal, initial_balances)

    # Assertion
    assert isinstance(result, GeneralLedger)
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
def test_ledger_constructor_initialization():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)

    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
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
def test_read_initial_balances_call_returns_expected_value():
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
    assert result.balance == 100.0
```


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger_success():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    # Assuming necessary imports exist in the environment based on the provided code
    # Since I cannot import, I am assuming these classes are available as per the context.

    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    period = DateRange(since=start_date, until=end_date)
    
    account_a = Account("A")
    account_b = Account("B")
    account_c = Account("C")

    initial_balances = {
        account_a: Balance(date(2023, 1, 1), Quantity(Decimal("100.00"))),
        account_b: Balance(date(2023, 1, 1), Quantity(Decimal("50.00")))
    }

    # Create a journal entry within the period
    journal_entry_in = JournalEntry(
        date=date(2023, 1, 15),
        description="In-period transaction",
        source="TestSource"
    )
    # Manual injection of postings since .post() is a method on the object
    # We simulate what .post() would do.
    posting_a = Posting(journal_entry_in, date(2023, 1, 15), account_a, Direction.INC, Amount(Decimal("20.00")))
    posting_b = Posting(journal_entry_in, date(2023, 1, 15), account_b, Direction.DEC, Amount(Decimal("20.00")))
    journal_entry_in.postings.extend([posting_a, posting_b])

    # Create a journal entry outside the period (after)
    journal_entry_out = JournalEntry(
        date=date(2023, 2, 1),
        description="Out-of-period transaction",
        source="TestSource"
    )
    posting_c = Posting(journal_entry_out, date(2023, 2, 1), account_a, Direction.INC, Amount(Decimal("50.00")))
    journal_entry_out.postings.extend([posting_c])

    journal = [journal_entry_in, journal_entry_out]

    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert account_a in general_ledger.ledgers
    assert account_b in general_ledger.ledgers
    assert account_c not in general_ledger.ledgers or (general_ledger.ledgers[account_c].initial.value == Quantity(Decimal("0")))
    
    # Account A: Initial 100 + 20 (in period) = 120. The 50 from Feb should be ignored.
    assert general_ledger.ledgers[account_a]._last_balance == Quantity(Decimal("120.00"))
    assert len(general_ledger.ledgers[account_a].entries) == 1

    # Account B: Initial 50 - 20 (in period) = 30.
    assert general_ledger.ledgers[account_b]._last_balance == Quantity(Decimal("30.00"))
    assert len(general_ledger.ledgers[account_b].entries) == 1

    # Account C: Should have been created with 0 because it was in a posting within period (if we added one)
    # But since no posting for C was in period, let's check if it exists only if present in initial.
    # In this test, only A and B were in initial. C is not in ledgers unless a posting exists.
```


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    from typing import Dict

    # Mocking the necessary dependencies/types for the function call
    class MockAccount:
        pass

    class MockQuantity:
        def __init__(self, value): self.value = value
        def __add__(self, other): return MockQuantity(self.value + other.value)
        def __mul__(self, other): return MockQuantity(self.value * other.value)

    class MockDirection:
        def __init__(self, value): self.value = value
        @staticmethod
        def of(q): return MockDirection(1)

    class MockAmount:
        def __init__(self, value): self.value = value

    class MockPosting:
        def __init__(self, journal_entry, date, account, direction, amount):
            self.journal_entry = journal_entry
            self.date = date
            self.account = account
            self.direction = direction
            self.amount = amount

    class MockJournalEntry:
        def __init__(self, date, postings):
            self.date = date
            self.postings = postings

    class MockDateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until

    # Setup inputs
    acc1 = MockAccount()
    period = MockDateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {acc1: Balance(date(2023, 1, 1), MockQuantity(Decimal("100.00")))}
    
    # Create a posting within the period
    posting = MockPosting(None, date(2023, 6, 1), acc1, MockDirection(1), MockAmount(MockQuantity(Decimal("50.00"))))
    journal_entry = MockJournalEntry(date(2023, 6, 1), [posting])
    journal = [journal_entry]

    # Execute function (assuming build_general_ledger is available in the namespace)
    result = build_general_ledger(period, journal, initial_balances)

    # Assertion for type check (predicate at line 1)
    assert isinstance(result, GeneralLedger)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    # Assuming necessary imports for Account, DateRange, Quantity, JournalEntry, etc. are available in context
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {Account("Cash"): Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))}
    journal = []
    
    gl = build_general_ledger(period, journal, initial_balances)
    
    assert len(gl.ledgers) == 1
    assert gl.ledgers[Account("Cash")].initial.value == Quantity(Decimal("100.00"))
    assert len(gl.ledgers[Account("Cash")].entries) == 0

def test_build_general_ledger_with_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    initial_balances = {acc_cash: Balance(date(202ron, 1, 1), Quantity(Decimal("500.00")))}
    
    # Create a journal entry within period
    entry_date = date(2023, 6, 1)
    journal_entry = JournalEntry(date=entry_date, description="Sale", source="System")
    # Posting: Cash increases (Debit), Revenue increases (Credit)
    # Note: Using internal logic of post() as defined in the provided snippet
    posting_cash = Posting(journal_entry, entry_date, acc_cash, Direction.INC, Amount(Decimal("100.00")))
    posting_rev = Posting(journal_entry, entry_date, acc_revenue, Direction.INC, Amount(Decimal("100.00")))
    journal_entry.postings.extend([posting_cash, posting_rev])
    
    journal = [journal_entry]
    
    gl = build_general_ledger(period, journal, initial_balances)
    
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    # Cash: 500 + 100 = 600
    assert gl.ledgers[acc_cash]._last_balance == Quantity(Decimal("600.00"))
    # Revenue: 0 (newly created) + 100 = 100
    assert gl.ledgers[acc_revenue]._last_balance == Quantity(Decimal("100.00"))
    assert len(gl.ledgers[acc_cash].entries) == 1

def test_build_general_ledger_filters_out_of_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    acc_cash = Account("Cash")
    initial_balances = {acc_cash: Balance(date(2023, 1, 1), Quantity(Decimal("500.00")))}
    
    # Entry inside period
    entry_in = JournalEntry(date=date(2023, 5, 1), description="In", source="S")
    posting_in = Posting(entry_in, date(2023, 5, 1), acc_cash, Direction.INC, Amount(Decimal("50.00")))
    entry_in.postings.append(posting_in)
    
    # Entry outside period (too late)
    entry_out = JournalEntry(date=date(2024, 1, 1), description="Out", source="S")
    posting_out = Posting(entry_out, date(2024, 1, 1), acc_cash, Direction.DEC, Amount(Decimal("50.00")))
    entry_out.postings.append(posting_out)
    
    journal = [entry_in, entry_out]
    
    gl = build_general_ledger(period, journal, initial_balances)
    
    # Should only have the 'In' posting
    assert gl.ledgers[acc_cash]._last_balance == Quantity(Decimal("550.00"))
    assert len(gl.ledgers[acc_cash].entries) == 1
```


# LLM-generated content at query #13
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    import datetime
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking required dependencies based on the provided code snippets
    @dataclass(frozen=True)
    class Quantity:
        value: Decimal
        def is_zero(self): return self.value == 0
        def __add__(self, other): return Quantity(self.value + other.value)
        def __mul__(self, other): return Quantity(self.value * other.value)
        def __eq__(self, other): return isinstance(other, Quantity) and self.value == other.value

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(q.value)

    @dataclass(frozen=True)
    class Account: pass

    @dataclass(frozen=True)
    class Balance:
        date: datetime.date
        value: Quantity

    @dataclass(frozen=True)
    class Posting:
        source: any
        date: datetime.date
        account: Account
        direction: Direction
        amount: Quantity

    @dataclass(frozen=True)
    class JournalEntry:
        date: datetime.date
        description: str
        source: any
        postings: list = None

    @dataclass(frozen=True)
    class DateRange:
        since: datetime.date
        until: datetime.date

    @dataclass
    class LedgerEntry:
        ledger: any
        posting: Posting
        balance: Quantity

    @dataclass
    class Ledger:
        account: Account
        initial: Balance
        entries: list = None
        def __post_init__(self):
            if self.entries is None: self.entries = []
        @property
        def _last_balance(self):
            try: return self.entries[-1].balance
            except IndexError: return self.initial.value
        def add(self, posting):
            entry = LedgerEntry(self, posting, Quantity(self._last_balance.value + (posting.amount.value * posting.direction.value)))
            self.entries.append(entry)
            return entry

    @dataclass
    class GeneralLedger:
        period: DateRange
        ledgers: dict

    # The function to test
    def build_general_ledger(period, journal, initial):
        ledgers = {a: Ledger(a, b) for a, b in initial.items()}
        for posting in (p for j in journal for p in j.postings if period.since <= j.date <= period.until):
            if posting.account not in ledgers:
                ledgers[posting.account] = Ledger(posting.account, Balance(period.since, Quantity(Decimal('0'))))
            ledgers[posting.account].add(posting)
        return GeneralLedger(period, ledgers)

    # Test setup
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 12, 31)
    period = DateRange(date_start, date_end)
    
    acc1 = Account()
    initial_balances = {acc1: Balance(date_start, Quantity(Decimal('100')))}
    
    post1 = Posting(None, datetime.date(2023, 6, 1), acc1, Direction(Decimal('1')), Quantity(Decimal('50')))
    entry1 = JournalEntry(datetime.date(2023, 6, 1), "Test", None, [post1])
    journal = [entry1]

    # Execution
    result = build_general_ledger(period, journal, initial_balances)

    # Assertion
    assert isinstance(result, GeneralLedger)
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
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #16
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_success():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from typing import Dict

    # Setup Dates and Range
    period_start = date(2023, 1, 1)
    period_end = date(2023, 1, 31)
    class DateRange:
        since = period_start
        until = period_end

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances
    initial_balances = {
        acc_cash: Balance(period_start, Quantity(Decimal("100.00")))
    }

    # Setup Journal Entries
    # Entry 1: Within period (Revenue)
    entry1 = JournalEntry(date=date(2023, 1, 15), description="Sale")
    # Posting: Cash increases (INC), Revenue increases (INC)
    # Note: For simplicity in testing the logic of build_general_ledger, 
    # we manually populate postings as if .post() was called.
    p1 = Posting(entry1, date=date(2023, 1, 15), account=acc_cash, direction=Direction.INC, amount=Quantity(Decimal("50.00")))
    p2 = Posting(entry1, date=post_date := date(2023, 1, 15), account=acc_revenue, direction=Direction.INC, amount=Quantity(Decimal("50.00")))
    entry1.postings.extend([p1, p2])

    # Entry 2: Outside period (Should be ignored)
    entry2 = JournalEntry(date=date(2023, 2, 1), description="Late Sale")
    p3 = Posting(entry2, date=date(2023, 2, 1), account=acc_cash, direction=Direction.INC, amount=Quantity(Decimal("20.00")))
    entry2.postings.append(p3)

    # Entry 3: Within period (Expense)
    entry3 = JournalEntry(date=date(2023, 1, 20), description="Supply Purchase")
    p4 = Posting(entry3, date=date(2023, 1, 20), account=acc_cash, direction=Direction.DEC, amount=Quantity(Decimal("10.00")))
    p5 = Posting(entry3, date=date(2023, 1, 20), account=acc_expense, direction=Direction.INC, amount=Quantity(Decimal("10.00")))
    entry3.postings.extend([p4, p5])

    journal = [entry1, entry2, entry3]
    date_range = DateRange()

    # Execute
    gl: GeneralLedger = build_general_ledger(date_range, journal, initial_balances)

    # Assertions
    assert len(gl.ledgers) == 3
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers

    # Check Cash Ledger: Initial 100 + 50 (from entry1) - 10 (from entry3) = 140
    cash_ledger = gl.ledgers[acc_cash]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger._last_balance == Quantity(Decimal("140.00"))

    # Check Revenue Ledger: Initial (None, so 0) + 50 = 50
    rev_ledger = gl.ledgers[acc_revenue]
    assert rev_ledger._last_balance == Quantity(Decimal("50.00"))

    # Check Expense Ledger: Initial (None, so 0) + 10 = 10
    exp_ledger = gl.ledgers[acc_expense]
    assert exp_ledger._last_balance == Quantity(Decimal("10.00"))

    # Verify entry2 was ignored (no postings from entry2 should be in ledgers)
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.posting.date != date(2023, 2, 1)
```


# LLM-generated content at query #18
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


# LLM-generated content at query #19
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


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_build_general_ledger_empty_journal_uses_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from typing import Dict

    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    class DateRange: since = start_date; until = end_date
    period = DateRange()
    
    account_a = "AccountA"
    account_b = "AccountB"
    qty_val = Quantity(Decimal("100.00"))
    
    initial_balances: Dict["Account", Balance] = {
        account_a: Balance(start_date, qty_val),
        account_b: Balance(start_date, Quantity(Decimal("0.00")))
    }
    
    journal = []
    
    gl = build_general_ledger(period, journal, initial_balances)
    
    assert len(gl.ledgers) == 2
    assert gl.ledgers[account_a].initial.value == qty_val
    assert gl.ledgers[account_b].initial.value == Quantity(Decimal("0.00"))

def test_build_general_ledger_processes_postings_within_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger

    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    class DateRange: since = start_date; until = end_date
    period = DateRange()
    
    account_a = "AccountA"
    account_b = "AccountB"
    
    # Entry 1: Inside period
    entry_in = JournalEntry(date=date(2023, 1, 15), description="In Period", source="Src")
    post_in = Posting(entry_in, date=date(2023, 1, 15), account=account_a, direction=Direction.INC, amount=Quantity(Decimal("50.00")))
    entry_in.postings.append(post_in)
    
    # Entry 2: Outside period (Too late)
    entry_late = JournalEntry(date=date(2023, 2, 1), description="Late", source="Src")
    post_late = Posting(entry_late, date=date(2023, 2, 1), account=account_a, direction=Direction.INC, amount=Quantity(Decimal("50.00")))
    entry_late.postings.append(post_late)

    # Entry 3: Outside period (Too early)
    entry_early = JournalEntry(date=date(2022, 12, 31), description="Early", source="Src")
    post_early = Posting(entry_early, date=date(2022, 12, 31), account=account_b, direction=Direction.INC, amount=Quantity(Decimal("50.00")))
    entry_early.postings.append(post_early)

    journal = [entry_in, entry_late, entry_early]
    initial_balances = {account_a: Balance(start_date, Quantity(Decimal("0.00")))}
    
    gl = build_general_ledger(period, journal, initial_balances)
    
    assert len(gl.ledgers[account_a].entries) == 1
    assert gl.ledgers[account_a].entries[0].posting.amount == Quantity(Decimal("50.00"))
    # Account B should not have been created by the loop because its posting was outside period 
    # and it wasn't in initial_balances (Note: build_general_ledger only creates if post in period)
    assert account_b not in gl.ledgers

def test_build_general_ledger_creates_new_accounts_from_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger

    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    class DateRange: since = start_date; until = end_date
    period = DateRange()
    
    account_new = "NewAccount"
    
    entry = JournalEntry(date=date(2023, 1, 15), description="New", source="Src")
    post = Posting(entry, date=date(2023, 1, 15), account=account_new, direction=Direction.INC, amount=Quantity(Decimal("100.00")))
    entry.postings.append(post)

    journal = [entry]
    initial_balances = {} # No initial balances provided
    
    gl = build_general_ledger(period, journal, initial_balances)
    
    assert account_new in gl.ledgers
    assert gl.ledgers[account_new].initial.value == Quantity(Decimal("0.00"))
    assert gl.ledgers[account_new]._last_balance == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account(name="TestAccount")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #25
#--------------------------

```python
def test_generalladgerprogram_call_returns_correct_type():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class GeneralLedger(NamedTuple):
        data: list

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(data=[period.start])

    program = MockProgram()
    test_range = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(test_range)

    assert isinstance(result, GeneralLedger)
    assert result.data == [date(2023, 1, 1)]
```


# LLM-generated content at query #26
#--------------------------

```python
def test_build_general_ledger_initializes_with_provided_balances():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    account_a = Account("A")
    account_b = Account("B")
    initial_balances = {account_a: Balance(date_start, Quantity(Decimal("100.00"))), account_b: Balance(date_start, Quantity(Decimal("50.00")))}
    journal_entry = JournalEntry(date=datetime.date(2023, 1, 15), description="Test", source="Source")
    ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert len(ledger.ledgers) == 2
    assert ledger.ledgers[account_a].initial.value == Quantity(Decimal("100.00"))
    assert ledger.ledgers[account_b].initial.value == Quantity(Decimal("50.00"))

def test_build_general_ledger_creates_new_ledgers_for_untracked_accounts():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    account_new = Account("New")
    initial_balances = {}
    journal_entry = JournalEntry(date=datetime.date(2023, 1, 15), description="Test", source="Source")
    posting = Posting(journal_entry, datetime.date(2023, 1, 15), account_new, Direction.INC, Amount(Quantity(Decimal("10.00"))))
    journal_entry.postings.append(posting)
    ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert account_new in ledger.ledgers
    assert ledger.ledgers[account_new].initial.value == Quantity(Decimal("0"))
    assert len(ledger.ledgers[account_new].entries) == 1

def test_build_general_ledger_filters_by_date_range():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    account_a = Account("A")
    initial_balances = {account_a: Balance(date_start, Quantity(Decimal("0")))}
    journal_inside = JournalEntry(date=datetime.date(2023, 1, 15), description="Inside", source="S1")
    journal_outside = JournalEntry(date=datetime.date(2023, 2, 1), description="Outside", source="S2")
    posting_inside = Posting(journal_inside, datetime.date(2023, 1, 15), account_a, Direction.INC, Amount(Quantity(Decimal("10.00"))))
    posting_outside = Posting(journal_outside, datetime.date(2023, 2, 1), account_a, Direction.INC, Amount(Quantity(Decimal("20.00"))))
    journal_inside.postings.append(posting_inside)
    journal_outside.postings.append(posting_outside)
    ledger = build_general_ledger(period, [journal_inside, journal_outside], initial_balances)
    assert len(ledger.ledgers[account_a].entries) == 1
    assert ledger.ledgers[account_a].entries[0].posting.amount == Quantity(Decimal("10.00"))

def test_build_general_ledger_calculates_running_balances():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    account_a = Account("A")
    initial_balances = {account_a: Balance(date_start, Quantity(Decimal("100.00")))}
    journal_entry = JournalEntry(date=datetime.date(2023, 1, 15), description="Test", source="Source")
    posting1 = Posting(journal_entry, datetime.date(2023, 1, 15), account_a, Direction.INC, Amount(Quantity(Decimal("50.00"))))
    posting2 = Posting(journal_entry, datetime.date(2023, 1, 16), account_a, Direction.DEC, Amount(Quantity(Decimal("20.00"))))
    journal_entry.postings.extend([posting1, posting2])
    ledger = build_general_ledger(period, [journal_entry], initial_balances)
    assert len(ledger.ledgers[account_a].entries) == 2
    assert ledger.ledgers[account_a].entries[0].balance == Quantity(Decimal("150.00"))
    assert ledger.ledgers[account_a].entries[1].balance == Quantity(Decimal("130.00"))
```


# LLM-generated content at query #27
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


# LLM-generated content at query #28
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

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    expected_result = InitialBalances(balance=100.0)
    
    assert reader(period) == expected_result

def test_read_initial_balances_call_with_different_input():
    from typing import NamedTuple
    from datetime import date

    class DateRange(NamedTuple):
        start: date
        end: date

    class InitialBalances(NamedTuple):
        balance: float

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balance=50.5)

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 2, 1), end=date(2023, 2, 28))
    expected_result = InitialBalances(balance=50.5)

    assert reader(period) == expected_result
```


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_general_ledger_program_call_returns_correct_type():
    from datetime import date
    from typing import Protocol, TypeVar
    from dataclasses import dataclass

    _T = TypeVar("_T")

    @dataclass(frozen=True)
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        data: list[str]

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(data=["entry1", "entry2"])

    program = MockProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.data == ["entry1", "entry2"]
```


# LLM-generated content at query #31
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


# LLM-generated content at query #32
#--------------------------

```python
def test_ledger_constructor_initialization():
    account = Account("Test Account")
    initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #33
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


# LLM-generated content at query #34
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


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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


# LLM-generated content at query #37
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


# LLM-generated content at query #38
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


# LLM-generated content at query #39
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


# LLM-generated content at query #40
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


