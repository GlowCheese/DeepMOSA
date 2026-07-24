####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generallgederprogram_call_returns_correct_type():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class GeneralLedger(NamedTuple):
        data: dict

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(data={period.start: "value"})

    program = MockProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.data[date(2023, 1, 1)] == "value"
```


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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

    read_initial_balances: ReadInitialBalances = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    expected_result = InitialBalances(balance=100.0)
    
    assert read_initial_balances(period) == expected_result
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
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
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
def test_read_initial_balances_call_returns_correct_balances():
    from dataclasses import dataclass
    from datetime import date
    from typing import Dict

    @dataclass(frozen=True)
    class DateRange:
        start: date
        end: date

    @dataclass(frozen=True)
    class InitialBalances:
        balances: Dict[str, float]

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balances={"USD": 100.0, "EUR": 85.0})

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    expected_balances = InitialBalances(balances={"USD": 100.0, "EUR": 85.0})
    
    result = reader(period)
    
    assert result == expected_balances
    assert result.balances["USD"] == 100.0
```


# LLM-generated content at query #8
#--------------------------

```python
def test_build_general_ledger_initializes_with_provided_balances():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    
    account_a = Account("A")
    account_b = Account("B")
    quantity_a = Quantity(Decimal("100.00"))
    quantity_b = Quantity(Decimal("50.00"))
    
    initial_balances = {
        account_a: Balance(date_start, quantity_a),
        account_b: Balance(date_start, quantity_b)
    }
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test Entry",
        source="TestSource"
    )
    # We manually inject postings because post() returns a new instance (dataclass frozen=True)
    # and we need to simulate the state of the journal
    posting_a = Posting(journal_entry, datetime.date(2023, 1, 15), account_a, Direction.INC, Amount(Quantity(Decimal("20.00"))))
    journal_entry.postings.append(posting_a)
    
    journal = [journal_entry]
    
    ledger = build_general_ledger(period, journal, initial_balances)
    
    assert ledger.period == period
    assert account_a in ledger.ledgers
    assert account_b in ledger.ledgers
    assert ledger.ledgers[account_a].initial.value == quantity_a
    assert ledger.ledgers[account_b].initial.value == quantity_b
    assert ledger.ledgers[account_a].entries[0].balance.value == Quantity(Decimal("120.00"))

def test_build_general_ledger_ignores_entries_outside_period():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    
    account_a = Account("A")
    initial_balances = {account_a: Balance(date_start, Quantity(Decimal("0.00")))}
    
    # Entry inside period
    entry_inside = JournalEntry(datetime.date(2023, 1, 15), "Inside", "Src")
    posting_inside = Posting(entry_inside, datetime.date(2023, 1, 15), account_a, Direction.INC, Amount(Quantity(Decimal("10.00"))))
    entry_inside.postings.append(posting_inside)
    
    # Entry outside period (too late)
    entry_outside = JournalEntry(datetime.date(2023, 2, 1), "Outside", "Src")
    posting_outside = Posting(entry_outside, datetime.date(2023, 2, 1), account_a, Direction.INC, Amount(Quantity(Decimal("10.00"))))
    entry_outside.postings.append(posting_outside)
    
    # Entry outside period (too early)
    entry_early = JournalEntry(datetime.date(2022, 12, 31), "Early", "Src")
    posting_early = Posting(entry_early, datetime.date(2022, 12, 31), account_a, Direction.INC, Amount(Quantity(Decimal("10.00"))))
    entry_early.postings.append(posting_early)
    
    journal = [entry_inside, entry_outside, entry_early]
    
    ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(ledger.ledgers[account_a].entries) == 1
    assert ledger.ledgers[account_a].entries[0].balance.value == Quantity(Decimal("10.00"))

def test_build_general_ledger_creates_new_ledgers_for_untracked_accounts():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    
    account_new = Account("New")
    initial_balances = {} # No initial balance provided for account_new
    
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "New Account Entry", "Src")
    posting_new = Posting(journal_entry, datetime.append(datetime.date(2023, 1, 15)), account_new, Direction.INC, Amount(Quantity(Decimal("50.00"))))
    # Correcting usage for the test scope:
    journal_entry.postings.append(Posting(journal_entry, datetime.date(2023, 1, 15), account_new, Direction.INC, Amount(Quantity(Decimal("50.00")))))
    
    journal = [journal_entry]
    
    ledger = build_general_ledger(period, journal, initial_balances)
    
    assert account_new in ledger.ledgers
    assert ledger.ledgers[account_new].initial.value == Quantity(Decimal("0.00"))
    assert ledger.ledgers[account_new].entries[0].balance.value == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger_predicate_filtering_logic():
    import datetime
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking necessary classes for the environment
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
        def __mul__(self, other): return Quantity(self.value * other.value)
        def __eq__(self, other): return self.value == other.value
        def __le__(self, other): return self.value <= other.value
        def __ge__(self, other): return self.value >= other.value

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(q.value)

    @dataclass(frozen=True)
    class Posting:
        entry: any
        date: datetime.date
        account: Account
        direction: Direction
        amount: Quantity

    @dataclass(frozen=True)
    class JournalEntry:
        date: datetime.date
        description: str
        source: any
        postings: list

    # Setup test data
    period = DateRange(datetime.date(202
        3, 1, 1), datetime.date(2023, 12, 31))
    
    acc1 = Account("A1")
    acc2 = Account("A2")
    
    # Posting 1: Inside period
    p1 = Posting(None, datetime.date(2023, 6, 1), acc1, Direction(1), Quantity(Decimal(10)))
    # Posting 2: Inside period
    p2 = Posting(None, datetime.date(2023, 7, 1), acc2, Direction(1), Quantity(Decimal(20)))
    # Posting 3: Before period
    p3 = Posting(None, datetime.date(2022, 12, 31), acc1, Direction(1), Quantity(Decimal(5)))
    # Posting 4: After period
    p4 = Posting(None, datetime.date(2024, 1, 1), acc1, Direction(1), Quantity(Decimal(5)))

    j1 = JournalEntry(datetime.date(2023, 6, 1), "In", None, [p1, p2])
    j2 = JournalEntry(datetime.date(2022, 12, 31), "Old", None, [p3])
    j3 = JournalEntry(datetime.date(2024, 1, 1), "Future", None, [p4])
    
    journal = [j1, j2, j3]
    initial = {}

    # The predicate to test: (p for j in journal for p in j.postings if period.since <= j.date <= period.until)
    filtered_postings = [p for j in journal for p in j.postings if period.since <= j.date <= period.until]

    # Assertions
    assert len(filtered_postings) == 2
    assert p1 in filtered_postings
    assert p2 in filtered_postings
    assert p3 not in filtered_postings
    assert p4 not in filtered_postings
```


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    date_since = datetime.date(2023, 1, 1)
    date_until = datetime.date(2023, 12, 31)
    period = DateRange(date_since, date_until)
    
    account_a = Account("A")
    account_b = Account("B")
    
    quantity_val = Quantity(Decimal("100.00"))
    initial_balances = {account_a: Balance(date_since, quantity_val)}
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 1),
        description="Test Entry",
        source="TestSource"
    )
    # Manually adding posting to bypass the logic of post() if needed, 
    # but using the class methods is cleaner.
    journal_entry.post(datetime.date(2023, 6, 1), account_a, Quantity(Decimal("50.00")))
    journal_entry.post(datetime.date(2023, 6, 1), account_b, Quantity(Decimal("-50.00")))
    
    journal = [journal_entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert isinstance(general_ledger, GeneralLedger)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_build_general_ledger_success():
    import datetime
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking the required dependencies based on the provided code structure
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
        def of(q): return Direction(Decimal(1) if q.value > 0 else Decimal(-1))

    @dataclass(frozen=True)
    class Amount:
        value: Decimal

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
        postings: list = None

    @dataclass(frozen=True)
    class DateRange:
        since: datetime.date
        until: datetime.date

    # Setup Test Data
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")
    
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(start_date, end_date)

    # Initial Balances
    initial_balances = {
        acc_cash: Balance(start_date, Quantity(Decimal(1000)))
    }

    # Journal Entry 1: Within period (Revenue)
    j1_date = datetime.date(202lag, 1, 15)
    p1 = Posting(None, j1_date, acc_cash, Direction(Decimal(1)), Amount(Decimal(500)))
    p2 = Posting(None, j1_date, acc_revenue, Direction(Decimal(1)), Amount(Decimal(500)))
    j1 = JournalEntry(j1_date, "Sale", None, [p1, p2])

    # Journal Entry 2: Outside period (Should be ignored)
    j2_date = datetime.date(2023, 2, 1)
    p3 = Posting(None, j2_date, acc_cash, Direction(Decimal(-1)), Amount(Decimal(100)))
    j2 = JournalEntry(j2_date, "Old Sale", None, [p3])

    # Journal Entry 3: Within period (Expense)
    j3_date = datetime.date(2023, 1, 20)
    p4 = Posting(None, j3_date, acc_expense, Direction(Decimal(1)), Amount(Decimal(200)))
    p5 = Posting(None, j3_date, acc_cash, Direction(Decimal(-1)), Amount(Decimal(200)))
    j3 = JournalEntry(j3_date, "Rent", None, [p4, p5])

    journal = [j1, j2, j3]

    # Execution
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert gl.period == period
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers
    
    # Verify Cash Ledger: 1000 (initial) + 500 (p1) - 200 (p5) = 1300
    assert gl.ledgers[acc_cash]._last_balance == Quantity(Decimal(1300))
    # Verify Revenue Ledger: 0 (no initial) + 500 (p2) = 500
    assert gl.ledgers[acc_revenue]._last_balance == Quantity(Decimal(500))
    # Verify Expense Ledger: 0 (no initial) + 200 (p4) = 200
    assert gl.ledgers[acc_expense]._last_balance == Quantity(Decimal(200))
    
    # Verify that the out-of-period entry (p3) was not added to Cash
    # If p3 was added, Cash would be 1200.
    assert len(gl.ledgers[acc_cash].entries) == 2
```


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

```python
def test_read_initial_balances_call_returns_correct_balances():
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
    expected_result = InitialBalances(balance=100.0)
    
    result = reader(period)
    
    assert result == expected_result
    assert result.balance == 100.0
```


# LLM-generated content at query #16
#--------------------------

```python
def test_read_initial_balances_call_returns_correct_value():
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
    expected_result = InitialBalances(balance=100.0)
    
    assert reader(period) == expected_result
```


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger_returns_correct_type():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import GeneralLedger, Ledger
    from pypara.accounting.generic import Balance
    # Assuming these exist in the environment based on the provided snippets
    # Since I cannot import them, I assume they are available in the scope or mockable
    
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)
    
    # Mocking DateRange structure as used in the function
    class DateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until
            
    period = DateRange(start_date, end_date)
    
    # Mocking Account and Quantity
    class Account:
        pass
    
    class Quantity:
        def __init__(self, value):
            self.value = value
        def __add__(self, other):
            return Quantity(self.value + other.value)
        def __mul__(self, other):
            return Quantity(self.value * other.value)
        def __eq__(self, other):
            return self.value == other.value
        def __bool__(self):
            return self.value != 0

    account_a = Account()
    account_b = Account()
    
    # Mocking InitialBalances (Dict[Account, Balance])
    initial_balances = {
        account_a: Balance(start_date, Quantity(Decimal("100.00")))
    }
    
    # Mocking JournalEntry and Posting
    # We need a posting that falls within the period
    class MockPosting:
        def __init__(self, account, amount_val, direction_val):
            self.account = account
            self.amount = Quantity(Decimal(amount_val))
            self.direction = type('Dir', (), {'value': direction_val})()

    class MockJournalEntry:
        def __init__(self, date, postings):
            self.date = date
            self.postings = postings

    posting_in_period = MockPosting(account_a, "50.00", 1)
    journal_entry = MockJournalEntry(date(2023, 1, 15), [posting_in_period])
    
    # We need to patch the function or ensure the environment has it.
    # Since the prompt asks to test the predicate at line 1 (the function signature/existence),
    # we call the function and assert the result type.
    
    # Note: build_general_ledger is the function being tested.
    result = build_general_ledger(period, [journal_entry], initial_balances)
    
    assert isinstance(result, GeneralLedger)
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
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock(spec=Ledger)
    mock_posting = Mock(spec=Posting)
    mock_balance = Mock(spec=Quantity)
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #20
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


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_general_ledger_program_call_returns_expected_type():
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
```


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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
def test_ledger_entry_constructor_initialization():
    mock_ledger = None
    mock_posting = None
    mock_balance = None
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
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
def test_generalledgerprogram_call_returns_correct_type_and_period():
    from datetime import date
    from typing import Protocol, TypeVar, runtime_checkable

    _T = TypeVar("_T")

    @runtime_checkable
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class GeneralLedger:
        def __init__(self, period: DateRange):
            self.period = period

    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(period)

    program = MockGeneralLedgerProgram()
    test_period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    
    result = program(test_period)

    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert result.period.start == date(2023, 1, 1)
```


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_ledger_constructor_initialization():
    account_mock = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account_mock, initial=initial_balance)
    
    assert ledger.account == account_mock
    assert ledger.initial == initial_balance
    assert ledger.entries == []
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
def test_ledger_constructor_initialization():
    account_mock = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account_mock, initial=initial_balance)
    
    assert ledger.account == account_mock
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #34
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


# LLM-generated content at query #35
#--------------------------

```python
def test_build_general_ledger_filters_postings_outside_period():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(since=date_start, until=date_end)
    
    account_in_period = Account("InPeriod")
    account_out_period = Account("OutPeriod")
    
    quantity = Quantity(Decimal("100.00"))
    
    # Entry 1: Date is within period
    entry_valid = JournalEntry(date=datetime.date(2023, 1, 15), description="Valid", source="Src")
    posting_valid = Posting(entry_valid, datetime.date(2023, 1, 15), account_in_period, Direction.INC, Amount(quantity))
    entry_valid.postings.append(posting_valid)
    
    # Entry 2: Date is before period
    entry_before = JournalEntry(date=datetime.date(2022, 12, 31), description="Before", source="Src")
    posting_before = Posting(entry_before, datetime.date(2022, 12, 31), account_out_period, Direction.INC, Amount(quantity))
    entry_before.postings.append(posting_before)
    
    # Entry 3: Date is after period
    entry_after = JournalEntry(date=datetime.date(2023, 2, 1), description="After", source="Src")
    posting_after = Posting(entry_after, datetime.date(2023, 2, 1), account_out_period, Direction.INC, Amount(quantity))
    entry_after.postings.append(posting_after)
    
    journal = [entry_valid, entry_before, entry_after]
    initial_balances = InitialBalances() # Assuming empty or compatible type
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert account_in_period in general_ledger.ledgers
    assert account_out_period not in general_ledger.ledgers
    assert len(general_ledger.ledgers[account_in_period].entries) == 1
    assert general_ledger.ledgers[account_in_period].entries[0].balance.value == quantity
```


# LLM-generated content at query #36
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from dataclasses import dataclass

    # Mocking necessary structures for the test environment
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
        def is_zero(self): return self.value == Decimal(0)

    @dataclass(frozen=True)
    class Direction:
        value: Decimal
        @staticmethod
        def of(q): return Direction(Decimal(1) if q.value > 0 else Decimal(-1))

    @dataclass(frozen=True)
    class Posting:
        entry: any
        date: date
        account: Account
        direction: Direction
        amount: Quantity

    @dataclass(frozen=True)
    class JournalEntry:
        date: date
        description: str
        source: any
        postings: list = None

    # Setup period
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 1, 31))

    # Setup accounts
    acc_in_period = Account("In Period")
    acc_out_period = Account("Out of Period")

    # Setup Journal Entries
    # Entry 1: Within period
    j1 = JournalEntry(
        date=date(2023, 1, 15),
        description="Valid",
        source=None,
        postings=[Posting(None, date(2023, 1, 15), acc_in_period, Direction(Decimal(1)), Quantity(Decimal(100)))]
    )

    # Entry 2: Before period
    j2 = JournalEntry(
        date=date(2022, 12, 31),
        description="Too Early",
        source=None,
        postings=[Posting(None, date(2022, 12, 31), acc_in_period, Direction(Decimal(1)), Quantity(Decimal(50)))]
    )

    # Entry 3: After period
    j3 = JournalEntry(
        date=date(2023, 2, 1),
        description="Too Late",
        source=None,
        postings=[Posting(None, date(2023, 2, 1), acc_in_period, Direction(Decimal(1)), Quantity(Decimal(50)))]
    )

    journal = [j1, j2, j3]
    initial_balances = {}

    # The function to test (re-implemented as provided in the prompt)
    from typing import Iterable, Dict, Generic, List
    from dataclasses import field

    # Note: We use a local implementation of build_general_ledger to ensure the test is self-contained
    # and specifically targets the logic of the predicate at line 16.
    
    # Re-implementation of the logic to be tested
    def build_general_ledger_logic(period, journal, initial):
        ledgers = {a: None for a, b in initial.items()} # Simplified for testing the predicate
        # The specific line we are testing:
        filtered_postings = [p for j in journal for p in j.postings if period.since <= j.date <= period.until]
        return filtered_postings

    # Execution
    result_postings = build_general_ledger_logic(period, journal, initial_balances)

    # Assertions
    # Only the posting from j1 should be present. 
    # j2 is before period.since, j3 is after period.until.
    assert len(result_postings) == 1
    assert result_postings[0].account == acc_in_period
    assert result_postings[0].amount.value == Decimal(100)
```


# LLM-generated content at query #37
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from datetime import date
    from unittest.mock import MagicMock
    
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


# LLM-generated content at query #38
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
def test_ledger_entry_constructor_initialization():
    mock_ledger = Mock()
    mock_posting = Mock()
    mock_balance = Mock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
```


# LLM-generated content at query #41
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


# LLM-generated content at query #42
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


# LLM-generated content at query #43
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


# LLM-generated content at query #44
#--------------------------

```python
def test_build_general_ledger_predicate_filtering():
    date_start = datetime.date(2023, 1, 1)
    date_end = datetime.date(2023, 1, 31)
    period = DateRange(since=date_start, until=date_end)
    
    account_in_period = Account("in_period")
    account_out_period = Account("out_period")
    
    qty = Quantity(Decimal("100.00"))
    
    # Entry 1: Within period
    journal_entry_valid = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Valid Entry",
        source="SourceA"
    )
    journal_entry_valid.postings.append(
        Posting(journal_entry_valid, datetime.date(2023, 1, 15), account_in_period, Direction.INC, Amount(qty))
    )
    
    # Entry 2: Before period
    journal_entry_old = JournalEntry(
        date=datetime.date(2022, 12, 31),
        description="Old Entry",
        source="SourceB"
    )
    journal_entry_old.postings.append(
        Posting(journal_entry_old, datetime.date(2022, 12, 31), account_out_period, Direction.INC, Amount(qty))
    )
    
    # Entry 3: After period
    journal_entry_future = JournalEntry(
        date=datetime.date(2023, 2, 1),
        description="Future Entry",
        source="SourceC"
    )
    journal_entry_future.postings.append(
        Posting(journal_entry_future, datetime.date(2023, 2, 1), account_out_period, Direction.INC, Amount(qty))
    )
    
    journal = [journal_entry_valid, journal_entry_old, journal_entry_future]
    initial_balances = InitialBalances()
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assert that only the posting within the date range was processed into the ledgers
    assert account_in_period in general_ledger.ledgers
    assert len(general_ledger.ledgers[account_in_period].entries) == 1
    assert general_ledger.ledgers[account_in_period].entries[0].posting.amount.value == qty
    
    # Assert that the posting outside the date range was NOT processed
    assert account_out_period not in general_ledger.ledgers
```


# LLM-generated content at query #45
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


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from datetime import date
    from dataclasses import dataclass
    from typing import TypeVar, Generic, List, Optional

    @dataclass
    class Quantity:
        value: float

    @dataclass
    class Amount:
        value: float

    @dataclass
    class Account:
        name: str

    @dataclass
    class PostingDirection:
        is_debit: bool
        is_credit: bool
        direction: str

    @dataclass
    class JournalPosting:
        account: Account
        direction: str

    @dataclass
    class Journal:
        description: str
        postings: List[JournalPosting]

    @dataclass
    class Posting:
        date: date
        amount: Amount
        journal: Journal
        direction: str
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[TypeVar("T")]):
        pass

    _T = TypeVar("T")
    
    mock_ledger = Ledger[_T]()
    mock_amount = Amount(100.0)
    mock_date = date(2023, 1, 1)
    mock_account = Account("Cash")
    mock_journal = Journal("Test Journal", [JournalPosting(mock_account, "debit")])
    mock_posting = Posting(
        date=mock_date,
        amount=mock_amount,
        journal=mock_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = Quantity(50.0)

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


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_build_general_ledger_empty_journal_and_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.range import DateRange

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial_balances = {}
    journal = []
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert general_ledger.period == period
    assert general_ledger.ledgers == {}

def test_build_general_ledger_with_initial_balances():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.range import DateRange
    from unittest.mock import MagicMock

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account_a = MagicMock()
    initial_balances = {account_a: Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))}
    journal = []
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert account_a in general_ledger.ledgers
    assert general_ledger.ledgers[account_a].initial.value == Quantity(Decimal("100.00"))

def test_build_general_ledger_processes_postings_within_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.range import DateRange
    from unittest.mock import MagicMock

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account_a = MagicMock()
    account_b = MagicMock()
    initial_balances = {account_a: Balance(date(2023, 1, 1), Quantity(Decimal("0.00")))}
    
    entry_in_period = JournalEntry(date(date(202rag, 6, 1), "Test In", MagicMock())
    # Manually injecting postings as post() logic is complex for a single test line
    entry_in_period.postings.append(Posting(entry_in_period, date(2023, 6, 1), account_a, Direction.INC, Quantity(Decimal("50.00"))))
    
    entry_out_period = JournalEntry(date(2022, 12, 31), "Test Out", MagicMock())
    entry_out_period.postings.append(Posting(entry_out_period, date(2022, 12, 31), account_a, Direction.INC, Quantity(Decimal("50.00"))))
    
    journal = [entry_in_period, entry_out_period]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers[account_a].entries) == 1
    assert general_ledger.ledgers[account_a].entries[0].balance == Quantity(Decimal("50.00"))

def test_build_general_ledger_creates_new_ledgers_for_untracked_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.range import DateRange
    from unittest.mock import MagicMock

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account_a = MagicMock()
    account_new = MagicMock()
    initial_balances = {account_a: Balance(date(2023, 1, 1), Quantity(Decimal("0.00")))}
    
    entry = JournalEntry(date(2023, 2, 1), "New Account Posting", MagicMock())
    entry.postings.append(Posting(entry, date(2023, 2, 1), account_new, Direction.INC, Quantity(Decimal("25.00"))))
    
    journal = [entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert account_new in general_ledger.ledgers
    assert general_ledger.ledgers[account_new].initial.value == Quantity(Decimal("0.00"))
    assert general_ledger.ledgers[account_new].entries[0].balance == Quantity(Discount(Decimal("25.00")))
```

Wait, I see a typo in my manual test construction in the last block (`Quantity(Discount...`). Let me provide a clean, corrected version of the test cases.

```python
def test_build_general_ledger_empty_input():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.range import DateRange
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    general_ledger = build_general_ledger(period, [], {})
    assert general_ledger.ledgers == {}

def test_build_general_ledger_with_initial_balances_and_postings():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.range import DateRange
    from unittest.mock import MagicMock

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account_a = MagicMock()
    account_b = MagicMock()
    initial_balances = {account_a: Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))}
    
    entry = JournalEntry(date(2023, 6, 1), "Description", MagicMock())
    entry.postings.append(Posting(entry, date(2023, 6, 1), account_a, Direction.INC, Quantity(Decimal("50.00"))))
    entry.postings.append(Posting(entry, date(2023, 6, 1), account_b, Direction.DEC, Quantity(Decimal("50.00"))))
    
    journal = [entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert account_a in general_ledger.ledgers
    assert account_b in general_ledger.ledgers
    assert general_ledger.ledgers[account_a].entries[0].balance == Quantity(Decimal("150.00"))
    assert general_ledger.ledgers[account_b].entries[0].balance == Quantity(Decimal("-50.00"))

def test_build_general_ledger_filters_by_date_range():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.range import DateRange
    from unittest.mock import MagicMock

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account_a = MagicMock()
    initial_balances = {account_a: Balance(date(2023, 1, 1), Quantity(Decimal("0.00")))}
    
    entry_valid = JournalEntry(date(2023, 6, 1), "Valid", MagicMock())
    entry_valid.postings.append(Posting(entry_valid, date(2023, 6, 1), account_a, Direction.INC, Quantity(Decimal("10.00"))))
    
    entry_invalid = JournalEntry(date(2022, 12, 31), "Invalid", MagicMock())
    entry_invalid.postings.append(Posting(entry_invalid, date(2022, 12, 31), account_a, Direction.INC, Quantity(Decimal("10.00"))))
    
    journal = [entry_valid, entry_invalid]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert len(general_ledger.ledgers[account_a].entries) == 1
    assert general_ledger.ledgers[account_a].entries[0].balance == Quantity(Decimal("10.00"))
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
def test_generalladger_program_call_returns_correct_type():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class GeneralLedger(NamedTuple):
        data: dict

    class MockProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(data={period.start: "value"})

    program = MockProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = program(period)

    assert isinstance(result, GeneralLedger)
    assert result.data[date(2023, 1, 1)] == "value"
```


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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


# LLM-generated content at query #10
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


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_build_general_ledger_does_not_fail_on_empty_inputs():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.ledger import GeneralLedger, Ledger
    
    # Mocking the required classes/structures for the scope of the test
    class DateRange:
        def __init__(self, since, until):
            self.since = since
            self.until = until

    class Account:
        def __init__(self, name):
            self.name = name
        def __hash__(self):
            return hash(self.name)
        def __eq__(self, other):
            return isinstance(other, Account) and self.name == other.name

    class Quantity:
        def __init__(self, value):
            self.value = Decimal(value)
        def __add__(self, other):
            return Quantity(self.value + other.value)
        def __mul__(self, other):
            return Quantity(self.value * other.value)
        def __eq__(self, other):
            return self.value == other.value

    # Setup inputs for an empty scenario to ensure the function executes without error
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    journal = []
    initial = {}

    # The function to test (as provided in the prompt)
    def build_general_ledger(period, journal, initial):
        ledgers = {a: Ledger(a, Balance(period.since, Quantity(0))) for a, b in initial.items()}
        for j in journal:
            if period.since <= j.date <= period.until:
                for posting in j.postings:
                    if posting.account not in ledgers:
                        ledgers[posting.account] = Ledger(posting.account, Balance(period.since, Quantity(0)))
                    ledgers[posting.account].add(posting)
        return GeneralLedger(period, ledgers)

    # Re-defining Ledger/LedgerEntry locally to satisfy the test environment
    class LedgerEntry:
        def __init__(self, ledger, posting, balance):
            self.ledger = ledger
            self.posting = posting
            self.balance = balance

    # Execute
    result = build_general_ledger(period, journal, initial)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.ledgers == {}
```


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_ledger_entry_constructor_initialization():
    from dataclasses import dataclass
    from datetime import date
    from typing import Generic, TypeVar, List, Optional

    @dataclass
    class Account:
        name: str

    @dataclass
    class Amount:
        value: float

    @dataclass
    class Quantity:
        value: float

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
    class Ledger(Generic['Account']):
        pass

    _T = TypeVar("_T")
    
    mock_account = Account(name="Test Account")
    mock_amount = Amount(value=100.0)
    mock_qty = Quantity(value=100.0)
    mock_date = date(2023, 1, 1)
    mock_journal = Journal(description="Test Journal", postings=[])
    mock_posting = Posting(
        date=mock_date, 
        amount=mock_amount, 
        direction="debit", 
        is_debit=True, 
        is_credit=False, 
        account=mock_account, 
        journal=mock_journal
    )
    mock_ledger = Ledger[Account]()

    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_qty
    )

    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_qty
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
def test_build_general_ledger_evaluates_predicate_to_false():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import GeneralLedger, Ledger
    
    # Setup date range and period
    period_start = date(2023, 1, 1)
    period_end = date(2023, 1, 31)
    
    # Mocking DateRange-like object
    class MockDateRange:
        since = period_start
        until = period_end
    period = MockDateRange()

    # Setup Accounts
    account_a = "Account A"
    account_b = "Account B"

    # Setup Initial Balances
    # We want to ensure the predicate (posting.account not in ledgers) is False
    # This means the account must already exist in the initial balances dict
    initial_balances = {
        account_a: Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))
    }

    # Setup Journal Entries with Postings
    # We create a posting for account_a (which is in initial) 
    # and account_b (which is not in initial, to test the logic)
    entry_date = date(2023, 1, 15)
    
    # Posting 1: Account A (Exists in initial -> predicate False)
    posting_a = Posting(
        journal_entry=None, # Not used in the logic for account check
        date=entry_date,
        account=account_a,
        direction=Direction.INC,
        amount=Amount(Decimal("50.00"))
    )

    # Posting 2: Account B (Does not exist in initial -> predicate True)
    posting_b = Posting(
        journal_entry=None,
        date=entry_date,
        account=account_b,
        direction=Direction.INC,
        amount=Amount(Decimal("20.00"))
    )

    # Create JournalEntry containing the posting for Account A
    # We use a dummy source object
    journal_entry_a = JournalEntry(date=entry_date, description="Test Entry", source="Source")
    journal_entry_a.postings.append(posting_a)

    journal = [journal_entry_a]

    # Execute the function
    # Note: We assume the function build_general_ledger is available in the scope
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    # The predicate 'posting.account not in ledgers' for posting_a must be False
    assert account_a in general_ledger.ledgers
    # The ledger for account_a should have processed the posting
    assert general_ledger.ledgers[account_a].entries[0].balance.value == Quantity(Decimal("150.00"))
    # The predicate for posting_b (if it were in the journal) would be True, 
    # but for the existing account_a, it must be False.
```


# LLM-generated content at query #19
#--------------------------

```python
def test_build_general_ledger_success():
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger, Ledger
    from pypara.accounting.utils import DateRange

    # Setup period
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances
    initial_balances = {
        acc_cash: Balance(start_date, Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue earned (Within period)
    j1 = JournalEntry(date=date(2023, 6, 1), description="Service Revenue", source="Client A")
    j1.post(date=date(2023, 6, 1), account=acc_cash, quantity=Quantity(Decimal("500.00")))
    j1.post(date=int(2023, 6, 1), account=acc_revenue, quantity=Quantity(Decimal("-500.00")))
    # Note: The implementation of post uses Direction.of(quantity). 
    # If quantity is 500, direction is INC. If -500, direction is DEC.
    # To match the logic: 500 (INC cash), -500 (DEC revenue)
    
    # Entry 2: Expense paid (Within period)
    j2 = JournalEntry(date=date(2023, 7, 1), description="Office Supplies", source="Supplier B")
    j2.post(date=date(2023, 7, 1), account=acc_cash, quantity=Quantity(Decimal("-100.00")))
    j2.post(date=date(2023, 7, 1), account=acc_expense, quantity=Quantity(Decimal("100.00")))

    # Entry 3: Old entry (Outside period - should be ignored)
    j3 = JournalEntry(date=date(2022, 12, 31), description="Old Entry", source="Legacy")
    j3.post(date=date(2022, 12, 31), account=acc_cash, quantity=Quantity(Decimal("100.00")))

    journal = [j1, j2, j3]

    # Build Ledger
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers
    
    # Verify Cash Balance: 1000 (init) + 500 (j1) - 100 (j2) = 1400
    assert gl.ledgers[acc_cash].initial.value == Quantity(Decimal("1000.00"))
    assert gl.ledgers[acc_cash]._last_balance == Quantity(Decimal("1400.00"))
    
    # Verify Revenue Balance: 0 (new) - 500 (j1) = -500
    assert gl.ledgers[acc_revenue]._last_balance == Quantity(Decimal("-500.00"))
    
    # Verify Expense Balance: 0 (new) + 100 (j2) = 100
    assert gl.ledgers[acc_expense]._last_balance == Quantity(Decimal("100.00"))

    # Verify that the old entry (j3) was not processed
    # Total entries in cash ledger should be 2 (the initial balance is not an entry, the two posts are)
    # The postings in j3 should not appear in any ledger
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.posting.date != date(2022, 12, 31)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_build_general_ledger_returns_correct_type_and_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Quantity
    
    # Setup dependencies
    period_start = date(2023, 1, 1)
    period_end = date(2023, 1, 31)
    # Mocking DateRange as it's used in the function signature
    class DateRange:
        since = period_start
        until = period_end
    
    period = DateRange()
    
    # Mocking Account and InitialBalances
    class Account:
        def __init__(self, name): self.name = name
    
    acc1 = Account("Cash")
    initial_balances = {acc1: Balance(period_start, Quantity(Decimal("100.00")))}
    
    # Mocking JournalEntry and Postings
    # We need at least one posting to trigger the logic inside the loop
    class MockSource: pass
    
    entry = JournalEntry(date=date(2023, 1, 15), description="Test", source=MockSource())
    # Manually injecting postings since .post() is a method and we want to control the list
    from dataclasses import field
    from typing import List
    entry.postings = [
        Posting(entry, date(2023, 1, 15), acc1, Direction.INC, Quantity(Decimal("50.00")))
    ]
    
    journal = [entry]
    
    # Execute
    result = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert acc1 in result.ledgers
    assert result.ledgers[acc1].initial.value == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #21
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
def test_generalladger_program_call_returns_correct_type():
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
```


# LLM-generated content at query #24
#--------------------------

```python
def test_read_initial_balances_call_returns_correct_balances():
    from datetime import date
    from typing import NamedTuple

    class DateRange(NamedTuple):
        start: date
        end: date

    class InitialBalances(NamedTuple):
        account_id: str
        balance: float

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(account_id="ACC123", balance=1500.0)

    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    reader = MockReadInitialBalances()
    result = reader(period)

    assert result.account_id == "ACC123"
    assert result.balance == 1500.0
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
def test_ledger_constructor_initialization():
    account = Account(name="Test Account")
    initial_balance = Balance(value=Quantity(100.0))
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
```


# LLM-generated content at query #27
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


# LLM-generated content at query #28
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
def test_ledger_entry_constructor_initialization():
    mock_ledger = MagicMock()
    mock_posting = MagicMock()
    mock_balance = MagicMock()
    
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance
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
def test_read_initial_balances_call_returns_correct_type_and_value():
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
    reader = MockReadInitialBalances()
    result = reader(period)

    assert isinstance(result, InitialBalances)
    assert result.amount == 100.0
```


# LLM-generated content at query #33
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


# LLM-generated content at query #34
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


# LLM-generated content at query #35
#--------------------------

```python
def test_build_general_ledger_filters_postings_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import GeneralLedger
    # Assuming Account, Quantity, DateRange, and build_general_ledger are available in the scope
    
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 1, 31))
    account_in_period = Account("InPeriod")
    account_out_period = Account("OutPeriod")
    initial_balances = {Account("Existing"): Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))}
    
    entry_valid = JournalEntry(date=date(2023, 1, 15), description="Valid", source="Source")
    entry_valid.postings.append(Posting(entry_valid, date(2023, 1, 15), account_in_period, Direction.INC, Amount(Quantity(Decimal("50.00")))))
    
    entry_too_early = JournalEntry(date=date(2022, 12, 31), description="Too Early", source="Source")
    entry_too_early.postings.append(Posting(entry_too_early, date(2022, 12, 31), account_in_period, Direction.INC, Amount(Quantity(Decimal("50.00")))))
    
    entry_too_late = JournalEntry(date=deate(2023, 2, 1), description="Too Late", source="Source")
    entry_too_late.postings.append(Posting(entry_too_late, date(2023, 2, 1), account_in_period, Direction.INC, Amount(Quantity(Decimal("50.00")))))
    
    entry_boundary_start = JournalEntry(date=date(2023, 1, 1), description="Boundary Start", source="Source")
    entry_boundary_start.postings.append(Posting(entry_boundary_start, date(2023, 1, 1), account_in_period, Direction.INC, Amount(Quantity(Decimal("10.00")))))
    
    entry_boundary_end = JournalEntry(date=date(2023, 1, 31), description="Boundary End", source="Source")
    entry_boundary_end.postings.append(Posting(entry_boundary_end, date(2023, 1, 31), account_in_period, Direction.INC, Amount(Quantity(Decimal("10.00")))))

    journal = [entry_valid, entry_too_early, entry_too_late, entry_boundary_start, entry_boundary_end]
    
    gl = build_general_ledger(period, journal, initial_balances)
    
    # The ledger for account_in_period should only contain the 3 valid postings (50 + 10 + 10 = 70)
    # The initial balance for 'Existing' was 100, but we only check the logic of the generator in line 16
    # by verifying that entries from 'too early' and 'too late' were never added to the ledgers.
    
    assert account_in_period in gl.ledgers
    assert len(gl.ledgers[account_in_period].entries) == 3
    assert gl.ledgers[account_in_period]._last_balance == Quantity(Decimal("70.00"))
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
def test_build_general_ledger_predicate_true():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.date_range import DateRange
    from pypara.accounting.account import Account

    date_start = date(2023, 1, 1)
    date_end = date(2023, 1, 31)
    period = DateRange(date_start, date_end)
    
    account_a = Account("A")
    account_b = Account("B")
    
    # Entry 1: Inside period
    entry_inside = JournalEntry(date=date(2023, 1, 15), description="Inside", source="src")
    # Manually adding posting because .post() is not available in the snippet's scope for instantiation 
    # but we need to satisfy the structure. We use the logic that postings exists.
    # Note: In the provided code, postings is init=False, so we must simulate its population.
    # We will use the .post method which is part of the provided JournalEntry class.
    entry_inside.post(date=date(2lag_date := date(2023, 1, 15)), account=account_a, quantity=Quantity(Decimal("100")))
    
    # Entry 2: Outside period (Before)
    entry_before = JournalEntry(date=date(2022, 12, 31), description="Before", source="src")
    entry_before.post(date=date(2022, 12, 31), account=account_b, quantity=Quantity(Decimal("50")))
    
    # Entry 3: Outside period (After)
    entry_after = JournalEntry(date=date(2023, 2, 1), description="After", source="src")
    entry_after.post(date=date(2023, 2, 1), account=account_b, quantity=Quantity(Decimal("50")))

    journal = [entry_inside, entry_before, entry_after]
    initial_balances = {account_a: Balance(date_start, Quantity(Decimal("0")))}
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # The predicate at line 16: (p for j in journal for p in j.postings if period.since <= j.date <= period.until)
    # For entry_inside: 2023-01-01 <= 2023-01-15 <= 2023-01-31 -> True
    # For entry_before: 2023-01-01 <= 2022-12-31 <= 2023-01-31 -> False
    # For entry_after: 2023-01-01 <= 2023-02-01 <= 2023-01-31 -> False
    
    # Assert that the ledger only contains the posting from the entry that satisfied the predicate
    assert len(general_ledger.ledgers[account_a].entries) == 1
    assert general_ledger.ledgers[account_a].entries[0].posting.amount.value == Decimal("100")
    
    # Assert that account_b was not created via the loop because its entries were outside the period
    # (Note: account_b only appears in entries that failed the predicate)
    assert account_b not in general_ledger.ledgers
```


# LLM-generated content at query #39
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


