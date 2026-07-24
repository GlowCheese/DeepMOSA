####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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
        pass
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_posting = MockPosting(
        date=date(2024, 1, 15),
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_posting._journal = MockJournal(
        description="Test Transaction",
        postings=[test_posting]
    )
    test_balance = MockQuantity(value=1000.0)
    test_ledger = MockLedger()
    
    # Test constructor
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_balance
    )
    
    # Assert constructor properly initializes all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_entries_ignored():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=Quantity(1000))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #3
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


# LLM-generated content at query #4
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
    posting = Posting(
        date=date(2023, 1, 15),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        direction="debit",
        journal=Journal("Test transaction", []),
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.date == date(2023, 1, 15)
    assert entry.description == "Test transaction"
    assert entry.amount == Amount(100.0, "USD")
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #5
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
        account: MockAccount
        amount: MockAmount
        direction: str
        journal: MockJournal
        is_debit: bool
        is_credit: bool

    @dataclass
    class MockLedger:
        name: str

    ledger = MockLedger(name="Test Ledger")
    posting = MockPosting(
        date=date(2023, 1, 1),
        account=MockAccount(name="Cash"),
        amount=MockAmount(value=100.0, currency="USD"),
        direction="debit",
        journal=MockJournal(description="Test Entry", postings=[]),
        is_debit=True,
        is_credit=False
    )
    balance = MockQuantity(value=1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


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
    class MockLedger:
        name: str = "Test Ledger"
    
    @dataclass
    class MockQuantity:
        value: float
        currency: str
    
    # Create test instances
    test_ledger = MockLedger(name="Test Ledger")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_account = MockAccount(name="Cash")
    test_posting = MockPosting(
        date=date(2023, 1, 15),
        amount=test_amount,
        account=test_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = MockQuantity(value=500.0, currency="USD")
    
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


# LLM-generated content at query #7
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
    mock_journal = MockJournal(description="Test Transaction", postings=[])
    mock_amount = MockAmount(value=100.0)
    mock_posting = MockPosting(
        date=date(2024, 1, 15),
        journal=mock_journal,
        amount=mock_amount,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_ledger = MockLedger()
    mock_balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor properly assigned all fields
    assert ledger_entry.ledger is mock_ledger
    assert ledger_entry.posting is mock_posting
    assert ledger_entry.balance is mock_balance


# LLM-generated content at query #8
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
    test_journal = MockJournal(description="Test Transaction", postings=[])
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
    ledger_entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor sets attributes correctly
    assert ledger_entry.ledger is test_ledger
    assert ledger_entry.posting is test_posting
    assert ledger_entry.balance is test_quantity


# LLM-generated content at query #9
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
        def __call__(self, period: DateRange) -> GeneralLedger[int]:
            return GeneralLedger([1, 2, 3])
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert result.data == [1, 2, 3]
    assert date_range.start == date(2023, 1, 1)
    assert date_range.end == date(2023, 12, 31)


# LLM-generated content at query #10
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
    test_account = MockAccount(name="TestAccount")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_posting = MockPosting(
        date=test_date,
        account=test_account,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=MockJournal(description="Test Description", postings=[])
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    # Assert constructor properly assigns all fields
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance


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
            return self._journal
        
        def set_journal(self, journal):
            self._journal = journal
            return self
    
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
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_quantity = MockQuantity(value=100.0)
    mock_ledger = MockLedger()
    
    mock_posting = MockPosting(
        date=date(2024, 1, 1),
        amount=mock_amount,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_posting.set_journal(MockJournal(description="Test Entry", postings=[mock_posting]))
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assert constructor sets all attributes correctly
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_quantity


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
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(1000.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger is not None
    assert entry.posting is not None
    assert entry.balance is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_read_initial_balances_protocol_call():
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
    assert period.start == date(2023, 1, 1)
    assert period.end == date(2023, 12, 31)


# LLM-generated content at query #14
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="ASSET")
    initial_balance = Balance(value=Quantity(100))
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []
    assert isinstance(ledger.entries, list)


# LLM-generated content at query #15
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar
    
    _T = TypeVar('_T')
    
    # Create mock objects for dependencies
    mock_account = type('Account', (), {})()
    mock_posting = type('Posting', (), {
        'date': date(2024, 1, 15),
        'journal': type('Journal', (), {
            'description': 'Test transaction',
            'postings': []
        })(),
        'amount': type('Amount', (), {'value': 100})(),
        'direction': 'debit',
        'is_debit': True,
        'is_credit': False,
        'account': mock_account
    })()
    mock_ledger = type('Ledger', (), {})()
    mock_balance = type('Quantity', (), {'value': 500})()
    
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

    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(account=account, amount=amount, date=date(2023, 1, 1), journal=journal, direction="debit")
    ledger = Ledger()
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #17
#--------------------------

```python
def test_general_ledger_program_call():
    from datetime import date
    from typing import Any, Dict, List
    
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    
    class GeneralLedger:
        def __init__(self, entries: List[Dict[str, Any]]):
            self.entries = entries
    
    class TestGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger([{"date": period.start, "amount": 100}])
    
    program = TestGeneralLedgerProgram()
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert len(result.entries) == 1
    assert result.entries[0]["date"] == start_date
    assert result.entries[0]["amount"] == 100


# LLM-generated content at query #18
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


# LLM-generated content at query #19
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

        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"

        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class MockLedger:
        pass

    # Setup test data
    test_date = date(2024, 1, 15)
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_journal = MockJournal(description="Test Transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        account=test_account
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
    assert ledger_entry.date == test_date
    assert ledger_entry.description == "Test Transaction"
    assert ledger_entry.amount is test_amount
    assert ledger_entry.is_debit is True
    assert ledger_entry.is_credit is False
    assert ledger_entry.debit is test_amount
    assert ledger_entry.credit is None


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
    assert entry.ledger.name == "General Ledger"
    assert entry.posting.account.name == "Cash"
    assert entry.balance.value == 500.0


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
        journal: Journal
        amount: Amount
        account: Account
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class Ledger(Generic[_T]):
        pass
    
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2024, 1, 1),
        journal=journal,
        amount=amount,
        account=account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    ledger = Ledger()
    balance = Quantity(value=100.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


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
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2023, 1, 1), account=account, amount=amount, direction="debit", journal=journal)
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


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
            return GeneralLedger({"start": period.start, "end": period.end})
    
    program = ConcreteGeneralLedgerProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.data["start"] == date(2023, 1, 1)
    assert result.data["end"] == date(2023, 12, 31)


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
    posting = Posting(
        date=date(2024, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        account=Account(name="Cash"),
        journal=Journal(description="Test transaction", postings=[]),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=1000.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #25
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
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 1),
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger()
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_balance)
    
    # Assertions
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_balance
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance


# LLM-generated content at query #26
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
    account = Account(name="Test Account")
    amount = Amount(value=100.0, currency="USD")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(date=date(2024, 1, 1), journal=journal, amount=amount, account=account, direction="debit")
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #27
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.quantity import Quantity
    from pypara.core.amount import Amount
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Setup test data
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    # Create initial balances
    initial_balance1 = Balance(start_date, Quantity(Decimal("1000")))
    initial_balance2 = Balance(start_date, Quantity(Decimal("500")))
    initial = {account1: initial_balance1, account2: initial_balance2}
    
    # Create journal entries
    entry1 = JournalEntry(datetime.date(2024, 1, 15), "Test entry 1", "source1")
    entry1.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal("-100")))
    entry1.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal("100")))
    
    entry2 = JournalEntry(datetime.date(2024, 2, 20), "Test entry 2", "source2")
    entry2.post(datetime.date(2024, 2, 20), account1, Quantity(Decimal("50")))
    entry2.post(datetime.date(2024, 2, 20), account2, Quantity(Decimal("-50")))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial)
    
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
    assert ledger1.entries[0].balance == Quantity(Decimal("900"))
    assert ledger1.entries[1].balance == Quantity(Decimal("950"))
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balance2
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal("600"))
    assert ledger2.entries[1].balance == Quantity(Decimal("550"))


def test_build_general_ledger_with_new_account():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Setup test data
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    account3 = Account("3000", "Revenue")
    
    # Only account1 has initial balance
    initial = {account1: Balance(start_date, Quantity(Decimal("1000")))}
    
    # Create journal entries with account2 and account3 (not in initial)
    entry1 = JournalEntry(datetime.date(2024, 1, 15), "Test entry", "source1")
    entry1.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal("-100")))
    entry1.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal("100")))
    
    entry2 = JournalEntry(datetime.date(2024, 2, 20), "Test entry 2", "source2")
    entry2.post(datetime.date(2024, 2, 20), account3, Quantity(Decimal("50")))
    entry2.post(datetime.date(2024, 2, 20), account1, Quantity(Decimal("-50")))
    
    journal = [entry1, entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Assertions
    assert len(general_ledger.ledgers) == 3
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert account3 in general_ledger.ledgers
    
    # Check that account2 and account3 were created with zero balance
    assert general_ledger.ledgers[account2].initial.value == Quantity(Decimal("0"))
    assert general_ledger.ledgers[account3].initial.value == Quantity(Decimal("0"))
    
    # Check ledger entries
    assert len(general_ledger.ledgers[account2].entries) == 1
    assert len(general_ledger.ledgers[account3].entries) == 1


def test_build_general_ledger_filters_by_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.quantity import Quantity
    from pypara.core.account import Account
    from pypara.core.daterange import DateRange
    
    # Setup test data
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account("1000", "Cash")
    initial = {account1: Balance(start_date, Quantity(Decimal("1000")))}
    
    # Create entries both inside and outside the period
    entry_inside = JournalEntry(datetime.date(2024, 6, 15), "Inside period", "source1")
    entry_inside.post(datetime.date(2024, 6, 15), account1, Quantity(Decimal("-100")))
    
    entry_before = JournalEntry(datetime.date(2023, 12, 31), "Before period", "source2")
    entry_before.post(datetime.date(2023, 12, 31), account1, Quantity(Decimal("-50")))
    
    entry_after = JournalEntry(datetime.date(2025, 1, 1), "After period", "source3")
    entry_after.post(datetime.date(2025, 1, 1), account1, Quantity(Decimal("-25")))
    
    journal = [entry_before, entry_inside, entry_after]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial)
    
    # Only entry inside the period should be included
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].


# LLM-generated content at query #28
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
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=journal
    )
    ledger = Ledger(name="General Ledger")
    balance = Quantity(value=500.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #29
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
    class MockPosting:
        date: date
        amount: MockAmount
        journal: MockJournal
        direction: str
        account: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create instances
    mock_journal = MockJournal(description="Test Description", postings=[])
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        amount=mock_amount,
        journal=mock_journal,
        direction="debit",
        account="Test Account",
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
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_quantity = MockQuantity(value=100.0)
    test_date = date(2023, 1, 1)
    test_journal = MockJournal(description="Test Journal", postings=[])
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
    
    # Test constructor
    entry = LedgerEntry(ledger=test_ledger, posting=test_posting, balance=test_quantity)
    
    assert entry.ledger is test_ledger
    assert entry.posting is test_posting
    assert entry.balance is test_quantity


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.posting import Posting, Account, Amount
    from pypara.core.quantity import Quantity
    from pypara.core.range import DateRange
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account1: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(1000))),
        account2: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(500)))
    }
    
    # Create journal entries with postings
    entry1 = JournalEntry(datetime.date(2024, 1, 15), "Test entry 1", "source1")
    entry1.post(datetime.date(2024, 1, 15), account1, Quantity(Decimal(-100)))
    entry1.post(datetime.date(2024, 1, 15), account2, Quantity(Decimal(100)))
    
    entry2 = JournalEntry(datetime.date(2024, 2, 1), "Test entry 2", "source2")
    entry2.post(datetime.date(2024, 2, 1), account1, Quantity(Decimal(200)))
    entry2.post(datetime.date(2024, 2, 1), account2, Quantity(Decimal(-200)))
    
    journal = [entry1, entry2]
    
    # Call build_general_ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
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
    assert ledger1.entries[1].balance == Quantity(Decimal(1100))
    
    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal(500))
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal(600))
    assert ledger2.entries[1].balance == Quantity(Decimal(400))


def test_build_general_ledger_with_out_of_period_entries():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.posting import Account
    from pypara.core.quantity import Quantity
    from pypara.core.range import DateRange
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    account = Account("1000", "Cash")
    initial_balances = {
        account: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(1000)))
    }
    
    # Create entries outside the period
    entry_before = JournalEntry(datetime.date(2023, 12, 31), "Before period", "source1")
    entry_before.post(datetime.date(2023, 12, 31), account, Quantity(Decimal(-100)))
    
    entry_in_period = JournalEntry(datetime.date(2024, 6, 1), "In period", "source2")
    entry_in_period.post(datetime.date(2024, 6, 1), account, Quantity(Decimal(50)))
    
    entry_after = JournalEntry(datetime.date(2025, 1, 1), "After period", "source3")
    entry_after.post(datetime.date(2025, 1, 1), account, Quantity(Decimal(200)))
    
    journal = [entry_before, entry_in_period, entry_after]
    
    # Call build_general_ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Only the in-period entry should be included
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(1050))


def test_build_general_ledger_with_new_account():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.posting import Account
    from pypara.core.quantity import Quantity
    from pypara.core.range import DateRange
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    account_initial = Account("1000", "Cash")
    account_new = Account("2000", "Revenue")
    
    initial_balances = {
        account_initial: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(1000)))
    }
    
    # Create entry with a new account not in initial balances
    entry = JournalEntry(datetime.date(2024, 3, 1), "New account entry", "source")
    entry.post(datetime.date(2024, 3, 1), account_initial, Quantity(Decimal(-100)))
    entry.post(datetime.date(2024, 3, 1), account_new, Quantity(Decimal(100)))
    
    journal = [entry]
    
    # Call build_general_ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Both accounts should be in the ledgers
    assert len(general_ledger.ledgers) == 2
    assert account_initial in general_ledger.ledgers
    assert account_new in general_ledger.ledgers
    
    # Check the new account was initialized with zero balance
    assert general_ledger.ledgers[account_new].initial.value == Quantity(Decimal(0))
    assert general_ledger.ledgers[account_new].entries[0].balance == Quantity(Decimal(100))


def test_build_general_ledger_empty_journal():
    import datetime
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Balance
    from pypara.accounting.posting import Account
    from pypara.core.quantity import Quantity
    from pypara.core.range import DateRange
    
    # Setup test data
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    account = Account("1000", "Cash")
    initial_balances = {
        account: Balance(datetime.date(2024, 1, 1), Quantity(Decimal(1000)))
    }
    
    journal = []
    
    # Call build_general_ledger
    general_ledger = build_general_ledger(period,


# LLM-generated content at query #2
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import List
    
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
        postings: List
    
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
        date=date(2023, 1, 1),
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
    
    # Assert constructor properly assigns all fields
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_quantity


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
    journal = Journal(description="Test Description", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        account=account,
        amount=amount,
        direction="debit",
        journal=journal
    )
    balance = Quantity(value=100.0)
    ledger = Ledger()

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test Description"
    assert entry.amount is amount
    assert entry.is_debit is True
    assert entry.is_credit is False
    assert entry.debit is amount
    assert entry.credit is None


# LLM-generated content at query #4
#--------------------------

```python
def test_build_general_ledger_creates_ledgers_for_all_accounts():
    from datetime import date
    from decimal import Decimal
    from pypara.core import DateRange, Quantity, Account, Amount, Direction
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Posting
    from pypara.accounting.ledger import build_general_ledger, Ledger
    
    # Setup test data
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    # Create initial balances
    initial = {account1: Balance(period.since, Quantity(Decimal(1000)))}
    
    # Create a journal entry with postings
    entry = JournalEntry(date(2024, 6, 15), "Test entry", "source")
    entry.post(date(2024, 6, 15), account1, Quantity(Decimal(-100)))
    entry.post(date(2024, 6, 15), account2, Quantity(Decimal(100)))
    
    journal = [entry]
    
    # Build general ledger
    result = build_general_ledger(period, journal, initial)
    
    # Verify that all accounts are present in the ledgers
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers) == 2
    assert isinstance(result.ledgers[account1], Ledger)
    assert isinstance(result.ledgers[account2], Ledger)
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account2].account == account2
    assert result.period == period


# LLM-generated content at query #5
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import DateRange, Quantity, Account, Amount
    
    # Setup test data
    account = Account(name="Test Account", number="1000")
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    initial_balances = {account: Balance(date(2023, 1, 1), Quantity(Decimal(0)))}
    
    # Create journal entries with dates inside and outside the period
    entry_inside = JournalEntry(date=date(2023, 6, 15), description="Inside period", source="test1")
    entry_inside.post(date(2023, 6, 15), account, Quantity(Decimal(100)))
    
    entry_before = JournalEntry(date=date(2022, 12, 31), description="Before period", source="test2")
    entry_before.post(date(2022, 12, 31), account, Quantity(Decimal(50)))
    
    entry_after = JournalEntry(date=date(2024, 1, 1), description="After period", source="test3")
    entry_after.post(date(2024, 1, 1), account, Quantity(Decimal(75)))
    
    journal = [entry_before, entry_inside, entry_after]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # The predicate at line 16 should filter to only include postings where period.since <= j.date <= period.until
    # So only entry_inside should be included
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting.journal_entry.date == date(2023, 6, 15)


# LLM-generated content at query #6
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_date_range():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.generic import Quantity
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Posting, Amount
    from pypara.core.ranges import DateRange
    from pypara.accounting.accounts import Account
    
    # Create a date range for the accounting period
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create accounts
    account_a = Account("1000", "Cash", None)
    account_b = Account("2000", "Payable", None)
    
    # Create initial balances
    initial_balances = {
        account_a: Balance(period_start, Quantity(Decimal(1000))),
        account_b: Balance(period_start, Quantity(Decimal(0))),
    }
    
    # Create journal entries
    # Entry within period
    entry_within = JournalEntry(datetime.date(2023, 6, 15), "Within period", None)
    
    # Entry before period
    entry_before = JournalEntry(datetime.date(2022, 12, 31), "Before period", None)
    
    # Entry after period
    entry_after = JournalEntry(datetime.date(2024, 1, 1), "After period", None)
    
    # Add postings to entries
    entry_within.post(datetime.date(2023, 6, 15), account_a, Quantity(Decimal(-100)))
    entry_before.post(datetime.date(2022, 12, 31), account_a, Quantity(Decimal(-50)))
    entry_after.post(datetime.date(2024, 1, 1), account_a, Quantity(Decimal(-200)))
    
    # Build general ledger
    journal = [entry_within, entry_before, entry_after]
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Verify that only postings within the period are included
    # entry_within has 1 posting that should be included
    # entry_before has 1 posting that should NOT be included
    # entry_after has 1 posting that should NOT be included
    assert len(general_ledger.ledgers[account_a].entries) == 1
    assert general_ledger.ledgers[account_a].entries[0].posting.journal_entry.date == datetime.date(2023, 6, 15)


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
def test_build_general_ledger_posting_account_not_in_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction
    from pypara.accounting.generic import Balance, Account, Quantity
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core.types import DateRange
    
    # Create test data
    test_date = date(2023, 1, 15)
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Create a simple source object
    source = "TestSource"
    
    # Create a journal entry with a posting
    journal_entry = JournalEntry(test_date, "Test Entry", source)
    
    # Create an account that is NOT in initial balances
    test_account = Account(name="TestAccount", number="1000")
    
    # Add posting to journal entry
    journal_entry.post(test_date, test_account, Quantity(Decimal(100)))
    
    # Create initial balances with a different account
    initial_account = Account(name="InitialAccount", number="2000")
    initial_balances = {initial_account: Balance(period.since, Quantity(Decimal(50)))}
    
    # Build general ledger
    result = build_general_ledger(period, [journal_entry], initial_balances)
    
    # The predicate "posting.account not in ledgers" should evaluate to False
    # for the posting we added, because after the check on line 18,
    # the account will be added to ledgers on line 19 before adding the posting.
    # This means when we check if the account is in the result ledgers, it should be True.
    assert test_account in result.ledgers
    assert len(result.ledgers[test_account].entries) == 1


# LLM-generated content at query #9
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
    class MockQuantity:
        value: float
    
    @dataclass
    class MockLedger:
        pass
    
    # Create test data
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Journal", postings=[])
    test_posting = MockPosting(
        date=date(2024, 1, 1),
        account=test_account,
        amount=test_amount,
        journal=test_journal,
        direction="debit",
        is_debit=True,
        is_credit=False
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
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_balance
    assert entry.date == date(2024, 1, 1)
    assert entry.description == "Test Journal"
    assert entry.amount == test_amount
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #10
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
        def __call__(self, period: DateRange) -> GeneralLedger[dict]:
            return GeneralLedger([{"date": period.start, "amount": 100}])
    
    program = ConcreteGeneralLedgerProgram()
    date_range = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    result = program(date_range)
    
    assert isinstance(result, GeneralLedger)
    assert len(result.entries) == 1
    assert result.entries[0]["date"] == date(2024, 1, 1)
    assert result.entries[0]["amount"] == 100


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.generic import Balance
    from pypara.accounting.accounts import Account
    from pypara.core import Quantity
    from pypara.utils.daterange import DateRange
    
    # Setup test data
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    initial_balances = {
        account1: Balance(period_start, Quantity(Decimal("1000"))),
        account2: Balance(period_start, Quantity(Decimal("500")))
    }
    
    # Create journal entries
    source_obj = "TestSource"
    entry1 = JournalEntry(date(2024, 1, 15), "Test entry 1", source_obj)
    entry1.post(date(2024, 1, 15), account1, Quantity(Decimal("-100")))
    entry1.post(date(2024, 1, 15), account2, Quantity(Decimal("100")))
    
    entry2 = JournalEntry(date(2024, 2, 20), "Test entry 2", source_obj)
    entry2.post(date(2024, 2, 20), account1, Quantity(Decimal("50")))
    entry2.post(date(2024, 2, 20), account2, Quantity(Decimal("-50")))
    
    journal = [entry1, entry2]
    
    # Execute
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 2
    assert len(general_ledger.ledgers[account2].entries) == 2
    assert general_ledger.ledgers[account1].entries[0].balance == Quantity(Decimal("900"))
    assert general_ledger.ledgers[account1].entries[1].balance == Quantity(Decimal("950"))
    assert general_ledger.ledgers[account2].entries[0].balance == Quantity(Decimal("600"))
    assert general_ledger.ledgers[account2].entries[1].balance == Quantity(Decimal("550"))


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
    posting_obj = Posting(
        date=date(2023, 1, 1),
        journal=Journal(description="Test Journal", postings=[]),
        amount=amount,
        account=account,
        direction="debit"
    )
    ledger = Ledger()
    balance = Quantity(value=100.0)
    
    entry = LedgerEntry(ledger=ledger, posting=posting_obj, balance=balance)
    
    assert entry.ledger is ledger
    assert entry.posting is posting_obj
    assert entry.balance is balance


# LLM-generated content at query #13
#--------------------------

```python
def test_build_general_ledger_filters_postings_by_period():
    import datetime
    from decimal import Decimal
    from pypara.accounting.generic import Balance
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, Ledger
    from pypara.core import Account, Amount, Quantity, DateRange
    
    # Setup
    account = Account("1000", "Test Account")
    period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    # Create journal entries with dates inside and outside the period
    entry_inside = JournalEntry(datetime.date(2024, 6, 15), "Inside period", "source1")
    entry_before = JournalEntry(datetime.date(2023, 12, 31), "Before period", "source2")
    entry_after = JournalEntry(datetime.date(2025, 1, 1), "After period", "source3")
    
    # Add postings to entries
    entry_inside.post(datetime.date(2024, 6, 15), account, Quantity(Decimal(100)))
    entry_before.post(datetime.date(2023, 12, 31), account, Quantity(Decimal(50)))
    entry_after.post(datetime.date(2025, 1, 1), account, Quantity(Decimal(75)))
    
    # Validate all entries
    entry_inside.validate()
    entry_before.validate()
    entry_after.validate()
    
    # Build general ledger
    initial_balances = {}
    journal_entries = [entry_inside, entry_before, entry_after]
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # The predicate at line 16 should evaluate to True only for postings within the period
    # This means only entry_inside's postings should be included
    ledger = general_ledger.ledgers[account]
    
    # Verify that only the posting from entry_inside was added
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.journal_entry == entry_inside


# LLM-generated content at query #14
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
    class MockLedger:
        pass
    
    @dataclass
    class MockQuantity:
        value: float
    
    # Create instances
    account = MockAccount(name="Test Account")
    amount = MockAmount(value=100.0, currency="USD")
    journal = MockJournal(description="Test Description", postings=[])
    posting = MockPosting(
        date=date(2023, 1, 1),
        amount=amount,
        account=account,
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    posting._journal = journal
    
    ledger = MockLedger()
    balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)
    
    # Assert constructor properly assigned all fields
    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


# LLM-generated content at query #15
#--------------------------

```python
def test_ledger_entry_constructor():
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    # Create mock objects for dependencies
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
        is_debit: bool
        is_credit: bool
        account: Account

    @dataclass
    class Ledger(Generic[_T]):
        name: str

    # Create test data
    test_account = Account(name="Test Account")
    test_amount = Amount(value=100.0, currency="USD")
    test_quantity = Quantity(value=100.0)
    test_date = date(2023, 1, 15)
    
    counter_account = Account(name="Counter Account")
    test_journal = Journal(
        description="Test Transaction",
        postings=[
            Posting(date=test_date, journal=None, amount=test_amount, direction="debit", is_debit=True, is_credit=False, account=test_account),
            Posting(date=test_date, journal=None, amount=test_amount, direction="credit", is_debit=False, is_credit=True, account=counter_account)
        ]
    )
    
    test_posting = Posting(
        date=test_date,
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        is_debit=True,
        is_credit=False,
        account=test_account
    )
    
    test_ledger = Ledger(name="Test Ledger")

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=test_ledger,
        posting=test_posting,
        balance=test_quantity
    )

    # Assert constructor set all fields correctly
    assert entry.ledger == test_ledger
    assert entry.posting == test_posting
    assert entry.balance == test_quantity


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
        amount: Amount
        journal: Journal
        account: Account
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

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(100.0, "USD"),
        journal=Journal("Test transaction", []),
        account=Account("TestAccount"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance
    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


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
    test_journal = Journal(description="Test Entry", postings=[])
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
    assert entry.ledger.name == "Test Ledger"
    assert entry.posting.date == date(2024, 1, 1)
    assert entry.balance.value == 100.0


# LLM-generated content at query #18
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
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_quantity = MockQuantity(value=100.0)
    mock_journal = MockJournal(description="Test Journal", postings=[])
    mock_posting = MockPosting(
        date=date(2023, 1, 1),
        account=mock_account,
        amount=mock_amount,
        direction="debit",
        journal=mock_journal
    )
    mock_ledger = MockLedger()
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )
    
    # Assertions
    assert ledger_entry.ledger is mock_ledger
    assert ledger_entry.posting is mock_posting
    assert ledger_entry.balance is mock_quantity
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_quantity


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
        journal: Journal
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

    ledger = Ledger(name="Test Ledger")
    journal = Journal(description="Test Journal", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
        amount=Amount(value=100.0, currency="USD"),
        journal=journal,
        account=Account(name="Test Account"),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(value=100.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger == ledger
    assert entry.posting == posting
    assert entry.balance == balance


# LLM-generated content at query #20
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
        is_debit: bool
        is_credit: bool

    @dataclass
    class Ledger(Generic[_T]):
        pass

    ledger = Ledger()
    posting = Posting(
        date=date(2023, 1, 15),
        amount=Amount(100.0, "USD"),
        account=Account("Cash"),
        journal=Journal("Test transaction", []),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    balance = Quantity(1000.0)

    entry = LedgerEntry(ledger=ledger, posting=posting, balance=balance)

    assert entry.ledger is ledger
    assert entry.posting is posting
    assert entry.balance is balance


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
        date=date(2024, 1, 15),
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


# LLM-generated content at query #23
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
    
    account = Account(name="Cash")
    amount = Amount(value=100.0, currency="USD")
    posting_obj = Posting(
        date=date(2024, 1, 15),
        amount=amount,
        journal=Journal(description="Test transaction", postings=[]),
        account=account,
        direction="debit"
    )
    balance = Quantity(value=500.0)
    ledger = Ledger(name="General Ledger")
    
    ledger_entry = LedgerEntry(ledger=ledger, posting=posting_obj, balance=balance)
    
    assert ledger_entry.ledger == ledger
    assert ledger_entry.posting == posting_obj
    assert ledger_entry.balance == balance


# LLM-generated content at query #24
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.dates import DateRange
    
    # Setup test data
    test_date = date(2024, 1, 15)
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    account_cash = Account("1000", "Cash")
    account_revenue = Account("4000", "Revenue")
    
    initial_balances = {
        account_cash: Balance(period_start, Quantity(Decimal("1000.00")))
    }
    
    # Create journal entries
    journal_entry_1 = JournalEntry(test_date, "Sales transaction", "source_obj_1")
    journal_entry_1.post(test_date, account_cash, Quantity(Decimal("500.00")))
    journal_entry_1.post(test_date, account_revenue, Quantity(Decimal("-500.00")))
    
    journal_entry_2 = JournalEntry(date(2024, 2, 10), "Additional sales", "source_obj_2")
    journal_entry_2.post(date(2024, 2, 10), account_cash, Quantity(Decimal("300.00")))
    journal_entry_2.post(date(2024, 2, 10), account_revenue, Quantity(Decimal("-300.00")))
    
    journal = [journal_entry_1, journal_entry_2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert account_cash in general_ledger.ledgers
    assert account_revenue in general_ledger.ledgers
    
    # Check cash ledger
    cash_ledger = general_ledger.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1800.00"))
    
    # Check revenue ledger
    revenue_ledger = general_ledger.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert revenue_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(revenue_ledger.entries) == 2
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500.00"))
    assert revenue_ledger.entries[1].balance == Quantity(Decimal("-800.00"))


def test_build_general_ledger_empty_journal():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.dates import DateRange
    
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    account_cash = Account("1000", "Cash")
    initial_balances = {
        account_cash: Balance(period_start, Quantity(Decimal("5000.00")))
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account_cash in general_ledger.ledgers
    assert len(general_ledger.ledgers[account_cash].entries) == 0
    assert general_ledger.ledgers[account_cash].initial.value == Quantity(Decimal("5000.00"))


def test_build_general_ledger_posting_outside_period():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger, GeneralLedger
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.dates import DateRange
    
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    account_cash = Account("1000", "Cash")
    
    initial_balances = {
        account_cash: Balance(period_start, Quantity(Decimal("1000.00")))
    }
    
    # Create journal entry outside period
    journal_entry = JournalEntry(date(2023, 12, 31), "Outside period", "source")
    journal_entry.post(date(2023, 12, 31), account_cash, Quantity(Decimal("100.00")))
    
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert len(general_ledger.ledgers[account_cash].entries) == 0


def test_build_general_ledger_creates_new_ledgers():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry
    from pypara.accounting.ledger import build_general_ledger
    from pypara.accounting.generic import Account, Balance, Quantity
    from pypara.dates import DateRange
    
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    account_existing = Account("1000", "Cash")
    account_new = Account("2000", "Accounts Payable")
    
    initial_balances = {
        account_existing: Balance(period_start, Quantity(Decimal("1000.00")))
    }
    
    journal_entry = JournalEntry(date(2024, 1, 15), "Transaction", "source")
    journal_entry.post(date(2024, 1, 15), account_existing, Quantity(Decimal("500.00")))
    journal_entry.post(date(2024, 1, 15), account_new, Quantity(Decimal("-500.00")))
    
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    
    assert len(general_ledger.ledgers) == 2
    assert account_existing in general_ledger.ledgers
    assert account_new in general_ledger.ledgers
    assert general_ledger.ledgers[account_new].initial.value == Quantity(Decimal("0.00"))
    assert general_ledger.ledgers[account_new].initial.date == period_start


# LLM-generated content at query #25
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
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_journal = MockJournal(description="Test Journal", postings=[])
    mock_posting = MockPosting(
        date=date(2024, 1, 1),
        amount=mock_amount,
        account=mock_account,
        direction="debit",
        is_debit=True,
        is_credit=False,
        journal=mock_journal
    )
    mock_ledger = MockLedger()
    mock_balance = MockQuantity(value=500.0)
    
    # Create LedgerEntry instance
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    
    # Assert constructor properly assigned all fields
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


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
        account: MockAccount
        amount: MockAmount
        direction: str
        journal: MockJournal
        
        @property
        def is_debit(self) -> bool:
            return self.direction == "debit"
        
        @property
        def is_credit(self) -> bool:
            return self.direction == "credit"

    @dataclass
    class MockLedger:
        name: str

    # Create test instances
    mock_account = MockAccount(name="Test Account")
    mock_amount = MockAmount(value=100.0, currency="USD")
    mock_quantity = MockQuantity(value=100.0)
    mock_journal = MockJournal(description="Test Description", postings=[])
    mock_posting = MockPosting(
        date=date(2024, 1, 1),
        account=mock_account,
        amount=mock_amount,
        direction="debit",
        journal=mock_journal
    )
    mock_ledger = MockLedger(name="Test Ledger")

    # Create LedgerEntry instance
    entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_quantity
    )

    # Assert constructor properly initializes all fields
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_quantity


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


# LLM-generated content at query #28
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
    journal = Journal(description="Test transaction", postings=[])
    posting = Posting(
        date=date(2023, 1, 1),
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


# LLM-generated content at query #29
#--------------------------

```python
def test_ledger_constructor():
    account = Account(name="Test Account", account_type="asset")
    initial_balance = Balance(value=1000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_ledger_constructor_with_entries():
    account = Account(name="Test Account", account_type="asset")
    initial_balance = Balance(value=500)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


def test_ledger_constructor_preserves_account():
    account = Account(name="Checking", account_type="asset")
    initial_balance = Balance(value=2000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.account.name == "Checking"
    assert ledger.account.account_type == "asset"


def test_ledger_constructor_preserves_initial_balance():
    account = Account(name="Savings", account_type="asset")
    initial_balance = Balance(value=5000)
    
    ledger = Ledger(account=account, initial=initial_balance)
    
    assert ledger.initial.value == 5000


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
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test transaction"
    assert entry.amount == Amount(value=100.0, currency="USD")
    assert entry.is_debit is True
    assert entry.is_credit is False


# LLM-generated content at query #31
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


# LLM-generated content at query #32
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
    test_account = MockAccount(name="Cash")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_date = date(2024, 1, 15)
    test_journal = MockJournal(description="Test transaction", postings=[])
    test_posting = MockPosting(
        date=test_date,
        journal=test_journal,
        amount=test_amount,
        direction="debit",
        account=test_account
    )
    test_balance = MockQuantity(value=500.0)
    test_ledger = MockLedger()

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


# LLM-generated content at query #33
#--------------------------

```python
def test_build_general_ledger():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Direction
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger
    from pypara.core.amount import Amount, Quantity
    from pypara.core.commons import DateRange
    from pypara.accounting.accounts import Account
    
    # Setup
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Create accounts
    cash_account = Account("1000", "Cash")
    revenue_account = Account("4000", "Revenue")
    expense_account = Account("5000", "Expenses")
    
    # Create initial balances
    initial_balances = {
        cash_account: Balance(date(2023, 1, 1), Quantity(Decimal("1000")))
    }
    
    # Create journal entries
    je1 = JournalEntry(date(2023, 1, 15), "Sale", "source1")
    je1.post(date(2023, 1, 15), cash_account, Quantity(Decimal("500")))
    je1.post(date(2023, 1, 15), revenue_account, Quantity(Decimal("-500")))
    
    je2 = JournalEntry(date(2023, 2, 10), "Expense", "source2")
    je2.post(date(2023, 2, 10), cash_account, Quantity(Decimal("-200")))
    je2.post(date(2023, 2, 10), expense_account, Quantity(Decimal("200")))
    
    # Outside period - should not be included
    je3 = JournalEntry(date(2024, 1, 1), "Future", "source3")
    je3.post(date(2024, 1, 1), cash_account, Quantity(Decimal("100")))
    je3.post(date(2024, 1, 1), revenue_account, Quantity(Decimal("-100")))
    
    journal = [je1, je2, je3]
    
    # Execute
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    assert cash_account in general_ledger.ledgers
    assert revenue_account in general_ledger.ledgers
    assert expense_account in general_ledger.ledgers
    
    # Check cash ledger
    cash_ledger = general_ledger.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial.value == Quantity(Decimal("1000"))
    assert len(cash_ledger.entries) == 2  # Only 2 entries within period
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500"))  # 1000 + 500
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300"))  # 1500 - 200
    
    # Check revenue ledger
    revenue_ledger = general_ledger.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert len(revenue_ledger.entries) == 1  # Only 1 entry within period
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500"))
    
    # Check expense ledger
    expense_ledger = general_ledger.ledgers[expense_account]
    assert expense_ledger.account == expense_account
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200"))


# LLM-generated content at query #34
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
        journal: MockJournal
        direction: str
        is_debit: bool
        is_credit: bool
    
    @dataclass
    class MockLedger:
        name: str
    
    # Create instances
    mock_ledger = MockLedger(name="Test Ledger")
    mock_posting = MockPosting(
        date=date(2023, 1, 15),
        account=MockAccount(name="Cash"),
        amount=MockAmount(value=100.0, currency="USD"),
        journal=MockJournal(description="Test Journal", postings=[]),
        direction="debit",
        is_debit=True,
        is_credit=False
    )
    mock_balance = MockQuantity(value=1000.0)
    
    # Create LedgerEntry instance
    ledger_entry = LedgerEntry(
        ledger=mock_ledger,
        posting=mock_posting,
        balance=mock_balance
    )
    
    # Assert constructor sets attributes correctly
    assert ledger_entry.ledger == mock_ledger
    assert ledger_entry.posting == mock_posting
    assert ledger_entry.balance == mock_balance


# LLM-generated content at query #35
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
    
    # Create test instances
    test_account = MockAccount(name="Test Account")
    test_amount = MockAmount(value=100.0, currency="USD")
    test_journal = MockJournal(description="Test Entry", postings=[])
    test_posting = MockPosting(
        date=date(2023, 1, 15),
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


