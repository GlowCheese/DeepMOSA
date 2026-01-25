####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {}

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry]:
            return []

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    read_initial_balances = MockReadInitialBalances()
    read_journal_entries = MockReadJournalEntries()
    program = compile_general_ledger_program(read_initial_balances, read_journal_entries)
    general_ledger = program(period)

    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert general_ledger.ledgers == {}


# LLM-generated content at query #2
#--------------------------

def test_ReadInitialBalances___call__():
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Account1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}

    mock_read_initial_balances = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = mock_read_initial_balances(period)

    assert isinstance(initial_balances, dict)
    assert Account("Account1") in initial_balances
    assert initial_balances[Account("Account1")].value == Decimal(100)


# LLM-generated content at query #3
#--------------------------

def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    mock_read_initial_balances = MockReadInitialBalances()
    result = mock_read_initial_balances(period)

    assert isinstance(result, dict)
    assert Account("cash") in result
    assert result[Account("cash")].value == Quantity(Decimal(1000))


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    mock_reader = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    result = mock_reader(period)

    assert isinstance(result, dict)
    assert Account("cash") in result
    assert result[Account("cash")].value == Quantity(Decimal(1000))


# LLM-generated content at query #5
#--------------------------

def test_build_general_ledger():
    # Create test accounts
    account1 = Account("Assets", "Cash")
    account2 = Account("Liabilities", "Loans")
    
    # Create test date range
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    
    # Create initial balances
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(500))),
    }
    
    # Create test journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Test transaction",
            [
                Posting(account1, Amount(Decimal(200)), 1),
                Posting(account2, Amount(Decimal(200)), -1),
            ]
        )
    ]
    
    # Build the general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assert the period is correct
    assert general_ledger.period == period
    
    # Assert all accounts are present in ledgers
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    # Verify account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal(1200))
    
    # Verify account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal(300))
    
    # Test with new account not in initial balances
    account3 = Account("Income", "Sales")
    journal_entries_with_new_account = journal_entries + [
        JournalEntry(
            datetime.date(2023, 1, 20),
            "New account transaction",
            [
                Posting(account1, Amount(Decimal(100)), 1),
                Posting(account3, Amount(Decimal(100)), -1),
            ]
        )
    ]
    
    general_ledger_with_new_account = build_general_ledger(
        period, journal_entries_with_new_account, initial_balances
    )
    
    # Verify new account was created with zero initial balance
    assert account3 in general_ledger_with_new_account.ledgers
    ledger3 = general_ledger_with_new_account.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal(0))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal(-100))


# LLM-generated content at query #6
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            posting = Posting(
                Account("Cash"),
                Amount(Decimal(50)),
                Direction.DEBIT,
                datetime.date(2023, 1, 2),
                JournalEntry("Test Journal", datetime.date(2023, 1, 2), [])
            )
            journal_entry = JournalEntry(
                "Test Journal",
                datetime.date(2023, 1, 2),
                [posting]
            )
            return [journal_entry]

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    program = compile_general_ledger_program(MockReadInitialBalances(), MockReadJournalEntries())
    general_ledger = program(period)

    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert Account("Cash") in general_ledger.ledgers
    ledger = general_ledger.ledgers[Account("Cash")]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal(150))


# LLM-generated content at query #7
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock initial balances
    mock_initial_balances = {
        Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    }

    # Mock journal entries
    mock_journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 2),
            "Sample Entry",
            [
                Posting(Account("Cash"), Amount(Decimal(500)), Decimal(1)),
                Posting(Account("Revenue"), Amount(Decimal(500)), Decimal(-1)),
            ],
        )
    ]

    # Mock functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return mock_initial_balances

    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return mock_journal_entries

    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Define the period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert Account("Cash") in general_ledger.ledgers
    assert len(general_ledger.ledgers[Account("Cash")].entries) == 1
    assert general_ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(1500))


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from datetime import date
from decimal import Decimal
from typing import Dict
from dataclasses import dataclass
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance

@dataclass
class MockReadInitialBalances:
    """
    Mock implementation of ReadInitialBalances protocol for testing.
    """
    return_value: Dict[Account, Balance]

    def __call__(self, period: DateRange) -> Dict[Account, Balance]:
        return self.return_value

def test_ReadInitialBalances___call__():
    # Setup test data
    test_account = Account("Test Account")
    test_balance = Balance(date(2023, 1, 1), Decimal("1000.00"))
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Create mock implementation
    mock_initial_balances = {test_account: test_balance}
    mock_reader = MockReadInitialBalances(mock_initial_balances)
    
    # Test the __call__ method
    result = mock_reader(test_period)
    
    # Assertions
    assert isinstance(result, dict)
    assert test_account in result
    assert result[test_account] == test_balance
    assert len(result) == 1


# LLM-generated content at query #9
#--------------------------

def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances protocol
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {
                Account("Assets:Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00"))),
                Account("Liabilities:Loans"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal("-500.00")))
            }

    # Create instance of mock implementation
    mock_reader = MockReadInitialBalances()

    # Test period
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Call the protocol implementation
    result = mock_reader(test_period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Assets:Cash") in result
    assert Account("Liabilities:Loans") in result
    assert result[Account("Assets:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liabilities:Loans")].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #10
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account = Account("Cash")
        balance = Balance(period.since, Quantity(Decimal(100)))
        return {account: balance}

    # Create an instance of the protocol using the mock implementation
    read_initial_balances = ReadInitialBalances(mock_read_initial_balances)

    # Define a DateRange for testing
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Call the __call__ method
    result = read_initial_balances(period)

    # Assert the expected results
    assert isinstance(result, dict)
    assert len(result) == 1
    account = Account("Cash")
    assert account in result
    assert result[account].value == Quantity(Decimal(100))
    assert result[account].date == period.since


# LLM-generated content at query #11
#--------------------------

def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1234", "Test Account")
    initial_balances = {account: Balance(period.since, Quantity(Decimal(100)))}
    
    # Mock journal entries
    journal_entry = JournalEntry(
        datetime.date(2023, 6, 15),
        "Test Journal",
        [
            Posting(account, Decimal(50), datetime.date(2023, 6, 15), "Test Posting", None)
        ]
    )
    journal_entries = [journal_entry]

    # Mock read functions
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        assert p == period
        return initial_balances

    def mock_read_journal_entries(p: DateRange) -> Iterable[JournalEntry]:
        assert p == period
        return journal_entries

    # Create the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )

    # Execute the program
    result = program(period)

    # Verify results
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.initial == initial_balances[account]
    assert len(ledger.entries) == 1
    entry = ledger.entries[0]
    assert entry.balance == Quantity(Decimal(150))
    assert entry.posting == journal_entry.postings[0]


# LLM-generated content at query #12
#--------------------------

Here's a unit test for the `__call__` method of `GeneralLedgerProgram`:


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}

    mock_read_initial_balances = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = mock_read_initial_balances(period)

    assert isinstance(initial_balances, dict)
    assert Account("1") in initial_balances
    assert initial_balances[Account("1")].value == Quantity(Decimal(100))
    assert initial_balances[Account("1")].date == datetime.date(2023, 1, 1)


# LLM-generated content at query #14
#--------------------------

def test_build_general_ledger():
    # Create a sample account
    account = Account("Cash", "Asset")

    # Create a sample DateRange
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Create a sample JournalEntry with Posting
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 1),
        "Sample Description",
        [Posting(account, Amount(Decimal("100.00")), Decimal("1"))]
    )

    # Create initial balances
    initial_balances = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("0.00")))}

    # Build the general ledger
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].posting.amount == Amount(Decimal("100.00"))
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal("100.00"))


# LLM-generated content at query #15
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    mock_read_initial_balances = MockReadInitialBalances()
    initial_balances = mock_read_initial_balances(period)

    assert isinstance(initial_balances, dict)
    assert Account("Cash") in initial_balances
    assert initial_balances[Account("Cash")] == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))


# LLM-generated content at query #16
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Cash"): Balance(DateRange("2023-01-01", "2023-01-31"), Quantity(Decimal(1000)))}

    # Instantiate the mock
    mock_read_initial_balances = MockReadInitialBalances()

    # Define a test period
    test_period = DateRange("2023-01-01", "2023-01-31")

    # Call the __call__ method
    result = mock_read_initial_balances(test_period)

    # Assert the result
    assert isinstance(result, dict)
    assert Account("Cash") in result
    assert result[Account("Cash")].value == Quantity(Decimal(1000))


# LLM-generated content at query #17
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    mock_read_initial_balances = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = mock_read_initial_balances(period)

    assert isinstance(initial_balances, dict)
    assert Account("Cash") in initial_balances
    assert initial_balances[Account("Cash")].value == Quantity(Decimal(1000))


# LLM-generated content at query #18
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {Account("Assets:Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    class MockReadJournalEntries:
        def __call__(self, period):
            return [
                JournalEntry(
                    datetime.date(2023, 1, 2),
                    "Test Entry",
                    [
                        Posting(Account("Assets:Cash"), Amount(Decimal(200)), Decimal(1)),
                        Posting(Account("Expenses:Test"), Amount(Decimal(200)), Decimal(-1)),
                    ],
                )
            ]

    mock_read_initial_balances = MockReadInitialBalances()
    mock_read_journal_entries = MockReadJournalEntries()

    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    general_ledger = program(period)

    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert Account("Assets:Cash") in general_ledger.ledgers
    assert len(general_ledger.ledgers[Account("Assets:Cash")].entries) == 1
    assert general_ledger.ledgers[Account("Assets:Cash")].entries[0].balance == Quantity(Decimal(1200))
    assert Account("Expenses:Test") in general_ledger.ledgers
    assert len(general_ledger.ledgers[Account("Expenses:Test")].entries) == 1
    assert general_ledger.ledgers[Account("Expenses:Test")].entries[0].balance == Quantity(Decimal(200))


# LLM-generated content at query #19
#--------------------------

```python
def test_build_general_ledger():
    # Mock data
    account1 = Account(name="Cash")
    account2 = Account(name="Revenue")
    initial_balances = {
        account1: Balance(value=Quantity(Decimal(1000))),
        account2: Balance(value=Quantity(Decimal(0))),
    }
    journal_entries = [
        JournalEntry(
            description="Sale",
            date=datetime.date(2023, 10, 1),
            postings=[
                Posting(account=account2, amount=Amount(Decimal(500)), direction=1),
                Posting(account=account1, amount=Amount(Decimal(500)), direction=-1),
            ],
        ),
        JournalEntry(
            description="Expense",
            date=datetime.date(2023, 10, 2),
            postings=[
                Posting(account=account1, amount=Amount(Decimal(200)), direction=-1),
                Posting(account=account2, amount=Amount(Decimal(200)), direction=1),
            ],
        ),
    ]
    period = DateRange(since=datetime.date(2023, 10, 1), until=datetime.date(2023, 10, 31))

    # Call the function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert len(general_ledger.ledgers) == 2

    # Check Cash account ledger
    cash_ledger = general_ledger.ledgers[account1]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal(1500))
    assert cash_ledger.entries[1].balance == Quantity(Decimal(1300))

    # Check Revenue account ledger
    revenue_ledger = general_ledger.ledgers[account2]
    assert len(revenue_ledger.entries) == 2
    assert revenue_ledger.entries[0].balance == Quantity(Decimal(500))
    assert revenue_ledger.entries[1].balance == Quantity(Decimal(700))


# LLM-generated content at query #20
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            return [
                JournalEntry(
                    datetime.date(2023, 1, 2),
                    "Description",
                    [
                        Posting(Account("1"), Amount(Decimal(50)), 1),
                        Posting(Account("2"), Amount(Decimal(50)), -1),
                    ],
                )
            ]

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    read_initial_balances = MockReadInitialBalances()
    read_journal_entries = MockReadJournalEntries()
    program = compile_general_ledger_program(read_initial_balances, read_journal_entries)
    general_ledger = program(period)

    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert Account("1") in general_ledger.ledgers
    assert Account("2") in general_ledger.ledgers
    assert len(general_ledger.ledgers[Account("1")].entries) == 1
    assert len(general_ledger.ledgers[Account("2")].entries) == 1
    assert general_ledger.ledgers[Account("1")].entries[0].balance == Quantity(Decimal(150))
    assert general_ledger.ledgers[Account("2")].entries[0].balance == Quantity(Decimal(50))


# LLM-generated content at query #21
#--------------------------

def test_GeneralLedgerProgram___call__():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    mock_initial_balances = {
        Account("Assets:Cash"): Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    mock_journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Test transaction",
            [
                Posting(Account("Assets:Cash"), Amount(Decimal("500.00")), -1),
                Posting(Account("Income:Salary"), Amount(Decimal("500.00")), 1)
            ]
        )
    ]

    # Mock functions
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        assert p == period
        return mock_initial_balances

    def mock_read_journal_entries(p: DateRange) -> List[JournalEntry]:
        assert p == period
        return mock_journal_entries

    # Create the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )

    # Execute the program
    result = program(period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2  # Should have ledgers for both accounts
    
    # Check Cash ledger
    cash_ledger = result.ledgers[Account("Assets:Cash")]
    assert cash_ledger.account == Account("Assets:Cash")
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    
    # Check Income ledger
    income_ledger = result.ledgers[Account("Income:Salary")]
    assert income_ledger.account == Account("Income:Salary")
    assert len(income_ledger.entries) == 1
    assert income_ledger.entries[0].balance == Quantity(Decimal("500.00"))


# LLM-generated content at query #22
#--------------------------

def test_build_general_ledger():
    # Test data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Create test accounts
    account1 = Account("1", "Cash")
    account2 = Account("2", "Revenue")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    # Create test journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Sale",
            [
                Posting(account1, Amount(Decimal("500.00")), 1),
                Posting(account2, Amount(Decimal("500.00")), -1)
            ]
        ),
        JournalEntry(
            datetime.date(2023, 2, 20),
            "Expense",
            [
                Posting(account1, Amount(Decimal("200.00")), -1),
                Posting(account2, Amount(Decimal("200.00")), 1)
            ]
        )
    ]
    
    # Build the general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2  # Should have ledgers for both accounts
    
    # Verify account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2
    
    # Check first entry for account1
    entry1 = ledger1.entries[0]
    assert entry1.posting.amount == Amount(Decimal("500.00"))
    assert entry1.balance == Quantity(Decimal("1500.00"))  # 1000 + 500
    
    # Check second entry for account1
    entry2 = ledger1.entries[1]
    assert entry2.posting.amount == Amount(Decimal("200.00"))
    assert entry2.balance == Quantity(Decimal("1300.00"))  # 1500 - 200
    
    # Verify account2 ledger (should have been created automatically)
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert len(ledger2.entries) == 2
    
    # Check first entry for account2
    entry3 = ledger2.entries[0]
    assert entry3.posting.amount == Amount(Decimal("500.00"))
    assert entry3.balance == Quantity(Decimal("-500.00"))  # 0 - 500
    
    # Check second entry for account2
    entry4 = ledger2.entries[1]
    assert entry4.posting.amount == Amount(Decimal("200.00"))
    assert entry4.balance == Quantity(Decimal("-300.00"))  # -500 + 200
    
    # Test with empty journal entries
    empty_ledger = build_general_ledger(period, [], initial_balances)
    assert len(empty_ledger.ledgers) == 1  # Only account1 should exist
    assert len(empty_ledger.ledgers[account1].entries) == 0
    
    # Test with no initial balances
    no_initial_ledger = build_general_ledger(period, journal_entries, {})
    assert len(no_initial_ledger.ledgers) == 2  # Both accounts should exist
    assert no_initial_ledger.ledgers[account1].initial == Balance(period.since, Quantity(Decimal("0.00")))


# LLM-generated content at query #23
#--------------------------

def test_ReadInitialBalances___call__():
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {
                Account("cash"): Balance(DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31)), Quantity(Decimal(1000))),
                Account("revenue"): Balance(DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31)), Quantity(Decimal(2000))),
            }

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    mock_read_initial_balances = MockReadInitialBalances()
    result = mock_read_initial_balances(period)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert isinstance(result[Account("cash")], Balance)
    assert result[Account("cash")].value == Quantity(Decimal(1000))
    assert isinstance(result[Account("revenue")], Balance)
    assert result[Account("revenue")].value == Quantity(Decimal(2000))


# LLM-generated content at query #24
#--------------------------

def test_GeneralLedgerProgram___call__():
    # Mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return []

    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Define a period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Execute the program
    result = program(period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("001"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    mock_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    mock_read_initial_balances = MockReadInitialBalances()

    result = mock_read_initial_balances(mock_period)

    assert isinstance(result, dict)
    assert Account("001") in result
    assert result[Account("001")].value == Quantity(Decimal(1000))


# LLM-generated content at query #26
#--------------------------

```python
def test_build_general_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    account1 = Account("001", "Cash")
    account2 = Account("002", "Revenue")
    
    initial_balances = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 2),
        "Sale",
        [
            Posting(account1, Amount(Decimal(500)), Direction.DEBIT),
            Posting(account2, Amount(Decimal(500)), Direction.CREDIT),
        ]
    )
    
    journal = [journal_entry]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    
    cash_ledger = general_ledger.ledgers[account1]
    assert cash_ledger.account == account1
    assert cash_ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Quantity(Decimal(1500))
    
    revenue_ledger = general_ledger.ledgers[account2]
    assert revenue_ledger.account == account2
    assert revenue_ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal(500))


# LLM-generated content at query #27
#--------------------------

def test_build_general_ledger():
    # Create test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("1", "Cash")
    account2 = Account("2", "Revenue")
    
    # Test initial balances
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000)))
    }
    
    # Test journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Sale",
            [
                Posting(account1, Amount(Decimal(500)), Decimal(1)),
                Posting(account2, Amount(Decimal(500)), Decimal(-1))
            ]
        )
    ]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2  # Should have both accounts
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal(1500))
    
    # Check account2 ledger (should be created automatically)
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal(-500))
    
    # Test entry properties
    entry = ledger1.entries[0]
    assert entry.date == datetime.date(2023, 1, 15)
    assert entry.description == "Sale"
    assert entry.amount == Amount(Decimal(500))
    assert entry.is_debit
    assert not entry.is_credit
    assert entry.debit == Amount(Decimal(500))
    assert entry.credit is None
    assert account2 in entry.cntraccts
    
    # Test with empty journal
    empty_ledger = build_general_ledger(period, [], initial_balances)
    assert len(empty_ledger.ledgers) == 1  # Only the initial account
    assert len(empty_ledger.ledgers[account1].entries) == 0


# LLM-generated content at query #28
#--------------------------

Here's a unit test for the `__call__` method of `GeneralLedgerProgram`:


# LLM-generated content at query #29
#--------------------------

Here's a unit test for the `__call__` method of the `ReadInitialBalances` protocol class:


# LLM-generated content at query #30
#--------------------------

def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1", "Cash")
        account2 = Account("2", "Revenue")
        return {
            account1: Balance(period.since, Quantity(Decimal(1000))),
            account2: Balance(period.since, Quantity(Decimal(5000)))
        }

    # Create a test instance using the mock
    read_initial_balances = ReadInitialBalances(mock_read_initial_balances)

    # Test period
    since = datetime.date(2023, 1, 1)
    until = datetime.date(2023, 12, 31)
    period = DateRange(since, until)

    # Call the protocol method
    result = read_initial_balances(period)

    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == since
        assert isinstance(balance.value, Quantity)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger():
    # Mock data
    account1 = Account("Account1")
    account2 = Account("Account2")
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial_balances = {account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    
    journal_entry1 = JournalEntry(
        description="Entry1",
        date=datetime.date(2023, 1, 2),
        postings=[
            Posting(account=account1, amount=Amount(Decimal(50)), direction=1),
            Posting(account=account2, amount=Amount(Decimal(50)), direction=-1)
        ]
    )
    
    journal_entry2 = JournalEntry(
        description="Entry2",
        date=datetime.date(2023, 1, 3),
        postings=[
            Posting(account=account1, amount=Amount(Decimal(20)), direction=-1),
            Posting(account=account2, amount=Amount(Decimal(20)), direction=1)
        ]
    )
    
    journal_entries = [journal_entry1, journal_entry2]
    
    # Test function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2  # account1 and account2
    
    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal(150))
    assert ledger1.entries[1].balance == Quantity(Decimal(130))
    
    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal(50))
    assert ledger2.entries[1].balance == Quantity(Decimal(30))


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger():
    # Define test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("001", "Account 1")
    account2 = Account("002", "Account 2")
    initial_balances = {account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(100)))}
    
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 10),
            "Entry 1",
            [
                Posting(account1, Amount(Decimal(50)), Decimal(1)),
                Posting(account2, Amount(Decimal(50)), Decimal(-1)),
            ],
        ),
        JournalEntry(
            datetime.date(2023, 1, 20),
            "Entry 2",
            [
                Posting(account1, Amount(Decimal(30)), Decimal(-1)),
                Posting(account2, Amount(Decimal(30)), Decimal(1)),
            ],
        ),
    ]
    
    # Build the general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert len(general_ledger.ledgers) == 2
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    
    ledger1 = general_ledger.ledgers[account1]
    ledger2 = general_ledger.ledgers[account2]
    
    assert len(ledger1.entries) == 2
    assert len(ledger2.entries) == 2
    
    assert ledger1.initial.value == Decimal(100)
    assert ledger2.initial.value == Decimal(0)
    
    assert ledger1.entries[0].balance == Quantity(Decimal(150))
    assert ledger1.entries[1].balance == Quantity(Decimal(120))
    assert ledger2.entries[0].balance == Quantity(Decimal(-50))
    assert ledger2.entries[1].balance == Quantity(Decimal(-20))


# LLM-generated content at query #3
#--------------------------

def test_ReadInitialBalances___call__():
    # Test with empty period
    empty_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    
    def mock_read_initial(period: DateRange) -> InitialBalances:
        assert period == empty_period
        return {}
    
    reader = ReadInitialBalances(mock_read_initial)
    result = reader(empty_period)
    assert result == {}
    
    # Test with non-empty period and balances
    test_period = DateRange(datetime.date(2023, 2, 1), datetime.date(2023, 2, 28))
    test_account = Account("Assets", "Cash")
    test_balance = Balance(datetime.date(2023, 1, 31), Quantity(Decimal(1000)))
    
    def mock_read_with_balance(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {test_account: test_balance}
    
    reader_with_balance = ReadInitialBalances(mock_read_with_balance)
    result = reader_with_balance(test_period)
    assert result == {test_account: test_balance}
    assert test_account in result
    assert result[test_account] == test_balance


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("TestAccount"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}

    mock_instance = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    result = mock_instance(period)
    
    assert isinstance(result, dict)
    assert len(result) == 1
    assert Account("TestAccount") in result
    assert result[Account("TestAccount")].value == Decimal(100)


# LLM-generated content at query #5
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    # Mock dependencies
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Assets"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry]:
            return [
                JournalEntry(
                    datetime.date(2023, 1, 2),
                    "Test Entry",
                    [Posting(Account("Assets"), Amount(Decimal(50)), Decimal(1))]
                )
            ]

    # Create instances of mocks
    read_initial_balances = MockReadInitialBalances()
    read_journal_entries = MockReadJournalEntries()

    # Compile the program
    program = compile_general_ledger_program(read_initial_balances, read_journal_entries)

    # Define the period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert Account("Assets") in general_ledger.ledgers
    ledger = general_ledger.ledgers[Account("Assets")]
    assert ledger.account == Account("Assets")
    assert ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    assert len(ledger.entries) == 1
    entry = ledger.entries[0]
    assert entry.posting.account == Account("Assets")
    assert entry.posting.amount == Amount(Decimal(50))
    assert entry.balance == Quantity(Decimal(150))


# LLM-generated content at query #6
#--------------------------

def test_GeneralLedgerProgram___call__():
    # Mock dependencies
    class MockReadInitialBalances:
        def __call__(self, period):
            return {
                Account("Assets:Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000))),
                Account("Liabilities:Loan"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(-500)))
            }

    class MockReadJournalEntries:
        def __call__(self, period):
            return [
                JournalEntry(
                    datetime.date(2023, 1, 2),
                    "Test entry",
                    [
                        Posting(Account("Assets:Cash"), Amount(Decimal(200)), 1),
                        Posting(Account("Revenue:Sales"), Amount(Decimal(200)), -1)
                    ]
                )
            ]

    # Create test period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Compile the program
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries()
    )

    # Execute the program
    result = program(period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 3  # Cash, Loan, and Revenue accounts

    # Verify cash ledger
    cash_ledger = result.ledgers[Account("Assets:Cash")]
    assert cash_ledger.initial.value == Decimal(1000)
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Decimal(1200)

    # Verify loan ledger (no entries, just initial balance)
    loan_ledger = result.ledgers[Account("Liabilities:Loan")]
    assert loan_ledger.initial.value == Decimal(-500)
    assert len(loan_ledger.entries) == 0

    # Verify revenue ledger (created automatically)
    revenue_ledger = result.ledgers[Account("Revenue:Sales")]
    assert revenue_ledger.initial.value == Decimal(0)
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Decimal(200)


# LLM-generated content at query #7
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("asset"): Balance(period.since, Quantity(Decimal(100)))}

    mock_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    mock_read_initial_balances = MockReadInitialBalances()
    result = mock_read_initial_balances(mock_period)

    assert isinstance(result, dict)
    assert len(result) == 1
    assert isinstance(result[Account("asset")], Balance)
    assert result[Account("asset")].value == Quantity(Decimal(100))


# LLM-generated content at query #8
#--------------------------

def test_GeneralLedgerProgram___call__():
    # Mock implementations for ReadInitialBalances and ReadJournalEntries
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Account1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100))),
            Account("Account2"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(200))),
        }

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry[str]]:
        return [
            JournalEntry(
                datetime.date(2023, 1, 2),
                "Description1",
                [
                    Posting(Account("Account1"), Amount(Decimal(50)), Direction.DEBIT),
                    Posting(Account("Account2"), Amount(Decimal(50)), Direction.CREDIT),
                ],
            )
        ]

    # Compile the GeneralLedgerProgram
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)

    # Define the period for the test
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Execute the program
    general_ledger = program(period)

    # Assertions to verify the correctness of the GeneralLedger
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2

    # Verify ledger for Account1
    account1_ledger = general_ledger.ledgers[Account("Account1")]
    assert len(account1_ledger.entries) == 1
    assert account1_ledger.entries[0].balance == Quantity(Decimal(150))

    # Verify ledger for Account2
    account2_ledger = general_ledger.ledgers[Account("Account2")]
    assert len(account2_ledger.entries) == 1
    assert account2_ledger.entries[0].balance == Quantity(Decimal(150))


# LLM-generated content at query #9
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    mock_read_initial_balances = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    initial_balances = mock_read_initial_balances(period)

    assert isinstance(initial_balances, dict)
    assert Account("Cash") in initial_balances
    assert initial_balances[Account("Cash")].value == Quantity(Decimal(1000))


# LLM-generated content at query #10
#--------------------------

def test_build_general_ledger():
    # Test data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("Assets", "Cash")
    account2 = Account("Liabilities", "Loan")
    
    # Initial balances
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(500)))
    }
    
    # Journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Test transaction",
            [
                Posting(account1, Amount(Decimal(200)), 1),
                Posting(account2, Amount(Decimal(200)), -1)
            ]
        )
    ]
    
    # Call the function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal(1200))
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal(300))
    
    # Test with new account not in initial balances
    account3 = Account("Expenses", "Rent")
    journal_entries_with_new_account = [
        JournalEntry(
            datetime.date(2023, 1, 20),
            "Rent payment",
            [
                Posting(account1, Amount(Decimal(500)), -1),
                Posting(account3, Amount(Decimal(500)), 1)
            ]
        )
    ]
    
    general_ledger_new_account = build_general_ledger(period, journal_entries_with_new_account, initial_balances)
    
    assert len(general_ledger_new_account.ledgers) == 3
    assert account3 in general_ledger_new_account.ledgers
    ledger3 = general_ledger_new_account.ledgers[account3]
    assert ledger3.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal(500))


# LLM-generated content at query #11
#--------------------------

```python
def test_ReadInitialBalances___call__():
    class MockReadInitialBalances(ReadInitialBalances):
        def __call__(self, period: DateRange) -> InitialBalances:
            return {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    mock_read_initial_balances = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    initial_balances = mock_read_initial_balances(period)

    assert isinstance(initial_balances, dict)
    assert Account("Cash") in initial_balances
    assert initial_balances[Account("Cash")].value == Quantity(Decimal(1000))


# LLM-generated content at query #12
#--------------------------

def test_build_general_ledger():
    # Test data setup
    from decimal import Decimal
    import datetime
    from dataclasses import dataclass
    from typing import Dict, List

    # Mock classes for testing
    @dataclass
    class MockAccount:
        name: str

    @dataclass
    class MockBalance:
        date: datetime.date
        value: Decimal

    @dataclass
    class MockPosting:
        account: MockAccount
        amount: Decimal
        direction: int
        date: datetime.date
        journal: 'MockJournalEntry'

    @dataclass
    class MockJournalEntry:
        date: datetime.date
        description: str
        postings: List[MockPosting]

    # Create test accounts
    account1 = MockAccount("Asset")
    account2 = MockAccount("Liability")
    account3 = MockAccount("Income")

    # Create test date range
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(start_date, end_date)

    # Create initial balances
    initial_balances: Dict[MockAccount, MockBalance] = {
        account1: MockBalance(start_date, Decimal(100)),
        account2: MockBalance(start_date, Decimal(50))
    }

    # Create test journal entries
    journal_entry1 = MockJournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction 1",
        postings=[
            MockPosting(
                account=account1,
                amount=Decimal(20),
                direction=1,
                date=datetime.date(2023, 1, 15),
                journal=None
            ),
            MockPosting(
                account=account2,
                amount=Decimal(20),
                direction=-1,
                date=datetime.date(2023, 1, 15),
                journal=None
            )
        ]
    )
    journal_entry1.postings[0].journal = journal_entry1
    journal_entry1.postings[1].journal = journal_entry1

    journal_entry2 = MockJournalEntry(
        date=datetime.date(2023, 1, 20),
        description="Test transaction 2",
        postings=[
            MockPosting(
                account=account3,
                amount=Decimal(30),
                direction=1,
                date=datetime.date(2023, 1, 20),
                journal=None
            ),
            MockPosting(
                account=account1,
                amount=Decimal(30),
                direction=-1,
                date=datetime.date(2023, 1, 20),
                journal=None
            )
        ]
    )
    journal_entry2.postings[0].journal = journal_entry2
    journal_entry2.postings[1].journal = journal_entry2

    # Test journal entries outside period (should be ignored)
    journal_entry_outside = MockJournalEntry(
        date=datetime.date(2022, 12, 31),
        description="Outside period",
        postings=[
            MockPosting(
                account=account1,
                amount=Decimal(100),
                direction=1,
                date=datetime.date(2022, 12, 31),
                journal=None
            )
        ]
    )

    # Build general ledger
    journal_entries = [journal_entry1, journal_entry2, journal_entry_outside]
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3  # Should include all accounts touched

    # Check account1 ledger
    account1_ledger = general_ledger.ledgers[account1]
    assert account1_ledger.initial.value == Decimal(100)
    assert len(account1_ledger.entries) == 2
    assert account1_ledger.entries[0].balance == Decimal(120)  # 100 + 20
    assert account1_ledger.entries[1].balance == Decimal(90)   # 120 - 30

    # Check account2 ledger
    account2_ledger = general_ledger.ledgers[account2]
    assert account2_ledger.initial.value == Decimal(50)
    assert len(account2_ledger.entries) == 1
    assert account2_ledger.entries[0].balance == Decimal(30)  # 50 - 20

    # Check account3 ledger (should be created automatically)
    account3_ledger = general_ledger.ledgers[account3]
    assert account3_ledger.initial.value == Decimal(0)
    assert len(account3_ledger.entries) == 1
    assert account3_ledger.entries[0].balance == Decimal(30)  # 0 + 30

    # Verify journal entry outside period was ignored
    assert account1_ledger.entries[-1].balance == Decimal(90)  # Not 190 (100+100-20-30)


# LLM-generated content at query #13
#--------------------------

def test_GeneralLedgerProgram___call__():
    # Mock implementations for ReadInitialBalances and ReadJournalEntries
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}

    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return [
            JournalEntry(
                date=datetime.date(2023, 1, 2),
                description="Test Entry",
                postings=[
                    Posting(
                        account=Account("Cash"),
                        amount=Amount(Decimal(100)),
                        direction=1,
                        journal=None
                    )
                ]
            )
        ]

    # Compile the GeneralLedgerProgram
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Define the period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Execute the program
    general_ledger = program(period)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert Account("Cash") in general_ledger.ledgers
    assert len(general_ledger.ledgers[Account("Cash")].entries) == 1
    assert general_ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(1100))


# LLM-generated content at query #14
#--------------------------

def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {
                Account("Assets:Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00"))),
                Account("Liabilities:Loan"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal("-500.00")))
            }

    # Create test instance
    mock_reader = MockReadInitialBalances()
    
    # Test period
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Call the method
    result = mock_reader(test_period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Assets:Cash") in result
    assert Account("Liabilities:Loan") in result
    assert result[Account("Assets:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liabilities:Loan")].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #15
#--------------------------

def test_build_general_ledger():
    # Test data setup
    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    initial_balance1 = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    initial_balance2 = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(200)))
    initial_balances = {account1: initial_balance1, account2: initial_balance2}
    
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    
    # Create journal entries
    journal_entry1 = JournalEntry(
        datetime.date(2023, 1, 2),
        "Test entry 1",
        [
            Posting(account1, Amount(Decimal(50)), 1),
            Posting(account2, Amount(Decimal(50)), -1)
        ]
    )
    
    journal_entry2 = JournalEntry(
        datetime.date(2023, 1, 3),
        "Test entry 2",
        [
            Posting(account1, Amount(Decimal(30)), -1),
            Posting(account2, Amount(Decimal(30)), 1)
        ]
    )
    
    journal = [journal_entry1, journal_entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    
    # Verify account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balance1
    assert len(ledger1.entries) == 2
    
    entry1 = ledger1.entries[0]
    assert entry1.amount == Amount(Decimal(50))
    assert entry1.balance == Quantity(Decimal(150))
    assert entry1.is_debit
    
    entry2 = ledger1.entries[1]
    assert entry2.amount == Amount(Decimal(30))
    assert entry2.balance == Quantity(Decimal(120))
    assert entry2.is_credit
    
    # Verify account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balance2
    assert len(ledger2.entries) == 2
    
    entry1 = ledger2.entries[0]
    assert entry1.amount == Amount(Decimal(50))
    assert entry1.balance == Quantity(Decimal(150))
    assert entry1.is_credit
    
    entry2 = ledger2.entries[1]
    assert entry2.amount == Amount(Decimal(30))
    assert entry2.balance == Quantity(Decimal(180))
    assert entry2.is_debit
    
    # Test with new account not in initial balances
    account3 = Account("3", "Account 3")
    journal_entry3 = JournalEntry(
        datetime.date(2023, 1, 4),
        "Test entry 3",
        [Posting(account3, Amount(Decimal(100)), 1)]
    )
    
    journal_with_new_account = [journal_entry3]
    general_ledger_new_account = build_general_ledger(period, journal_with_new_account, initial_balances)
    
    assert len(general_ledger_new_account.ledgers) == 3
    ledger3 = general_ledger_new_account.ledgers[account3]
    assert ledger3.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal(100))


# LLM-generated content at query #16
#--------------------------

```python
def test_build_general_ledger():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1001", "Cash")
    initial_balances = {account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    
    # Mock JournalEntry and Posting
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 15),
        "Initial deposit",
        [
            Posting(account, Amount(Decimal(500)), Decimal(1), datetime.date(2023, 1, 15))
        ]
    )
    journal = [journal_entry]

    # Call the function
    general_ledger = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].entries[0].balance == Quantity(Decimal(1500))


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger():
    # Mock data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Mock initial balances
    initial_balances = {
        Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000))),
        Account("Receivables"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(500))),
    }
    
    # Mock journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Sale",
            [
                Posting(Account("Cash"), Decimal(200), 1),
                Posting(Account("Receivables"), Decimal(200), -1),
            ],
        ),
        JournalEntry(
            datetime.date(2023, 2, 20),
            "Purchase",
            [
                Posting(Account("Cash"), Decimal(150), -1),
                Posting(Account("Receivables"), Decimal(150), 1),
            ],
        ),
    ]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    
    # Check Cash ledger
    cash_ledger = general_ledger.ledgers[Account("Cash")]
    assert cash_ledger.account == Account("Cash")
    assert cash_ledger.initial == initial_balances[Account("Cash")]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal(1200))
    assert cash_ledger.entries[1].balance == Quantity(Decimal(1050))
    
    # Check Receivables ledger
    receivables_ledger = general_ledger.ledgers[Account("Receivables")]
    assert receivables_ledger.account == Account("Receivables")
    assert receivables_ledger.initial == initial_balances[Account("Receivables")]
    assert len(receivables_ledger.entries) == 2
    assert receivables_ledger.entries[0].balance == Quantity(Decimal(300))
    assert receivables_ledger.entries[1].balance == Quantity(Decimal(450))


# LLM-generated content at query #18
#--------------------------

Here's a unit test for the `__call__` method of `GeneralLedgerProgram`:


# LLM-generated content at query #19
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """
    Test the __call__ method of GeneralLedgerProgram.
    """
    from datetime import date
    from decimal import Decimal
    from typing import Dict
    from unittest.mock import Mock

    # Create mock objects for dependencies
    mock_read_initial_balances = Mock()
    mock_read_journal_entries = Mock()

    # Setup test data
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    test_account = Account("Test Account")
    test_initial_balances = {test_account: Balance(date(2023, 1, 1), Quantity(Decimal('1000')))}
    test_journal_entries = []

    # Configure mocks
    mock_read_initial_balances.return_value = test_initial_balances
    mock_read_journal_entries.return_value = test_journal_entries

    # Create the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )

    # Execute the program
    result = program(test_period)

    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 1
    assert test_account in result.ledgers
    assert result.ledgers[test_account].initial == test_initial_balances[test_account]
    assert len(result.ledgers[test_account].entries) == 0

    # Verify mocks were called correctly
    mock_read_initial_balances.assert_called_once_with(test_period)
    mock_read_journal_entries.assert_called_once_with(test_period)

    # Test with journal entries
    test_journal_entry = JournalEntry(
        date=date(2023, 1, 15),
        description="Test Entry",
        postings=[
            Posting(
                account=test_account,
                amount=Amount(Decimal('100')),
                direction=1
            )
        ]
    )
    mock_read_journal_entries.return_value = [test_journal_entry]
    result_with_entries = program(test_period)
    
    assert len(result_with_entries.ledgers[test_account].entries) == 1
    entry = result_with_entries.ledgers[test_account].entries[0]
    assert entry.posting == test_journal_entry.postings[0]
    assert entry.balance == Quantity(Decimal('1100'))


# LLM-generated content at query #20
#--------------------------

def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Create test accounts
    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(500)))
    }
    
    # Create test journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Test entry 1",
            [
                Posting(account1, Decimal(100), Direction.DEBIT),
                Posting(account2, Decimal(100), Direction.CREDIT)
            ]
        ),
        JournalEntry(
            datetime.date(2023, 2, 20),
            "Test entry 2",
            [
                Posting(account1, Decimal(200), Direction.CREDIT),
                Posting(account2, Decimal(200), Direction.DEBIT)
            ]
        )
    ]
    
    # Call the function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal(1100))
    assert ledger1.entries[1].balance == Quantity(Decimal(900))
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 2
    assert ledger2.entries[0].balance == Quantity(Decimal(400))
    assert ledger2.entries[1].balance == Quantity(Decimal(600))
    
    # Test with new account not in initial balances
    account3 = Account("3", "Account 3")
    journal_entries_with_new_account = [
        JournalEntry(
            datetime.date(2023, 3, 1),
            "Test entry with new account",
            [
                Posting(account1, Decimal(50), Direction.DEBIT),
                Posting(account3, Decimal(50), Direction.CREDIT)
            ]
        )
    ]
    
    general_ledger_with_new_account = build_general_ledger(
        period, 
        journal_entries_with_new_account, 
        initial_balances
    )
    
    assert len(general_ledger_with_new_account.ledgers) == 3
    assert account3 in general_ledger_with_new_account.ledgers
    ledger3 = general_ledger_with_new_account.ledgers[account3]
    assert ledger3.initial.value == Decimal(0)
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal(-50))


# LLM-generated content at query #21
#--------------------------

def test_GeneralLedgerProgram___call__():
    # Mock dependencies
    class MockReadInitialBalances:
        def __call__(self, period):
            return {Account("cash"): Balance(datetime.date(2023,1,1), Quantity(Decimal(100)))}

    class MockReadJournalEntries:
        def __call__(self, period):
            return [
                JournalEntry(
                    datetime.date(2023,1,2),
                    "Test entry",
                    [
                        Posting(Account("cash"), Amount(Decimal(50)), 1),
                        Posting(Account("revenue"), Amount(Decimal(50)), -1)
                    ]
                )
            ]

    # Test data
    period = DateRange(datetime.date(2023,1,1), datetime.date(2023,1,31))
    
    # Create program
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries()
    )
    
    # Execute program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2  # cash and revenue accounts
    
    # Check cash ledger
    cash_ledger = result.ledgers[Account("cash")]
    assert cash_ledger.account == Account("cash")
    assert cash_ledger.initial.value == Decimal(100)
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Decimal(150)
    
    # Check revenue ledger
    revenue_ledger = result.ledgers[Account("revenue")]
    assert revenue_ledger.account == Account("revenue")
    assert revenue_ledger.initial.value == Decimal(0)
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Decimal(50)


# LLM-generated content at query #22
#--------------------------

def test_build_general_ledger():
    # Mock data
    from datetime import date
    from decimal import Decimal

    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    account1 = Account("Cash", "Asset")
    account2 = Account("Revenue", "Income")
    initial_balances = {account1: Balance(start_date, Quantity(Decimal(1000)))}

    journal_entry = JournalEntry(
        date=date(2023, 6, 15),
        description="Sale",
        postings=[
            Posting(account1, Amount(Decimal(-500)), Direction.DEBIT),
            Posting(account2, Amount(Decimal(500)), Direction.CREDIT),
        ],
    )

    journal_entries = [journal_entry]

    # Build the general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert general_ledger.period == period

    assert account1 in general_ledger.ledgers
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal(500))

    assert account2 in general_ledger.ledgers
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(start_date, Quantity(Decimal(0)))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal(500))


# LLM-generated content at query #23
#--------------------------

```python
def test_build_general_ledger():
    # Setup test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("Account1")
    account2 = Account("Account2")
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000)))
    }
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 10),
        "Test Entry",
        [
            Posting(account1, Amount(Decimal(500)), Direction.DEBIT),
            Posting(account2, Amount(Decimal(500)), Direction.CREDIT)
        ]
    )
    journal_entries = [journal_entry]

    # Execute the function
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert general_ledger.period == period
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers

    # Check ledger for account1
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == Balance(datetime.date(2022, 12, 31), Quantity(Decimal(1000)))
    assert len(ledger1.entries) == 1
    entry1 = ledger1.entries[0]
    assert entry1.posting == Posting(account1, Amount(Decimal(500)), Direction.DEBIT)
    assert entry1.balance == Quantity(Decimal(1500))

    # Check ledger for account2
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))
    assert len(ledger2.entries) == 1
    entry2 = ledger2.entries[0]
    assert entry2.posting == Posting(account2, Amount(Decimal(500)), Direction.CREDIT)
    assert entry2.balance == Quantity(Decimal(500))


# LLM-generated content at query #24
#--------------------------

```python
def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("1234"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000))),
            Account("5678"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(2000))),
        }

    # Create an instance of ReadInitialBalances using the mock implementation
    read_initial_balances = ReadInitialBalances(mock_read_initial_balances)

    # Define a DateRange for the test
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

    # Call the __call__ method
    initial_balances = read_initial_balances(period)

    # Assert the expected results
    assert isinstance(initial_balances, dict)
    assert len(initial_balances) == 2
    assert Account("1234") in initial_balances
    assert Account("5678") in initial_balances
    assert initial_balances[Account("1234")].value == Quantity(Decimal(1000))
    assert initial_balances[Account("5678")].value == Quantity(Decimal(2000))


# LLM-generated content at query #25
#--------------------------

Here's a unit test for the `__call__` method of `GeneralLedgerProgram`:


# LLM-generated content at query #26
#--------------------------

def test_ReadInitialBalances___call__():
    # Mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            Account("Assets"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000))),
            Account("Liabilities"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(500))),
        }
    
    # Create an instance of ReadInitialBalances using the mock
    read_initial_balances = ReadInitialBalances(mock_read_initial_balances)
    
    # Define test period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Call the __call__ method
    result = read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Assets") in result
    assert Account("Liabilities") in result
    assert result[Account("Assets")].value == Quantity(Decimal(1000))
    assert result[Account("Liabilities")].value == Quantity(Decimal(500))


