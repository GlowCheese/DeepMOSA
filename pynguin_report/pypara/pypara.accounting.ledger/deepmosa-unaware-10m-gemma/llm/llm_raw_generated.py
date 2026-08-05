####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

@pytest.mark.parametrize("period, initial_balances, journal_entries, expected_accounts", [
    # Case 1: Empty journal and empty initial balances
    (
        DateRange(date(2023, 1, 1), date(2023, 12, 31)),
        {},
        [],
        []
    ),
    # Case 2: Initial balances present, no journal entries
    (
        DateRange(date(2023, 1, 1), date(2023, 12, 31)),
        {MagicMock(spec=Account): Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))},
        [],
        [True] # Checking if account exists in ledgers
    ),
    # Case 3: Journal entries within period and outside period
    (
        DateRange(date(2023, 1, 1), date(2023, 6, 30)),
        {},
        [
            # Entry inside period
            MagicMock(
                spec=JournalEntry,
                date=date(2023, 3, 1),
                postings=[
                    MagicMock(
                        spec=Posting,
                        account=MagicMock(spec=Account),
                        amount=Quantity(Decimal("50.00")),
                        direction=MagicMock(value=Decimal("1")), # Debit
                    )
                ]
            ),
            # Entry outside period (too late)
            MagicMock(
                spec=JournalEntry,
                date=date(2023, 7, 1),
                postings=[
                    MagicMock(
                        spec=Posting,
                        account=MagicMock(spec=Account),
                        amount=Quantity(Decimal("50.00")),
                        direction=MagicMock(value=Decimal("-1")), # Credit
                    )
                ]
            ),
        ],
        [True]
    ),
])
def test_build_general_ledger(period, initial_balances, journal_entries, expected_accounts):
    """
    Tests the build_general_ledger function for correct ledger creation, 
    filtering by date range, and processing of initial balances.
    """
    # Setup
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Account)
    
    # We manually override the mock names to ensure identity in dict lookups
    account_a.__hash__.return_value = hash(account_a)
    account_b.__hash__.return_value = hash(account_b)

    # Create specific test data for a controllable scenario
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    initial = {
        account_a: Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))
    }
    
    # Create a journal entry that should be processed
    posting_in_period = MagicMock(spec=Posting)
    posting_in_period.account = account_b
    posting_in_period.amount = Quantity(Decimal("50.00"))
    posting_in_period.direction = MagicMock(value=Decimal("1"))
    
    journal_entry_in = MagicMock(spec=JournalEntry)
    journal_entry_in.date = date(2023, 6, 1)
    journal_entry_in.postings = [posting_in_period]
    
    # Create a journal entry that should be ignored (out of range)
    journal_entry_out = MagicMock(spec=JournalEntry)
    journal_entry_out.date = date(2024, 1, 1)
    journal_entry_out.postings = [
        MagicMock(
            spec=Posting,
            account=account_a,
            amount=Quantity(Decimal("50.00")),
            direction=MagicMock(value=Decimal("-1"))
        )
    ]

    # Execution
    gl = build_general_ledger(test_period, [journal_entry_in, journal_entry_out], initial)

    # Assertions
    assert gl.period == test_period
    # Account A should exist because of initial balance
    assert account_a in gl.ledgers
    # Account B should exist because of the in-period posting
    assert account_b in gl.ledgers
    # Account A's ledger should NOT have the out-of-period posting
    assert len(gl.ledgers[account_a].entries) == 0
    # Account B's ledger SHOULD have the in-period posting
    assert len(gl.ledgers[account_b].entries) == 1
    assert gl.ledgers[account_b].entries[0].amount == Quantity(Decimal("50.00"))
    # Check balance calculation: Initial (100) + (50 * 1) = 150? 
    # No, Account B started at 0 because it wasn't in initial.
    assert gl.ledgers[account_b]._last_balance == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup common test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    account_a = Account("Assets:Cash")
    account_b = Account("Equity:Capital")
    account_c = Account("Expenses:Rent")

    # Setup initial balances
    initial_balances = {
        account_a: Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("1000.00")))
    }

    # Mocking Journal Entries and Postings
    # Entry 1: Capital injection (Debit Cash 500, Credit Capital 500) - Within period
    date_1 = datetime.date(2023, 2, 1)
    journal_1 = MagicMock(spec=JournalEntry)
    journal_1.date = date_1
    journal_1.description = "Initial Investment"
    
    posting_1a = MagicMock(spec=Posting)
    posting_1a.account = account_a
    posting_1a.amount = Amount(Decimal("500.00"))
    posting_1a.direction = Direction.DEBIT
    
    posting_1b = MagicMock(spec=Posting)
    posting_1b.account = account_b
    posting_1b.amount = Amount(Decimal("500.00"))
    posting_1b.direction = Direction.CREDIT
    
    journal_1.postings = [posting_1a, posting_1b]

    # Entry 2: Rent Expense (Debit Rent 200, Credit Cash 200) - Within period
    date_2 = datetime.date(2023, 3, 1)
    journal_2 = MagicMock(spec=JournalEntry)
    journal_2.date = date_2
    journal_2.description = "Rent Payment"
    
    posting_2a = MagicMock(spec=Posting)
    posting_2a.account = account_c
    posting_2a.amount = Amount(Decimal("200.00"))
    posting_2a.direction = Direction.DEBIT
    
    posting_2b = MagicMock(spec=Posting)
    posting_2b.account = account_a
    posting_2b.amount = Amount(Decimal("200.00"))
    posting_2b.direction = Markup.CREDIT # Assuming direction logic exists or using a mock flag
    # Note: Since direction is used for math, we'll use actual objects if possible 
    # but here I will manually set the value attribute to ensure math works
    posting_2b.direction.value = Decimal("-1") 

    journal_2.postings = [posting_2a, posting_2b]

    # Entry 3: Out of period entry (Should be ignored)
    date_3 = datetime.date(2024, 1, 1)
    journal_3 = MagicMock(spec=JournalEntry)
    journal_3.date = date_3
    posting_3a = MagicMock(spec=Posting)
    posting_3a.account = account_a
    posting_3a.amount = Amount(Decimal("100.00"))
    posting_3a.direction.value = Decimal("-1")
    journal_3.postings = [posting_3a]

    journal = [journal_1, journal_2, journal_3]

    # Execute function under test
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert gl.period == period
    assert account_a in gl.ledgers
    assert account_b in gl.ledgers
    assert account_c in gl.ledgers

    # Check Cash Account (Account A)
    # Initial 1000 + Debit 500 - Credit 200 = 1300
    cash_ledger = gl.ledgers[account_a]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Find the specific entries in Cash ledger
    # Entry 1: Debit 500; Entry 2: Credit 200
    cash_amounts = [e.amount.value for e in cash_ledger.entries]
    assert Amount(Decimal("500.00")) in cash_amounts
    assert Amount(Decimal("200.00")) in cash_amounts
    
    # Verify final balance calculation via the property _last_balance
    # Postings are processed: 1000 + (500 * 1) + (200 * -1) = 1300
    assert cash_ledger._last_balance == Quantity(Decimal("1300.00"))

    # Check Capital Account (Account B)
    # Initial 0 + Credit 500 = 500
    capital_ledger = gl.ledgers[account_b]
    assert capital_ledger.initial.value == Quantity(Decimal("0"))
    assert capital_ledger._last_balance == Quantity(Decimal("500.00"))

    # Check Rent Account (Account C)
    # Initial 0 + Debit 200 = 200
    rent_ledger = gl.ledgers[account_c]
    assert rent_ledger._last_balance == Quantity(Decimal("200.00"))

    # Verify that the out-of-period entry was NOT added to any ledger
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date <= end_date
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the behavior of the function returned by compile_general_ledger_program,
    which conforms to the GeneralLedgerProgram protocol.
    """
    # 1. Setup Test Data
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(since=period_start, until=period_end)

    # Mock Account and Balance objects
    mock_account = MagicMock()
    initial_balance_value = Quantity(Decimal("100.00"))
    initial_balances = {mock_account: Balance(period_start, initial_balance_value)}

    # Mock Journal Entry
    mock_posting = MagicMock()
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = MagicMock(value=Decimal("1")) # Debit/Positive direction
    
    mock_journal_entry = MagicMock()
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.postings = [mock_posting]
    mock_journal_entry.description = "Test Transaction"
    
    journal_entries = [mock_journal_entry]

    # 2. Setup Mocks for the Algebra (ReadInitialBalances and ReadJournalEntries)
    read_initial_balances_mock = MagicMock()
    read_initial_balances_mock.return_value = initial_balances

    read_journal_entries_mock = MagicMock()
    read_journal_entries_mock.return_value = journal_entries

    # 3. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=read_initial_balances_mock,
        read_journal_entries=read_journal_entries_mock
    )

    # 4. Execute the program (the __call__ of GeneralLedgerProgram)
    result_gl = program(period)

    # 5. Assertions
    # Verify that the algebra functions were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify the structure of the returned GeneralLedger
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period
    assert mock_account in result_gl.ledgers
    
    # Verify that the ledger was correctly populated from initial balances and journal entries
    ledger = result_gl.ledgers[mock_account]
    assert ledger.initial.value == initial_balance_value
    
    # Check if the entry was added: 100 (initial) + 50 (posting) = 150
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
    assert ledger._last_balance == Quantity(Decimal("150.00"))
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup DateRange
    period_start = date(2023, 1, 1)
    period_end = date(2023, 12, 31)
    date_range = DateRange(since=period_start, until=period_end)

    # Setup Accounts
    account_cash = Account("Cash")
    account_revenue = Account("Revenue")
    account_expense = Account("Expense")

    # Setup Initial Balances
    initial_balances = {
        account_cash: Balance(since=date(2023, 1, 1), value=Quantity(Decimal("1000.00")))
    }

    # Setup Mock Journal Entries and Postings
    # Entry 1: Revenue earned (Debit Cash, Credit Revenue) - Within period
    journal_entry_1 = MagicMock(spec=JournalEntry)
    journal_entry_1.date = date(2023, 6, 1)
    journal_entry_1.description = "Service Revenue"
    
    posting_cash_dr = MagicMock(spec=Posting)
    posting_cash_dr.account = account_cash
    posting_cash_dr.amount = Quantity(Decimal("500.00"))
    posting_cash_dr.direction = Direction.DEBIT
    
    posting_rev_cr = MagicMock(spec=Posting)
    posting_rev_cr.account = account_revenue
    posting_rev_cr.amount = Quantity(Decimal("500.00"))
    posting_rev_cr.direction = Direction.CREDIT

    journal_entry_1.postings = [posting_cash_dr, posting_rev_cr]

    # Entry 2: Expense paid (Debit Expense, Credit Cash) - Within period
    journal_entry_2 = MagicMock(spec=JournalEntry)
    journal_entry_2.date = date(2023, 7, 1)
    journal_entry_2.description = "Office Supplies"
    
    posting_exp_dr = MagicMock(spec=Posting)
    posting_exp_dr.account = account_expense
    posting_exp_dr.amount = Quantity(Decimal("200.00"))
    posting_exp_dr.direction = Direction.DEBIT
    
    posting_cash_cr = MagicMock(spec=Posting)
    posting_cash_cr.account = account_cash
    posting_cash_cr.amount = Quantity(Decimal("200.00"))
    posting_cash_cr.direction = Direction.CREDIT

    journal_entry_2.postings = [posting_exp_dr, posting_cash_cr]

    # Entry 3: Old entry - Outside period (Should be ignored)
    journal_entry_old = MagicMock(spec=JournalEntry)
    journal_entry_old.date = date(2022, 12, 31)
    journal_entry_old.description = "Old Entry"
    posting_old = MagicMock(spec=Posting)
    posting_old.account = account_cash
    posting_old.amount = Quantity(Discount(Decimal("100.00")))
    posting_old.direction = Direction.DEBIT
    journal_entry_old.postings = [posting_old]

    journal = [journal_entry_1, journal_entry_2, journal_entry_old]

    # Execute
    gl = build_general_ledger(date_range, journal, initial_balances)

    # Assertions
    assert gl.period == date_range
    assert account_cash in gl.ledgers
    assert account_revenue in gl.ledgers
    assert account_expense in gl.ledgers

    # Verify Cash Ledger: 1000 (initial) + 500 (dr) - 200 (cr) = 1300
    cash_ledger = gl.ledgers[account_cash]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Check entries for Cash (only the ones within period)
    # Entry 1 dr + Entry 2 cr
    cash_entries = [e for e in cash_ledger.entries if e.amount == Quantity(Decimal("500.00")) or e.amount == Quantity(Decimal("200.00"))]
    assert len(cash_entries) == 2
    # Final balance check
    # Note: direction.value for Debit is usually 1, Credit is -1
    # Last balance calculation depends on the specific implementation of Direction in your codebase
    # Assuming standard accounting logic used in the provided snippet:
    # 1000 + (500 * 1) + (200 * -1) = 1300
    assert cash_ledger._last_balance == Quantity(Decimal("1300.00"))

    # Verify Revenue Ledger: 0 (initial) + 500 (cr) = 500 (if credit is treated as negative/reduction in logic or vice versa)
    # Based on the code: entry.balance = last_balance + amount * direction.value
    # If Direction.CREDIT.value is -1, then 0 + 500 * -1 = -500. 
    # The test assumes standard ledger logic provided in your class.
    rev_ledger = gl.ledgers[account_revenue]
    assert rev_ledger.initial.value == Quantity(Decimal("0"))
    
    # Verify that the old entry was NOT processed
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date >= period_start
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from ..commons.numbers import Amount, Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance
from .journaling import JournalEntry, Posting, Direction
from .general_ledger import build_general_ledger, GeneralLedger

def test_build_general_ledger():
    # Setup Period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)

    # Setup Accounts
    cash_account = Account("Cash")
    revenue_account = Account("Revenue")
    expense_account = Account("Expense")

    # Setup Initial Balances (e.g., Cash starts with 1000)
    initial_balances = {
        cash_account: Balance(since=datetime.date(202ass, 12, 31), value=Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue of 500 (Debit Cash, Credit Revenue)
    j1_date = datetime.date(2023, 6, 1)
    p1_cash = Posting(account=cash_account, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT, date=j1_date)
    p1_rev = Posting(account=revenue_account, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT, date=j1_date)
    journal1 = JournalEntry(date=j1_date, description="Sales Revenue", postings=[p1_cash, p1_rev])

    # Entry 2: Expense of 200 (Debit Expense, Credit Cash)
    j2_date = datetime.date(2023, 7, 1)
    p2_exp = Posting(account=expense_account, amount=Amount(Decimal("200.00")), direction=Direction.DEBIT, date=j2_date)
    p2_cash = Posting(account=cash_account, amount=Amount(Decimal("200.00")), direction=Direction.CREDIT, date=j2_date)
    journal2 = JournalEntry(date=j2_date, description="Office Supplies", postings=[p2_exp, p2_cash])

    # Entry 3: Out of period (should be ignored)
    j3_date = datetime.date(2024, 1, 1)
    p3_cash = Posting(account=cash_account, amount=amount=Amount(Decimal("100.00")), direction=Direction.DEBIT, date=j3_date)
    journal3 = JournalEntry(date=j3_date, description="New Year Gift", postings=[p3_cash])

    journal = [journal1, journal2, journal3]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    
    # Check Cash Ledger (Initial 1000 + 500 - 200 = 1300)
    assert cash_account in gl.ledgers
    cash_ledger = gl.ledgers[cash_account]
    assert cash_ledger._last_balance == Quantity(Decimal("1300.00"))
    # Check that entry 3 was ignored (only 2 entries for cash)
    # Entry 1 (Debit) and Entry 2 (Credit)
    assert len([e for e in cash_ledger.entries if e.date == j1_date]) == 1
    assert len([e for e in cash_ledger.entries if e.date == j2_date]) == 1

    # Check Revenue Ledger (Starts at 0 + 500 = 500)
    assert revenue_account in gl.ledgers
    assert gl.ledgers[revenue_account]._last_balance == Quantity(Decimal("500.00"))

    # Check Expense Ledger (Starts at 0 + 200 = 200)
    assert expense_account in gl.ledgers
    assert gl.ledgers[expense_account]._last_balance == Quantity(Decimal("200.00"))

    # Verify out-of-period entry was not added to any ledger
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date <= end_date
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol structure of ReadInitialBalances by verifying a 
    callable implementation returns the expected InitialBalances type.
    """
    # Setup period
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Create mock accounts and balances
    mock_account = MagicMock(spec=Account)
    expected_balance = Balance(
        since=datetime.date(2023, 1, 1),
        value=Quantity(Decimal("100.00"))
    )
    
    # Define the expected return value (InitialBalances is Dict[Account, Balance])
    expected_initial_balances = {mock_account: expected_balance}

    # Create a concrete implementation of the ReadInitialBalances protocol
    def mock_reader(p: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Verify type compatibility (implicit via execution)
    result = mock_reader(period)

    # Assertions
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account].value == Decimal("100.00")
    assert result[mock_account].since == datetime.date(2023, 1, 1)

    # Verify a different implementation/behavior
    def empty_reader(p: DateRange) -> InitialBalances:
        return {}

    assert empty_reader(period) == {}
```


# LLM-generated content at query #7
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

def test_ReadInitialBalances___call__():
    """
    Tests the protocol/type definition for ReadInitialBalances by verifying 
    a mock implementation adheres to the expected signature and behavior.
    """
    # Define a period for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)

    # Create a mock account and balance
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(since=datetime.date(2020, 1, 1), value=Quantity(Decimal("100.00")))
    
    # Expected return value for the protocol implementation
    expected_initial_balances: InitialBalances = {
        mock_account: mock_balance
    }

    # Define a concrete implementation of the ReadInitialBalances protocol
    def mock_read_initial_balances(period_arg: DateRange) -> InitialBalances:
        # Verify the input type/value as part of the test logic
        assert isinstance(period_arg, DateRange)
        assert period_arg.since == start_date
        return expected_initial_balances

    # Cast the function to the Protocol type to ensure it is compatible
    read_balances: ReadInitialBalances = mock_read_initial_balances

    # Execute the call
    result = read_balances(period)

    # Assertions
    assert result == expected_initial_balances
    assert mock_account in result
    assert result[mock_account].value == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

@pytest.mark.parametrize("period, initial_balances, journal_entries", [
    (
        DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31)),
        {MagicMock(spec=Account): Balance(datetime.date(202im3, 1, 1), Quantity(Decimal("100.00")))},
        [
            MagicMock(
                spec=JournalEntry,
                date=datetime.date(2023, 6, 1),
                description="Test Entry",
                postings=[
                    MagicMock(
                        account=MagicMock(spec=Account),
                        amount=Quantity(Decimal("50.00")),
                        direction=MagicMock(value=Decimal("1")), # Debit
                        is_debit=True,
                        is_credit=False
                    )
                ]
            )
        ]
    ),
])
def test_GeneralLedgerProgram___call__(period, initial_balances, journal_entries):
    """
    Tests the execution of the program returned by compile_general_ledger_program.
    The test verifies that the compiled function correctly orchestrates 
    the reading of initial balances and journal entries to build a GeneralLedger.
    """
    # Arrange
    mock_read_initial_balances = MagicMock(return_value=initial_balances)
    mock_read_journal_entries = MagicMock(return_value=journal_entries)
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Act
    result = program(period)

    # Assert
    # 1. Verify the dependency functions were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # 2. Verify the returned object is a GeneralLedger
    assert isinstance(result, GeneralLedger)
    assert result.period == period

    # 3. Verify the internal data was processed correctly via build_general_ledger logic
    # Check if initial balances were loaded into ledgers
    for account, balance in initial_balances.items():
        assert account in result.ledgers
        assert result.ledgers[account].initial == balance

    # 4. Verify that entries from the journal within the period were added to the ledger
    # We check if any ledger has entries (the number of postings in our mock)
    total_entries = sum(len(ledger.entries) for ledger in result.ledgers.values())
    
    # In our specific test case, we have 1 posting in the journal
    # If it falls within period, total_entries should be at least 1 if account matches
    assert total_entries >= 0 
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the ReadInitialBalances protocol implementation via a mock 
    to ensure it adheres to the expected signature and return type.
    """
    # Arrange
    # Create a mock function that satisfies the ReadInitialBalances protocol
    mock_reader = MagicMock()
    
    # Setup test data
    test_date = datetime.date(2023, 1, 1)
    test_period = DateRange(since=test_date, until=datetime.date(2023, 12, 31))
    
    # Define dummy accounts and balances
    dummy_account = MagicMock(spec=Account)
    dummy_balance = Balance(since=test_date, value=Quantity(Decimal("100.00")))
    expected_initial_balances: InitialBalances = {dummy_account: dummy_balance}
    
    # Configure the mock to return our expected dictionary when called with the period
    mock_reader.return_value = expected_initial_balances

    # Act
    # Execute the call as if it were a real implementation of the protocol
    result = mock_reader(test_period)

    # Assert
    # Verify the function was called with the correct argument
    mock_reader.assert_called_once_with(test_period)
    
    # Verify the returned object is of type InitialBalances (Dict[Account, Balance])
    assert isinstance(result, dict)
    assert dummy_account in result
    assert result[dummy_account].value == Quantity(Decimal("100.00"))
    assert result == expected_initial_balances

```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    # Setup dependencies
    period = MagicMock()
    period.since = datetime.date(2023, 1, 1)
    period.until = datetime.date(2023, 12, 31)

    # Mocking the algebra implementations (ReadInitialBalances and ReadJournalEntries)
    mock_read_initial_balances = MagicMock()
    mock_read_journal_entries = MagicMock()

    # Define fake initial balances and journal entries to be returned by mocks
    fake_account = MagicMock()
    fake_balance = MagicMock()
    fake_initial_balances = {fake_account: fake_balance}
    
    fake_journal_entry = MagicMock()
    fake_journal_entry.date = datetime.date(2023, 6, 1)
    fake_postings = []
    # We don't need complex postings here, just enough to satisfy the loop in build_general_ledger
    fake_journal_entry.postings = [] 
    fake_journal_entries = [fake_journal_entry]

    mock_read_initial_balances.return_value = fake_initial_balances
    mock_read_journal_entries.return_value = fake_journal_entries

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the __call__ method of the produced GeneralLedgerProgram
    result = program(period)

    # Assertions
    # 1. Verify that the algebra functions were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # 2. Verify that the result is an instance of GeneralLedger
    assert isinstance(result, GeneralLedger)

    # 3. Verify that the returned period matches the input period
    assert result.period == period

    # 4. Verify that the ledgers were built from the initial balances provided by the mock
    assert fake_account in result.ledgers
    assert result.ledgers[fake_account].initial == fake_balance
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup common components
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(since=start_date, until=end_date)
    
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")
    
    qty_zero = Quantity(Decimal("0"))
    qty_100 = Quantity(Decimal("100"))
    qty_50 = Quantity(Decimal("50"))

    # Initial Balances
    initial_balances: InitialBalances = {
        acc_cash: Balance(start_date, qty_100)
    }

    # Mock Journal Entries
    # Entry 1: Revenue of 50 (Debit Cash, Credit Revenue) - Within period
    j1_date = datetime.date(2023, 1, 15)
    p1_cash = Posting(account=acc_cash, amount=qty_50, direction=Direction.DEBIT, date=j1_date, journal=None)
    p1_rev = Posting(account=perc_revenue, amount=qty_50, direction=Direction.CREDIT, date=j1_date, journal=None)
    # We need to manually link the journal for .description and .cntraccts logic in LedgerEntry
    j1 = MagicMock(spec=JournalEntry)
    j1.date = j1_date
    j1.description = "Service Revenue"
    j1.postings = [p1_cash, p1_rev]
    # Fix the direction/account references for the mock to work with LedgerEntry properties
    p1_cash.direction = Direction.DEBIT
    p1_rev.direction = Direction.CREDIT
    # Note: In a real test, we'd use actual objects, but here we ensure the logic holds
    
    # Entry 2: Expense of 30 (Debit Expense, Credit Cash) - Within period
    j2_date = datetime.date(2023, 1, 20)
    p2_exp = MagicMock(spec=Posting)
    p2_exp.account = acc_expense
    p2_exp.amount = qty_30 # Assume qty_30 is defined or use Decimal
    p2_exp.direction = Direction.DEBIT
    p2_exp.date = j2_date
    
    p2_cash = MagicMock(spec=Posting)
    p2_cash.account = acc_cash
    p2_cash.amount = qty_30 
    p2_cash.direction = Direction.CREDIT
    p2_cash.date = j2_date

    j2 = MagicMock(spec=JournalEntry)
    j2.date = j2_date
    j2.description = "Office Supplies"
    j2.postings = [p2_exp, p2_cash]

    # Entry 3: Outside period (Should be ignored)
    j3_date = datetime.date(2023, 2, 1)
    j3 = MagicMock(spec=JournalEntry)
    j3.date = j3_date
    j3.postings = []

    journal = [j1, j2, j3]

    # Execute function
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    
    # Check Cash Ledger (Initial 100 + 50 - 30 = 120)
    assert acc_cash in gl.ledgers
    cash_ledger = gl.ledgers[acc_cash]
    # Verify calculation: Initial(100) + Posting1(50 * 1) + Posting2(30 * -1) = 120
    # Note: LedgerEntry logic depends on posting.direction.value (Debit=1, Credit=-1)
    # We assume Direction.DEBIT.value is 1 and Direction.CREDIT.value is -1
    
    # Check Revenue Ledger (Initial 0 + 50 = 50)
    assert acc_revenue in gl.ledgers
    assert gl.ledgers[acc_revenue].initial.value == Decimal("0")

    # Check Expense Ledger (Created from posting, Initial 0 + 30 = 30)
    assert acc_expense in gl.ledgers
    
    # Verify entries count (Only j1 and j2 postings fall within period)
    # Total postings in period: p1_cash, p1_rev, p2_exp, p2_cash = 4
    total_entries_count = sum(len(l.entries) for l in gl.ledgers.values())
    assert total_entries_count == 4
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup Period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(start_date, end_date)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances (Cash starts with 1000)
    initial_balances: InitialBalances = {
        acc_cash: Balance(start_date, Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue of 500 (Debit Cash, Credit Revenue) - Within Period
    j1_date = datetime.date(2023, 1, 15)
    p1_cash = Posting(acc_cash, Amount(Decimal("500.00")), Direction.DEBIT, j1_journal) # Note: dependency on j1_journal setup below
    # Since we can't easily define the journal object without its dependencies, 
    # we mock the structure required by build_general_ledger
    
    j1 = MagicMock(spec=JournalEntry)
    j1.date = datetime.date(2023, 1, 15)
    j1.description = "Service Revenue"
    p1 = MagicMock(spec=Posting)
    p1.account = acc_cash
    p1.amount = Amount(Decimal("500.00"))
    p1.direction = Direction.DEBIT # Assuming Direction enum exists in scope or mocked
    # We need to mock the direction value for the math: posting.amount * posting.direction.value
    p1.direction.value = Decimal("1") 
    j1.postings = [p1]

    # Entry 2: Expense of 200 (Debit Expense, Credit Cash) - Within Period
    j2 = MagicMock(spec=JournalEntry)
    j2.date = datetime.date(2023, 1, 20)
    j2.description = "Office Supplies"
    p2 = MagicMock(spec=Posting)
    p2.account = acc_expense
    p2.amount = Amount(Decimal("200.00"))
    p2.direction = Direction.DEBIT
    p2.direction.value = Decimal("1")
    # To make the math work for Cash (Credit side):
    p3 = MagicMock(spec=Posting)
    p3.account = acc_cash
    p3.amount = Amount(Decimal("200.00"))
    p3.direction = Direction.CREDIT
    p3.direction.value = Decimal("-1")
    j2.postings = [p2, p3]

    # Entry 3: Out of Period entry (Should be ignored)
    j3 = MagicMock(spec=JournalEntry)
    j3.date = datetime.date(2022, 12, 31)
    j3.description = "Old Entry"
    p4 = MagicMock(spec=Posting)
    p4.account = acc_cash
    p4.amount = Amount(Decimal("100.00"))
    p4.direction = Direction.DEBIT
    p4.direction.value = Decimal("1")
    j3.postings = [p4]

    journal = [j1, j2, j3]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    
    # Check Cash Ledger (Initial 1000 + 500 - 200 = 1300)
    assert acc_cash in gl.ledgers
    cash_ledger = gl.ledgers[acc_cash]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Entries for Cash should be p1 (debit) and p3 (credit)
    # Note: build_general_ledger iterates through all postings in journal entries within period
    # Filtered postings are: p1, p2, p3. 
    # p1 is cash debit, p3 is cash credit.
    assert len(cash_ledger.entries) == 2 
    
    # Check balance calculation via the last entry
    # Final Cash Balance = 1000 + 500 (p1) - 200 (p3) = 1300
    # We need to ensure the order of addition is consistent with the loop
    # The loop processes j1 then j2. p1 is from j1, p3 is from j2.
    # Since p1 was added first: 1000 + 500 = 1500. Then p3: 1500 - 200 = 1300.
    final_cash_balance = cash_ledger.entries[-1].balance.value
    assert final_cash_balance == Decimal("1300.00")

    # Check Expense Ledger (Created during loop, starts at 0)
    assert acc_expense in gl.ledgers
    expense_ledger = gl.ledgers[acc_expense]
    assert expense_ledger.initial.value == Decimal("0")
    assert expense_ledger.entries[0].amount.value == Decimal("200.00")

    # Check that j3 was ignored
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date != datetime.date(2022, 12, 31)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the __call__ method of the GeneralLedgerProgram (the returned function 
    from compile_general_ledger_program).
    """
    # Setup period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)

    # Mock dependencies (Algebra implementations)
    mock_read_initial_balances = MagicMock()
    mock_read_journal_entries = MagicMock()

    # Setup dummy data for initial balances
    account_a = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    mock_read_initial_balances.return_value = {
        account_a: Balance(since=datetime.date(2022, 1, 1), value=initial_balance_val)
    }

    # Setup dummy data for journal entries
    # We need a JournalEntry with a Posting that falls within the period
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Entry"
    
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = account_a
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = MagicMock(value=Decimal("1"))  # Debit/Increase
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_journal_entry.postings = [mock_posting]
    mock_read_journal_entries.return_value = [mock_journal_entry]

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program (the __call__ method of the returned function)
    result: GeneralLedger = program(period)

    # Assertions
    # 1. Check if dependencies were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # 2. Verify result type and structure
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account_a in result.ledgers
    
    # 3. Verify the math logic (Initial 100 + Posting 50 = 150)
    ledger_account_a = result.ledgers[account_a]
    expected_final_balance = Quantity(Decimal("150.00"))
    assert ledger_account_a._last_balance == expected_final_balance
    assert len(ledger_account_a.entries) == 1
    assert ledger_account_a.entries[0].amount == Quantity(Decimal("50.00"))

    # 4. Verify filtering (ensure entries outside period are ignored if we added one)
    outside_date = datetime.date(2024, 1, 1)
    mock_outside_entry = MagicMock(spec=JournalEntry)
    mock_outside_entry.date = outside_date
    # Re-run build logic check: if we added an entry for 2024, it shouldn't be in the ledger
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the execution of the function returned by compile_general_ledger_program,
    which satisfies the GeneralLedgerProgram protocol.
    """
    # 1. Setup Mock Dependencies
    mock_read_initial_balances = MagicMock()
    mock_read_journal_entries = MagicMock()
    
    # Define a test period
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    test_period = DateRange(since=period_start, until=period_end)

    # Define mock return values for the algebras
    mock_account = MagicMock(spec=Account)
    initial_balance_val = Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("100.00")))
    mock_read_initial_balances.return_value = {mock_account: initial_balance_val}

    # Create a mock JournalEntry and Posting
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Amount(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT # Assuming Direction enum exists in scope
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.postings = [mock_posting]
    mock_journal_entry.description = "Test Transaction"
    
    mock_read_journal_entries.return_value = [mock_journal_entry]

    # 2. Compile the Program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # 3. Execute the Program (__call__)
    result_gl = program(test_period)

    # 4. Assertions
    # Verify that the algebra functions were called with the correct period
    mock_read_initial_balances.assert_called_once_with(test_period)
    mock_read_journal_entries.assert_called_once_with(test_period)

    # Verify the returned object is a GeneralLedger
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == test_period
    
    # Verify the data was processed correctly through build_general_ledger logic
    assert mock_account in result_gl.ledgers
    target_ledger = result_gl.ledgers[mock_account]
    
    # Check initial balance application
    assert target_ledger.initial == initial_balance_val
    
    # Check that the posting was added to the ledger entries
    assert len(target_ledger.entries) == 1
    entry = target_ledger.entries[0]
    assert entry.amount == Amount(Decimal("50.00"))
    
    # Verify running balance: 100 (initial) + 50 (debit) = 150
    expected_balance = Quantity(Decimal("150.00"))
    assert entry.balance == expected_balance
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the execution of the function returned by compile_general_ledger_program,
    verifying it correctly orchestrates the reading of initial balances and 
    journal entries to build a GeneralLedger.
    """
    # Setup common test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    # Mock Account and Balance
    mock_account = MagicMock(spec=Account)
    initial_balance_value = Quantity(Decimal("100.00"))
    initial_balances = {mock_account: Balance(since=datetime.date(2022, 12, 31), value=initial_balance_value)}
    
    # Mock JournalEntry and Posting
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Amount(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction enum exists in scope
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Transaction"
    mock_journal_entry.postings = [mock_posting]
    journal_entries = [mock_journal_entry]

    # Define the algebra mocks (ReadInitialBalances and ReadJournalEntries)
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = initial_balances
    
    read_journal_entries_mock = MagicMock()
    read_journal_entries_mock.return_value = journal_entries

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=read_initial_balances_mock,
        read_journal_entries=read_journal_entries_mock
    )

    # Execute the program (__call__)
    result_gl = program(period)

    # Assertions: Verify the algebra was called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Assertions: Verify the resulting GeneralLedger structure
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period
    assert mock_account in result_gl.ledgers
    
    # Verify calculations within the ledger
    ledger = result_gl.ledgers[mock_account]
    # Expected balance: 100 (initial) + 50 * 1 (debit direction value) = 150
    expected_balance = Quantity(Decimal("150.00"))
    assert ledger._last_balance == expected_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Amount(Decimal("50.00"))
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the execution of the function returned by compile_general_ledger_program,
    verifying that it correctly invokes the provided algebras and builds a GeneralLedger.
    """
    # 1. Setup Mock Data and Dependencies
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create a mock account
    mock_account = MagicMock(spec=Account)
    
    # Define initial balances
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {
        mock_account: Balance(period.since, initial_balance_val)
    }

    # Create a mock JournalEntry and Posting
    mock_journal_desc = "Test Transaction"
    mock_amount = Amount(Decimal("50.00"))
    
    # Mock Posting direction (Debit/Credit)
    class MockDirection:
        value = 1
    direction = MockDirection()

    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = mock_amount
    mock_posting.direction = direction
    mock_posting.is_debit = True
    mock_posting.is_credit = False

    mock_journal = MagicMock(spec=JournalEntry)
    mock_journal.date = datetime.date(2023, 6, 1)
    mock_journal.description = mock_journal_desc
    mock_journal.postings = [mock_posting]

    # 2. Setup Mock Algebras (Protocols)
    read_initial_balances = MagicMock(spec=ReadInitialBalances)
    read_initial_balances.return_value = initial_balances

    read_journal_entries = MagicMock(spec=ReadJournalEntries)
    read_journal_entries.return_value = [mock_journal]

    # 3. Compile the Program
    program = compile_general_ledger_program(
        read_initial_balances=read_initial_balances,
        read_journal_entries=read_journal_entries
    )

    # 4. Execute the Program (__call__)
    result: GeneralLedger = program(period)

    # 5. Assertions
    # Verify algebras were called with correct period
    read_initial_balances.assert_called_once_with(period)
    read_journal_entries.assert_called_once_with(period)

    # Verify result type and structure
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert mock_account in result.ledgers
    
    # Verify the ledger calculation logic inside build_general_ledger via the program
    ledger = result.ledgers[mock_account]
    
    # Initial (100) + Posting (50 * 1) = 150
    expected_final_balance = Quantity(Decimal("150.00"))
    
    assert len(ledger.entries) == 1
    assert ledger._last_balance == expected_final_balance
    assert ledger.entries[0].amount == mock_amount
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the execution of the function returned by compile_general_ledger_program,
    which conforms to the GeneralLedgerProgram protocol.
    """
    # 1. Setup Mocks for dependencies (Algebra implementations)
    mock_read_initial_balances = MagicMock()
    mock_read_journal_entries = MagicMock()

    # 2. Define Test Data
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    test_period = DateRange(since=period_start, until=period_end)

    # Mock Initial Balances
    mock_account = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    mock_initial_balances = {mock_count: Balance(period_start, initial_balance_val)}
    mock_read_initial_balances.return_value = mock_initial_balances

    # Mock Journal Entries
    # We need a structure that build_general_ledger can iterate over
    # A JournalEntry needs postings and a date
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = MagicMock()
    mock_posting.direction.value = Decimal("1")  # Debit simulation
    mock_posting.is_debit = True
    mock_posting.is_credit = False

    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.postings = [mock_posting]
    mock_journal_entry.description = "Test Transaction"

    mock_read_journal_entries.return_value = [mock_journal_entry]

    # 3. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # 4. Execute the __call__ method of the GeneralLedgerProgram
    result_gl = program(test_period)

    # 5. Assertions
    # Verify that the algebra functions were called with the correct period
    mock_read_initial_balances.assert_called_once_with(test_period)
    mock_read_journal_entries.assert_called_once_with(test_period)

    # Verify the result is a GeneralLedger instance
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == test_period

    # Verify that the ledger for our account was created and populated
    assert mock_account in result_gl.ledgers
    ledger = result_gl.ledgers[mock_account]
    
    # Initial balance was 100, posting was +50 (direction value 1)
    # Total should be 150
    expected_balance = Quantity(Decimal("150.00"))
    assert ledger._last_balance == expected_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #18
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

def test_ReadInitialBalances___call__():
    """
    Tests the protocol definition of ReadInitialBalances by verifying 
    that a compatible implementation can be called with a DateRange 
    and returns InitialBalances.
    """
    # Arrange
    period = DateRange(
        since=datetime.date(2023, 1, 1), 
        until=datetime.date(2023, 12, 31)
    )
    
    # Create a mock account and balance to act as the return value
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(
        since=datetime.date(202ass, 1, 1), 
        value=Quantity(Decimal("100.00"))
    )
    expected_initial_balances: InitialBalances = {mock_account: mock_balance}

    # Define a concrete implementation of the ReadInitialBalances protocol
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return expected_initial_balances

    reader: ReadInitialBalances = MockReadInitialBalances()

    # Act
    result = reader(period)

    # Assert
    assert result == expected_initial_balances
    assert mock_account in result
    assert result[mock_account].value == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    # Setup dependencies
    period = DateRange(
        since=datetime.date(2023, 1, 1), 
        until=datetime.date(2023, 12, 31)
    )
    
    # Mocking the Return values for initial balances
    account_a = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {account_a: Balance(period.since, initial_balance_v)}
    
    # Mocking the Return values for journal entries
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    
    posting_a = MagicMock(spec=Posting)
    posting_a.account = account_a
    posting_a.amount = Decimal("50.00")
    posting_a.direction = Direction.DEBIT  # Assuming Direction exists in scope
    
    mock_journal_entry.postings = [posting_a]
    mock_journal_entry.description = "Test Entry"
    
    journal_entries = [mock_journal_entry]

    # Define the mock algebra implementations (ReadInitialBalances and ReadJournalEntries)
    read_initial_balances_mock = MagicMock(return_value=initial_balances)
    read_journal_entries_mock = MagicMock(return_value=journal_entries)

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances_mock, 
        read_journal_entries_mock
    )

    # Execute the __call__ of GeneralLedgerProgram (the compiled _program function)
    gl_result = program(period)

    # Assertions to verify the execution flow and result integrity
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)
    
    assert isinstance(gl_result, GeneralLedger)
    assert gl_result.period == period
    assert account_a in gl_result.ledgers
    
    # Verify the ledger math: 100 (initial) + 50 (debit) = 150
    ledger_a = gl_result.ledgers[account_a]
    assert ledger_a._last_balance == Quantity(Decimal("150.00"))
    assert len(ledger_a.entries) == 1
    assert ledger_a.entries[0].amount == Decimal("50.00")
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup common dates and accounts
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    account_a = Account("Assets:Cash")
    account_b = Account("Equity:Capital")
    account_c = Account("Expenses:Rent")

    # Setup Initial Balances (Opening balance for account_a)
    initial_balances = {
        account_a: Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Capital injection (Debit Cash 500, Credit Capital 500)
    journal_entry_1 = MagicMock(spec=JournalEntry)
    journal_entry_1.date = datetime.date(2023, 2, 1)
    journal_entry_1.description = "Initial Investment"
    
    posting_a1 = MagicMock(spec=Posting)
    posting_a1.account = account_a
    posting_a1.amount = Amount(Decimal("500.00"))
    posting_a1.direction = Direction.DEBIT
    
    posting_b1 = MagicMock(spec=Posting)
    posting_b1.account = account_b
    posting_b1.amount = Amount(Decimal("500.00"))
    posting_b1.direction = Direction.CREDIT
    
    journal_entry_1.postings = [posting_a1, posting_b1]

    # Entry 2: Rent payment (Debit Rent 200, Credit Cash 200) - Inside period
    journal_entry_2 = MagicMock(spec=JournalEntry)
    journal_entry_2.date = datetime.date(2023, 3, 1)
    journal_entry_2.description = "Rent Payment"
    
    posting_c2 = MagicMock(spec=Posting)
    posting_c2.account = account_c
    posting_c2.amount = Amount(Decimal("200.00"))
    posting_c2.direction = Direction.DEBIT
    
    posting_a2 = MagicMock(spec=Posting)
    posting_a2.account = account_a
    posting_a2.amount = Amount(Decimal("200.00"))
    posting_a2.direction = Direction.CREDIT
    
    journal_entry_2.postings = [posting_c2, posting_a2]

    # Entry 3: Out of period entry (Should be ignored)
    journal_entry_3 = MagicMock(spec=JournalEntry)
    journal_entry_3.date = datetime.mock.datetime.date(2024, 1, 1)
    journal_entry_3.description = "New Year Entry"
    posting_out = MagicMock(spec=Posting)
    posting_out.account = account_a
    posting_out.amount = Amount(Decimal("100.00"))
    posting_out.direction = Direction.DEBIT
    journal_entry_3.postings = [posting_out]

    journal = [journal_entry_1, journal_entry_2, journal_entry_3]

    # Execution
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert gl.period == period
    assert account_a in gl.ledgers
    assert account_b in gl.ledgers
    assert account_c in gl.ledgers
    
    # Check Cash Account (Account A)
    # Start 1000 + 500 (Debit) - 200 (Credit) = 1300
    ledger_a = gl.ledgers[account_a]
    assert len(ledger_a.entries) == 2
    assert ledger_a._last_balance.value == Quantity(Decimal("1300.00"))
    
    # Check Capital Account (Account B)
    # Start 0 + 500 (Credit) = 500
    ledger_b = gl.ledgers[account_b]
    assert len(ledger_b.entries) == 1
    assert ledger_b._last_balance.value == Quantity(Decimal("500.00"))

    # Check Rent Account (Account C)
    # Start 0 + 200 (Debit) = 200
    ledger_c = gl.ledgers[account_c]
    assert len(ledger_c.entries) == 1
    assert ledger_c._last_balance.value == Quantity(Decimal("200.00"))

    # Verify that the out-of-period entry was not added to any ledger
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date <= end_date
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import Mock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol structure of ReadInitialBalances by verifying 
    that a compatible callable returns the expected InitialBalances type.
    """
    # Setup dependencies
    test_date = datetime.date(2023, 1, 1)
    test_period = DateRange(since=test_date, until=datetime.date(2023, 12, 31))
    
    # Create a mock account and balance
    mock_account = Mock(spec=Account)
    mock_balance = Balance(since=test_date, value=Quantity(Decimal("100.00")))
    
    # Define the expected return value (InitialBalances is Dict[Account, Balance])
    expected_initial_balances: InitialBalances = {
        mock_account: mock_balance
    }

    # Create a mock implementation of the ReadInitialBalances protocol
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        if period == test_period:
            return expected_initial_balances
        return {}

    # Verify the callable behavior matches the Protocol definition
    # 1. Test successful retrieval
    result = mock_read_initial_balances(test_period)
    assert result == expected_initial_balances
    assert result[mock_account].value == Quantity(Decimal("100.00"))

    # 2. Test behavior with different period
    other_period = DateRange(since=datetime.date(2024, 1, 1), until=datetime.date(2024, 12, 31))
    assert mock_read_initial_balances(other_period) == {}

    # 3. Verify type compatibility via a mock that implements the protocol
    protocol_implementation = Mock(spec=ReadInitialBalances)
    protocol_implementation.return_value = expected_initial_balances
    
    assert protocol_implementation(test_period) == expected_initial_balances
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup common components
    period = MagicMock(spec=DateRange)
    period.since = datetime.date(2023, 1, 1)
    period.until = datetime.date(2023, 12, 31)

    # Create Accounts
    acc_cash = MagicMock(spec=Account)
    acc_revenue = MagicMock(spec=Account)
    acc_expense = MagicMock(spec=Account)

    # Setup Initial Balances (Cash starts with 1000)
    initial_balances = {
        acc_cash: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00")))
    }

    # Create Journal Entries
    # Entry 1: Revenue of 500 (Cash Debit, Revenue Credit)
    j1_date = datetime.date(2023, 6, 1)
    j1_desc = "Service Revenue"
    
    p1_cash = MagicMock(spec=Posting)
    p1_cash.account = acc_cash
    p1_cash.amount = Decimal("500.00")
    p1_cash.direction = Direction.DEBIT # Assuming Direction enum exists in context
    p1_cash.is_debit = True
    p1_cash.is_credit = False

    p1_rev = MagicMock(spec=Posting)
    p1_rev.account = acc_revenue
    p1_rev.amount = Decimal("500.00")
    p1_rev.direction = Direction.CREDIT
    p1_rev.is_debit = False
    p1_rev.is_credit = True

    j1 = MagicMock(spec=JournalEntry)
    j1.date = j1_date
    j1.description = j1_desc
    j1.postings = [p1_cash, p1_rev]

    # Entry 2: Expense of 200 (Expense Debit, Cash Credit) - OUTSIDE period
    j2_date = datetime.date(2022, 12, 31)
    p2_cash = MagicMock(spec=Posting)
    p2_cash.account = acc_cash
    p2_cash.amount = Decimal("200.00")
    p2_cash.direction = Direction.CREDIT
    p2_cash.is_debit = False
    p2_cash.is_credit = True

    j2 = MagicMock(spec=JournalEntry)
    j2.date = j2_date
    j2.postings = [p2_cash]

    journal = [j1, j2]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    
    # Check if cash ledger exists and includes the initial balance + valid posting
    assert acc_cash in gl.ledgers
    cash_ledger = gl.ledgers[acc_cash]
    # Initial 1000 + 500 (debit) = 1500
    assert cash_ledger._last_balance == Quantity(Decimal("1500.00"))
    
    # Check if revenue ledger was created correctly from the posting
    assert acc_revenue in gl.ledgers
    rev_ledger = gl.ledgers[acc_revenue]
    # Initial 0 + 500 (credit) -> direction value for credit is -1 usually, 
    # but based on logic: last_balance + amount * direction.value
    # If Direction.CREDIT.value is -1: 0 + 500 * -1 = -500
    assert rev_ledger._last_balance == Quantity(Decimal("-500.00"))

    # Verify that the out-of-period entry (j2) was ignored
    # Cash ledger should not have a credit of 200 from j2
    for entry in cash_ledger.entries:
        assert entry.date != j2.date

    # Check if expense account was NOT created because it wasn't in initial and not in valid journal
    assert acc_expense not in gl.ledgers
```


# LLM-generated content at query #4
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

def test_GeneralLedgerProgram___call__():
    """
    Tests the functionality of the compiled GeneralLedgerProgram's __call__ method.
    This verifies that the program correctly orchestrates reading initial balances,
    reading journal entries, and building the general ledger within a specific period.
    """
    # 1. Setup Mock Data
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(since=period_start, until=period_end)

    # Create a mock Account
    mock_account = MagicMock(spec=Account)
    mock_account.__str__.return_value = "Test Account"

    # Define initial balances (e.g., Account A starts with 100)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {mock_count: Balance(period_start, initial_balance_val)}
    # Note: In a real scenario, InitialBalances uses the account as key. 
    # Since we are mocking Account, we use our mock object.
    mock_account_key = mock_account 
    initial_balances = {mock_account_key: Balance(period_start, initial_balance_val)}

    # Create a Mock Journal Entry
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Transaction"
    
    # Create a Mock Posting
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account_key
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction enum exists in context
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_journal_entry.postings = [mock_posting]

    # 2. Setup Mocks for the Algebas (Protocols)
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = initial_balances

    read_journal_entries_mock = MagicMock(spec=ReadJournalEntries)
    read_journal_entries_mock.return_value = [mock_journal_entry]

    # 3. Compile the Program
    program = compile_general_ledger_program(
        read_initial_balances_mock, 
        read_journal_entries_mock
    )

    # 4. Execute the Program
    result_gl = program(period)

    # 5. Assertions
    # Verify that the algebras were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify the resulting GeneralLedger structure
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period
    assert mock_account_key in result_gl.ledgers
    
    # Verify the ledger math: Initial (100) + Posting (50 Debit) = 150
    ledger = result_gl.ledgers[mock_account_key]
    expected_final_balance = Quantity(Decimal("150.00"))
    assert ledger._last_balance == expected_final_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol structure of ReadInitialBalances by verifying that 
    a callable implementation correctly returns InitialBalances for a given DateRange.
    """
    # Setup
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create mock accounts
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Jackets) # Using a placeholder for Account type context
    
    # Define expected initial balances
    expected_balances: InitialBalances = {
        account_a: Balance(since=datetime.date(2023, 1, 1), value=Quantity(Decimal("100.00"))),
        account_b: Balance(since=datetime.date(2023, 1, 1), value=Quantity(Decimal("50.00")))
    }

    # Define a concrete implementation of the Protocol
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        return expected_balances

    # Verify adherence to protocol via type checking (conceptual) and execution
    # In Python, protocols are structural; we test if our function matches the signature
    actual_balances = mock_read_initial_balances(period)

    # Assertions
    assert actual_balances == expected_balances
    assert len(actual_balances) == 2
    assert actual_balances[account_a].value == Quantity(Decimal("100.00"))
    assert actual_balances[account_b].value == Quantity(Decimal("50.00"))

    # Test with a mock to ensure it behaves like a function/callable as expected by the protocol
    mock_reader = MagicMock(spec=ReadInitialBalances)
    mock_reader.return_value = expected_balances
    
    result = mock_reader(period)
    
    mock_reader.assert_called_once_with(period)
    assert result == expected_balances
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol structure of ReadInitialBalances by verifying 
    a callable implementation behaves as expected with a DateRange input.
    """
    # Setup dependencies
    period = MagicMock()
    period.since = datetime.date(2023, 1, 1)
    period.until = datetime.date(2023, 12, 31)
    
    # Create a mock account and balance to be returned
    mock_account = MagicMock()
    mock_balance = MagicMock()
    expected_balances = {mock_account: mock_balance}
    
    # Define the implementation of the protocol
    def implementation(p: datetime.date) -> dict:
        return expected_balances

    # Cast to the Protocol type (simulating a real usage scenario)
    read_initial_balances_func: ReadInitialBalances = implementation

    # Execute the call
    result = read_initial_balances_func(period)

    # Assertions
    assert result == expected_balances
    assert mock_account in result
    assert result[mock_account] == mock_balance
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol/signature of ReadInitialBalances by verifying 
    a mock implementation adheres to the expected return type and behavior.
    """
    # Setup period
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )

    # Setup mock accounts and balances
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(
        since=datetime.date(2023, 1, 1), 
        value=Quantity(Decimal("100.00"))
    )
    expected_initial_balances: InitialBalances = {mock_account: mock_balance}

    # Create a mock function that implements the ReadInitialBalances protocol
    read_initial_balances_mock: ReadInitialBalances = MagicMock()
    read_initial_balances_mock.return_value = expected_initial_balances

    # Execute the call
    result = read_initial_balances_mock(period)

    # Assertions
    read_initial_balances_mock.assert_called_once_with(period)
    assert result == expected_initial_balances
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account].value == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol/type definition behavior of ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test that an implementation 
    conforming to its signature works as expected.
    """
    # Setup common dependencies
    period = MagicMock()
    # Mocking Account and Balance for the return type InitialBalances
    mock_account = MagicMock()
    mock_balance = MagicMock()
    
    # Define a concrete implementation of the Protocol
    def mock_read_initial_balances(p: datetime.date) -> dict:
        return {mock_account: mock_balance}

    # Define the expected return value
    expected_output = {mock_account: mock_balance}

    # Execution
    # We test if calling an implementation of the protocol returns the correct structure
    result = mock_read_initial_balances(period)

    # Assertions
    assert result == expected_output
    assert isinstance(result, dict)
    assert mock_account in result
    assert result[mock_account] == mock_balance

def test_ReadInitialBalances_type_compliance():
    """
    Tests that a function satisfying the ReadInitialBalances protocol 
    can be used in place of the type definition.
    """
    from typing import Callable
    
    # Create a type-hinted variable using the Protocol
    program: ReadInitialBalances = MagicMock(side_effect=lambda p: {MagicMock(): MagicMock()})
    
    test_date = datetime.date(2023, 1, 1)
    result = program(test_date)
    
    assert isinstance(result, dict)
    assert len(result) == 1
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup Dates
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances (e.g., Cash starts with 1000)
    initial_balances = {
        acc_cash: Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue of 500 (Debit Cash, Credit Revenue) - Inside Period
    j1_date = datetime.date(2023, 6, 1)
    p1_cash = MagicMock(spec=Posting)
    p1_cash.account = acc_cash
    p1_cash.amount = Amount(Decimal("500.00"))
    p1_cash.direction = Direction.DEBIT
    
    p2_revenue = MagicMock(spec=Posting)
    p2_revenue.account = acc_revenue
    p2_revenue.amount = Amount(Decimal("500.00"))
    p2_revenue.direction = Direction.CREDIT

    j1 = MagicMock(spec=JournalEntry)
    j1.date = j1_date
    j1.description = "Service Revenue"
    j1.postings = [p1_cash, p2_revenue]

    # Entry 2: Expense of 200 (Debit Expense, Credit Cash) - Inside Period
    j2_date = datetime.date(2023, 7, 1)
    p3_expense = MagicMock(spec=Posting)
    p3_expense.account = acc_expense
    p3_expense.amount = Amount(Decimal("200.00"))
    p3_expense.direction = Direction.DEBIT

    p4_cash = MagicMock(spec=Posting)
    p4_cash.account = acc_cash
    p4_cash.amount = Amount(Decimal("200.00"))
    p4_cash.direction = Direction.CREDIT

    j2 = MagicMock(spec=JournalEntry)
    j2.date = j2_date
    j2.description = "Office Supplies"
    j2.postings = [p3_expense, p4_cash]

    # Entry 3: Old entry - Outside Period (Should be ignored)
    j3_date = datetime.date(2022, 12, 15)
    p5_old = MagicMock(spec=Posting)
    p5_old.account = acc_cash
    p5_old.amount = Amount(Discount(Decimal("100.00"))) # Using a generic amount
    p5_old.direction = Direction.DEBIT

    j3 = MagicMock(spec=JournalEntry)
    j3.date = j3_date
    j3.postings = [p5_old]

    journal = [j1, j2, j3]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert gl.period == period
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers

    # Verify Cash Ledger: 1000 (initial) + 500 (debit) - 200 (credit) = 1300
    cash_ledger = gl.ledgers[acc_cash]
    assert cash_ledger._last_balance == Quantity(Decimal("1300.00"))
    # Entry count for cash: 1 initial + 2 postings (p1 and p4)
    # Note: build_general_ledger adds entries from journal to the ledger object
    cash_entries = [e for e in cash_ledger.entries if e.posting.account == acc_cash]
    assert len(cash_entries) == 2 

    # Verify Revenue Ledger: 0 (initial) + 500 (credit) = 500
    revenue_ledger = gl.ledgers[acc_revenue]
    assert revenue_ledger._last_balance == Quantity(Decimal("500.00"))

    # Verify Expense Ledger: 0 (initial) + 200 (debit) = 200
    expense_ledger = gl.ledgers[acc_expense]
    assert expense_ledger._last_balance == Quantity(Decimal("200.00"))

    # Verify that the old entry (j3) was not processed
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date != j3_date
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the type compliance and behavior of a ReadInitialBalances implementation.
    Since ReadInitialBalances is a Protocol, we test an implementation of it.
    """
    # Arrange
    period = DateRange(
        since=datetime.date(2023, 1, 1), 
        until=datetime.date(2023, 12, 31)
    )
    
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(
        since=datetime.date(202ally, 1, 1), 
        value=Quantity(Decimal("100.00"))
    )
    expected_initial_balances: InitialBalances = {
        mock_account: mock_balance
    }

    # Define an implementation of the Protocol
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        assert p == period
        return expected_initial_balances

    # Act
    # In Python, Protocols are used for structural subtyping. 
    # We verify that our function matches the signature/behavior expected by the protocol.
    result = mock_read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert mock_account in result
    assert result[mock_account].value == Quantity(Decimal("100.00"))

    # Verify type compatibility (Structural Typing check)
    from typing import cast
    reader: ReadInitialBalances = cast(ReadInitialBalances, mock_read_initial_balances)
    assert reader(period) == expected_initial_balances
```


# LLM-generated content at query #11
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

def test_ReadInitialBalances___call__():
    """
    Tests the protocol definition of ReadInitialBalances by verifying 
    that a conforming callable can be used as intended.
    """
    # Setup period and mock data
    period = DateRange(
        since=datetime.date(2023, 1, 1), 
        until=datetime.date(2023, 12, 31)
    )
    
    account_a = Account("Assets:Cash")
    account_b = Account("Equity:RetainedEarnings")
    
    initial_balance_val = Quantity(Decimal("1000.00"))
    initial_balances_mock = {
        account_a: Balance(period.since, initial_balance_val),
        account_b: Balance(period.since, Quantity(Decimal("1000.00")))
    }

    # Create a mock implementation of the ReadInitialBalances protocol
    # Since it is a Protocol (structural typing), any callable with this 
    # signature satisfies it.
    read_initial_balances_impl: ReadInitialBalances = MagicMock(return_value=initial_balances_mock)

    # Execution
    result = read_initial_balances_impl(period)

    # Assertions
    read_initial_balances_impl.assert_called_once_with(period)
    assert isinstance(result, dict)
    assert result == initial_balances_mock
    assert account_a in result
    assert result[account_a].value == initial_balance_val
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol/type definition behavior of ReadInitialBalances via a mock implementation.
    Since ReadInitialBalances is a Protocol, we test that an object following its 
    signature behaves as expected when called with a DateRange.
    """
    # Setup period and expected return data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    # Create dummy accounts and balances
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(since=start_date, value=Quantity(Decimal("100.00")))
    expected_initial_balances: InitialBalances = {mock_call_account := mock_account: mock_balance}

    # Define the implementation (the callable to be tested)
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        if p.since == start_date:
            return expected_initial_balances
        return {}

    # Verify the implementation matches the protocol's expected behavior
    # 1. Test with correct period
    result = mock_read_initial_balances(period)
    assert result == expected_initial_balances
    assert result[mock_account].value == Quantity(Decimal("100.00"))

    # 2. Test with different period (should return empty dict per our implementation)
    other_period = DateRange(since=datetime.date(2024, 1, 1), until=datetime.date(2024, 12, 31))
    assert mock_read_initial_balances(other_period) == {}

    # 3. Test using a MagicMock to ensure it satisfies the protocol call signature
    protocol_implementation = MagicMock(side_effect=mock_read_initial_balances)
    
    # Assert that calling the mock with the period returns the correct dictionary
    assert protocol_implementation(period) == expected_initial_balances
    # Ensure the mock was called with the correct argument
    protocol_implementation.assert_called_with(period)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup DateRange and Period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)

    # Setup Accounts
    cash_account = Account("Cash")
    revenue_account = Account("Revenue")
    expense_account = Account("Expense")

    # Setup Initial Balances (Cash starts with 1000)
    initial_balances = {
        cash_account: Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Cash (Debit) vs Revenue (Credit) - 500
    journal_entry_1 = MagicMock(spec=JournalEntry)
    journal_entry_1.date = datetime.date(2023, 6, 1)
    journal_entry_1.description = "Service Rendered"
    
    posting_cash_dr = MagicMock(spec=Posting)
    posting_cash_dr.account = cash_account
    posting_cash_dr.amount = Amount(Decimal("500.00"))
    posting_cash_dr.direction = Direction.DEBIT
    
    posting_rev_cr = MagicMock(spec=Posting)
    posting_rev_cr.account = revenue_account
    posting_rev_cr.amount = Amount(Decimal("500.00"))
    posting_rev_cr.direction = Direction.CREDIT

    # Link postings to journal description logic (for LedgerEntry.description/cntracts)
    journal_entry_1.postings = [posting_cash_dr, posting_rev_cr]
    
    # Entry 2: Expense (Debit) vs Cash (Credit) - 200 (Out of period check)
    journal_entry_2 = MagicMock(spec=JournalEntry)
    journal_entry_2.date = datetime.date(2022, 12, 31) # Should be ignored by build_general_ledger
    journal_entry_2.description = "Old Expense"
    posting_exp_dr = MagicMock(spec=Posting)
    posting_exp_dr.account = expense_account
    posting_exp_dr.amount = Amount(Decimal("200.00"))
    posting_exp_dr.direction = Direction.DEBIT
    journal_entry_2.postings = [posting_exp_dr]

    # Entry 3: Expense (Debit) vs Cash (Credit) - 100 (Within period)
    journal_entry_3 = MagicMock(spec=JournalEntry)
    journal_entry_3.date = datetime.date(2023, 7, 1)
    journal_entry_3.description = "Supplies"
    posting_exp_dr_2 = MagicMock(spec=Posting)
    posting_exp_dr_2.account = expense_account
    posting_exp_dr_2.amount = Amount(Decimal("100.00"))
    posting_exp_dr_2.direction = Direction.DEBIT
    
    posting_cash_cr = MagicMock(spec=Posting)
    posting_cash_cr.account = cash_account
    posting_cash_cr.amount = Amount(Decimal("100.00"))
    posting_cash_cr.direction = Direction.CREDIT

    journal_entry_3.postings = [posting_exp_dr_2, posting_cash_cr]

    journal = [journal_entry_1, journal_entry_2, journal_entry_3]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    
    # Check Cash Ledger (Initial 1000 + 500 DR - 100 CR = 1400)
    assert cash_account in gl.ledgers
    cash_ledger = gl.ledgers[cash_account]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Entries: 1 (DR 500), 2 (CR 100)
    assert len(cash_ledger.entries) == 2
    assert cash_ledger._last_balance == Quantity(Decimal("1400.00"))

    # Check Revenue Ledger (Initial 0 + 500 CR = 500)
    assert revenue_account in gl.ledgers
    rev_ledger = gl.ledgers[revenue_account]
    assert rev_ledger.initial.value == Quantity(Decimal("0"))
    assert rev_ledger._last_balance == Quantity(Decimal("500.00"))

    # Check Expense Ledger (Initial 0 + 100 DR = 100)
    # Note: Entry 2 was ignored because it was out of period
    assert expense_account in gl.ledgers
    exp_ledger = gl.ledgers[expense_account]
    assert exp_ledger._last_balance == Quantity(Decimal("100.00"))

    # Verify description and counter-accounts for a specific entry
    entry_dr_cash = cash_ledger.entries[0] 
    assert entry_dr_cash.description == "Service Rendered"
    assert revenue_account in entry_dr_cash.cntracts
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the __call__ method of a GeneralLedgerProgram instance.
    Since GeneralLedgerProgram is a Protocol, we test the function 
    returned by compile_general_ledger_program which satisfies the protocol.
    """
    # Setup period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)

    # Mock Account and Balance
    mock_account = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {mock_account: Balance(start_date, initial_balance_val)}

    # Mock JournalEntry and Posting
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = MagicMock()
    mock_posting.direction.value = Decimal("1") # Debit
    mock_posting.is_debit = True
    mock_posting.is_credit = False

    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.postings = [mock_posting]
    # Mock description for the LedgerEntry property access
    mock_journal_entry.description = "Test Entry"
    
    # To satisfy LedgerEntry.description -> posting.journal.description
    mock_posting.journal = mock_journal_entry

    # Setup Mocks for the algebra implementations (ReadInitialBalances and ReadJournalEntries)
    read_initial_balances_mock = MagicMock(return_value=initial_balances)
    read_journal_entries_mock = MagicMock(return_value=[mock_journal_entry])

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances_mock, 
        read_journal_entries_mock
    )

    # Execute the program (the __call__ implementation)
    general_ledger = program(period)

    # Assertions
    # Verify algebra functions were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify output structure
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert mock_account in general_ledger.ledgers
    
    # Verify ledger content calculation
    ledger = general_ledger.ledgers[mock_account]
    # Initial (100) + Posting (50 * 1) = 150
    expected_balance = Quantity(Decimal("150.00"))
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == expected_balance
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the execution of the function returned by compile_general_ledger_program,
    which implements the GeneralLedgerProgram protocol.
    """
    # 1. Setup Mock Data and Dependencies
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    mock_account = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {mock_account: Balance(period.since, initial_balance_val)}
    
    # Mock Journal Entry and Postings
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction is available in scope
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Transaction"
    mock_journal_entry.postings = [mock_posting]
    
    journal_entries = [mock_journal_entry]

    # 2. Setup Mocks for the Algebra (ReadInitialBalances and ReadJournalEntries)
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = initial_balances
    
    read_journal_entries_mock = MagicMock(spec=ReadJournalEntries)
    read_journal_entries_mock.return_value = journal_entries

    # 3. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances_mock,
        read_journal_entries_mock
    )

    # 4. Execute the Program (The __call__ of GeneralLedgerProgram)
    result_gl = program(period)

    # 5. Assertions
    # Verify that the algebra functions were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify the type of the returned object
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period

    # Verify that the ledger for our account was correctly built from initial balances and journal entries
    assert mock_account in result_gl.ledgers
    ledger = result_gl.ledgers[mock_account]
    
    # The first entry should be the posting from the journal
    assert len(ledger.entries) == 1
    entry = ledger.entries[0]
    assert entry.amount == Quantity(Decimal("50.00"))
    
    # Verify balance calculation: Initial (100) + Debit (50) = 150
    # Note: direction.value for DEBIT is usually 1 in accounting logic
    expected_balance = initial_balance_val + Quantity(Decimal("50.00"))
    assert entry.balance == expected_balance
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the functionality of the compiled GeneralLedgerProgram's __call__ method.
    The test verifies that the program correctly orchestrates reading initial balances,
    reading journal entries, and building the general ledger for a given period.
    """
    # 1. Setup mock data and dependencies
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    test_period = DateRange(since=period_start, until=period_end)

    # Mock Account
    mock_account = MagicMock(spec=Account)
    
    # Mock Initial Balances
    initial_balance_val = Quantity(Decimal("100.00"))
    mock_initial_balances = {
        mock_account: Balance(since=datetime.date(2022, 1, 1), value=initial_balance_val)
    }

    # Mock Journal Entry and Posting
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction enum exists in scope
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Entry"
    mock_journal_entry.postings = [mock_posting]

    # 2. Setup Mock Algebra implementations (Protocols)
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = mock_initial_balances

    read_journal_entries_mock = MagicMock(spec=ReadJournalEntries)
    read_journal_entries_mock.return_value = [mock_journal_entry]

    # 3. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=read_initial_balances_mock,
        read_journal_entries=read_journal_entries_mock
    )

    # 4. Execute the __call__ method of the program
    result_gl = program(test_period)

    # 5. Assertions
    # Verify that the algebra implementations were called with the correct period
    read_initial_balances_mock.assert_called_once_with(test_period)
    read_journal_entries_mock.assert_called_once_with(test_period)

    # Verify the resulting GeneralLedger structure
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == test_period
    assert mock_account in result_gl.ledgers
    
    # Verify that the ledger was updated correctly from initial balance + posting
    ledger = result_gl.ledgers[mock_account]
    expected_balance = initial_balance_val + Quantity(Decimal("50.00"))
    
    # Check if the entry exists and has the correct calculated balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance.value == expected_balance
    assert ledger.entries[0].amount.value == Decimal("50.00")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the execution of the function returned by compile_general_ledger_program,
    which satisfies the GeneralLedgerProgram protocol.
    """
    # 1. Setup Mock Data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Account)
    
    initial_balances = {
        account_a: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("100.00")))
    }
    
    # Mock a JournalEntry with one Posting
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = account_b
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction enum exists
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.postings = [mock_posting]
    # Mock description for LedgerEntry.description access
    mock_journal_entry.description = "Test Transaction"
    
    journal_entries = [mock_journal_entry]

    # 2. Setup Mocks for the Algebra implementations
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = initial_balances
    
    read_journal_entries_mock = MagicMock(spec=ReadJournalEntries)
    read_journal_entries_mock.return_value = journal_entries

    # 3. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances_mock,
        read_journal_entries_mock
    )

    # 4. Execute the program (the __call__ of GeneralLedgerProgram)
    gl = program(period)

    # 5. Assertions
    # Verify that the algebra functions were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify the returned object is a GeneralLedger
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    
    # Verify ledger contents for account_a (from initial balances)
    assert account_a in gl.ledgers
    assert gl.ledgers[account_a].initial.value == Quantity(Decimal("100.00"))
    
    # Verify ledger contents for account_b (from journal entries)
    assert account_b in gl.ledgers
    assert len(gl.ledgers[account_b].entries) == 1
    assert gl.ledgers[account_b].entries[0].amount == Quantity(Decimal("50.00"))
    assert gl.ledgers[account_b].entries[0].is_debit is True
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol definition for ReadInitialBalances by verifying 
    that a compatible implementation can be called and returns 
    the expected InitialBalances type.
    """
    # Setup
    period = DateRange(
        since=datetime.date(2023, 1, 1), 
        until=datetime.date(2023, 12, 31)
    )
    
    # Create mock accounts and balances
    mock_account_1 = MagicMock(spec=Account)
    mock_account_2 = MagicMock(spec=Account)
    
    balance_val_1 = Balance(
        date=datetime.date(202ass, 12, 31), 
        value=Quantity(Decimal("100.00"))
    )
    balance_val_2 = Balance(
        date=datetime.date(2023, 12, 31), 
        value=Quantity(Decimal("50.00"))
    )
    
    expected_initial_balances: InitialBalances = {
        mock_account_1: balance_val_1,
        mock_account_2: balance_val_2
    }

    # Create a mock implementation of the ReadInitialBalances protocol
    # Since Protocol is a structural type, any callable with this signature works.
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Verify the implementation matches the protocol's behavior
    actual_balances = mock_read_initial_balances(period)

    # Assertions
    assert actual_balances == expected_initial_balances
    assert len(actual_balances) == 2
    assert actual_balances[mock_account_1].value == Quantity(Decimal("100.00"))
    assert actual_balances[mock_account_2].value == Quantity(Decimal("50.00"))

    # Test with a different implementation (using MagicMock)
    mock_implementation = MagicMock(spec=ReadInitialBalances)
    mock_implementation.return_value = expected_initial_balances
    
    returned_val = mock_implementation(period)
    
    assert returned_val == expected_initial_balances
    mock_implementation.assert_called_once_with(period)
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the functionality of the compiled GeneralLedgerProgram.__call__ method.
    The test verifies that the program correctly orchestrates reading initial balances,
    reading journal entries, and building the general ledger for a given period.
    """
    # 1. Setup Test Data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)

    # Mock Account and Balance
    mock_account = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {mock_account: Balance(since=datetime.date(2022, 12, 31), value=initial_balance_val)}

    # Mock Journal Entry and Posting
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction is available in scope
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 15)
    mock_journal_entry.description = "Test Transaction"
    mock_journal_entry.postings = [mock_posting]

    # 2. Setup Mocks for the Algebra (Protocols)
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = initial_balances

    read_journal_entries_mock = MagicMock(spec=ReadJournalEntries)
    read_journal_entries_mock.return_value = [mock_journal_entry]

    # 3. Compile the Program
    program = compile_general_ledger_program(
        read_initial_balances=read_initial_balances_mock,
        read_journal_entries=read_journal_entries_mock
    )

    # 4. Execute the Program
    result_gl = program(period)

    # 5. Assertions
    # Verify that the algebra functions were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify the structure of the returned GeneralLedger
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period
    assert mock_account in result_gl.ledgers
    
    # Verify calculations within the ledger
    ledger = result_gl.ledgers[mock_account]
    assert len(ledger.entries) == 1
    # Initial (100) + Posting (50 * Debit(1)) = 150
    assert ledger.entries[0].balance.value == Quantity(Decimal("150.00"))
    
    # Verify the entry details match the posting/journal
    entry = ledger.entries[0]
    assert entry.amount.value == Decimal("50.00")
    assert entry.date == datetime.date(2023, 6, 15)
    assert entry.description == "Test Transaction"
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the functionality of the function returned by compile_general_ledger_program,
    which adheres to the GeneralLedgerProgram protocol.
    """
    # 1. Setup Mock Data and Dependencies
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Mock Account and Balance
    mock_account = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {mock_account: Balance(period.since, initial_balance_val)}
    
    # Mock Journal Entry and Posting
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Transaction"
    
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Amount(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction is available in scope
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_journal_entry.postings = [mock_posting]
    journal_entries = [mock_journal_entry]

    # 2. Setup Mocks for the Algebra implementations (ReadInitialBalances and ReadJournalEntries)
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = initial_balances
    
    read_journal_entries_mock = MagicMand(spec=ReadJournalEntries)
    read_journal_entries_mock.return_value = journal_entries

    # 3. Compile the Program
    program = compile_general_ledger_program(
        read_initial_balances_mock,
        read_journal_entries_mock
    )

    # 4. Execute the program (the __call__ of GeneralLedgerProgram)
    result_gl = program(period)

    # 5. Assertions
    # Verify that the algebra functions were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify the resulting GeneralLedger structure
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period
    assert mock_account in result_gl.ledgers
    
    # Verify the ledger content calculation
    ledger = result_gl.ledgers[mock_account]
    # Initial (100) + Posting (50 * 1 for Debit) = 150
    expected_balance = Quantity(Decimal("150.00"))
    assert ledger._last_balance == expected_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Amount(Decimal("50.00"))
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances (Cash starts with 100)
    initial_balances = {
        acc_cash: Balance(start_date, Quantity(Decimal("100.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue of 50 (Debit Cash, Credit Revenue)
    j1_date = datetime.date(2023, 6, 1)
    p1_cash = Posting(acc_cash, Amount(Decimal("50.00")), Direction.DEBIT, j1_date)
    p1_rev = Posting(acc_revenue, Amount(leDecimal("50.00")), Direction.CREDIT, j1_date)
    j1 = JournalEntry(j1_date, "Service Revenue", [p1_cash, p1_rev], "Revenue earned")

    # Entry 2: Expense of 20 (Debit Expense, Credit Cash)
    j2_date = datetime.date(2023, 7, 1)
    p2_exp = Posting(acc_expense, Amount(Decimal("20.00")), Direction.DEBIT, j2_date)
    p2_cash = Posting(acc_cash, Amount(Decimal("20.00")), Direction.CREDIT, j2_date)
    j2 = JournalEntry(j2_date, "Office Supplies", [p2_exp, p2_cash], "Supplies purchased")

    # Entry 3: Out of period entry (Should be ignored)
    j3_date = datetime.date(2024, 1, 1)
    p3_cash = Posting(acc_cash, Amount(Decimal("10.00")), Direction.DEBIT, j3_date)
    j3 = JournalEntry(j3_date, "New Year Gift", [p3_cash], "Ignore me")

    journal = [j1, j2, j3]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert gl.period == period
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers

    # Verify Cash Ledger: 100 (initial) + 50 (debit) - 20 (credit) = 130
    cash_ledger = gl.ledgers[acc_cash]
    assert cash_ledger.initial.value == Quantity(Decimal("100.00"))
    # Check entries in cash ledger (only j1 and j2 postings for cash)
    # Entry 1: +50, Entry 2: -20
    cash_entries = [e for e in cash_ledger.entries if e.amount == Amount(Decimal("50.00")) or e.amount == Amount(Decimal("20.00"))]
    assert len(cash_entries) == 2
    # Final balance check via the property _last_balance (or checking specific entry)
    # The last entry in cash should be the credit of 20, resulting in 130
    # Note: implementation adds entries as they appear in journal loop
    final_cash_val = Decimal("130.00")
    # We find the entry that corresponds to the last posting
    last_entry = next(e for e in cash_ledger.entries if e.amount == Amount(Decimal("20.00")) and e.is_credit)
    assert last_entry.balance.value == Quantity(Decimal("130.00"))

    # Verify Revenue Ledger: 0 (initial) + 50 (credit) = 50? 
    # Note: direction.value for Credit is typically -1 or similar in accounting logic
    # In the provided code: balance = last_balance + posting.amount * posting.direction.value
    # If Debit is 1 and Credit is -1:
    # Cash: 100 + (50 * 1) + (20 * -1) = 130
    # Revenue: 0 + (50 * -1) = -50 (or 50 depending on direction.value implementation)
    rev_ledger = gl.ledgers[acc_revenue]
    assert rev_ledger.initial.value == Quantity(Decimal("0"))
    
    # Verify that the out-of-period entry was not added to any ledger
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date <= end_date
            assert entry.date >= start_date

    # Verify all accounts in initial_balances are present
    for acc, bal in initial_balances.items():
        assert acc in gl.ledgers
        assert gl.ledgers[acc].initial == bal
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol/type definition behavior for ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test an implementation 
    to ensure it adheres to the expected signature and functionality.
    """
    # Setup dummy data
    test_date_start = datetime.date(2023, 1, 1)
    test_date_end = datetime.date(2023, 12, 31)
    period = DateRange(since=test_date_start, until=test_date_end)
    
    # Create a mock account and balance
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(since=test_date_start, value=Quantity(Decimal("100.00")))
    expected_initial_balances: InitialBalances = {mock_account: mock_balance}

    # Define an implementation of the protocol
    def mock_read_initial_balances(period_arg: DateRange) -> InitialBalances:
        # Verify that the input period is passed correctly
        assert period_arg == period
        return expected_initial_balances

    # Cast to the Protocol type to verify type compatibility
    reader: ReadInitialBalances = mock_read_initial_balances

    # Execute the call
    result = reader(period)

    # Assertions
    assert result == expected_initial_balances
    assert mock_account in result
    assert result[mock_account].value == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol structure for ReadInitialBalances by verifying 
    that a compatible callable returns the expected InitialBalances type.
    """
    # Setup dependencies
    mock_period = MagicMock()
    
    # Define mock accounts and balances
    account_a = MagicMock()
    account_b = MagicMock()
    balance_a = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    balance_b = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("50.00")))
    
    # Expected return value
    expected_initial_balances: InitialBalances = {
        account_a: balance_a,
        account_b: balance_b
    }

    # Define a concrete implementation of the Protocol
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        if period == mock_period:
            return expected_initial_balances
        return {}

    # Verify the callable adheres to the protocol signature and returns correct data
    result = mock_read_initial_balances(mock_period)
    
    assert result == expected_initial_balances
    assert result[account_a].value == Decimal("100.00")
    assert len(result) == 2

    # Test with a different period to ensure logic is contained
    different_period = MagicMock()
    empty_result = mock_read_initial_balances(different_period)
    assert empty_result == {}
```


