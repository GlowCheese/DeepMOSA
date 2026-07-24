####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the __call__ method of the function returned by compile_general_ledger_program.
    The test verifies that the program correctly orchestrates the reading of initial balances
    and journal entries, and uses build_general_ledger to produce the final result.
    """
    # 1. Setup Test Data
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(since=period_start, until=period_end)
    
    # Mock Account and Balance
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(period_start, Quantity(Decimal("100.00")))
    initial_balances = {mock_account: mock_balance}
    
    # Mock Journal Entry
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    # Mock a posting within the period
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT # Assuming Direction is available
    mock_journal_entry.postings = [mock_posting]
    
    journal_entries = [mock_journal_entry]

    # 2. Setup Mocks for the Algebra (Protocols)
    read_initial_balances_mock = MagicMock(spec=ReadInitialBalances)
    read_initial_balances_mock.return_value = initial_balances
    
    read_journal_entries_mock = MagicMock(spec=ReadJournalEntries)
    read_journal_entries_mock.return_value = journal_entries

    # 3. Compile the Program
    program = compile_general_ledger_program(
        read_initial_balances=read_initial_balances_mock,
        read_journal_entries=read_journal_entries_mock
    )

    # 4. Execute the Program
    result_gl = program(period)

    # 5. Assertions
    # Verify that the algebra implementations were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)

    # Verify the type of the returned object
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period

    # Verify that the ledger for the account was created and populated
    assert mock_account in result_gl.ledgers
    ledger = result_gl.ledgers[mock_account]
    
    # Check if the posting was applied to the ledger
    # Initial 100 + (50 * 1) = 150
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
    assert ledger.entries[0].balance.value == Quantity(Decimal("150.00"))
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Since ReadInitialBalances is a Protocol (interface), we cannot instantiate it directly.
    This test verifies that a valid implementation of the protocol behaves as expected.
    """
    # Setup
    period = MagicMock()
    # Define a mock implementation of the protocol
    class MockReadInitialBalances:
        def __call__(self, period) -> InitialBalances:
            # Return a dummy balance for a dummy account
            dummy_account = MagicMock(spec=Account)
            return {dummy_account: Balance(period, Quantity(Decimal("100.00")))}

    read_initial_balances = MockReadInitialBalances()
    
    # Define expected outputs
    expected_account = MagicMock(spec=Account)
    expected_balance_value = Decimal("100.00")

    # Execute
    result = read_initial_balances(period)

    # Assert
    assert isinstance(result, dict)
    assert len(result) == 1
    # Check that the returned balance corresponds to the period passed
    for account, balance in result.items():
        assert balance.value == Quantity(expected_balance_value)
        # Depending on implementation, we check if the logic inside the mock worked
        assert balance.date_range == period
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol implementation/usage of ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test a mock/concrete 
    implementation that adheres to its signature.
    """
    # Setup dependencies
    test_date_range = MagicMock()
    test_account = MagicMock()
    test_balance = MagicMock()
    
    # Define the expected return value
    expected_initial_balances = {
        test_account: test_balance
    }

    # Create a concrete implementation of the Protocol
    def mock_read_initial_balances(period) -> InitialBalances:
        assert period == test_date_range
        return expected_initial_balances

    # Verify the function signature matches the Protocol
    # We use a type hint check or simply verify the execution logic
    read_initial_balances_impl: ReadInitialBalances = mock_read_initial_balances

    # Execute the call
    result = read_initial_balances_impl(test_date_range)

    # Assertions
    assert result == expected_initial_balances
    assert test_account in result
    assert result[test_account] == test_balance
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol implementation of ReadInitialBalances by verifying 
    that a conforming callable returns the expected InitialBalances dictionary.
    """
    # Setup
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create dummy accounts
    account_a = Account("Asset Account")
    account_b = Account("Liability Account")
    
    # Define expected return value
    expected_balances = {
        account_a: Balance(period_start, Quantity(Decimal("100.00"))),
        account_b: Balance(period_start, Quantity(Decimal("50.00")))
    }
    
    # Create a mock implementation of the ReadInitialBalances protocol
    # Since ReadInitialBalances is a Protocol, any callable with the 
    # correct signature satisfies it.
    mock_reader = MagicMock(spec=ReadInitialBalances)
    mock_reader.return_value = expected_balances

    # Execution
    result = mock_reader(period)

    # Assertion
    assert result == expected_balances
    assert result[account_a].value == Quantity(Decimal("100.00"))
    assert result[account_b].value == Quantity(Decimal("50.00"))
    mock_reader.assert_called_once_with(period)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup dates
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(period_start, period_end)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances
    # Cash starts with 1000
    initial_balances = {
        acc_cash: Balance(period_start, Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue of 500 (Debit Cash, Credit Revenue) - Inside period
    date1 = datetime.date(2023, 6, 1)
    journal_entry1 = MagicMock(spec=JournalEntry)
    journal_entry1.date = date1
    posting1_debit = MagicMock(spec=Posting)
    posting1_debit.account = acc_cash
    posting1_debit.amount = Amount(Decimal("500.00"))
    posting1_debit.direction = Direction.DEBIT
    
    posting1_credit = MagicMock(spec=Posting)
    posting1_credit.account = acc_revenue
    posting1_credit.amount = Amount(Decimal("500.00"))
    posting1_credit.direction = Direction.CREDIT
    
    journal_entry1.postings = [posting1_debit, posting1_credit]
    journal_entry1.description = "Service Revenue"

    # Entry 2: Expense of 200 (Debit Expense, Credit Cash) - Inside period
    date2 = datetime.date(2023, 7, 1)
    journal_entry2 = MagicMock(spec=JournalEntry)
    journal_entry_2.date = date2
    posting2_debit = MagicMock(spec=Posting)
    posting2_debit.account = acc_expense
    posting2_debit.amount = Amount(Decimal("200.00"))
    posting2_debit.direction = Direction.DEBIT
    
    posting2_credit = MagicMock(spec=Posting)
    posting2_credit.account = acc_cash
    posting2_credit.amount = Amount(Decimal("200.00"))
    posting2_credit.direction = Direction.CREDIT
    
    journal_entry2.postings = [posting2_debit, posting2_credit]
    journal_entry2.description = "Office Supplies"

    # Entry 3: Out of period (Should be ignored)
    date3 = datetime.date(2022, 12, 31)
    journal_entry3 = MagicMock(spec=JournalEntry)
    journal_entry3.date = date3
    posting3 = MagicMock(spec=Posting)
    posting3.account = acc_cash
    posting3.amount = Amount(Decimal("100.00"))
    posting3.direction = Direction.DEBIT
    journal_entry3.postings = [posting3]
    journal_entry3.description = "Old Entry"

    journal = [journal_entry1, journal_entry2, journal_entry3]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    # 1. Check that the period is correct
    assert gl.period == period

    # 2. Check Cash Ledger (Initial 1000 + 500 - 200 = 1300)
    assert acc_cash in gl.ledgers
    cash_ledger = gl.ledgers[acc_cash]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Entries in Cash: +500 (from entry 1), -200 (from entry 2)
    # Note: build_general_ledger adds entries in order of iteration through journal postings
    assert len(cash_ledger.entries) == 2
    
    # Check final balance of Cash
    # Last entry for cash should be the 200 credit
    # We find the entry where amount is 200 and direction was credit
    cash_entries = [e for e in cash_ledger.entries if e.amount == Amount(Decimal("200.00"))]
    assert len(cash_entries) == 1
    # 1000 + 500 - 200 = 1300
    assert cash_ledger._last_balance == Quantity(Decimal("1300.00"))

    # 3. Check Revenue Ledger (Started at 0, +500)
    assert acc_revenue in gl.ledgers
    revenue_ledger = gl.ledgers[acc_revenue]
    assert revenue_ledger.initial.value == Quantity(Decimal("0.00"))
    assert revenue_ledger._last_balance == Quantity(Decimal("500.00"))

    # 4. Check Expense Ledger (Started at 0, +200)
    assert acc_expense in gl.ledgers
    expense_ledger = gl.ledgers[acc_expense]
    assert expense_ledger._last_balance == Quantity(Decimal("200.00"))

    # 5. Verify that the out-of-period entry was not processed
    # Check that no ledger has an entry with amount 100
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.amount != Amount(Decimal("100.00"))
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
    Tests the protocol implementation of ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test a concrete implementation 
    to verify the expected behavior of a callable following this signature.
    """
    # Arrange
    period = MagicMock()
    # Define expected return value: a dictionary mapping Accounts to Balances
    mock_account = MagicMock()
    mock_balance = MagicMock()
    expected_balances = {mock_account: mock_balance}

    # Create a concrete implementation of the protocol
    def mock_read_initial_balances(p) -> InitialBalances:
        if p == period:
            return expected_balances
        return {}

    # Act
    # We simulate the behavior of a function/class instance that implements the protocol
    result = mock_read_initial_balances(period)

    # Assert
    assert result == expected_balances
    assert result[mock_account] == mock_balance
    assert len(result) == 1

    # Test with different period to ensure logic is tied to the input
    assert mock_read_initial_balances(MagicMock()) == {}
```


# LLM-generated content at query #7
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
from .general_ledger import build_general_ledger, GeneralLedger, Ledger

def test_build_general_ledger():
    # 1. Setup Test Data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(start_date, end_date)

    # Mock Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Initial Balances (Cash starts with 100)
    initial_balances = {
        acc_cash: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    }

    # Mock Journal Entry 1: Revenue of 50 (Debit Cash, Credit Revenue)
    # Note: In accounting logic, Revenue increases with Credit. 
    # Here we use the provided Posting structure.
    j1_date = datetime.date(2023, 1, 15)
    j1_desc = "Service Revenue"
    
    # Posting 1: Debit Cash 50
    p1 = MagicMock(spec=Posting)
    p1.account = acc_cash
    p1.amount = Quantity(Decimal("50.00"))
    p1.direction = Direction.DEBIT
    p1.is_debit = True
    p1.is_credit = False

    # Posting 2: Credit Revenue 50
    p2 = MagicMock(spec=Posting)
    p2.account = acc_revenue
    p2.amount = Quantity(Decimal("50.00"))
    p2.direction = Direction.CREDIT
    p2.is_debit = False
    p2.is_credit = True

    j1_postings = [p1, p2]
    j1_journal = MagicMock(spec=JournalEntry)
    j1_journal.date = j1_date
    j1_journal.description = j1_desc
    j1_journal.postings = j1_postings
    
    # Mock Journal Entry 2: Expense of 20 (Debit Expense, Credit Cash)
    # This happens OUTSIDE the period to test filtering
    j2_date = datetime.date(2023, 2, 1)
    j2_journal = MagicMock(spec=JournalEntry)
    j2_journal.date = j2_date
    j2_journal.postings = [] # No postings for simplicity in this specific branch
    
    journal = [j1_journal, j2_journal]

    # 2. Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # 3. Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period
    
    # Check if Cash ledger exists and contains the initial balance + the posting
    assert acc_cash in gl.ledgers
    cash_ledger = gl.ledgers[acc_cash]
    
    # Verify Cash: Initial 100 + Debit 50 = 150
    # We check the last entry's balance
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance.value == Quantity(Decimal("150.00"))
    
    # Check if Revenue ledger was created (it wasn't in initial, so it should be 0 + 50 credit)
    # Note: Depending on Direction implementation, direction.value is usually 1 or -1
    # We assume direction.value for Credit is -1 or logic handles it. 
    # Based on the code: Quantity(self._last_balance + posting.amount * posting.direction.value)
    # If direction.value for Credit is -1, then 0 + 50 * -1 = -50.
    assert acc_revenue in gl.ledgers
    revenue_ledger = gl.ledgers[acc_revenue]
    
    # Verify filtering: The entry from j2_journal should not be in the ledger because date is Feb
    # (Since j2 had no postings, we just ensure no extra entries exist)
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert period.since <= entry.date <= period.until

    # Check that all accounts mentioned in postings within period are present
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    # Setup dependencies and mocks
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Mock InitialBalances
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial_balances = {mock_account: mock_balance}
    
    # Mock JournalEntry
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction is available in scope
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Transaction"
    mock_journal_entry.postings = [mock_posting]
    
    journal_entries = [mock_journal_entry]
    
    # Define the algebras (the inputs to the compiler)
    read_initial_balances_mock = MagicMock(return_value=initial_balances)
    read_journal_entries_mock = MagicMock(return_value=journal_entries)
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances_mock,
        read_journal_entries_mock
    )
    
    # Execute the program (the __call__ of the returned function)
    result_gl = program(period)
    
    # Assertions
    # 1. Verify algebras were called with the correct period
    read_initial_balances_mock.assert_called_once_with(period)
    read_journal_entries_mock.assert_called_once_with(period)
    
    # 2. Verify the resulting GeneralLedger structure
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period
    assert mock_account in result_gl.ledgers
    
    # 3. Verify the ledger content
    ledger = result_gl.ledgers[mock_account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
    # 100 (initial) + 50 (debit) = 150
    assert ledger.entries[0].balance.value == Quantity(Decimal("150.00"))
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
    Tests the protocol behavior of ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test a valid implementation 
    to ensure it adheres to the expected signature and returns the correct type.
    """
    # Setup
    test_date_range = MagicMock()
    
    # Create a concrete implementation of the Protocol
    class MockReadInitialBalances:
        def __call__(self, period) -> InitialBalances:
            # Return a dummy balance for a dummy account
            dummy_account = MagicMock(spec=Account)
            dummy_balance = Balance(period, Quantity(Decimal("100.00")))
            return {dummy_account: dummy_balance}

    reader = MockReadInitialBalances()
    
    # Execution
    result = reader(test_date_range)
    
    # Verification
    assert isinstance(result, dict)
    assert len(result) == 1
    
    # Verify the content matches the implementation logic
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.value == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the __call__ method of the function returned by compile_general_ledger_program.
    This verifies that the compiled program correctly orchestrates the reading of 
    initial balances and journal entries to build a GeneralLedger.
    """
    # 1. Setup Mock Data
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(since=period_start, until=period_end)

    # Mock Account
    mock_account = MagicMock(spec=Account)
    
    # Mock Initial Balances
    initial_balance_val = Quantity(Decimal("100.00"))
    mock_initial_balances = {
        mock_account: Balance(since=datetime.date(2022, 12, 31), value=initial_balance_val)
    }

    # Mock Journal Entry
    # Create a posting that falls within the period
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT # Assuming Direction is available in scope
    mock_posting.is_debit = True
    mock_posting.is_credit = False

    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Transaction"
    mock_journal_entry.postings = [mock_posting]

    # 2. Setup Mocks for the Algebra (Protocols)
    mock_read_initial_balances = MagicMock(spec=ReadInitialBalances)
    mock_read_initial_balances.return_value = mock_initial_balances

    mock_read_journal_entries = MagicMock(spec=ReadJournalEntries)
    mock_read_journal_entries.return_value = [mock_journal_entry]

    # 3. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # 4. Execute the program
    result_gl = program(period)

    # 5. Assertions
    # Verify the algebra functions were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # Verify the output type
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period

    # Verify the content of the General Ledger
    assert mock_account in result_gl.ledgers
    ledger = result_gl.ledgers[mock_account]
    
    # Verify initial balance was correctly applied
    assert ledger.initial.value == initial_balance_val
    
    # Verify the posting was processed
    assert len(ledger.entries) == 1
    entry = ledger.entries[0]
    assert entry.amount == Quantity(Decimal("50.00"))
    assert entry.is_debit is True
    
    # Verify the running balance calculation (100 + 50 = 150)
    expected_balance = Quantity(Decimal("150.00"))
    assert entry.balance == expected_balance
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup DateRange
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances
    # Cash starts with 1000
    initial_balances = {
        acc_cash: Balance(start_date, Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue of 500 (Debit Cash, Credit Revenue)
    # Date: 2023-06-01
    j1_posting_cash = MagicMock(spec=Posting)
    j1_posting_cash.account = acc_cash
    j1_posting_cash.amount = Quantity(Decimal("500.00"))
    j1_posting_cash.direction = Direction.DEBIT
    
    j1_posting_rev = MagicMock(spec=Posting)
    j1_posting_rev.account = acc_revenue
    j1_posting_rev.amount = Quantity(Decimal("500.00"))
    j1_posting_rev.direction = Direction.CREDIT

    j1_journal = MagicMock(spec=JournalEntry)
    j1_journal.date = datetime.date(2023, 6, 1)
    j1_journal.description = "Service Revenue"
    j1_journal.postings = [j1_posting_cash, j1_posting_rev]

    # Entry 2: Expense of 200 (Debit Expense, Credit Cash)
    # Date: 2023-07-01
    j2_posting_exp = MagicMock(spec=Posting)
    j2_posting_exp.account = acc_expense
    j2_posting_exp.amount = Quantity(Decimal("200.00"))
    j2_posting_exp.direction = Direction.DEBIT

    j2_posting_cash_out = MagicMock(spec=Posting)
    j2_posting_cash_out.account = acc_cash
    j2_posting_cash_out.amount = Quantity(Decimal("200.00"))
    j2_posting_cash_out.direction = Direction.CREDIT

    j2_journal = MagicMock(spec=JournalEntry)
    j2_journal.date = datetime.date(2023, 7, 1)
    j2_journal.description = "Office Supplies"
    j2_journal.postings = [j2_posting_exp, j2_posting_cash_out]

    # Entry 3: Out of period (Date: 2022-12-31)
    j3_journal = MagicMock(spec=JournalEntry)
    j3_journal.date = datetime.date(2022, 12, 31)
    j3_journal.postings = []

    journal_entries = [j1_journal, j2_journal, j3_journal]

    # Execute
    gl = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    # 1. Check if all accounts are present in ledgers
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers

    # 2. Check Cash Ledger calculations
    # Initial 1000 + 500 (Debit) - 200 (Credit) = 1300
    cash_ledger = gl.ledgers[acc_cash]
    assert len(cash_ledger.entries) == 2
    # Check final balance via last entry
    # We need to find the entry that corresponds to the specific posting
    # Since order in entries depends on iteration, we check the logic
    final_cash_balance = cash_ledger._last_balance
    assert final_cash_balance == Quantity(Decimal("1300.00"))

    # 3. Check Revenue Ledger (Starts at 0)
    rev_ledger = gl.ledgers[acc_revenue]
    assert rev_ledger.initial.value == Quantity(Decimal("0.00"))
    # Revenue is Credit, so 0 - 500 = -500 (or depending on how direction.value is implemented)
    # Based on code: balance + amount * direction.value. 
    # If Credit direction.value is -1: 0 + 500 * -1 = -500
    assert rev_ledger._last_balance == Quantity(Decimal("-500.00"))

    # 4. Check Expense Ledger (Starts at 0)
    exp_ledger = gl.ledgers[acc_expense]
    assert exp_ledger._last_balance == Quantity(Decimal("200.00"))

    # 5. Verify Out of Period entry was ignored
    # Total entries across all ledgers should be 4 (2 from j1, 2 from j2)
    total_entries = sum(len(l.entries) for l in gl.ledgers.values())
    assert total_entries == 4
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
    Tests the protocol/type definition behavior for ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test that a compatible 
    callable implementation functions as expected.
    """
    # Setup period
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(since=period_start, until=period_end)

    # Setup mock accounts and balances
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(since=period_start, value=Quantity(Decimal("100.00")))
    expected_initial_balances = {mock_account: mock_balance}

    # Define a concrete implementation of the ReadInitialBalances protocol
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        return expected_initial_balances

    # Verify implementation matches the protocol signature and returns expected data
    result = mock_read_initial_balances(period)

    assert result == expected_initial_balances
    assert result[mock_account].value == Quantity(Decimal("100.00"))
    assert result[mock_account].since == period_start

    # Test with a different period to ensure it's a functional callable
    def mock_read_initial_balances_empty(p: DateRange) -> InitialBalances:
        return {}

    assert mock_read_initial_balances_empty(period) == {}
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
    Tests the functionality of the compiled GeneralLedgerProgram's __call__ method.
    The test verifies that the program correctly orchestrates reading initial balances,
    reading journal entries, and building the GeneralLedger within the specified period.
    """
    # 1. Setup Mock Data and Dependencies
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(period_start, period_end)

    # Mock Account
    mock_account = MagicMock(spec=Account)
    
    # Mock Initial Balances
    mock_initial_balances = {
        mock_account: Balance(period_start, Quantity(Decimal("100.00")))
    }

    # Mock Journal Entry and Postings
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT  # Assuming Direction enum exists
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    mock_journal_entry.description = "Test Transaction"
    mock_journal_entry.postings = [mock_posting]

    # 2. Setup Mock Algebra Implementations (Protocols)
    mock_read_initial_balances = MagicMock(spec=ReadInitialBalances)
    mock_read_initial_balances.return_value = mock_initial_balances

    mock_read_journal_entries = MagicMock(spec=ReadJournalEntries)
    mock_read_journal_entries.return_value = [mock_journal_entry]

    # 3. Compile the Program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # 4. Execute the Program (__call__)
    general_ledger = program(period)

    # 5. Assertions
    # Verify the algebra implementations were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # Verify the resulting GeneralLedger structure
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert mock_account in general_ledger.ledgers
    
    # Verify the ledger state calculation
    ledger = general_ledger.ledgers[mock_account]
    assert ledger.initial.value == Quantity(Decimal("100.00"))
    
    # Check if the posting was processed
    # 100 (initial) + 50 (debit) = 150
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
    assert ledger.entries[0].balance.value == Quantity(Decimal("150.00"))
    assert ledger.entries[0].is_debit is True
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol implementation by verifying that a callable 
    conforming to ReadInitialBalances returns the expected InitialBalances.
    """
    # Setup
    test_date_start = datetime.date(2023, 1, 1)
    test_date_end = datetime.date(2023, 12, 31)
    period = DateRange(since=test_date_start, until=test_date_end)
    
    # Mocking accounts
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Account)
    
    # Mocking initial balances data
    balance_a = Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("100.00")))
    balance_b = Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("50.00")))
    expected_initial_balances: InitialBalances = {
        account_a: balance_a,
        account_b: balance    # Note: Using variable names to match logic
    }
    # Re-defining clearly for the test
    expected_initial_balances = {
        account_a: balance_a,
        account_b: Balance(since=datetime.date(2022, 12, 31), value=Quantity(Decimal("50.00")))
    }

    # Define the mock implementation of the protocol
    def mock_read_initial_balances(period_arg: DateRange) -> InitialBalances:
        assert period_arg == period
        return expected_initial_balances

    # Execute
    result = mock_read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert result[account_a].value == Decimal("100.00")
    assert result[account_b].value == Decimal("50.00")
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
    Tests the __call__ method of the function returned by compile_general_ledger_program,
    which satisfies the GeneralLedgerProgram protocol.
    """
    # 1. Setup Mock Dependencies
    # We need a mock for ReadInitialBalances
    mock_read_initial_balances = MagicMock()
    # We need a mock for ReadJournalEntries
    mock_read_journal_entries = MagicMock()

    # 2. Define Test Data
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(since=period_start, until=period_end)

    # Mock initial balances
    account_a = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {account_a: Balance(since=datetime.date(2022, 12, 31), value=initial_balance_val)}
    mock_read_initial_balances.return_value = initial_balances

    # Mock journal entries
    # We need a mock JournalEntry with a date within the period and a posting
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    
    account_b = MagicMock(spec=Account)
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = account_b
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = MagicMock()
    mock_posting.direction.value = Decimal("1") # Debit
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_journal_entry.postings = [mock_posting]
    mock_read_journal_entries.return_value = [mock_journal_entry]

    # 3. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # 4. Execute the program (The __call__ implementation)
    gl = program(period)

    # 5. Assertions
    # Verify dependencies were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # Verify the returned object is a GeneralLedger
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period

    # Verify the ledger for Account A (from initial balances) exists and has initial value
    assert account_a in gl.ledgers
    assert gl.ledgers[account_a].initial.value == initial_balance_val
    # Account A should have no entries because no postings were made to it in our mock
    assert len(gl.ledgers[account_a].entries) == 0

    # Verify the ledger for Account B (from journal entries) exists and was updated
    assert account_b in gl.ledgers
    assert len(gl.ledgers[account_b].entries) == 1
    assert gl.ledgers[account_b].entries[0].amount == Quantity(Decimal("50.00"))
    assert gl.ledgers[account_b]._last_balance == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    # Arrange
    # Setup period
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(period_start, period_end)

    # Mock the dependencies (the algebra)
    # We need to mock the function returned by compile_general_ledger_program
    # which satisfies the GeneralLedgerProgram protocol.
    mock_read_initial_balances = MagicMock()
    mock_read_journal_entries = MagicMock()

    # Mock return values for the algebra
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(period_start, Quantity(Decimal("100.00")))
    mock_initial_balances = {mock_account: mock_balance}
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = mock_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = Direction.DEBIT # Assuming Direction enum exists
    mock_journal_entry.postings = [mock_posting]
    
    mock_journal_entries = [mock_journal_entry]

    mock_read_initial_balances.return_value = mock_initial_balances
    mock_read_journal_entries.return_value = mock_journal_entries

    # The program is the function produced by compile_general_ledger_program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Act
    result = program(period)

    # Assert
    # Verify the algebra was called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # Verify the returned object is a GeneralLedger
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    
    # Verify the contents of the ledger were built correctly from the mocks
    assert mock_account in result.ledgers
    ledger = result.ledgers[mock_account]
    
    # Check that the initial balance was applied
    assert ledger.initial.value == Decimal("100.00")
    
    # Check that the posting was processed
    # 100 (initial) + 50 (debit) = 150
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
    assert ledger.entries[0].balance.value == Decimal("150.00")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the protocol/type definition behavior of ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test that a compatible 
    callable object behaves as expected when invoked.
    """
    # Setup
    period = MagicMock() # Represents DateRange
    mock_account = MagicMock() # Represents Account
    mock_balance_value = MagicMock() # Represents Quantity/Balance
    
    # Create a concrete implementation of the protocol
    def mock_read_initial_balances(p) -> dict:
        # Return a dummy InitialBalances dict
        return {mock_account: MagicMock(value=mock_balance_value)}

    # Define the expected return value
    expected_initial_balances = {mock_account: MagicMock(value=mock_balance_value)}

    # Execute
    result = mock_read_initial_balances(period)

    # Assert
    assert result == expected_initial_balances
    assert mock_account in result
    assert result[mock_account].value == mock_balance_value
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal
from ..commons.numbers import Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance

def test_ReadInitialBalances___call__():
    """
    Tests the protocol-based signature of ReadInitialBalances by verifying 
    that a callable implementation returns the expected InitialBalances.
    """
    # Setup test data
    test_period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Create mock accounts and balances
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Account)
    
    balance_a = Balance(since=date(2023, 1, 1), value=Quantity(Decimal("100.00")))
    balance_b = Balance(since=date(2023, 1, 1), value=Quantity(Decimal("50.00")))
    
    expected_initial_balances = {
        account_a: balance_a,
        account_b: balance_b
    }

    # Define a mock implementation of the ReadInitialBalances protocol
    def mock_read_initial_balances(period: DateRange) -> dict:
        # In a real scenario, this would use the 'period' to query a database
        if period == test_period:
            return expected_initial_balances
        return {}

    # Verify the callable behavior
    # Test 1: Correct period returns correct balances
    result = mock_read_initial_balances(test_period)
    assert result == expected_initial_balances
    assert result[account_a].value == Quantity(Decimal("100.00"))
    assert result[account_b].value == Quantity(Decimal("50.00"))

    # Test 2: Different period returns empty balances (as per our mock logic)
    other_period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    assert mock_read_initial_balances(other_period) == {}
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup common test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create mock accounts
    account_cash = MagicMock(spec=Account)
    account_revenue = MagicMock(spec=Account)
    account_expense = MagicMock(spec=Account)
    
    # Setup Initial Balances
    # Cash starts with 1000
    initial_balances = {
        account_cash: Balance(start_date, Quantity(Decimal("1000.00")))
    }
    
    # Setup Journal Entries
    # Entry 1: Revenue of 500 (Debit Cash, Credit Revenue)
    journal_entry_1 = MagicMock(spec=JournalEntry)
    journal_entry_1.date = datetime.date(2023, 6, 1)
    journal_entry_1.description = "Service Revenue"
    
    posting_cash_dr = MagicMock(spec=Posting)
    posting_cash_dr.account = account_cash
    posting_cash_dr.amount = Amount(Decimal("500.00"))
    posting_cash_dr.direction = Direction.DEBIT
    
    posting_rev_cr = MagicMock(spec=Posting)
    posting_rev_cr.account = account_revenue
    posting_rev_cr.amount = Amount(Decimal("500.00"))
    posting_rev_cr.direction = Direction.CREDIT
    
    journal_entry_1.postings = [posting_cash_dr, posting_rev_cr]
    
    # Entry 2: Expense of 200 (Debit Expense, Credit Cash) - OUTSIDE period
    journal_entry_out_of_period = MagicMock(spec=JournalEntry)
    journal_entry_out_of_period.date = datetime.date(2024, 1, 1)
    
    posting_exp_dr = MagicMock(spec=Posting)
    posting_exp_dr.account = account_expense
    posting_exp_dr.amount = Amount(Decimal("200.00"))
    posting_exp_dr.direction = Direction.DEBIT
    
    posting_cash_cr = MagicMock(spec=Posting)
    posting_cash_cr.account = account_cash
    posting_cash_cr.amount = Amount(Decimal("200.00"))
    posting_cash_cr.direction = Direction.CREDIT
    
    journal_entry_out_of_period.postings = [posting_exp_dr, posting_cash_cr]
    
    # Entry 3: Expense of 100 (Debit Expense, Credit Cash) - INSIDE period
    journal_entry_2 = MagicMock(spec=JournalEntry)
    journal_entry_2.date = datetime.date(2023, 7, 1)
    journal_entry_2.description = "Office Supplies"
    
    posting_exp_dr_2 = MagicMock(spec=Posting)
    posting_exp_dr_2.account = account_expense
    posting_exp_dr_2.amount = Amount(Decimal("100.00"))
    posting_exp_dr_2.direction = Direction.DEBIT
    
    posting_cash_cr_2 = MagicMock(spec=Posting)
    posting_cash_cr_2.account = account_cash
    posting_cash_cr_2.amount = Amount(Decimal("100.00"))
    posting_cash_cr_2.direction = Direction.CREDIT
    
    journal_entry_2.postings = [posting_exp_dr_2, posting_cash_cr_2]

    journal = [journal_entry_1, journal_entry_out_of_period, journal_entry_2]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert gl.period == period
    
    # Verify Cash Ledger (Initial 1000 + 500 - 100 = 1400)
    assert account_cash in gl.ledgers
    cash_ledger = gl.ledgers[account_cash]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # 1000 + 500 (dr) - 100 (cr) = 1400
    assert cash_ledger._last_balance == Quantity(Decimal("1400.00"))
    # Should have 2 entries (the 500 dr and the 100 cr)
    assert len(cash_ledger.entries) == 2

    # Verify Revenue Ledger (Started at 0 + 500 cr = 500)
    assert account_revenue in gl.ledgers
    rev_ledger = gl.ledgers[account_revenue]
    assert rev_ledger._last_balance == Quantity(Decimal("500.00"))

    # Verify Expense Ledger (Started at 0 + 100 dr = 100)
    # Note: The 200 expense was out of period, so it shouldn't be here
    assert account_expense in gl.ledgers
    exp_ledger = gl.ledgers[account_expense]
    assert exp_ledger._last_balance == Quantity(Decimal("100.00"))
    
    # Verify that the out-of-period entry was ignored
    # The expense ledger should only have the 100 entry
    assert len(exp_ledger.entries) == 1
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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
from .general_ledger import build_general_ledger, GeneralLedger, Ledger

def test_build_general_ledger():
    # Setup dates
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 1, 31)
    period = DateRange(start_date, end_date)

    # Setup Accounts
    cash_account = Account("Cash")
    revenue_account = Account("Revenue")
    expense_account = Account("Expense")

    # Setup Initial Balances
    # Cash starts with 1000
    initial_balances = {
        cash_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # 1. Revenue Entry: Cash increases (Debit), Revenue increases (Credit)
    # Date: 2023-01-15
    journal_desc = "Sales Revenue"
    j1_date = datetime.date(2023, 1, 15)
    
    posting_cash_dr = Posting(
        account=cash_account,
        amount=Amount(Decimal("500.00")),
        direction=Direction.DEBIT,
        date=j1_date,
        journal=MagicMock(description=journal_desc)
    )
    # We need to mock the journal object to return the correct postings for the counter-account logic
    j1_journal = MagicMock()
    j1_journal.description = journal_desc
    j1_journal.postings = [
        MagicMock(account=cash_account, direction=Direction.DEBIT, amount=Amount(Decimal("500.00"))),
        MagicMock(account=revenue_account, direction=Direction.CREDIT, amount=Amount(Decimal("500.00")))
    ]
    
    posting_rev_cr = Posting(
        account=revenue_account,
        amount=Amount(Discount(Decimal("500.00"))), # Using amount logic
        direction=Direction.CREDIT,
        date=j1_date,
        journal=j1_journal
    )
    # Re-assigning amount correctly for the test
    posting_rev_cr.amount = Amount(Decimal("500.00"))

    j1 = JournalEntry(date=j1_date, postings=[posting_cash_dr, posting_rev_cr], description=journal_desc)

    # 2. Out of period Entry (should be ignored)
    j2_date = datetime.date(2022, 12, 25)
    j2_posting = Posting(
        account=cash_account,
        amount=Amount(Decimal("100.00")),
        direction=Direction.DEBIT,
        date=j2_date,
        journal=MagicMock(description="Old Entry")
    )
    j2 = JournalEntry(date=j2_date, postings=[j2_posting], description="Old Entry")

    # 3. Expense Entry: Expense increases (Debit), Cash decreases (Credit)
    # Date: 2023-01-20
    j3_date = datetime.date(2023, 1, 20)
    j3_journal = MagicMock()
    j3_journal.description = "Office Supplies"
    j3_journal.postings = [
        MagicMock(account=expense_account, direction=Direction.DEBIT, amount=Amount(Decimal("50.00"))),
        MagicMock(account=cash_account, direction=Direction.CREDIT, amount=Amount(Decimal("50.00")))
    ]
    
    posting_exp_dr = Posting(
        account=expense_account,
        amount=Amount(Decimal("50.00")),
        direction=Direction.DEBIT,
        date=j3_date,
        journal=j3_journal
    )
    posting_cash_cr = Posting(
        account=cash_account,
        amount=Amount(Decimal("50.00")),
        direction=Direction.CREDIT,
        date=j3_date,
        journal=j3_journal
    )
    j3 = JournalEntry(date=j3_date, postings=[posting_exp_dr, posting_cash_cr], description="Office Supplies")

    journal_entries = [j1, j2, j3]

    # Execution
    gl = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert isinstance(gl, GeneralLedger)
    assert gl.period == period

    # Check Cash Ledger
    # Initial 1000 + 500 (Dr) - 50 (Cr) = 1450
    assert cash_account in gl.ledgers
    cash_ledger = gl.ledgers[cash_account]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Check that the out-of-period entry was ignored
    # Only 2 entries should exist in Cash: the 500 Dr and the 50 Cr
    assert len(cash_ledger.entries) == 2
    assert cash_ledger._last_balance.value == Quantity(Decimal("1450.00"))

    # Check Revenue Ledger
    # Started at 0 (since not in initial) + 500 (Cr) = 500
    assert revenue_account in gl.ledgers
    rev_ledger = gl.ledgers[revenue_account]
    assert rev_ledger.initial.value == Quantity(Decimal("0.00"))
    assert rev_ledger._last_balance.value == Quantity(Decimal("500.00"))

    # Check Expense Ledger
    # Started at 0 + 50 (Dr) = 50
    assert expense_account in gl.ledgers
    exp_ledger = gl.ledgers[expense_account]
    assert exp_ledger._last_balance.value == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup dates
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 1, 31)
    period = DateRange(period_start, period_end)

    # Setup Accounts
    account_cash = Account("Cash")
    account_revenue = Account("Revenue")
    account_expense = Account("Expense")

    # Setup Initial Balances
    # Cash starts with 1000
    initial_balances = {
        account_cash: Balance(period_start, Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # 1. Revenue entry (Debit Cash 500, Credit Revenue 500) - Within period
    journal_entry_1 = MagicMock(spec=JournalEntry)
    journal_entry_1.date = datetime.date(2023, 1, 15)
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

    # 2. Expense entry (Debit Expense 200, Credit Cash 200) - Within period
    journal_entry_2 = MagicMock(spec=JournalEntry)
    journalger_entry_2.date = datetime.date(2023, 1, 20)
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

    # 3. Out of period entry (Should be ignored)
    journal_entry_3 = MagicMock(spec=JournalEntry)
    journal_entry_3.date = datetime.date(2022, 12, 31)
    journal_entry_3.description = "Old Entry"
    posting_old = MagicMock(spec=Posting)
    posting_old.account = account_cash
    posting_old.amount = Quantity(Disimal("100.00"))
    posting_old.direction = Direction.DEBIT
    journal_entry_3.postings = [posting_old]

    journal_iterable = [journal_entry_1, journal_entry_2, journal_entry_3]

    # Execute
    gl = build_general_ledger(period, journal_iterable, initial_balances)

    # Assertions
    assert gl.period == period
    assert account_cash in gl.ledgers
    assert account_revenue in gl.ledgers
    assert account_expense in gl.ledgers

    # Verify Cash Ledger calculation: 1000 (initial) + 500 (dr) - 200 (cr) = 1300
    cash_ledger = gl.ledgers[account_cash]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Check that the entries only include the valid period ones
    # Entries: [Dr 500, Cr 200]
    assert len(cash_ledger.entries) == 2
    
    # Verify final balance of cash
    last_cash_entry = cash_ledger.entries[-1]
    assert last_cash_entry.balance.value == Quantity(Decimal("1300.00"))

    # Verify Revenue Ledger: 0 (initial) + 500 (cr) = 500 (if we treat credit as positive in this context)
    # Note: The logic in the provided code uses direction.value (Debit=1, Credit=-1 usually)
    # Let's check the balance based on the code's logic: balance + amount * direction.value
    rev_ledger = gl.ledgers[account_revenue]
    # 0 + 500 * (-1) = -500
    assert rev_ledger.entries[0].balance.value == Quantity(Decimal("-500.00"))

    # Verify Expense Ledger: 0 + 200 * (1) = 200
    exp_ledger = gl.ledgers[account_expense]
    assert exp_ledger.entries[0].balance.value == Quantity(Decimal("200.00"))
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_ReadInitialBalances___call__():
    """
    Tests the behavior of a function implementing the ReadInitialBalances protocol.
    Since ReadInitialBalances is a Protocol (interface), we test a concrete 
    implementation (a mock or a lambda) to verify it adheres to the signature.
    """
    # Setup
    test_date_range = MagicMock()
    test_date_range.since = datetime.date(2023, 1, 1)
    test_date_range.until = datetime.date(2023, 12, 31)
    
    # Create dummy accounts
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Account)
    
    # Define the expected return value (InitialBalances is Dict[Account, Balance])
    expected_balances = {
        account_a: Balance(test_date_range.since, Quantity(Decimal("100.00"))),
        account_b: Balance(test_date_range.since, Quantity(Decimal("50.00")))
    }

    # Implementation of the protocol using a Mock
    # This simulates a function that 'reads' balances for a given period
    read_initial_balances_impl: ReadInitialBalances = MagicMock(return_value=expected_balances)

    # Execute
    result = read_initial_balances_impl(test_date_range)

    # Assertions
    read_initial_balances_impl.assert_called_once_with(test_date_range)
    assert result == expected_balances
    assert result[account_a].value == Decimal("100.00")
    assert result[account_b].value == Decimal("50.00")
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal
from ..commons.numbers import Quantity
from ..commons.zeitgeist import DateRange
from ..accounts import Account
from ..generic import Balance

def test_ReadInitialBalances___call__():
    """
    Tests the protocol behavior of ReadInitialBalances by verifying that 
    a callable implementation correctly returns InitialBalances.
    """
    # Setup
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))
    
    expected_initial_balances = {
        mock_account: mock_balance
    }

    # Create a concrete implementation of the ReadInitialBalances protocol
    def mock_read_initial_balances(period: DateRange) -> dict:
        # Verify the input passed to the callable
        assert period == test_period
        return expected_initial_balances

    # The protocol is a type hint, so we test a function matching the signature
    # We use a spy/mock to ensure the call is intercepted
    spy_read_initial_balances = MagicMock(side_effect=mock_read_initial_balances)

    # Execution
    result = spy_read_initial_balances(test_period)

    # Assertions
    spy_read_initial_balances.assert_called_once_with(test_period)
    assert result == expected_initial_balances
    assert result[mock_account].value == Quantity(Decimal("100.00"))
```


# LLM-generated content at query #5
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

    # Mocking the algebras (ReadInitialBalances and ReadJournalEntries)
    mock_read_initial_balances = MagicMock()
    mock_read_journal_entries = MagicMock()

    # Mocking the data returned by the algebras
    mock_initial_balances = {
        MagicMock(spec=Account): Balance(period.since, Quantity(Decimal("100.00")))
    }
    mock_journal_entries = [
        MagicMock(
            spec=JournalEntry,
            date=datetime.date(2023, 6, 1),
            postings=[
                MagicMock(
                    spec=Posting,
                    account=MagicMock(spec=Account),
                    amount=Quantity(Decimal("50.00")),
                    direction=MagicMock(value=Decimal("1")),
                    is_debit=True,
                    is_credit=False
                )
            ]
        )
    ]

    mock_read_initial_balances.return_value = mock_initial_balances
    mock_read_journal_entries.return_value = mock_journal_entries

    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )

    # Execute the program (the __call__ method of the returned function)
    result = program(period)

    # Assertions
    # 1. Verify that the algebras were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)

    # 2. Verify the result is a GeneralLedger instance
    assert isinstance(result, GeneralLedger)
    assert result.period == period

    # 3. Verify the ledgers were built correctly from the mocked data
    assert len(result.ledgers) > 0
    
    # Verify that the data flow from algebra to build_general_ledger was intact
    # If the program works, the internal build_general_ledger should have processed the mock_journal_entries
    # We check if the ledger for the account in our mock journal exists and has entries
    found_entry_processed = False
    for ledger in result.ledgers.values():
        if len(ledger.entries) > 0:
            found_entry_processed = True
            break
    
    assert found_entry_processed, "The program should have processed journal entries into the ledger"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

def test_ReadInitialBalances___call__():
    """
    Tests the protocol definition/interface of ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test it by verifying 
    that a compatible function implementation behaves as expected.
    """
    # Setup period
    period_start = date(2023, 1, 1)
    period_end = date(2023, 12, 31)
    period = DateRange(period_start, period_end)

    # Setup mock accounts and balances
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(period_start, Quantity(Decimal("100.00")))
    expected_initial_balances = {mock_account: mock_balance}

    # Define a concrete implementation of the protocol
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        # In a real scenario, this would use the period to query a database
        if p.since == period_start:
            return expected_initial_balances
        return {}

    # Verify the implementation matches the protocol signature and returns correct data
    # Testing the logic of the function that satisfies the ReadInitialBalances protocol
    result = mock_read_initial_balances(period)
    
    assert result == expected_initial_balances
    assert result[mock_account].value == Decimal("100.00")
    assert result[mock_account].date == period_start

    # Test with a different period to ensure it handles the logic as defined
    other_period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    result_empty = mock_read_initial_balances(other_period)
    assert result_empty == {}
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
    Tests the ReadInitialBalances protocol behavior by verifying a 
    mock implementation adheres to the expected signature and return type.
    """
    # Setup period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    # Create mock accounts
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Account)

    # Define expected balances
    balance_a = Balance(start_date, Quantity(Decimal("100.00")))
    balance_b = Balance(start_date, Quantity(Decimal("50.00")))
    expected_initial_balances: InitialBalances = {
        account_a: balance_a,
        account_b: balance_b
    }

    # Define the mock implementation of the protocol
    mock_reader: ReadInitialBalances = MagicMock(return_value=expected_initial_balances)

    # Execute the call
    result = mock_reader(period)

    # Assertions
    mock_reader.assert_called_once_with(period)
    assert result == expected_initial_balances
    assert len(result) == 2
    assert result[account_a] == balance_a
    assert result[account_b] == balance_b
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

@pytest.mark import patch

def test_ReadInitialBalances___call__():
    """
    Tests the protocol-defined behavior of ReadInitialBalances.
    Since ReadInitialBalances is a Protocol, we test a concrete implementation 
    to ensure it adheres to the expected signature and return type.
    """
    # Arrange
    test_period = MagicMock()
    test_period.since = date(2023, 1, 1)
    test_period.until = date(2023, 12, 31)
    
    # Create a mock account and balance
    mock_account = MagicMock()
    mock_balance = MagicMock()
    mock_balance.value = Decimal("100.00")
    
    # Define the expected return value
    expected_initial_balances = {mock_account: mock_balance}
    
    # Create a concrete implementation of the protocol
    def mock_read_initial_balances(period) -> InitialBalances:
        # Simulate logic that depends on the period
        if period.since == date(2023, 1, 1):
            return expected_initial_balances
        return {}

    # Act
    # We cast to the protocol type to ensure type compliance during the test
    reader: ReadInitialBalances = mock_read_initial_balances
    actual_balances = reader(test_period)
    
    # Assert
    assert actual_balances == expected_initial_balances
    assert mock_account in actual_balances
    assert actual_balances[mock_account].value == Decimal("100.00")

    # Test with a different period to ensure it handles the input
    different_period = MagicMock()
    different_period.since = date(2024, 1, 1)
    assert reader(different_period) == {}
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark import patch

def test_ReadInitialBalances___call__():
    """
    Tests the protocol structure of ReadInitialBalances by verifying a 
    mock implementation satisfies the protocol and returns the expected type.
    """
    # Define test data
    test_date_range = MagicMock()
    test_account = MagicMock()
    test_balance_value = Decimal("100.00")
    
    # Create a mock implementation of the ReadInitialBalances protocol
    # Since it is a Protocol, any callable matching the signature works.
    mock_reader = MagicMock()
    
    # Define the expected return value (InitialBalances = Dict[Account, Balance])
    expected_balance = MagicMock()
    expected_balance.value = test_balance_value
    expected_initial_balances = {test_account: expected_balance}
    
    # Configure the mock to return our expected balances when called
    mock_reader.return_value = expected_initial_balances

    # Execution
    result = mock_reader(test_date_range)

    # Assertions
    mock_reader.assert_called_once_with(test_date_range)
    assert isinstance(result, dict)
    assert test_account in result
    assert result[test_account].value == test_balance_value
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup DateRange
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances
    # Cash starts with 1000
    initial_balances = {
        acc_cash: Balance(start_date, Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # 1. Revenue Entry (Debit Cash, Credit Revenue) - Within period
    j1_date = datetime.date(2023, 6, 1)
    j1_desc = "Service Revenue"
    p1_cash = MagicMock(spec=Posting)
    p1_cash.account = acc_cash
    p1_cash.amount = Quantity(Decimal("500.00"))
    p1_cash.direction = Direction.DEBIT
    
    p1_rev = MagicMock(spec=Posting)
    p1_rev.account = acc_revenue
    p1_rev.amount = Quantity(Decimal("500.00"))
    p1_rev.direction = Direction.CREDIT

    j1_journal = MagicMock(spec=JournalEntry)
    j1_journal.date = j1_date
    j1_journal.description = j1_desc
    j1_journal.postings = [p1_cash, p1_rev]

    # 2. Expense Entry (Debit Expense, Credit Cash) - Within period
    j2_date = datetime.date(2023, 7, 1)
    j2_desc = "Office Supplies"
    p2_exp = MagicMock(spec=Posting)
    p2_exp.account = acc_expense
    p2_exp.amount = Quantity(Decimal("100.00"))
    p2_exp.direction = Direction.DEBIT

    p2_cash = MagicMock(spec=Posting)
    p2_cash.account = acc_cash
    p2_cash.amount = Quantity(Decimal("100.00"))
    p2_cash.direction = Direction.CREDIT

    j2_journal = MagicMock(spec=JournalEntry)
    j2_journal.date = j2_date
    j2_journal.description = j2_desc
    j2_journal.postings = [p2_exp, p2_cash]

    # 3. Out of period Entry - Should be ignored
    j3_date = datetime.date(2024, 1, 1)
    j3_journal = MagicMock(spec=JournalEntry)
    j3_journal.date = j3_date
    j3_journal.postings = []

    journal = [j1_journal, j2_journal, j3_journal]

    # Execute
    gl = build_general_ledger(period, journal, initial_balances)

    # Assertions
    assert gl.period == period
    assert acc_cash in gl.ledgers
    assert acc_revenue in gl.ledgers
    assert acc_expense in gl.ledgers

    # Verify Cash Ledger (Initial 1000 + 500 - 100 = 1400)
    cash_ledger = gl.ledgers[acc_cash]
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    # Entries for Cash: 1. Debit 500, 2. Credit 100
    # We filter entries by looking at the posting amount/direction via the LedgerEntry
    # Note: build_general_ledger adds entries to the ledger list
    assert len(cash_ledger.entries) == 2
    
    # Check final balance of Cash
    # Last entry should be the credit of 100
    last_cash_entry = cash_ledger.entries[-1]
    assert last_cash_entry.balance.value == Quantity(Decimal("1400.00"))

    # Verify Revenue Ledger (Starts 0, + 500 = 500)
    rev_ledger = gl.ledgers[acc_revenue]
    assert rev_ledger.initial.value == Quantity(Decimal("0.00"))
    assert rev_ledger.entries[0].amount.value == Quantity(Decimal("500.00"))
    assert rev_ledger.entries[0].balance.value == Quantity(Decimal("500.00"))

    # Verify Expense Ledger (Starts 0, + 100 = 100)
    exp_ledger = gl.ledgers[acc_expense]
    assert exp_ledger.entries[0].amount.value == Quantity(Decimal("100.00"))
    assert exp_ledger.entries[0].balance.value == Quantity(Decimal("100.00"))

    # Verify that the out-of-period journal entry was not processed
    # (Check that no ledger has entries from j3)
    for ledger in gl.ledgers.values():
        for entry in ledger.entries:
            assert entry.date < datetime.date(2024, 1, 1)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal
from ..commons.numbers import Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance

def test_ReadInitialBalances___call__():
    """
    Tests the behavior of a function implementing the ReadInitialBalances protocol.
    Since ReadInitialBalances is a Protocol (interface), we test a concrete 
    implementation (a mock) to verify it adheres to the expected signature 
    and return type.
    """
    # Arrange
    test_period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Mocking Account and Balance data
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(since=date(2023, 1, 1), value=Quantity(Decimal("100.00")))
    expected_initial_balances = {mock_account: mock_balance}
    
    # Create a mock implementation of the ReadInitialBalances protocol
    # In Python, a Protocol is a structural type, so any callable matching 
    # the signature works.
    read_initial_balances_impl = MagicMock(return_value=expected_initial_balances)

    # Act
    # The protocol defines: __call__(self, period: DateRange) -> InitialBalances
    result = read_initial_balances_impl(test_period)

    # Assert
    # Verify the function was called with the correct period
    read_initial_balances_impl.assert_called_once_with(test_period)
    
    # Verify the returned object is of type InitialBalances (Dict[Account, Balance])
    assert isinstance(result, dict)
    assert mock_account in result
    assert isinstance(result[mock_account], Balance)
    assert result[mock_account].value == Quantity(Decimal("100.00"))
    assert result == expected_initial_balances
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    """
    Tests the __call__ method of the function returned by compile_general_ledger_program.
    The test verifies that the compiled program correctly orchestrates the 
    reading of initial balances and journal entries to build a GeneralLedger.
    """
    # 1. Setup Mocks for the Algebra implementations
    mock_read_initial_balances = MagicMock(spec=ReadInitialBalances)
    mock_read_journal_entries = MagicMock(spec=ReadJournalEntries)
    
    # 2. Define Test Data
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 1, 31)
    period = DateRange(since=period_start, until=period_end)
    
    # Mock Account and Balance
    test_account = MagicMock(spec=Account)
    initial_balance_val = Quantity(Decimal("100.00"))
    initial_balances = {test_account: Balance(period_start, initial_balance_val)}
    
    # Mock Journal Entry and Posting
    # We need a minimal structure that satisfies build_general_ledger requirements
    mock_posting = MagicMock(spec=Posting)
    mock_posting.account = test_account
    mock_posting.amount = Quantity(Decimal("50.00"))
    mock_posting.direction = MagicMock()
    mock_posting.direction.value = Decimal("1") # Debit
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 1, 15)
    mock_journal_entry.postings = [mock_posting]
    mock_journal_entry.description = "Test Transaction"
    
    journal_entries = [mock_journal_entry]
    
    # 3. Configure Mocks to return our test data
    mock_read_initial_balances.return_value = initial_balances
    mock_read_journal_entries.return_value = journal_entries
    
    # 4. Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # 5. Execute the program (the __call__ method)
    result_gl = program(period)
    
    # 6. Assertions
    # Verify that the algebra implementations were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)
    
    # Verify the structure of the returned GeneralLedger
    assert isinstance(result_gl, GeneralLedger)
    assert result_gl.period == period
    assert test_account in result_gl.ledgers
    
    # Verify the logic of the ledger calculation
    # Initial (100) + Posting (50 * 1) = 150
    ledger = result_gl.ledgers[test_account]
    expected_final_balance = Quantity(Decimal("150.00"))
    assert ledger._last_balance == expected_final_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].amount == Quantity(Decimal("50.00"))
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

def test_build_general_ledger():
    # Setup dates
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    # Setup Accounts
    acc_cash = Account("Cash")
    acc_revenue = Account("Revenue")
    acc_expense = Account("Expense")

    # Setup Initial Balances
    # Cash starts with 1000
    initial_balances = {
        acc_cash: Balance(start_date, Quantity(Decimal("1000.00")))
    }

    # Setup Journal Entries
    # Entry 1: Revenue of 500 (Debit Cash, Credit Revenue) - Inside period
    j1_date = datetime.date(2023, 6, 1)
    p1_cash = Posting(acc_cash, Amount(Decimal("500.00")), Direction.DEBIT, j1_journal)
    p1_rev = Posting(acc_revenue, Amount(cal_amount_for_credit(500)), Direction.CREDIT, j1_journal)
    
    # We need to mock the JournalEntry and its postings
    j1_journal = MagicMock(spec=JournalEntry)
    j1_journal.date = j1_date
    j1_journal.description = "Service Revenue"
    j1_postings = [
        Posting(acc_cash, Amount(Decimal("500.00")), Direction.DEBIT, j1_journal),
        Posting(acc_revenue, Amount(Decimal("500.00")), Direction.CREDIT, j1_journal)
    ]
    j1_journal.postings = j1_postings
    j1 = MagicMock(spec=JournalEntry)
    j1.date = j1_date
    j1.postings = j1_postings

    # Entry 2: Expense of 200 (Debit Expense, Credit Cash) - Inside period
    j2_date = datetime.mock_date_logic_here = datetime.date(2023, 7, 1)
    j2_journal = MagicMock(spec=JournalEntry)
    j2_journal.date = j2_date
    j2_journal.description = "Office Supplies"
    j2_postings = [
        Posting(acc_expense, Amount(Decimal("200.00")), Direction.DEBIT, j2_journal),
        Posting(acc_cash, Amount(Decimal("200.00")), Direction.CREDIT, j2_journal)
    ]
    j2_journal.postings = j2_postings
    j2 = MagicMock(spec=JournalEntry)
    j2.date = j2_date
    j2.postings = j2_postings

    # Entry 3: Out of period (Should be ignored)
    j3_date = datetime.date(2024, 1, 1)
    j3_journal = MagicMock(spec=JournalEntry)
    j3_journal.date = j3_date
    j3_postings = [
        Posting(acc_cash, Amount(Decimal("100.00")), Direction.DEBIT, j3_journal)
    ]
    j3_journal.postings = j3_postings
    j3 = MagicMock(spec=JournalEntry)
    j3.date = j3_date
    j3.postings = j3_postings

    journal_entries = [j1, j2, j3]

    # Execute
    gl = build_general_ledger(period, journal_entries, initial_balances)

    # Assertions
    assert gl.period == period
    
    # Check Cash Ledger: 1000 (init) + 500 (debit) - 200 (credit) = 1300
    assert acc_cash in gl.ledgers
    cash_ledger = gl.ledgers[acc_cash]
    assert cash_ledger.initial.value == Decimal("1000.00")
    # Last balance should be 1300
    assert cash_ledger._last_balance == Decimal("1300.00")
    # Should have 2 entries (the 500 debit and 200 credit)
    assert len(cash_ledger.entries) == 2

    # Check Revenue Ledger: 0 (init) + 500 (credit) = -500 (assuming credit is negative in direction logic)
    # Note: Direction.CREDIT.value is usually -1 or similar in accounting libs
    # Let's verify based on the provided logic: balance + amount * direction.value
    assert acc_revenue in gl.ledgers
    rev_ledger = gl.ledgers[acc_revenue]
    # Revenue starts at 0. Posting was 500 Credit. 
    # If Credit.value is -1, balance is -500.
    assert rev_ledger._last_balance == Decimal("-500.00")

    # Check Expense Ledger: 0 (init) + 200 (debit) = 200
    assert acc_expense in gl.ledgers
    exp_ledger = gl.ledgers[acc_expense]
    assert exp_ledger._last_balance == Decimal("200.00")

    # Verify Entry 3 was ignored
    assert not any(e.amount == Decimal("100.00") for l in gl.ledgers.values() for e in l.entries)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

@pytest.mark.parametrize("period, expected_balances", [
    (
        DateRange(date(2023, 1, 1), date(2023, 12, 31)),
        {
            Account("1001"): Balance(date(2023, 1, 1), Quantity(Decimal("100.00"))),
            Account("2001"): Balance(date(2023, 1, 1), Quantity(Decimal("50.00"))),
        }
    ),
    (
        DateRange(date(2024, 1, 1), date(2024, 12, 31)),
        {}
    ),
])
def test_ReadInitialBalances___call__(period, expected_balances):
    """
    Tests the implementation of a ReadInitialBalances protocol implementation.
    Since ReadInitialBalances is a Protocol, we test a mock/implementation 
    that adheres to the signature.
    """
    # Define a concrete implementation for testing the protocol behavior
    class InitialBalancesReader:
        def __init__(self, data_map):
            self.data_map = data_map

        def __call__(self, period: DateRange) -> InitialBalances:
            # In a real scenario, this would logic to fetch data based on period
            # Here we simulate returning the pre-defined balances
            return self.data_map

    # Arrange
    reader = InitialBalancesReader(expected_balances)

    # Act
    result = reader(period)

    # Assert
    assert result == expected_balances
    assert isinstance(result, dict)
    if expected_balances:
        for account, balance in expected_balances.items():
            assert account in result
            assert result[account] == balance
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal
from ..commons.numbers import Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance

def test_ReadInitialBalances___call__():
    """
    Tests the protocol/type definition of ReadInitialBalances by verifying 
    that a compatible callable returns the expected InitialBalances structure.
    """
    # Setup
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Create dummy accounts and balances
    account_a = MagicMock(spec=Account)
    account_b = MagicMock(spec=Account)
    
    balance_a = Balance(date(2023, 1, 1), Quantity(Decimal("100.00")))
    balance_b = Balance(date(2023, 1, 1), Quantity(Decimal("50.00")))
    
    expected_initial_balances = {
        account_a: balance_a,
        account_b: balance_b
    }

    # Define a mock implementation of the ReadInitialBalances protocol
    def mock_read_initial_balances(period: DateRange) -> dict:
        # In a real scenario, this would use the period to query a DB
        # For testing the protocol signature, we return our predefined dict
        return expected_initial_balances

    # Verify the callable matches the protocol requirements
    # 1. Test the return type and content
    result = mock_read_initial_balances(test_period)
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert result[account_a] == balance_a
    assert result[account_b] == balance_b
    
    # 2. Test with a different period to ensure it's a functional callable
    # (The protocol expects the function to accept a DateRange)
    other_period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    result_other = mock_read_initial_balances(other_period)
    assert result_other == expected_initial_balances
```


# LLM-generated content at query #16
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
    
    # Mock data
    mock_initial_balances = {
        MagicMock(spec=Account): Balance(period.since, Quantity(Decimal("100.00")))
    }
    mock_journal_entries = [
        MagicMock(spec=JournalEntry)
    ]
    
    # Mock the algebra implementations
    read_initial_balances = MagicMock(return_value=mock_initial_balances)
    read_journal_entries = MagicMock(return_value=mock_journal_entries)
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=read_initial_balances,
        read_journal_entries=read_journal_entries
    )
    
    # Execute the program
    general_ledger = program(period)
    
    # Assertions
    read_initial_balances.assert_called_once_with(period)
    read_journal_entries.assert_called_once_with(period)
    
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    
    # Verify the content was passed correctly to build_general_ledger
    # Since build_general_ledger is called internally, we check the result state
    first_account = list(mock_initial_balances.keys())[0]
    assert first_account in general_ledger.ledgers
    assert general_ledger.ledgers[first_account].initial.value == Decimal("100.00")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

def test_GeneralLedgerProgram___call__():
    # Arrange
    # Mocking the dependencies for the program
    mock_read_initial_balances = MagicMock()
    mock_read_journal_entries = MagicMock()
    
    # Define a mock period
    period_start = datetime.date(2023, 1, 1)
    period_end = datetime.date(2023, 12, 31)
    period = DateRange(since=period_start, until=period_end)
    
    # Define mock return values for the algebras
    mock_account = MagicMock(spec=Account)
    mock_balance = Balance(since=period_start, value=Quantity(Decimal("100.00")))
    mock_initial_balances = {mock_account: mock_balance}
    
    mock_journal_entry = MagicMock(spec=JournalEntry)
    mock_journal_entry.date = datetime.date(2023, 6, 1)
    
    # Setup the mocks to return our prepared data
    mock_read_initial_balances.return_value = mock_initial_balances
    mock_read_journal_entries.return_value = [mock_journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Act
    # Execute the __call__ of the GeneralLedgerProgram
    result = program(period)
    
    # Assert
    # Verify the algebra functions were called with the correct period
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)
    
    # Verify the returned object is a GeneralLedger
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    
    # Verify the ledger content is correctly built from the mock data
    assert mock_account in result.ledgers
    assert result.ledgers[mock_account].initial == mock_balance
```


