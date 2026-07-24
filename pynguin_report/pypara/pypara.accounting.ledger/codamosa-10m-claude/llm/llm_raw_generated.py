####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    from decimal import Decimal
    
    # Setup test data
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Create mock accounts
    account1 = Account(code="1000", name="Cash", account_type="Asset")
    account2 = Account(code="2000", name="Payable", account_type="Liability")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000"))),
        account2: Balance(date(2022, 12, 31), Quantity(Decimal("500"))),
    }
    
    # Create mock journal entries with postings
    from ..commons.journaling import Direction
    
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        date=date(2023, 1, 15),
        journal=JournalEntry(
            date=date(2023, 1, 15),
            description="Test transaction",
            postings=[]
        )
    )
    
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        date=date(2023, 1, 15),
        journal=JournalEntry(
            date=date(2023, 1, 15),
            description="Test transaction",
            postings=[]
        )
    )
    
    journal_entry = JournalEntry(
        date=date(2023, 1, 15),
        description="Test transaction",
        postings=[posting1, posting2]
    )
    
    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries,
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account2].account == account2
    assert result.ledgers[account1].initial == initial_balances[account1]
    assert result.ledgers[account2].initial == initial_balances[account2]
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #2
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """
    Test the __call__ method of ReadInitialBalances protocol.
    """
    # Create a sample DateRange
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create sample accounts and balances
    account1 = Account(code="1000", name="Cash", account_type="asset")
    account2 = Account(code="2000", name="Accounts Payable", account_type="liability")
    
    balance_date = datetime.date(2022, 12, 31)
    balance1 = Balance(balance_date, Quantity(Decimal("1000.00")))
    balance2 = Balance(balance_date, Quantity(Decimal("500.00")))
    
    initial_balances: InitialBalances = {
        account1: balance1,
        account2: balance2,
    }
    
    # Create a concrete implementation of ReadInitialBalances
    def read_initial_balances_impl(period: DateRange) -> InitialBalances:
        return initial_balances
    
    # Test the __call__ method
    result = read_initial_balances_impl(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1] == balance1
    assert result[account2] == balance2
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #3
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a sample DateRange
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create sample accounts and balances
    account1 = Account(name="Cash", number="1000")
    account2 = Account(name="Accounts Receivable", number="1200")
    
    balance1 = Balance(start_date, Quantity(Decimal("1000.00")))
    balance2 = Balance(start_date, Quantity(Decimal("500.00")))
    
    initial_balances: InitialBalances = {
        account1: balance1,
        account2: balance2,
    }
    
    # Create a concrete implementation of ReadInitialBalances
    def read_initial_balances_impl(period: DateRange) -> InitialBalances:
        return initial_balances
    
    # Call the function (which implements the protocol)
    result = read_initial_balances_impl(period)
    
    # Assert the results
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1] == balance1
    assert result[account2] == balance2
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(name="Cash", number="1000")
        account2 = Account(name="Equity", number="3000")
        balance_date = period.since
        return {
            account1: Balance(balance_date, Quantity(Decimal("1000.00"))),
            account2: Balance(balance_date, Quantity(Decimal("1000.00"))),
        }
    
    # Create a date range for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    assert all(isinstance(account, Account) for account in result.keys())
    assert all(isinstance(balance, Balance) for balance in result.values())
    
    # Verify the balances
    for account, balance in result.items():
        assert balance.date == start_date
        assert isinstance(balance.value, Quantity)
        assert balance.value == Quantity(Decimal("1000.00"))


# LLM-generated content at query #5
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    from decimal import Decimal
    
    # Create test accounts
    account_1 = Account(code="1000", name="Cash")
    account_2 = Account(code="2000", name="Payable")
    account_3 = Account(code="3000", name="Revenue")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account_1: Balance(date(2024, 1, 1), Quantity(Decimal("1000"))),
        account_2: Balance(date(2024, 1, 1), Quantity(Decimal("500"))),
    }
    
    # Create period
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    # Create posting direction (assuming it has a value attribute)
    from enum import Enum
    class Direction(Enum):
        DEBIT = 1
        CREDIT = -1
    
    # Create journal entries with postings
    posting_1 = Posting(
        account=account_1,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        journal=None  # Will be set in JournalEntry
    )
    posting_2 = Posting(
        account=account_3,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=None
    )
    
    journal_entry_1 = JournalEntry(
        date=date(2024, 6, 15),
        description="Test transaction",
        postings=[posting_1, posting_2]
    )
    
    # Update posting references to journal
    posting_1.journal = journal_entry_1
    posting_2.journal = journal_entry_1
    
    journal_entries = [journal_entry_1]
    
    # Build general ledger
    gl = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert gl.period == period
    assert len(gl.ledgers) == 3
    
    # Check that all accounts are in ledgers
    assert account_1 in gl.ledgers
    assert account_2 in gl.ledgers
    assert account_3 in gl.ledgers
    
    # Check initial balances were set correctly
    assert gl.ledgers[account_1].initial == initial_balances[account_1]
    assert gl.ledgers[account_2].initial == initial_balances[account_2]
    
    # Check account_3 has zero initial balance
    assert gl.ledgers[account_3].initial.value == Quantity(Decimal(0))
    
    # Check entries were added
    assert len(gl.ledgers[account_1].entries) == 1
    assert len(gl.ledgers[account_3].entries) == 1
    assert len(gl.ledgers[account_2].entries) == 0
    
    # Check ledger entry details
    entry_1 = gl.ledgers[account_1].entries[0]
    assert entry_1.posting == posting_1
    assert entry_1.date == date(2024, 6, 15)
    
    entry_3 = gl.ledgers[account_3].entries[0]
    assert entry_3.posting == posting_2


def test_build_general_ledger_empty():
    """Test build_general_ledger with empty journal."""
    from datetime import date
    from decimal import Decimal
    
    account_1 = Account(code="1000", name="Cash")
    
    initial_balances: InitialBalances = {
        account_1: Balance(date(2024, 1, 1), Quantity(Decimal("1000"))),
    }
    
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    gl = build_general_ledger(period, [], initial_balances)
    
    assert gl.period == period
    assert len(gl.ledgers) == 1
    assert account_1 in gl.ledgers
    assert len(gl.ledgers[account_1].entries) == 0
    assert gl.ledgers[account_1].initial == initial_balances[account_1]


def test_build_general_ledger_filters_by_period():
    """Test that build_general_ledger filters entries by period."""
    from datetime import date
    from decimal import Decimal
    
    account_1 = Account(code="1000", name="Cash")
    account_2 = Account(code="2000", name="Revenue")
    
    initial_balances: InitialBalances = {}
    period = DateRange(date(2024, 6, 1), date(2024, 6, 30))
    
    from enum import Enum
    class Direction(Enum):
        DEBIT = 1
        CREDIT = -1
    
    # Entry outside period (before)
    posting_before_1 = Posting(account=account_1, amount=Amount(Decimal("50")), direction=Direction.DEBIT, journal=None)
    posting_before_2 = Posting(account=account_2, amount=Amount(Decimal("50")), direction=Direction.CREDIT, journal=None)
    entry_before = JournalEntry(date=date(2024, 5, 15), description="Before", postings=[posting_before_1, posting_before_2])
    posting_before_1.journal = entry_before
    posting_before_2.journal = entry_before
    
    # Entry inside period
    posting_inside_1 = Posting(account=account_1, amount=Amount(Decimal("100")), direction=Direction.DEBIT, journal=None)
    posting_inside_2 = Posting(account=account_2, amount=Amount(Decimal("100")), direction=Direction.CREDIT, journal=None)
    entry_inside = JournalEntry(date=date(2024, 6, 15), description="Inside", postings=[posting_inside_1, posting_inside_2])
    posting_inside_1.journal = entry_inside
    posting_inside_2.journal = entry_inside
    
    # Entry outside period (after)
    posting_after_1 = Posting(account=account_1, amount=Amount(Decimal("75")), direction=Direction.DEBIT, journal=None)
    posting_after_2 = Posting(account=account_2, amount=Amount(Decimal("75")), direction=Direction.CREDIT, journal=None)
    entry_after = JournalEntry(date=date(2024, 7, 15), description="After", postings=[posting_after_1, posting_after_2])
    posting_after_1.journal = entry_after
    posting_after_2.journal = entry_after
    
    journal_entries = [entry_before, entry_inside, entry_after]
    
    gl = build_general_ledger(period, journal_entries, initial_balances)
    
    # Only the inside entry should be included
    assert len(gl.ledgers[account_1].entries) == 1
    assert len(gl.ledgers[account_2].entries) == 1
    assert gl.ledgers[account_1].entries[0].date == date(2024, 6, 15)


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock

def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup mock implementations
    mock_read_initial_balances = Mock()
    mock_read_journal_entries = Mock()
    
    # Create test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account()
    test_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000")))
    test_initial_balances = {test_account: test_balance}
    
    # Create mock journal entries
    mock_posting = Mock(spec=Posting)
    mock_posting.account = test_account
    mock_posting.amount = Decimal("100")
    mock_posting.direction = Mock(value=1)
    mock_posting.date = datetime.date(2023, 6, 15)
    mock_posting.is_debit = True
    mock_posting.is_credit = False
    
    mock_journal_entry = Mock(spec=JournalEntry)
    mock_journal_entry.postings = [mock_posting]
    mock_journal_entry.date = datetime.date(2023, 6, 15)
    mock_journal_entry.description = "Test entry"
    
    test_journal_entries = [mock_journal_entry]
    
    # Configure mocks to return test data
    mock_read_initial_balances.return_value = test_initial_balances
    mock_read_journal_entries.return_value = test_journal_entries
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert isinstance(result.ledgers, dict)
    assert test_account in result.ledgers
    
    # Verify mock calls
    mock_read_initial_balances.assert_called_once_with(test_period)
    mock_read_journal_entries.assert_called_once_with(test_period)
    
    # Verify ledger content
    ledger = result.ledgers[test_account]
    assert ledger.account == test_account
    assert ledger.initial == test_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting == mock_posting


# LLM-generated content at query #7
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    # Setup test data
    account1 = Account(code="1000", name="Cash")
    account2 = Account(code="2000", name="Accounts Payable")
    account3 = Account(code="3000", name="Revenue")
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account1: Balance(start_date, Quantity(Decimal("1000"))),
        account2: Balance(start_date, Quantity(Decimal("500"))),
    }
    
    # Create journal entries with postings
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=datetime.date(2024, 1, 15),
            description="Test entry 1",
            postings=[]
        )
    )
    
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        journal=JournalEntry(
            date=datetime.date(2024, 1, 15),
            description="Test entry 1",
            postings=[]
        )
    )
    
    posting3 = Posting(
        account=account3,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        journal=JournalEntry(
            date=datetime.date(2024, 2, 1),
            description="Test entry 2",
            postings=[]
        )
    )
    
    # Set postings in journal entries
    posting1.journal.postings = [posting1, posting2]
    posting2.journal.postings = [posting1, posting2]
    posting3.journal.postings = [posting3]
    
    journal_entries = [posting1.journal, posting3.journal]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    # Check account1 ledger
    assert account1 in general_ledger.ledgers
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal("1100"))
    
    # Check account2 ledger
    assert account2 in general_ledger.ledgers
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("600"))
    
    # Check account3 ledger (not in initial balances)
    assert account3 in general_ledger.ledgers
    ledger3 = general_ledger.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal("0"))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal("200"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    account1 = Account(code="1000", name="Cash")
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    initial_balances: InitialBalances = {
        account1: Balance(start_date, Quantity(Decimal("1000"))),
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0
    assert general_ledger.ledgers[account1].initial == initial_balances[account1]


def test_build_general_ledger_postings_outside_period():
    """Test build_general_ledger filters postings outside period."""
    account1 = Account(code="1000", name="Cash")
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    initial_balances: InitialBalances = {
        account1: Balance(start_date, Quantity(Decimal("1000"))),
    }
    
    # Create posting outside period
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=datetime.date(2023, 12, 31),
            description="Outside period",
            postings=[]
        )
    )
    posting1.journal.postings = [posting1]
    
    general_ledger = build_general_ledger(period, [posting1.journal], initial_balances)
    
    assert len(general_ledger.ledgers[account1].entries) == 0


# LLM-generated content at query #8
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    from datetime import date
    from decimal import Decimal
    
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(name="Cash", number="1000")
        account2 = Account(name="Accounts Receivable", number="1200")
        
        return {
            account1: Balance(date=period.since, value=Quantity(Decimal("1000.00"))),
            account2: Balance(date=period.since, value=Quantity(Decimal("500.00"))),
        }
    
    # Create a DateRange for testing
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Call the function as a protocol implementation
    result = mock_read_initial_balances(period)
    
    # Verify the result is a dictionary of initial balances
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure of returned balances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.value >= Quantity(Decimal("0"))
    
    # Verify specific values
    accounts = list(result.keys())
    assert any(acc.name == "Cash" for acc in accounts)
    assert any(acc.name == "Accounts Receivable" for acc in accounts)


def test_ReadInitialBalances___call___empty_balances():
    """Test ReadInitialBalances __call__ method with empty initial balances."""
    from datetime import date
    
    # Create a mock that returns empty balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___with_multiple_accounts():
    """Test ReadInitialBalances __call__ method with multiple accounts."""
    from datetime import date
    from decimal import Decimal
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        accounts = [
            Account(name="Cash", number="1000"),
            Account(name="Accounts Receivable", number="1200"),
            Account(name="Equipment", number="1500"),
            Account(name="Accounts Payable", number="2000"),
        ]
        
        return {
            acc: Balance(date=period.since, value=Quantity(Decimal("1000.00")))
            for acc in accounts
        }
    
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    result = mock_read_initial_balances(period)
    
    assert len(result) == 4
    for account, balance in result.items():
        assert balance.date == period.since
        assert balance.value == Quantity(Decimal("1000.00"))


# LLM-generated content at query #9
#--------------------------

```python
import datetime
from decimal import Decimal

import pytest

from commons.numbers import Amount, Quantity
from commons.zeitgeist import DateRange
from accounts import Account, AccountType
from generic import Balance
from journaling import JournalEntry, Posting, Direction


def test_build_general_ledger():
    """Test build_general_ledger function"""
    # Setup test data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create test accounts
    cash_account = Account(code="1000", name="Cash", type=AccountType.ASSET)
    revenue_account = Account(code="4000", name="Revenue", type=AccountType.REVENUE)
    expense_account = Account(code="5000", name="Expenses", type=AccountType.EXPENSE)
    
    # Create initial balances
    initial_balances = {
        cash_account: Balance(
            date=datetime.date(2022, 12, 31),
            value=Quantity(Decimal("1000"))
        )
    }
    
    # Create postings and journal entries
    posting1 = Posting(
        account=cash_account,
        amount=Amount(Decimal("500")),
        direction=Direction.DEBIT,
        journal=None  # Will be set by journal entry
    )
    posting2 = Posting(
        account=revenue_account,
        amount=Amount(Decimal("500")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Initial revenue",
        postings=[posting1, posting2]
    )
    posting1.journal = journal_entry1
    posting2.journal = journal_entry1
    
    posting3 = Posting(
        account=cash_account,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        journal=None
    )
    posting4 = Posting(
        account=expense_account,
        amount=Amount(Decimal("200")),
        direction=Direction.DEBIT,
        journal=None
    )
    
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Expense payment",
        postings=[posting3, posting4]
    )
    posting3.journal = journal_entry2
    posting4.journal = journal_entry2
    
    journal_entries = [journal_entry1, journal_entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    # Check cash account ledger
    cash_ledger = general_ledger.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial.value == Quantity(Decimal("1000"))
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300"))
    
    # Check revenue account ledger
    revenue_ledger = general_ledger.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500"))
    
    # Check expense account ledger
    expense_ledger = general_ledger.ledgers[expense_account]
    assert expense_ledger.account == expense_account
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal"""
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account = Account(code="1000", name="Cash", type=AccountType.ASSET)
    initial_balances = {
        account: Balance(
            date=datetime.date(2022, 12, 31),
            value=Quantity(Decimal("5000"))
        )
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[account].entries == []
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal("5000"))


def test_build_general_ledger_outside_period():
    """Test build_general_ledger ignores entries outside period"""
    period = DateRange(
        since=datetime.date(2023, 6, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account1 = Account(code="1000", name="Cash", type=AccountType.ASSET)
    account2 = Account(code="4000", name="Revenue", type=AccountType.REVENUE)
    
    initial_balances = {
        account1: Balance(
            date=datetime.date(2023, 5, 31),
            value=Quantity(Decimal("1000"))
        )
    }
    
    # Entry outside period (before)
    posting1 = Posting(account=account1, amount=Amount(Decimal("100")), direction=Direction.DEBIT, journal=None)
    posting2 = Posting(account=account2, amount=Amount(Decimal("100")), direction=Direction.CREDIT, journal=None)
    entry_before = JournalEntry(date=datetime.date(2023, 5, 15), description="Before", postings=[posting1, posting2])
    posting1.journal = entry_before
    posting2.journal = entry_before
    
    # Entry inside period
    posting3 = Posting(account=account1, amount=Amount(Decimal("200")), direction=Direction.DEBIT, journal=None)
    posting4 = Posting(account=account2, amount=Amount(Decimal("200")), direction=Direction.CREDIT, journal=None)
    entry_inside = JournalEntry(date=datetime.date(2023, 7, 15), description="Inside", postings=[posting3, posting4])
    posting3.journal = entry_inside
    posting4.journal = entry_inside
    
    journal_entries = [entry_before, entry_inside]
    
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Only the entry inside period should be included
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].amount == Amount(Decimal("200"))


# LLM-generated content at query #10
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(name="Cash", number="1000")
        account2 = Account(name="Revenue", number="4000")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("5000.00"))),
        }
    
    # Create a DateRange for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Verify the result is an InitialBalances dictionary
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure and values
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == period.since
        assert isinstance(balance.value, Quantity)
    
    # Verify specific values
    accounts = list(result.keys())
    assert accounts[0].name == "Cash"
    assert result[accounts[0]].value == Quantity(Decimal("1000.00"))
    assert accounts[1].name == "Revenue"
    assert result[accounts[1]].value == Quantity(Decimal("5000.00"))


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from decimal import Decimal
    
    # Setup test data
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create test accounts
    account_a = Account(code="1000", name="Cash")
    account_b = Account(code="2000", name="Payable")
    account_c = Account(code="3000", name="Revenue")
    
    # Create initial balances
    initial_balances = {
        account_a: Balance(start_date, Quantity(Decimal("1000"))),
        account_b: Balance(start_date, Quantity(Decimal("500"))),
    }
    
    # Create mock journal entries with postings
    from ..commons.numbers import Direction
    
    journal_entry_1 = JournalEntry(
        date=datetime.date(2024, 1, 15),
        description="Test transaction 1",
        postings=[
            Posting(account=account_a, amount=Decimal("100"), direction=Direction.DEBIT, journal=None),
            Posting(account=account_c, amount=Decimal("100"), direction=Direction.CREDIT, journal=None),
        ]
    )
    
    journal_entry_2 = JournalEntry(
        date=datetime.date(2024, 2, 20),
        description="Test transaction 2",
        postings=[
            Posting(account=account_b, amount=Decimal("50"), direction=Direction.DEBIT, journal=None),
            Posting(account=account_a, amount=Decimal("50"), direction=Direction.CREDIT, journal=None),
        ]
    )
    
    journal_entries = [journal_entry_1, journal_entry_2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    # Verify account_a ledger
    assert account_a in general_ledger.ledgers
    ledger_a = general_ledger.ledgers[account_a]
    assert ledger_a.account == account_a
    assert ledger_a.initial.value == Quantity(Decimal("1000"))
    assert len(ledger_a.entries) == 2
    
    # Verify account_b ledger
    assert account_b in general_ledger.ledgers
    ledger_b = general_ledger.ledgers[account_b]
    assert ledger_b.account == account_b
    assert ledger_b.initial.value == Quantity(Decimal("500"))
    assert len(ledger_b.entries) == 1
    
    # Verify account_c ledger (created without initial balance)
    assert account_c in general_ledger.ledgers
    ledger_c = general_ledger.ledgers[account_c]
    assert ledger_c.account == account_c
    assert ledger_c.initial.value == Quantity(Decimal("0"))
    assert len(ledger_c.entries) == 1


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account = Account(code="1000", name="Cash")
    initial_balances = {account: Balance(start_date, Quantity(Decimal("500")))}
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 0


def test_build_general_ledger_outside_period():
    """Test build_general_ledger filters entries outside period."""
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account_a = Account(code="1000", name="Cash")
    account_b = Account(code="2000", name="Payable")
    
    # Create journal entry outside period
    journal_entry = JournalEntry(
        date=datetime.date(2023, 12, 31),
        description="Outside period",
        postings=[
            Posting(account=account_a, amount=Decimal("100"), direction=Direction.DEBIT, journal=None),
            Posting(account=account_b, amount=Decimal("100"), direction=Direction.CREDIT, journal=None),
        ]
    )
    
    general_ledger = build_general_ledger(period, [journal_entry], {})
    
    assert len(general_ledger.ledgers) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup test data
    test_account = Account(code="1000", name="Test Account")
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create mock initial balances
    initial_balances = {
        test_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000")))
    }
    
    # Create mock journal entries
    posting = Posting(
        account=test_account,
        amount=Amount(Decimal("100")),
        direction=Balance(datetime.date(2023, 1, 15), Quantity(Decimal("1"))),
        journal=JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test entry",
            postings=[]
        )
    )
    
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Test entry",
            postings=[posting]
        )
    ]
    
    # Create mock reader functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return journal_entries
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Call the program with a test period
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert isinstance(result.ledgers, dict)
    assert test_account in result.ledgers
    assert result.ledgers[test_account].account == test_account
    assert result.ledgers[test_account].initial == initial_balances[test_account]
    assert len(result.ledgers[test_account].entries) > 0


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test ReadInitialBalances.__call__ method."""
    # Create a mock implementation of ReadInitialBalances
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account(name="Test Account", number="1000")
    test_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00")))
    expected_balances = {test_account: test_balance}
    
    # Define a concrete implementation of the protocol
    def read_balances(period: DateRange) -> InitialBalances:
        if period == test_period:
            return expected_balances
        return {}
    
    # Call the function and verify it returns the expected initial balances
    result = read_balances(test_period)
    
    assert result == expected_balances
    assert test_account in result
    assert result[test_account] == test_balance


def test_ReadInitialBalances___call___empty():
    """Test ReadInitialBalances.__call__ returns empty dict for unknown period."""
    test_period = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    def read_balances(period: DateRange) -> InitialBalances:
        return {}
    
    result = read_balances(test_period)
    
    assert result == {}
    assert len(result) == 0


def test_ReadInitialBalances___call___multiple_accounts():
    """Test ReadInitialBalances.__call__ with multiple accounts."""
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account(name="Asset", number="1000")
    account2 = Account(name="Liability", number="2000")
    balance1 = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("5000.00")))
    balance2 = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("2000.00")))
    expected_balances = {account1: balance1, account2: balance2}
    
    def read_balances(period: DateRange) -> InitialBalances:
        return expected_balances
    
    result = read_balances(test_period)
    
    assert len(result) == 2
    assert result[account1] == balance1
    assert result[account2] == balance2


# LLM-generated content at query #14
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a sample account and balance for testing
    account = Account(code="1000", name="Cash", kind=None)
    balance = Balance(date=datetime.date(2023, 1, 1), value=Quantity(Decimal("1000.00")))
    initial_balances = {account: balance}
    
    # Create a concrete implementation of ReadInitialBalances
    def read_initial_balances_impl(period: DateRange) -> InitialBalances:
        return initial_balances
    
    # Create a DateRange for testing
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Call the function
    result = read_initial_balances_impl(period)
    
    # Assert the result matches expected initial balances
    assert result == initial_balances
    assert account in result
    assert result[account] == balance
    assert result[account].value == Quantity(Decimal("1000.00"))


def test_ReadInitialBalances___call__empty():
    """Test ReadInitialBalances with empty initial balances."""
    def read_initial_balances_empty(period: DateRange) -> InitialBalances:
        return {}
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    result = read_initial_balances_empty(period)
    
    assert result == {}
    assert len(result) == 0


def test_ReadInitialBalances___call__multiple_accounts():
    """Test ReadInitialBalances with multiple accounts."""
    account1 = Account(code="1000", name="Cash", kind=None)
    account2 = Account(code="2000", name="Accounts Payable", kind=None)
    
    balance1 = Balance(date=datetime.date(2023, 1, 1), value=Quantity(Decimal("1000.00")))
    balance2 = Balance(date=datetime.date(2023, 1, 1), value=Quantity(Decimal("500.00")))
    
    initial_balances = {account1: balance1, account2: balance2}
    
    def read_initial_balances_multi(period: DateRange) -> InitialBalances:
        return initial_balances
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    result = read_initial_balances_multi(period)
    
    assert len(result) == 2
    assert result[account1] == balance1
    assert result[account2] == balance2
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #15
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    from decimal import Decimal
    
    # Create test accounts
    account1 = Account(code="1000", name="Cash")
    account2 = Account(code="2000", name="Accounts Payable")
    account3 = Account(code="3000", name="Revenue")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account1: Balance(date(2024, 1, 1), Quantity(Decimal("1000"))),
        account2: Balance(date(2024, 1, 1), Quantity(Decimal("500"))),
    }
    
    # Create test postings and journal entries
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        date=date(2024, 1, 15),
        journal=JournalEntry(
            description="Test entry 1",
            date=date(2024, 1, 15),
            postings=[]
        )
    )
    posting1.journal.postings = [posting1]
    
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        date=date(2024, 1, 15),
        journal=posting1.journal
    )
    posting1.journal.postings.append(posting2)
    
    posting3 = Posting(
        account=account3,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        date=date(2024, 1, 20),
        journal=JournalEntry(
            description="Test entry 2",
            date=date(2024, 1, 20),
            postings=[]
        )
    )
    posting3.journal.postings = [posting3]
    
    journal_entries = [posting1.journal, posting3.journal]
    period = DateRange(date(2024, 1, 1), date(2024, 1, 31))
    
    # Build general ledger
    gl = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert gl.period == period
    assert len(gl.ledgers) == 3
    
    # Check account1 ledger
    assert account1 in gl.ledgers
    ledger1 = gl.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal("1100"))
    
    # Check account2 ledger
    assert account2 in gl.ledgers
    ledger2 = gl.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("600"))
    
    # Check account3 ledger (not in initial balances)
    assert account3 in gl.ledgers
    ledger3 = gl.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal("0"))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal("-200"))
    
    # Check that entries are in chronological order
    assert ledger1.entries[0].date == date(2024, 1, 15)
    assert ledger3.entries[0].date == date(2024, 1, 20)


# LLM-generated content at query #16
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """
    Test the __call__ method of GeneralLedgerProgram.
    """
    from datetime import date
    from decimal import Decimal
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1000", "Cash")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account("1000", "Cash")
        account2 = Account("2000", "Accounts Payable")
        
        posting1 = Posting(
            account=account1,
            amount=Amount(Decimal("500")),
            direction=Direction.DEBIT,
            date=date(2023, 1, 15)
        )
        posting1.journal = JournalEntry(
            date=date(2023, 1, 15),
            description="Test transaction",
            postings=[posting1, Posting(
                account=account2,
                amount=Amount(Decimal("500")),
                direction=Direction.CREDIT,
                date=date(2023, 1, 15)
            )]
        )
        
        posting2 = Posting(
            account=account2,
            amount=Amount(Decimal("500")),
            direction=Direction.CREDIT,
            date=date(2023, 1, 15)
        )
        posting2.journal = posting1.journal
        
        return [posting1.journal]
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Create test period
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) >= 1
    
    # Verify that initial balances were applied
    account1 = Account("1000", "Cash")
    if account1 in result.ledgers:
        ledger = result.ledgers[account1]
        assert ledger.account == account1
        assert ledger.initial.value == Quantity(Decimal("1000"))


# LLM-generated content at query #17
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1000", "Cash", None)
        account2 = Account("2000", "Accounts Payable", None)
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create test period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure of returned balances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == period.since
        assert isinstance(balance.value, Quantity)
    
    # Verify specific values
    account1 = list(result.keys())[0]
    account2 = list(result.keys())[1]
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("500.00"))


def test_ReadInitialBalances___call___empty_balances():
    """Test ReadInitialBalances with empty initial balances."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___single_account():
    """Test ReadInitialBalances with a single account."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account = Account("1000", "Cash", None)
        return {account: Balance(period.since, Quantity(Decimal("5000.00")))}
    
    start_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert len(result) == 1
    account = list(result.keys())[0]
    assert account.code == "1000"
    assert result[account].value == Quantity(Decimal("5000.00"))
    assert result[account].date == period.since


# LLM-generated content at query #18
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Create mock implementations
    test_account = Account(code="1000", name="Test Account")
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create initial balances
    initial_balances = {
        test_account: Balance(
            date=datetime.date(2022, 12, 31),
            value=Quantity(Decimal("1000.00"))
        )
    }
    
    # Create journal entries with postings
    from .journaling import Direction
    posting1 = Posting(
        account=test_account,
        amount=Amount(Decimal("100.00")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 6, 15),
        journal=None  # Will be set by JournalEntry
    )
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test Entry",
        postings=[posting1]
    )
    
    # Mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert test_account in result.ledgers
    assert result.ledgers[test_account].account == test_account
    assert result.ledgers[test_account].initial == initial_balances[test_account]
    assert len(result.ledgers[test_account].entries) > 0


# LLM-generated content at query #19
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create mock accounts
    account_1 = Account(name="Cash", number="1000")
    account_2 = Account(name="Revenue", number="4000")
    
    # Create initial balances
    initial_balances = {
        account_1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000")))
    }
    
    # Create journal entries with postings
    posting_1 = Posting(
        account=account_1,
        amount=Amount(Decimal("500")),
        direction=1,  # Debit
        date=datetime.date(2023, 6, 15),
        journal=None  # Will be set by JournalEntry
    )
    
    posting_2 = Posting(
        account=account_2,
        amount=Amount(Decimal("500")),
        direction=-1,  # Credit
        date=datetime.date(2023, 6, 15),
        journal=None
    )
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test transaction",
        postings=[posting_1, posting_2]
    )
    
    # Update posting journal references
    posting_1.journal = journal_entry
    posting_2.journal = journal_entry
    
    journal_entries = [journal_entry]
    
    # Create mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return journal_entries
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert len(result.ledgers) == 2
    assert account_1 in result.ledgers
    assert account_2 in result.ledgers
    
    # Verify ledger for account_1
    ledger_1 = result.ledgers[account_1]
    assert ledger_1.account == account_1
    assert ledger_1.initial == initial_balances[account_1]
    assert len(ledger_1.entries) == 1
    
    # Verify ledger for account_2
    ledger_2 = result.ledgers[account_2]
    assert ledger_2.account == account_2
    assert ledger_2.initial.value == Quantity(Decimal(0))
    assert len(ledger_2.entries) == 1
    
    # Verify the entries
    entry_1 = ledger_1.entries[0]
    assert entry_1.amount == Amount(Decimal("500"))
    assert entry_1.is_debit is True
    assert entry_1.balance == Quantity(Decimal("1500"))
    
    entry_2 = ledger_2.entries[0]
    assert entry_2.amount == Amount(Decimal("500"))
    assert entry_2.is_credit is True
    assert entry_2.balance == Quantity(Decimal("-500"))


# LLM-generated content at query #20
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from ..commons.numbers import Amount, Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance
from .journaling import JournalEntry, Posting
from .general_ledger import (
    GeneralLedger,
    GeneralLedgerProgram,
    compile_general_ledger_program,
)


def test_GeneralLedgerProgram___call__():
    """Test that GeneralLedgerProgram.__call__ correctly builds a general ledger."""
    
    # Setup test data
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create test accounts
    account_1 = Mock(spec=Account)
    account_1.name = "Account 1"
    account_2 = Mock(spec=Account)
    account_2.name = "Account 2"
    
    # Create initial balances
    initial_balances = {
        account_1: Balance(start_date, Quantity(Decimal("1000"))),
        account_2: Balance(start_date, Quantity(Decimal("500"))),
    }
    
    # Create mock postings and journal entries
    posting_1 = Mock(spec=Posting)
    posting_1.account = account_1
    posting_1.amount = Amount(Decimal("100"))
    posting_1.date = datetime.date(2023, 6, 15)
    posting_1.direction = Mock(value=1)
    posting_1.is_debit = True
    posting_1.is_credit = False
    posting_1.journal = Mock()
    posting_1.journal.description = "Test entry"
    posting_1.journal.postings = [posting_1]
    
    posting_2 = Mock(spec=Posting)
    posting_2.account = account_2
    posting_2.amount = Amount(Decimal("50"))
    posting_2.date = datetime.date(2023, 7, 20)
    posting_2.direction = Mock(value=-1)
    posting_2.is_debit = False
    posting_2.is_credit = True
    posting_2.journal = Mock()
    posting_2.journal.description = "Test entry 2"
    posting_2.journal.postings = [posting_2]
    
    journal_entry_1 = Mock(spec=JournalEntry)
    journal_entry_1.date = datetime.date(2023, 6, 15)
    journal_entry_1.postings = [posting_1]
    
    journal_entry_2 = Mock(spec=JournalEntry)
    journal_entry_2.date = datetime.date(2023, 7, 20)
    journal_entry_2.postings = [posting_2]
    
    journal_entries = [journal_entry_1, journal_entry_2]
    
    # Create mock functions
    read_initial_balances = Mock(return_value=initial_balances)
    read_journal_entries = Mock(return_value=journal_entries)
    
    # Compile the program
    program = compile_general_ledger_program(read_initial_balances, read_journal_entries)
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert read_initial_balances.called
    assert read_journal_entries.called
    assert read_initial_balances.call_args[0][0] == period
    assert read_journal_entries.call_args[0][0] == period
    
    # Verify ledgers were created
    assert account_1 in result.ledgers
    assert account_2 in result.ledgers
    
    # Verify entries were added to ledgers
    assert len(result.ledgers[account_1].entries) == 1
    assert len(result.ledgers[account_2].entries) == 1
    
    # Verify balances were calculated correctly
    assert result.ledgers[account_1].entries[0].balance == Quantity(Decimal("1100"))
    assert result.ledgers[account_2].entries[0].balance == Quantity(Decimal("450"))


# LLM-generated content at query #21
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test ReadInitialBalances protocol __call__ method."""
    # Create a sample DateRange
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create a sample Account and Balance
    account = Account(name="Test Account", number="1000")
    balance = Balance(
        date=datetime.date(2023, 1, 1),
        value=Quantity(Decimal("1000.00"))
    )
    
    # Create expected initial balances
    expected_balances: InitialBalances = {account: balance}
    
    # Create a concrete implementation of ReadInitialBalances protocol
    def read_initial_balances_impl(period: DateRange) -> InitialBalances:
        return expected_balances
    
    # Call the function
    result = read_initial_balances_impl(period)
    
    # Assert the result matches expected
    assert result == expected_balances
    assert account in result
    assert result[account] == balance
    assert result[account].value == Quantity(Decimal("1000.00"))


def test_ReadInitialBalances___call___multiple_accounts():
    """Test ReadInitialBalances protocol __call__ with multiple accounts."""
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account1 = Account(name="Account 1", number="1000")
    account2 = Account(name="Account 2", number="2000")
    account3 = Account(name="Account 3", number="3000")
    
    balance1 = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("500.00")))
    balance2 = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1500.00")))
    balance3 = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("2000.00")))
    
    expected_balances: InitialBalances = {
        account1: balance1,
        account2: balance2,
        account3: balance3,
    }
    
    def read_initial_balances_impl(period: DateRange) -> InitialBalances:
        return expected_balances
    
    result = read_initial_balances_impl(period)
    
    assert len(result) == 3
    assert result[account1].value == Quantity(Decimal("500.00"))
    assert result[account2].value == Quantity(Decimal("1500.00"))
    assert result[account3].value == Quantity(Decimal("2000.00"))


def test_ReadInitialBalances___call___empty_balances():
    """Test ReadInitialBalances protocol __call__ with empty balances."""
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    expected_balances: InitialBalances = {}
    
    def read_initial_balances_impl(period: DateRange) -> InitialBalances:
        return expected_balances
    
    result = read_initial_balances_impl(period)
    
    assert result == {}
    assert len(result) == 0


# LLM-generated content at query #22
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """
    Test the __call__ method of GeneralLedgerProgram.
    """
    # Setup test data
    test_date = datetime.date(2023, 1, 1)
    test_period = DateRange(test_date, datetime.date(2023, 12, 31))
    
    # Create mock accounts
    account1 = Account(number="1000", name="Cash")
    account2 = Account(number="2000", name="Payable")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account1: Balance(test_date, Quantity(Decimal("1000"))),
        account2: Balance(test_date, Quantity(Decimal("500"))),
    }
    
    # Create journal entries with postings
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=None,
        date=test_date
    )
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        journal=None,
        date=test_date
    )
    
    journal_entry = JournalEntry(
        date=test_date,
        description="Test transaction",
        postings=[posting1, posting2]
    )
    journal_entries = [journal_entry]
    
    # Create mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        assert period == test_period
        return journal_entries
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries,
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account2].account == account2
    assert result.ledgers[account1].initial == initial_balances[account1]
    assert result.ledgers[account2].initial == initial_balances[account2]


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """
    Test the __call__ method of ReadInitialBalances protocol.
    """
    # Create a sample DateRange
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create sample accounts and balances
    account1 = Account(name="Cash", account_type="Asset")
    account2 = Account(name="Revenue", account_type="Revenue")
    
    balance1 = Balance(start_date, Quantity(Decimal("1000.00")))
    balance2 = Balance(start_date, Quantity(Decimal("500.00")))
    
    expected_initial_balances: InitialBalances = {
        account1: balance1,
        account2: balance2,
    }
    
    # Create a concrete implementation of ReadInitialBalances protocol
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return expected_initial_balances
    
    # Test that the callable returns the expected initial balances
    result = mock_read_initial_balances(period)
    
    assert result == expected_initial_balances
    assert account1 in result
    assert account2 in result
    assert result[account1] == balance1
    assert result[account2] == balance2


def test_ReadInitialBalances___call___empty():
    """
    Test the __call__ method of ReadInitialBalances protocol with empty balances.
    """
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create a concrete implementation that returns empty balances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    result = mock_read_initial_balances(period)
    
    assert result == {}
    assert len(result) == 0


def test_ReadInitialBalances___call___period_parameter():
    """
    Test that ReadInitialBalances __call__ correctly receives period parameter.
    """
    start_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2023, 6, 30)
    period = DateRange(start_date, end_date)
    
    account = Account(name="Expenses", account_type="Expense")
    balance = Balance(start_date, Quantity(Decimal("250.00")))
    
    received_period = None
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        nonlocal received_period
        received_period = period
        return {account: balance}
    
    result = mock_read_initial_balances(period)
    
    assert received_period == period
    assert received_period.since == start_date
    assert received_period.until == end_date
    assert result[account] == balance


# LLM-generated content at query #24
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    
    # Setup test data
    account1 = Account(code="1000", name="Cash")
    account2 = Account(code="2000", name="Payable")
    account3 = Account(code="3000", name="Revenue")
    
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account1: Balance(period_start, Quantity(Decimal("1000"))),
        account2: Balance(period_start, Quantity(Decimal("500"))),
    }
    
    # Create mock journal entries
    from ..commons.enums import Direction
    
    posting1 = Posting(
        account=account1,
        amount=Decimal("100"),
        direction=Direction.DEBIT,
        date=date(2024, 6, 15),
        journal=None
    )
    posting1.journal = JournalEntry(
        date=date(2024, 6, 15),
        description="Test entry 1",
        postings=[posting1, Posting(account=account3, amount=Decimal("100"), direction=Direction.CREDIT, date=date(2024, 6, 15), journal=None)]
    )
    posting1.journal.postings[1].journal = posting1.journal
    
    posting2 = Posting(
        account=account2,
        amount=Decimal("50"),
        direction=Direction.CREDIT,
        date=date(2024, 7, 20),
        journal=None
    )
    posting2.journal = JournalEntry(
        date=date(2024, 7, 20),
        description="Test entry 2",
        postings=[Posting(account=account1, amount=Decimal("50"), direction=Direction.DEBIT, date=date(2024, 7, 20), journal=None), posting2]
    )
    posting2.journal.postings[0].journal = posting2.journal
    
    journal_entries = [posting1.journal, posting2.journal]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert account3 in general_ledger.ledgers
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("1000"))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal("1100"))
    assert ledger1.entries[1].balance == Quantity(Decimal("1150"))
    
    # Check account2 ledger
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("500"))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("550"))
    
    # Check account3 ledger (no initial balance)
    ledger3 = general_ledger.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal("0"))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal("-100"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    from datetime import date
    
    account1 = Account(code="1000", name="Cash")
    period_start = date(2024, 1, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    initial_balances: InitialBalances = {
        account1: Balance(period_start, Quantity(Decimal("5000"))),
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal("5000"))
    assert len(general_ledger.ledgers[account1].entries) == 0


def test_build_general_ledger_outside_period():
    """Test build_general_ledger filters entries outside period."""
    from datetime import date
    
    account1 = Account(code="1000", name="Cash")
    account2 = Account(code="2000", name="Revenue")
    
    period_start = date(2024, 6, 1)
    period_end = date(2024, 12, 31)
    period = DateRange(period_start, period_end)
    
    initial_balances: InitialBalances = {
        account1: Balance(period_start, Quantity(Decimal("1000"))),
    }
    
    from ..commons.enums import Direction
    
    # Entry before period
    posting1 = Posting(
        account=account1,
        amount=Decimal("100"),
        direction=Direction.DEBIT,
        date=date(2024, 5, 15),
        journal=None
    )
    posting1.journal = JournalEntry(
        date=date(2024, 5, 15),
        description="Before period",
        postings=[posting1, Posting(account=account2, amount=Decimal("100"), direction=Direction.CREDIT, date=date(2024, 5, 15), journal=None)]
    )
    posting1.journal.postings[1].journal = posting1.journal
    
    journal_entries = [posting1.journal]
    
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert account2 not in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function with various scenarios."""
    
    # Setup test data
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create test accounts
    account_cash = Account(number="1000", name="Cash")
    account_revenue = Account(number="4000", name="Revenue")
    account_expense = Account(number="5000", name="Expense")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account_cash: Balance(start_date, Quantity(Decimal("1000.00"))),
        account_revenue: Balance(start_date, Quantity(Decimal("0.00"))),
    }
    
    # Create test postings and journal entries
    from ..commons.direction import Direction
    
    posting1 = Posting(
        account=account_cash,
        amount=Amount(Decimal("500.00")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=datetime.date(2024, 6, 15),
            description="Test transaction 1",
            postings=[]
        )
    )
    
    posting2 = Posting(
        account=account_revenue,
        amount=Amount(Decimal("500.00")),
        direction=Direction.CREDIT,
        journal=JournalEntry(
            date=datetime.date(2024, 6, 15),
            description="Test transaction 1",
            postings=[]
        )
    )
    
    posting1.journal.postings = [posting1, posting2]
    posting2.journal.postings = [posting1, posting2]
    
    posting3 = Posting(
        account=account_expense,
        amount=Amount(Decimal("200.00")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=datetime.date(2024, 7, 20),
            description="Test transaction 2",
            postings=[]
        )
    )
    
    posting4 = Posting(
        account=account_cash,
        amount=Amount(Decimal("200.00")),
        direction=Direction.CREDIT,
        journal=JournalEntry(
            date=datetime.date(2024, 7, 20),
            description="Test transaction 2",
            postings=[]
        )
    )
    
    posting3.journal.postings = [posting3, posting4]
    posting4.journal.postings = [posting3, posting4]
    
    journal_entries = [posting1.journal, posting3.journal]
    
    # Build general ledger
    gl = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert gl.period == period
    assert len(gl.ledgers) == 3
    
    # Check cash ledger (initial + debit - credit)
    cash_ledger = gl.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial == initial_balances[account_cash]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    
    # Check revenue ledger (initial + credit)
    revenue_ledger = gl.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500.00"))
    
    # Check expense ledger (created with zero initial balance + debit)
    expense_ledger = gl.ledgers[account_expense]
    assert expense_ledger.account == account_expense
    assert expense_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal entries."""
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account = Account(number="1000", name="Cash")
    initial_balances: InitialBalances = {
        account: Balance(start_date, Quantity(Decimal("5000.00"))),
    }
    
    # Build with empty journal
    gl = build_general_ledger(period, [], initial_balances)
    
    assert gl.period == period
    assert len(gl.ledgers) == 1
    assert gl.ledgers[account].account == account
    assert len(gl.ledgers[account].entries) == 0
    assert gl.ledgers[account].initial == initial_balances[account]


def test_build_general_ledger_out_of_period():
    """Test that postings outside period are not included."""
    
    start_date = datetime.date(2024, 6, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account = Account(number="1000", name="Cash")
    initial_balances: InitialBalances = {
        account: Balance(start_date, Quantity(Decimal("1000.00"))),
    }
    
    from ..commons.direction import Direction
    
    # Create posting outside period (before start)
    posting = Posting(
        account=account,
        amount=Amount(Decimal("100.00")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=datetime.date(2024, 5, 15),
            description="Before period",
            postings=[]
        )
    )
    posting.journal.postings = [posting]
    
    gl = build_general_ledger(period, [posting.journal], initial_balances)
    
    assert len(gl.ledgers[account].entries) == 0


# LLM-generated content at query #26
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """
    Test the __call__ method of ReadInitialBalances protocol.
    """
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(name="Cash", code="1000")
        account2 = Account(name="Accounts Receivable", code="1100")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create a DateRange for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function as per the protocol
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure of returned initial balances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == period.since
        assert isinstance(balance.value, Quantity)
    
    # Verify specific values
    accounts = list(result.keys())
    assert accounts[0].code == "1000"
    assert accounts[1].code == "1100"
    assert result[accounts[0]].value == Quantity(Decimal("1000.00"))
    assert result[accounts[1]].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #27
#--------------------------

```python
import datetime
from decimal import Decimal

import pytest

from ..commons.numbers import Amount, Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance
from .journaling import Direction, JournalEntry, Posting


def test_build_general_ledger():
    """Test build_general_ledger function."""
    
    # Setup test data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account_a = Account(name="Account A", number="1000")
    account_b = Account(name="Account B", number="2000")
    account_c = Account(name="Account C", number="3000")
    
    # Initial balances
    initial_balances = {
        account_a: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000"))),
        account_b: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("500"))),
    }
    
    # Create journal entries with postings
    posting_a1 = Posting(
        account=account_a,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 1, 15)
    )
    posting_b1 = Posting(
        account=account_b,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        date=datetime.date(2023, 1, 15)
    )
    
    posting_a2 = Posting(
        account=account_a,
        amount=Amount(Decimal("50")),
        direction=Direction.CREDIT,
        date=datetime.date(2023, 2, 10)
    )
    posting_c1 = Posting(
        account=account_c,
        amount=Amount(Decimal("50")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 2, 10)
    )
    
    # Create journal entries
    journal_entry_1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Transaction 1",
        postings=[posting_a1, posting_b1]
    )
    journal_entry_2 = JournalEntry(
        date=datetime.date(2023, 2, 10),
        description="Transaction 2",
        postings=[posting_a2, posting_c1]
    )
    
    journal = [journal_entry_1, journal_entry_2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    # Check Account A ledger
    ledger_a = general_ledger.ledgers[account_a]
    assert ledger_a.account == account_a
    assert ledger_a.initial.value == Quantity(Decimal("1000"))
    assert len(ledger_a.entries) == 2
    assert ledger_a.entries[0].balance == Quantity(Decimal("1100"))
    assert ledger_a.entries[1].balance == Quantity(Decimal("1050"))
    
    # Check Account B ledger
    ledger_b = general_ledger.ledgers[account_b]
    assert ledger_b.account == account_b
    assert ledger_b.initial.value == Quantity(Decimal("500"))
    assert len(ledger_b.entries) == 1
    assert ledger_b.entries[0].balance == Quantity(Decimal("400"))
    
    # Check Account C ledger
    ledger_c = general_ledger.ledgers[account_c]
    assert ledger_c.account == account_c
    assert ledger_c.initial.value == Quantity(Decimal("0"))
    assert len(ledger_c.entries) == 1
    assert ledger_c.entries[0].balance == Quantity(Decimal("50"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account_a = Account(name="Account A", number="1000")
    initial_balances = {
        account_a: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000"))),
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[account_a].entries == []
    assert general_ledger.ledgers[account_a].initial.value == Quantity(Decimal("1000"))


def test_build_general_ledger_no_initial_balances():
    """Test build_general_ledger with no initial balances."""
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account_a = Account(name="Account A", number="1000")
    account_b = Account(name="Account B", number="2000")
    
    posting_a = Posting(
        account=account_a,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 1, 15)
    )
    posting_b = Posting(
        account=account_b,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        date=datetime.date(2023, 1, 15)
    )
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Transaction",
        postings=[posting_a, posting_b]
    )
    
    general_ledger = build_general_ledger(period, [journal_entry], {})
    
    assert len(general_ledger.ledgers) == 2
    assert general_ledger.ledgers[account_a].initial.value == Quantity(Decimal("0"))
    assert general_ledger.ledgers[account_b].initial.value == Quantity(Decimal("0"))


def test_build_general_ledger_out_of_period_entries():
    """Test build_general_ledger filters entries outside period."""
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account_a = Account(name="Account A", number="1000")
    account_b = Account(name="Account B", number="2000")
    
    # Entry within period
    posting_a1 = Posting(
        account=account_a,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 6, 15)
    )
    posting_b1 = Posting(
        account=account_b,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        date=datetime.date(2023, 6, 15)
    )
    journal_entry_in_period = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="In period",
        postings=[posting_a1, posting_b1]
    )
    
    # Entry outside period
    posting_a2 = Posting(
        account=account_a,
        amount=Amount(Decimal("50")),
        direction=Direction.DEBIT,
        date=datetime.date(2024, 1, 


# LLM-generated content at query #28
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """
    Test that ReadInitialBalances protocol can be called with a DateRange
    and returns InitialBalances.
    """
    # Arrange
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account()
    account2 = Account()
    
    expected_balances = {
        account1: Balance(start_date, Quantity(Decimal("1000"))),
        account2: Balance(start_date, Quantity(Decimal("2000"))),
    }
    
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        """Mock implementation of ReadInitialBalances protocol."""
        if p == period:
            return expected_balances
        return {}
    
    # Act
    result = mock_read_initial_balances(period)
    
    # Assert
    assert result == expected_balances
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1].value == Quantity(Decimal("1000"))
    assert result[account2].value == Quantity(Decimal("2000"))


def test_ReadInitialBalances___call___empty():
    """
    Test that ReadInitialBalances protocol can return empty balances.
    """
    # Arrange
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        """Mock implementation returning empty balances."""
        return {}
    
    # Act
    result = mock_read_initial_balances(period)
    
    # Assert
    assert result == {}
    assert len(result) == 0


def test_ReadInitialBalances___call___different_periods():
    """
    Test that ReadInitialBalances protocol works with different periods.
    """
    # Arrange
    period1 = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    period2 = DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    
    account = Account()
    balance1 = Balance(period1.since, Quantity(Decimal("1000")))
    balance2 = Balance(period2.since, Quantity(Decimal("2000")))
    
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        """Mock implementation returning different balances per period."""
        if p == period1:
            return {account: balance1}
        elif p == period2:
            return {account: balance2}
        return {}
    
    # Act
    result1 = mock_read_initial_balances(period1)
    result2 = mock_read_initial_balances(period2)
    
    # Assert
    assert result1 == {account: balance1}
    assert result2 == {account: balance2}
    assert result1[account].value == Quantity(Decimal("1000"))
    assert result2[account].value == Quantity(Decimal("2000"))


# LLM-generated content at query #29
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function"""
    from datetime import date
    from decimal import Decimal
    
    # Setup test data
    test_date_start = date(2024, 1, 1)
    test_date_end = date(2024, 12, 31)
    period = DateRange(test_date_start, test_date_end)
    
    # Create test accounts
    account_cash = Account(code="1000", name="Cash")
    account_revenue = Account(code="4000", name="Revenue")
    account_expense = Account(code="5000", name="Expense")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account_cash: Balance(test_date_start, Quantity(Decimal("1000"))),
    }
    
    # Create test journal entries with postings
    from ..commons.zeitgeist import Direction
    
    posting1 = Posting(
        account=account_cash,
        amount=Amount(Decimal("500")),
        direction=Direction.DEBIT,
        journal=None  # Will be set by JournalEntry
    )
    posting2 = Posting(
        account=account_revenue,
        amount=Amount(Decimal("500")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    entry1 = JournalEntry(
        date=date(2024, 6, 15),
        description="Sales transaction",
        postings=[posting1, posting2]
    )
    
    # Update posting references to journal entry
    posting1.journal = entry1
    posting2.journal = entry1
    
    # Create additional posting for expense
    posting3 = Posting(
        account=account_cash,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        journal=None
    )
    posting4 = Posting(
        account=account_expense,
        amount=Amount(Decimal("200")),
        direction=Direction.DEBIT,
        journal=None
    )
    
    entry2 = JournalEntry(
        date=date(2024, 7, 20),
        description="Expense transaction",
        postings=[posting3, posting4]
    )
    
    posting3.journal = entry2
    posting4.journal = entry2
    
    journal_entries = [entry1, entry2]
    
    # Execute
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    # Check cash ledger
    assert account_cash in general_ledger.ledgers
    cash_ledger = general_ledger.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial.value == Quantity(Decimal("1000"))
    assert len(cash_ledger.entries) == 2
    # Final balance: 1000 + 500 - 200 = 1300
    assert cash_ledger.entries[-1].balance == Quantity(Decimal("1300"))
    
    # Check revenue ledger
    assert account_revenue in general_ledger.ledgers
    revenue_ledger = general_ledger.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert revenue_ledger.initial.value == Quantity(Decimal("0"))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500"))
    
    # Check expense ledger
    assert account_expense in general_ledger.ledgers
    expense_ledger = general_ledger.ledgers[account_expense]
    assert expense_ledger.account == account_expense
    assert expense_ledger.initial.value == Quantity(Decimal("0"))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal"""
    from datetime import date
    from decimal import Decimal
    
    test_date_start = date(2024, 1, 1)
    test_date_end = date(2024, 12, 31)
    period = DateRange(test_date_start, test_date_end)
    
    account = Account(code="1000", name="Cash")
    initial_balances: InitialBalances = {
        account: Balance(test_date_start, Quantity(Decimal("5000"))),
    }
    
    # Empty journal
    journal_entries = []
    
    # Execute
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 0
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal("5000"))


def test_build_general_ledger_entries_outside_period():
    """Test build_general_ledger filters entries outside period"""
    from datetime import date
    from decimal import Decimal
    
    test_date_start = date(2024, 6, 1)
    test_date_end = date(2024, 12, 31)
    period = DateRange(test_date_start, test_date_end)
    
    account_cash = Account(code="1000", name="Cash")
    account_revenue = Account(code="4000", name="Revenue")
    
    initial_balances: InitialBalances = {
        account_cash: Balance(test_date_start, Quantity(Decimal("1000"))),
    }
    
    # Create entries both inside and outside period
    from ..commons.zeitgeist import Direction
    
    # Entry before period
    posting1 = Posting(
        account=account_cash,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting2 = Posting(
        account=account_revenue,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        journal=None
    )
    entry_before = JournalEntry(
        date=date(2024, 5, 15),
        description="Before period",
        postings=[posting1, posting2]
    )
    posting1.journal = entry_before
    posting2.journal = entry_before
    
    # Entry within period
    posting3 = Posting(
        account=account_cash,
        amount=Amount(Decimal("200")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting4 = Posting(
        account=account_revenue,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        journal=None
    )
    entry_within = JournalEntry(
        date=date(2024, 7, 15),
        description="Within period",
        postings=[posting3, posting4]
    )
    posting3.journal = entry_within
    posting4.journal = entry_within
    
    journal_entries = [entry_before, entry_within]
    
    # Execute
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions - only the entry within period should be included
    assert len(general_ledger.ledgers[account_cash].entries) == 1
    assert general_ledger.ledgers[account_cash].entries[0].date == date(2024, 7, 15)


# LLM-generated content at query #30
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test that ReadInitialBalances protocol can be called with a DateRange and returns InitialBalances."""
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account()
        account1.code = "1000"
        account1.name = "Cash"
        
        account2 = Account()
        account2.code = "2000"
        account2.name = "Accounts Payable"
        
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create a DateRange for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the mock implementation
    result = mock_read_initial_balances(period)
    
    # Verify the result is a dictionary (InitialBalances)
    assert isinstance(result, dict)
    
    # Verify the result contains Account keys and Balance values
    assert len(result) == 2
    
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == start_date
        assert isinstance(balance.value, Quantity)
    
    # Verify specific balance values
    accounts = list(result.keys())
    balances = list(result.values())
    assert balances[0].value == Quantity(Decimal("1000.00"))
    assert balances[1].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #31
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """
    Test the __call__ method of ReadInitialBalances protocol.
    """
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1000", "Cash", None)
        account2 = Account("2000", "Accounts Payable", None)
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create a DateRange for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Check that all keys are Account instances
    for key in result.keys():
        assert isinstance(key, Account)
    
    # Check that all values are Balance instances
    for value in result.values():
        assert isinstance(value, Balance)
    
    # Verify specific balances
    accounts = list(result.keys())
    assert accounts[0].number == "1000"
    assert accounts[1].number == "2000"
    assert result[accounts[0]].value == Quantity(Decimal("1000.00"))
    assert result[accounts[1]].value == Quantity(Decimal("500.00"))


def test_ReadInitialBalances___call___empty():
    """
    Test the __call__ method of ReadInitialBalances protocol with empty balances.
    """
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___single_account():
    """
    Test the __call__ method of ReadInitialBalances protocol with single account.
    """
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account = Account("5000", "Revenue", None)
        return {
            account: Balance(period.since, Quantity(Decimal("0.00"))),
        }
    
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert len(result) == 1
    account = list(result.keys())[0]
    assert account.number == "5000"
    assert result[account].value == Quantity(Decimal("0.00"))


# LLM-generated content at query #32
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup test data
    test_account = Account(code="1000", name="Test Account", account_type="asset")
    test_date = datetime.date(2024, 1, 1)
    test_period = DateRange(since=test_date, until=datetime.date(2024, 12, 31))
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {
            test_account: Balance(date=period.since, value=Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        posting = Posting(
            account=test_account,
            amount=Decimal("100"),
            direction=1,
            journal=None
        )
        journal_entry = JournalEntry(
            date=test_date,
            description="Test Entry",
            postings=[posting]
        )
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert isinstance(result.ledgers, dict)
    assert test_account in result.ledgers
    assert result.ledgers[test_account].account == test_account
    assert result.ledgers[test_account].initial.value == Quantity(Decimal("1000"))
    assert len(result.ledgers[test_account].entries) > 0


# LLM-generated content at query #33
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    from datetime import date
    
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(number="1000", name="Cash")
        account2 = Account(number="2000", name="Accounts Payable")
        return {
            account1: Balance(date(2023, 1, 1), Quantity(Decimal("1000.00"))),
            account2: Balance(date(2023, 1, 1), Quantity(Decimal("500.00"))),
        }
    
    # Create a DateRange for testing
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assert the result is a dictionary
    assert isinstance(result, dict)
    
    # Assert the result contains the expected accounts
    assert len(result) == 2
    
    # Assert the balances are correct
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == date(2023, 1, 1)
        if account.number == "1000":
            assert balance.value == Quantity(Decimal("1000.00"))
        elif account.number == "2000":
            assert balance.value == Quantity(Decimal("500.00"))


def test_ReadInitialBalances___call___empty():
    """Test ReadInitialBalances with empty initial balances."""
    def mock_read_initial_balances_empty(period: DateRange) -> InitialBalances:
        return {}
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = mock_read_initial_balances_empty(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___with_period():
    """Test that ReadInitialBalances respects the period parameter."""
    def mock_read_initial_balances_period_aware(period: DateRange) -> InitialBalances:
        account = Account(number="1000", name="Cash")
        return {
            account: Balance(period.since, Quantity(Decimal("2000.00"))),
        }
    
    period1 = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    period2 = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    result1 = mock_read_initial_balances_period_aware(period1)
    result2 = mock_read_initial_balances_period_aware(period2)
    
    # Both should return balances but with different dates
    assert len(result1) == 1
    assert len(result2) == 1
    
    account1 = list(result1.keys())[0]
    account2 = list(result2.keys())[0]
    
    assert result1[account1].date == date(2023, 1, 1)
    assert result2[account2].date == date(2024, 1, 1)


# LLM-generated content at query #34
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    from decimal import Decimal
    
    # Setup test data
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create test accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    account3 = Account("3000", "Revenue")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account1: Balance(start_date, Quantity(Decimal("1000.00"))),
        account2: Balance(start_date, Quantity(Decimal("500.00"))),
    }
    
    # Create test postings and journal entries
    posting1 = Posting(account1, Amount(Decimal("100.00")), direction=1)
    posting2 = Posting(account2, Amount(Decimal("100.00")), direction=-1)
    journal_entry1 = JournalEntry("J001", date(2024, 6, 15), "Test entry 1", [posting1, posting2])
    
    posting3 = Posting(account3, Amount(Decimal("250.00")), direction=1)
    posting4 = Posting(account1, Amount(Decimal("250.00")), direction=-1)
    journal_entry2 = JournalEntry("J002", date(2024, 7, 20), "Test entry 2", [posting3, posting4])
    
    journal_entries = [journal_entry1, journal_entry2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    # Check account1 ledger
    assert account1 in general_ledger.ledgers
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("1000.00"))
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal("1100.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("850.00"))
    
    # Check account2 ledger
    assert account2 in general_ledger.ledgers
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("500.00"))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("400.00"))
    
    # Check account3 ledger (created without initial balance)
    assert account3 in general_ledger.ledgers
    ledger3 = general_ledger.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal("0"))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal("250.00"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    from datetime import date
    from decimal import Decimal
    
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account("1000", "Cash")
    initial_balances: InitialBalances = {
        account1: Balance(start_date, Quantity(Decimal("5000.00"))),
    }
    
    # Build general ledger with empty journal
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0
    assert general_ledger.ledgers[account1].initial.value == Quantity(Decimal("5000.00"))


def test_build_general_ledger_entries_outside_period():
    """Test build_general_ledger filters entries outside accounting period."""
    from datetime import date
    from decimal import Decimal
    
    start_date = date(2024, 6, 1)
    end_date = date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Expenses")
    
    initial_balances: InitialBalances = {
        account1: Balance(start_date, Quantity(Decimal("1000.00"))),
    }
    
    # Entry before period
    posting1 = Posting(account1, Amount(Decimal("100.00")), direction=1)
    posting2 = Posting(account2, Amount(Decimal("100.00")), direction=-1)
    entry_before = JournalEntry("J001", date(2024, 5, 15), "Before period", [posting1, posting2])
    
    # Entry within period
    posting3 = Posting(account1, Amount(Decimal("200.00")), direction=1)
    posting4 = Posting(account2, Amount(Decimal("200.00")), direction=-1)
    entry_within = JournalEntry("J002", date(2024, 7, 20), "Within period", [posting3, posting4])
    
    # Entry after period
    posting5 = Posting(account1, Amount(Decimal("300.00")), direction=1)
    posting6 = Posting(account2, Amount(Decimal("300.00")), direction=-1)
    entry_after = JournalEntry("J003", date(2025, 1, 15), "After period", [posting5, posting6])
    
    journal_entries = [entry_before, entry_within, entry_after]
    
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Only the entry within period should be processed
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert general_ledger.ledgers[account1].entries[0].date == date(2024, 7, 20)


# LLM-generated content at query #35
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    from decimal import Decimal
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1000", "Cash")
        return {account1: Balance(period.since, Quantity(Decimal("1000")))}
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account("1000", "Cash")
        account2 = Account("4000", "Revenue")
        journal_entry = JournalEntry(
            date=period.since,
            description="Test entry",
            postings=[
                Posting(account1, Quantity(Decimal("500")), "debit", journal=None),
                Posting(account2, Quantity(Decimal("500")), "credit", journal=None),
            ]
        )
        journal_entry.postings[0].journal = journal_entry
        journal_entry.postings[1].journal = journal_entry
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries,
    )
    
    # Create a test period
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) >= 1
    
    # Verify that initial balance account is in ledgers
    account1 = Account("1000", "Cash")
    assert any(ledger.account.code == "1000" for ledger in result.ledgers.values())
    
    # Verify that journal entry account is in ledgers
    assert any(ledger.account.code == "4000" for ledger in result.ledgers.values())


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    
    # Setup test data
    account1 = Account(code="1000", name="Cash")
    account2 = Account(code="2000", name="Accounts Payable")
    account3 = Account(code="3000", name="Revenue")
    
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    
    initial_balances = {
        account1: Balance(date=date(2024, 1, 1), value=Quantity(Decimal("1000"))),
        account2: Balance(date=date(2024, 1, 1), value=Quantity(Decimal("500"))),
    }
    
    # Create journal entries with postings
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=date(2024, 1, 15),
            description="Test entry 1",
            postings=[]
        )
    )
    
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("50")),
        direction=Direction.CREDIT,
        journal=JournalEntry(
            date=date(2024, 2, 1),
            description="Test entry 2",
            postings=[]
        )
    )
    
    posting3 = Posting(
        account=account3,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        journal=JournalEntry(
            date=date(2024, 3, 10),
            description="Test entry 3",
            postings=[]
        )
    )
    
    journal_entries = [
        JournalEntry(date=date(2024, 1, 15), description="Test entry 1", postings=[posting1]),
        JournalEntry(date=date(2024, 2, 1), description="Test entry 2", postings=[posting2]),
        JournalEntry(date=date(2024, 3, 10), description="Test entry 3", postings=[posting3]),
    ]
    
    # Execute
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assert
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 3
    
    # Check account1 ledger
    assert account1 in result.ledgers
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("1000"))
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal("1100"))
    
    # Check account2 ledger
    assert account2 in result.ledgers
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("500"))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("450"))
    
    # Check account3 ledger (created without initial balance)
    assert account3 in result.ledgers
    ledger3 = result.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal("0"))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal("-200"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    from datetime import date
    
    account1 = Account(code="1000", name="Cash")
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    
    initial_balances = {
        account1: Balance(date=date(2024, 1, 1), value=Quantity(Decimal("1000"))),
    }
    
    result = build_general_ledger(period, [], initial_balances)
    
    assert len(result.ledgers) == 1
    assert result.ledgers[account1].entries == []


def test_build_general_ledger_filters_by_period():
    """Test build_general_ledger filters entries outside period."""
    from datetime import date
    
    account1 = Account(code="1000", name="Cash")
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    
    initial_balances = {}
    
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=date(2023, 12, 31),
            description="Outside period",
            postings=[]
        )
    )
    
    posting2 = Posting(
        account=account1,
        amount=Amount(Decimal("50")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=date(2024, 6, 1),
            description="Inside period",
            postings=[]
        )
    )
    
    journal_entries = [
        JournalEntry(date=date(2023, 12, 31), description="Outside period", postings=[posting1]),
        JournalEntry(date=date(2024, 6, 1), description="Inside period", postings=[posting2]),
    ]
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert len(result.ledgers[account1].entries) == 1
    assert result.ledgers[account1].entries[0].date == date(2024, 6, 1)


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    
    # Create test accounts
    account_cash = Account(number="1000", name="Cash")
    account_revenue = Account(number="4000", name="Revenue")
    account_expense = Account(number="5000", name="Expense")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account_cash: Balance(date(2023, 1, 1), Quantity(Decimal("1000"))),
    }
    
    # Create journal entries with postings
    posting1 = Posting(
        account=account_cash,
        amount=Decimal("500"),
        direction=1,
        journal=JournalEntry(
            date=date(2023, 1, 15),
            description="Initial deposit",
            postings=[]
        )
    )
    posting2 = Posting(
        account=account_revenue,
        amount=Decimal("500"),
        direction=-1,
        journal=JournalEntry(
            date=date(2023, 1, 15),
            description="Initial deposit",
            postings=[]
        )
    )
    
    journal_entry1 = JournalEntry(
        date=date(2023, 1, 15),
        description="Initial deposit",
        postings=[posting1, posting2]
    )
    
    posting3 = Posting(
        account=account_expense,
        amount=Decimal("200"),
        direction=1,
        journal=JournalEntry(
            date=date(2023, 2, 1),
            description="Office supplies",
            postings=[]
        )
    )
    posting4 = Posting(
        account=account_cash,
        amount=Decimal("200"),
        direction=-1,
        journal=JournalEntry(
            date=date(2023, 2, 1),
            description="Office supplies",
            postings=[]
        )
    )
    
    journal_entry2 = JournalEntry(
        date=date(2023, 2, 1),
        description="Office supplies",
        postings=[posting3, posting4]
    )
    
    posting1.journal = journal_entry1
    posting2.journal = journal_entry1
    posting3.journal = journal_entry2
    posting4.journal = journal_entry2
    
    # Create period
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Build general ledger
    general_ledger = build_general_ledger(period, [journal_entry1, journal_entry2], initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    assert account_cash in general_ledger.ledgers
    assert account_revenue in general_ledger.ledgers
    assert account_expense in general_ledger.ledgers
    
    # Check cash ledger
    cash_ledger = general_ledger.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial == initial_balances[account_cash]
    assert len(cash_ledger.entries) == 2
    
    # Check revenue ledger (not in initial balances)
    revenue_ledger = general_ledger.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert revenue_ledger.initial.value == Quantity(Decimal(0))
    assert len(revenue_ledger.entries) == 1
    
    # Check expense ledger (not in initial balances)
    expense_ledger = general_ledger.ledgers[account_expense]
    assert expense_ledger.account == account_expense
    assert expense_ledger.initial.value == Quantity(Decimal(0))
    assert len(expense_ledger.entries) == 1
    
    # Check ledger entry values
    assert cash_ledger.entries[0].amount == Decimal("500")
    assert cash_ledger.entries[1].amount == Decimal("200")


# LLM-generated content at query #3
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(code="1000", name="Cash", account_type="Asset")
        account2 = Account(code="2000", name="Payable", account_type="Liability")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create a date range for the period
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function through the protocol
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the accounts and balances
    accounts = list(result.keys())
    assert accounts[0].code == "1000"
    assert accounts[1].code == "2000"
    
    balances = list(result.values())
    assert balances[0].value == Quantity(Decimal("1000.00"))
    assert balances[1].value == Quantity(Decimal("500.00"))
    
    # Verify balance dates match period start
    assert balances[0].date == start_date
    assert balances[1].date == start_date


def test_ReadInitialBalances___call___empty():
    """Test ReadInitialBalances protocol with empty balances."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___single_account():
    """Test ReadInitialBalances protocol with a single account."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account = Account(code="5000", name="Revenue", account_type="Revenue")
        return {account: Balance(period.since, Quantity(Decimal("0.00")))}
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert len(result) == 1
    account = list(result.keys())[0]
    balance = list(result.values())[0]
    
    assert account.code == "5000"
    assert balance.value == Quantity(Decimal("0.00"))


# LLM-generated content at query #4
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    from datetime import date
    from decimal import Decimal
    
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(name="Cash", code="1000")
        account2 = Account(name="Accounts Payable", code="2000")
        
        return {
            account1: Balance(date=period.since, value=Quantity(Decimal("1000.00"))),
            account2: Balance(date=period.since, value=Quantity(Decimal("500.00"))),
        }
    
    # Create a DateRange for testing
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify that all keys are Account instances
    for key in result.keys():
        assert isinstance(key, Account)
    
    # Verify that all values are Balance instances
    for value in result.values():
        assert isinstance(value, Balance)
    
    # Verify the balance values
    accounts = list(result.keys())
    balances = list(result.values())
    
    assert balances[0].value == Quantity(Decimal("1000.00"))
    assert balances[1].value == Quantity(Decimal("500.00"))
    
    # Verify the date in balances
    for balance in balances:
        assert balance.date == period.since


# LLM-generated content at query #5
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup test data
    test_date = datetime.date(2024, 1, 1)
    test_end_date = datetime.date(2024, 12, 31)
    period = DateRange(test_date, test_end_date)
    
    # Create test accounts
    account1 = Account(number="1000", name="Cash")
    account2 = Account(number="2000", name="Accounts Payable")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(test_date, Quantity(Decimal("1000"))),
        account2: Balance(test_date, Quantity(Decimal("500"))),
    }
    
    # Create test journal entries with postings
    from .journaling import Direction
    
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        date=datetime.date(2024, 6, 15),
        journal=None  # Will be set by JournalEntry
    )
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        date=datetime.date(2024, 6, 15),
        journal=None
    )
    
    journal_entry = JournalEntry(
        date=datetime.date(2024, 6, 15),
        description="Test transaction",
        postings=[posting1, posting2]
    )
    
    # Update posting references to journal entry
    posting1.journal = journal_entry
    posting2.journal = journal_entry
    
    journal_entries = [journal_entry]
    
    # Define mock implementations
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        assert p == period
        return initial_balances
    
    def mock_read_journal_entries(p: DateRange) -> Iterable[JournalEntry]:
        assert p == period
        return journal_entries
    
    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)
    
    # Execute the program by calling it
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account2].account == account2
    assert result.ledgers[account1].initial == initial_balances[account1]
    assert result.ledgers[account2].initial == initial_balances[account2]
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #6
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(code="1000", name="Test Account 1")
        return {
            account1: Balance(date=period.since, value=Quantity(Decimal("1000.00")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account(code="1000", name="Test Account 1")
        account2 = Account(code="2000", name="Test Account 2")
        
        from ..commons.journaling import Direction
        
        posting1 = Posting(
            account=account1,
            amount=Amount(Decimal("100.00")),
            direction=Direction.DEBIT,
            date=period.since
        )
        posting2 = Posting(
            account=account2,
            amount=Amount(Decimal("100.00")),
            direction=Direction.CREDIT,
            date=period.since
        )
        
        journal_entry = JournalEntry(
            date=period.since,
            description="Test Entry",
            postings=[posting1, posting2]
        )
        
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Create a test period
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) >= 1
    
    # Verify ledger structure
    for account, ledger in result.ledgers.items():
        assert isinstance(ledger, Ledger)
        assert ledger.account == account
        assert isinstance(ledger.entries, list)


# LLM-generated content at query #7
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    import datetime
    from decimal import Decimal
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(code="1000", name="Cash")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account(code="1000", name="Cash")
        account2 = Account(code="2000", name="Payable")
        
        from ..commons.journaling import Direction
        
        posting1 = Posting(account1, Decimal("100"), Direction.DEBIT, JournalEntry(
            date=period.since,
            description="Test entry",
            postings=[]
        ))
        posting2 = Posting(account2, Decimal("100"), Direction.CREDIT, JournalEntry(
            date=period.since,
            description="Test entry",
            postings=[]
        ))
        
        journal_entry = JournalEntry(
            date=period.since,
            description="Test entry",
            postings=[posting1, posting2]
        )
        
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Create a date range
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) >= 1
    
    # Verify ledger structure
    for account, ledger in result.ledgers.items():
        assert isinstance(ledger, Ledger)
        assert ledger.account == account
        assert isinstance(ledger.entries, list)


# LLM-generated content at query #8
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(code="1000", name="Cash", account_type="Asset")
        account2 = Account(code="2000", name="Payable", account_type="Liability")
        
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create a date range
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure of returned balances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == period.since
        assert isinstance(balance.value, Quantity)
    
    # Verify specific values
    accounts = list(result.keys())
    assert any(acc.code == "1000" for acc in accounts)
    assert any(acc.code == "2000" for acc in accounts)
    
    # Verify amounts
    balances = list(result.values())
    assert any(bal.value == Quantity(Decimal("1000.00")) for bal in balances)
    assert any(bal.value == Quantity(Decimal("500.00")) for bal in balances)


def test_ReadInitialBalances___call___empty():
    """Test ReadInitialBalances protocol with empty initial balances."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___multiple_accounts():
    """Test ReadInitialBalances protocol with multiple accounts."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        accounts_data = [
            ("1000", "Cash", Decimal("5000.00")),
            ("1100", "Accounts Receivable", Decimal("2500.00")),
            ("2000", "Accounts Payable", Decimal("1000.00")),
            ("3000", "Capital", Decimal("6500.00")),
        ]
        
        return {
            Account(code=code, name=name, account_type="Asset"): Balance(period.since, Quantity(amount))
            for code, name, amount in accounts_data
        }
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 4
    
    total_amount = sum(bal.value for bal in result.values())
    assert total_amount == Quantity(Decimal("15000.00"))


# LLM-generated content at query #9
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    
    # Setup test data
    account1 = Account(name="Cash", number="1000")
    account2 = Account(name="Revenue", number="4000")
    account3 = Account(name="Expenses", number="5000")
    
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    
    initial_balances = {
        account1: Balance(date=date(2024, 1, 1), value=Quantity(Decimal("1000"))),
        account2: Balance(date=date(2024, 1, 1), value=Quantity(Decimal("0"))),
    }
    
    # Create journal entries with postings
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("500")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("500")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    journal_entry1 = JournalEntry(
        date=date(2024, 6, 15),
        description="Test transaction",
        postings=[posting1, posting2]
    )
    
    posting3 = Posting(
        account=account3,
        amount=Amount(Decimal("200")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting4 = Posting(
        account=account1,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    journal_entry2 = JournalEntry(
        date=date(2024, 7, 20),
        description="Another transaction",
        postings=[posting3, posting4]
    )
    
    journal = [journal_entry1, journal_entry2]
    
    # Test build_general_ledger
    result = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert result.period == period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert account3 in result.ledgers
    
    # Check initial ledger
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account1].initial == initial_balances[account1]
    
    # Check ledger entries were added
    assert len(result.ledgers[account1].entries) == 2
    assert len(result.ledgers[account2].entries) == 1
    assert len(result.ledgers[account3].entries) == 1
    
    # Check that account3 was created with zero initial balance
    assert result.ledgers[account3].initial.value == Quantity(Decimal("0"))
    assert result.ledgers[account3].initial.date == period.since
    
    # Check balances are calculated correctly
    # account1: 1000 + 500 (debit) - 200 (credit) = 1300
    assert result.ledgers[account1].entries[-1].balance == Quantity(Decimal("1300"))
    
    # account2: 0 - 500 (credit) = -500
    assert result.ledgers[account2].entries[-1].balance == Quantity(Decimal("-500"))
    
    # account3: 0 + 200 (debit) = 200
    assert result.ledgers[account3].entries[-1].balance == Quantity(Decimal("200"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    account1 = Account(name="Cash", number="1000")
    
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    
    initial_balances = {
        account1: Balance(date=date(2024, 1, 1), value=Quantity(Decimal("5000"))),
    }
    
    journal = []
    
    result = build_general_ledger(period, journal, initial_balances)
    
    assert result.period == period
    assert account1 in result.ledgers
    assert len(result.ledgers[account1].entries) == 0
    assert result.ledgers[account1].initial == initial_balances[account1]


def test_build_general_ledger_out_of_period():
    """Test build_general_ledger filters postings outside period."""
    account1 = Account(name="Cash", number="1000")
    account2 = Account(name="Revenue", number="4000")
    
    period = DateRange(since=date(2024, 6, 1), until=date(2024, 12, 31))
    
    initial_balances = {
        account1: Balance(date=date(2024, 6, 1), value=Quantity(Decimal("1000"))),
    }
    
    # Create posting outside period
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    journal_entry_before = JournalEntry(
        date=date(2024, 5, 15),
        description="Before period",
        postings=[posting1, posting2]
    )
    
    journal_entry_after = JournalEntry(
        date=date(2025, 1, 15),
        description="After period",
        postings=[posting1, posting2]
    )
    
    journal = [journal_entry_before, journal_entry_after]
    
    result = build_general_ledger(period, journal, initial_balances)
    
    # No entries should be added as both are outside period
    assert len(result.ledgers) == 1
    assert account1 in result.ledgers
    assert account2 not in result.ledgers
    assert len(result.ledgers[account1].entries) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    from decimal import Decimal
    
    # Setup test data
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Create test accounts
    account_cash = Account(code="1000", name="Cash")
    account_revenue = Account(code="4000", name="Revenue")
    account_expense = Account(code="5000", name="Expense")
    
    # Create initial balances
    initial_balances: InitialBalances = {
        account_cash: Balance(start_date, Quantity(Decimal("1000")))
    }
    
    # Create journal entries with postings
    from .journaling import Direction
    
    posting1_cash = Posting(
        account=account_cash,
        amount=Amount(Decimal("500")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting1_revenue = Posting(
        account=account_revenue,
        amount=Amount(Decimal("500")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    posting2_expense = Posting(
        account=account_expense,
        amount=Amount(Decimal("200")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting2_cash = Posting(
        account=account_cash,
        amount=Amount(Decimal("200")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    journal_entry1 = JournalEntry(
        date=date(2023, 6, 15),
        description="Revenue transaction",
        postings=[posting1_cash, posting1_revenue]
    )
    posting1_cash.journal = journal_entry1
    posting1_revenue.journal = journal_entry1
    
    journal_entry2 = JournalEntry(
        date=date(2023, 6, 20),
        description="Expense transaction",
        postings=[posting2_expense, posting2_cash]
    )
    posting2_expense.journal = journal_entry2
    posting2_cash.journal = journal_entry2
    
    journal = [journal_entry1, journal_entry2]
    
    # Execute
    result = build_general_ledger(period, journal, initial_balances)
    
    # Assert
    assert result.period == period
    assert len(result.ledgers) == 3
    
    # Verify cash ledger
    assert account_cash in result.ledgers
    cash_ledger = result.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial == initial_balances[account_cash]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500"))  # 1000 + 500
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300"))  # 1500 - 200
    
    # Verify revenue ledger
    assert account_revenue in result.ledgers
    revenue_ledger = result.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500"))  # 0 - 500 (credit)
    
    # Verify expense ledger
    assert account_expense in result.ledgers
    expense_ledger = result.ledgers[account_expense]
    assert expense_ledger.account == account_expense
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200"))  # 0 + 200


def test_build_general_ledger_empty_period():
    """Test build_general_ledger with no entries in period."""
    from datetime import date
    from decimal import Decimal
    
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    account_cash = Account(code="1000", name="Cash")
    initial_balances: InitialBalances = {
        account_cash: Balance(start_date, Quantity(Decimal("1000")))
    }
    
    result = build_general_ledger(period, [], initial_balances)
    
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account_cash in result.ledgers
    assert len(result.ledgers[account_cash].entries) == 0


def test_build_general_ledger_excludes_entries_outside_period():
    """Test build_general_ledger excludes entries outside period."""
    from datetime import date
    from decimal import Decimal
    
    start_date = date(2023, 6, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    account = Account(code="1000", name="Account")
    initial_balances: InitialBalances = {}
    
    from .journaling import Direction
    
    # Entry before period
    posting_before = Posting(
        account=account,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=None
    )
    journal_before = JournalEntry(
        date=date(2023, 5, 15),
        description="Before",
        postings=[posting_before]
    )
    posting_before.journal = journal_before
    
    # Entry within period
    posting_within = Posting(
        account=account,
        amount=Amount(Decimal("200")),
        direction=Direction.DEBIT,
        journal=None
    )
    journal_within = JournalEntry(
        date=date(2023, 7, 15),
        description="Within",
        postings=[posting_within]
    )
    posting_within.journal = journal_within
    
    journal = [journal_before, journal_within]
    
    result = build_general_ledger(period, journal, initial_balances)
    
    assert len(result.ledgers) == 1
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].amount == Amount(Decimal("200"))


# LLM-generated content at query #11
#--------------------------

```python
import datetime
from decimal import Decimal

import pytest

from ..commons.numbers import Amount, Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance
from .journaling import JournalEntry, Posting, Direction


def test_build_general_ledger():
    """Test build_general_ledger function with various scenarios."""
    
    # Setup test data
    account_a = Account(code="1000", name="Cash", account_type="Asset")
    account_b = Account(code="2000", name="Payable", account_type="Liability")
    account_c = Account(code="3000", name="Revenue", account_type="Revenue")
    
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(since=period_start, until=period_end)
    
    # Initial balances
    initial_balances: InitialBalances = {
        account_a: Balance(period_start, Quantity(Decimal("1000.00"))),
        account_b: Balance(period_start, Quantity(Decimal("500.00"))),
    }
    
    # Create journal entries with postings
    journal_entry_1 = JournalEntry(
        date=datetime.date(2024, 1, 15),
        description="Initial transaction",
        postings=[
            Posting(account=account_a, amount=Amount(Decimal("-100.00")), direction=Direction.DEBIT, journal=None),
            Posting(account=account_b, amount=Amount(Decimal("100.00")), direction=Direction.CREDIT, journal=None),
        ]
    )
    journal_entry_1.postings[0].journal = journal_entry_1
    journal_entry_1.postings[1].journal = journal_entry_1
    
    journal_entry_2 = JournalEntry(
        date=datetime.date(2024, 2, 20),
        description="Revenue transaction",
        postings=[
            Posting(account=account_a, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT, journal=None),
            Posting(account=account_c, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT, journal=None),
        ]
    )
    journal_entry_2.postings[0].journal = journal_entry_2
    journal_entry_2.postings[1].journal = journal_entry_2
    
    journal_entries = [journal_entry_1, journal_entry_2]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 4  # account_a, account_b, account_c, and account_c created
    
    # Check account_a ledger
    ledger_a = general_ledger.ledgers[account_a]
    assert ledger_a.account == account_a
    assert ledger_a.initial.value == Quantity(Decimal("1000.00"))
    assert len(ledger_a.entries) == 2
    assert ledger_a.entries[0].balance == Quantity(Decimal("900.00"))
    assert ledger_a.entries[1].balance == Quantity(Decimal("1400.00"))
    
    # Check account_b ledger
    ledger_b = general_ledger.ledgers[account_b]
    assert ledger_b.account == account_b
    assert ledger_b.initial.value == Quantity(Decimal("500.00"))
    assert len(ledger_b.entries) == 1
    assert ledger_b.entries[0].balance == Quantity(Decimal("600.00"))
    
    # Check account_c ledger (created without initial balance)
    ledger_c = general_ledger.ledgers[account_c]
    assert ledger_c.account == account_c
    assert ledger_c.initial.value == Quantity(Decimal("0"))
    assert len(ledger_c.entries) == 1
    assert ledger_c.entries[0].balance == Quantity(Decimal("-500.00"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal entries."""
    
    account_a = Account(code="1000", name="Cash", account_type="Asset")
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(since=period_start, until=period_end)
    
    initial_balances: InitialBalances = {
        account_a: Balance(period_start, Quantity(Decimal("1000.00"))),
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert general_ledger.ledgers[account_a].entries == []
    assert general_ledger.ledgers[account_a].initial.value == Quantity(Decimal("1000.00"))


def test_build_general_ledger_out_of_period():
    """Test build_general_ledger filters entries outside the period."""
    
    account_a = Account(code="1000", name="Cash", account_type="Asset")
    account_b = Account(code="2000", name="Payable", account_type="Liability")
    
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(since=period_start, until=period_end)
    
    initial_balances: InitialBalances = {
        account_a: Balance(period_start, Quantity(Decimal("1000.00"))),
    }
    
    # Entry outside period
    journal_entry = JournalEntry(
        date=datetime.date(2023, 12, 31),
        description="Out of period",
        postings=[
            Posting(account=account_a, amount=Amount(Decimal("-100.00")), direction=Direction.DEBIT, journal=None),
            Posting(account=account_b, amount=Amount(Decimal("100.00")), direction=Direction.CREDIT, journal=None),
        ]
    )
    journal_entry.postings[0].journal = journal_entry
    journal_entry.postings[1].journal = journal_entry
    
    general_ledger = build_general_ledger(period, [journal_entry], initial_balances)
    
    assert len(general_ledger.ledgers) == 1  # Only account_a from initial balances
    assert general_ledger.ledgers[account_a].entries == []


def test_build_general_ledger_no_initial_balances():
    """Test build_general_ledger with no initial balances."""
    
    account_a = Account(code="1000", name="Cash", account_type="Asset")
    account_b = Account(code="2000", name="Payable", account_type="Liability")
    
    period_start = datetime.date(2024, 1, 1)
    period_end = datetime.date(2024, 12, 31)
    period = DateRange(since=period_start, until=period_end)
    
    journal_entry = JournalEntry(
        date=datetime.date(2024, 1, 15),
        description="Transaction",
        postings=[
            Posting(account=account_a, amount=Amount(Decimal("500.00")), direction=Direction.DEBIT, journal=None),
            Posting(account=account_b, amount=Amount(Decimal("500.00")), direction=Direction.CREDIT, journal=None),
        ]
    )
    journal_entry.postings[0].journal = journal_entry
    journal_entry.postings[1].journal = journal_entry
    
    general_ledger


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

from ..commons.numbers import Amount, Quantity
from ..commons.zeitgeist import DateRange
from .accounts import Account
from .generic import Balance
from .journaling import JournalEntry, Posting, Direction


def test_build_general_ledger():
    """Test build_general_ledger function with various scenarios."""
    
    # Setup test data
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account_cash = Account("1000", "Cash", "asset")
    account_revenue = Account("4000", "Revenue", "income")
    account_expense = Account("5000", "Expense", "expense")
    
    # Create initial balances
    initial_balances = {
        account_cash: Balance(
            date=datetime.date(2023, 1, 1),
            value=Quantity(Decimal("1000.00"))
        )
    }
    
    # Create mock journal entries with postings
    posting1 = Posting(
        account=account_cash,
        amount=Amount(Decimal("500.00")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 1, 15),
        journal=None
    )
    
    posting2 = Posting(
        account=account_revenue,
        amount=Amount(Decimal("500.00")),
        direction=Direction.CREDIT,
        date=datetime.date(2023, 1, 15),
        journal=None
    )
    
    posting3 = Posting(
        account=account_cash,
        amount=Amount(Decimal("200.00")),
        direction=Direction.CREDIT,
        date=datetime.date(2023, 2, 1),
        journal=None
    )
    
    posting4 = Posting(
        account=account_expense,
        amount=Amount(Decimal("200.00")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 2, 1),
        journal=None
    )
    
    # Create journal entries
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test entry 1",
        postings=[posting1, posting2]
    )
    
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 1),
        description="Test entry 2",
        postings=[posting3, posting4]
    )
    
    # Update journal references in postings
    posting1.journal = journal_entry1
    posting2.journal = journal_entry1
    posting3.journal = journal_entry2
    posting4.journal = journal_entry2
    
    journal = [journal_entry1, journal_entry2]
    
    # Execute the function
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    assert account_cash in general_ledger.ledgers
    assert account_revenue in general_ledger.ledgers
    assert account_expense in general_ledger.ledgers
    
    # Check cash ledger
    cash_ledger = general_ledger.ledgers[account_cash]
    assert cash_ledger.account == account_cash
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    
    # Check revenue ledger
    revenue_ledger = general_ledger.ledgers[account_revenue]
    assert revenue_ledger.account == account_revenue
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500.00"))
    
    # Check expense ledger
    expense_ledger = general_ledger.ledgers[account_expense]
    assert expense_ledger.account == account_expense
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account = Account("1000", "Cash", "asset")
    initial_balances = {
        account: Balance(
            date=datetime.date(2023, 1, 1),
            value=Quantity(Decimal("500.00"))
        )
    }
    
    general_ledger = build_general_ledger(period, [], initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 0
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal("500.00"))


def test_build_general_ledger_no_initial_balances():
    """Test build_general_ledger with no initial balances."""
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    account = Account("1000", "Cash", "asset")
    
    posting = Posting(
        account=account,
        amount=Amount(Decimal("100.00")),
        direction=Direction.DEBIT,
        date=datetime.date(2023, 1, 15),
        journal=None
    )
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test",
        postings=[posting]
    )
    
    posting.journal = journal_entry
    
    general_ledger = build_general_ledger(period, [journal_entry], {})
    
    assert len(general_ledger.ledgers) == 1
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1
    assert general_ledger.ledgers[account].initial.value == Quantity(Decimal("0"))


# LLM-generated content at query #13
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from decimal import Decimal
    
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Create mock accounts
    account1 = Account(number="1000", name="Cash", account_type="Asset")
    account2 = Account(number="2000", name="Accounts Payable", account_type="Liability")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000"))),
        account2: Balance(datetime.date(2023, 1, 1), Quantity(Decimal("500"))),
    }
    
    # Create mock journal entries with postings
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=None
    )
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        journal=None
    )
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test transaction",
        postings=[posting1, posting2]
    )
    
    # Create mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Execute the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].initial == initial_balances[account1]
    assert result.ledgers[account2].initial == initial_balances[account2]


# LLM-generated content at query #14
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(number="1000", name="Cash")
        return {account1: Balance(period.since, Quantity(Decimal("1000")))}
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account(number="1000", name="Cash")
        account2 = Account(number="2000", name="Accounts Payable")
        
        from ..commons.direction import Direction
        
        posting1 = Posting(account1, Direction.DEBIT, Quantity(Decimal("100")), None)
        posting2 = Posting(account2, Direction.CREDIT, Quantity(Decimal("100")), None)
        
        journal = JournalEntry(
            date=date(2023, 6, 15),
            description="Test transaction",
            postings=[posting1, posting2]
        )
        return [journal]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Create a date range
    period = DateRange(since=date(2023, 1, 1), until=date(2023, 12, 31))
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) >= 1
    
    # Verify ledger structure
    for account, ledger in result.ledgers.items():
        assert isinstance(ledger, Ledger)
        assert ledger.account == account
        assert isinstance(ledger.entries, list)
        assert isinstance(ledger.initial, Balance)


# LLM-generated content at query #15
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    # Create a test implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(number="1000", name="Cash")
        account2 = Account(number="2000", name="Accounts Payable")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create a date range
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure of returned initial balances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == period.since
        assert isinstance(balance.value, Quantity)
    
    # Verify specific values
    accounts = list(result.keys())
    assert accounts[0].number == "1000"
    assert accounts[1].number == "2000"
    assert result[accounts[0]].value == Quantity(Decimal("1000.00"))
    assert result[accounts[1]].value == Quantity(Decimal("500.00"))


def test_ReadInitialBalances___call___empty():
    """Test ReadInitialBalances with empty initial balances."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___single_account():
    """Test ReadInitialBalances with a single account."""
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account = Account(number="5000", name="Revenue")
        return {account: Balance(period.since, Quantity(Decimal("0.00")))}
    
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert len(result) == 1
    account = list(result.keys())[0]
    assert account.number == "5000"
    assert result[account].value == Quantity(Decimal("0.00"))


# LLM-generated content at query #16
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    from decimal import Decimal
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(code="1000", name="Cash", account_type="Asset")
        return {
            account1: Balance(date=period.since, value=Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account(code="1000", name="Cash", account_type="Asset")
        account2 = Account(code="2000", name="Liability", account_type="Liability")
        
        from ..commons.directions import Direction
        
        journal = JournalEntry(
            date=period.since,
            description="Test transaction",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("100")), direction=Direction.DEBIT, journal=None),
                Posting(account=account2, amount=Amount(Decimal("100")), direction=Direction.CREDIT, journal=None),
            ]
        )
        journal.postings[0].journal = journal
        journal.postings[1].journal = journal
        
        return [journal]
    
    # Compile the program
    program = compile_general_ledger_program(mock_read_initial_balances, mock_read_journal_entries)
    
    # Create a test period
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) >= 1
    
    # Verify ledger entries were created
    for account, ledger in result.ledgers.items():
        assert isinstance(ledger, Ledger)
        assert ledger.account == account


# LLM-generated content at query #17
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(name="Cash", number="1000")
        account2 = Account(name="Accounts Receivable", number="1200")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000.00"))),
            account2: Balance(period.since, Quantity(Decimal("500.00"))),
        }
    
    # Create a DateRange for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    # Call the function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    accounts = list(result.keys())
    assert all(isinstance(acc, Account) for acc in accounts)
    
    balances = list(result.values())
    assert all(isinstance(bal, Balance) for bal in balances)
    
    # Verify the balances are correct
    for account, balance in result.items():
        assert balance.date == start_date
        if account.number == "1000":
            assert balance.value == Quantity(Decimal("1000.00"))
        elif account.number == "1200":
            assert balance.value == Quantity(Decimal("500.00"))


def test_ReadInitialBalances___call___empty():
    """Test ReadInitialBalances with empty initial balances."""
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {}
    
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert isinstance(result, dict)
    assert len(result) == 0


def test_ReadInitialBalances___call___single_account():
    """Test ReadInitialBalances with a single account."""
    
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account = Account(name="Bank", number="1010")
        return {
            account: Balance(period.since, Quantity(Decimal("5000.50"))),
        }
    
    start_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)
    
    result = mock_read_initial_balances(period)
    
    assert len(result) == 1
    account = list(result.keys())[0]
    balance = result[account]
    
    assert account.name == "Bank"
    assert balance.date == start_date
    assert balance.value == Quantity(Decimal("5000.50"))


# LLM-generated content at query #18
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup test data
    test_date = datetime.date(2023, 1, 1)
    period = DateRange(test_date, datetime.date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Payable")
    
    initial_balances = {
        account1: Balance(test_date, Quantity(Decimal("1000"))),
        account2: Balance(test_date, Quantity(Decimal("500"))),
    }
    
    # Create mock journal entries
    journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test transaction",
        postings=[
            Posting(account1, Decimal("100"), Direction.DEBIT, journal_entry=None),
            Posting(account2, Decimal("100"), Direction.CREDIT, journal_entry=None),
        ]
    )
    
    # Mock reader functions
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(p: DateRange) -> Iterable[JournalEntry]:
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account2].account == account2
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].amount == Decimal("100")
    assert result.ledgers[account2].entries[0].amount == Decimal("100")


# LLM-generated content at query #19
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    from decimal import Decimal
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1000", "Cash", None)
        return {
            account1: Balance(period.since, Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account("1000", "Cash", None)
        account2 = Account("2000", "Payable", None)
        
        from ..commons.zeitgeist import Direction
        
        posting1 = Posting(account1, Decimal("500"), Direction.DEBIT)
        posting2 = Posting(account2, Decimal("500"), Direction.CREDIT)
        
        journal_entry = JournalEntry(
            date=period.since,
            description="Test Entry",
            postings=[posting1, posting2]
        )
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Create a test period
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Call the program
    general_ledger = program(period)
    
    # Assertions
    assert general_ledger is not None
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    assert all(isinstance(ledger, Ledger) for ledger in general_ledger.ledgers.values())


# LLM-generated content at query #20
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup test data
    test_account = Account(code="1000", name="Test Account", account_type="asset")
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create mock initial balances
    initial_balances = {
        test_account: Balance(
            date=datetime.date(2022, 12, 31),
            value=Quantity(Decimal("1000"))
        )
    }
    
    # Create mock journal entries
    posting = Posting(
        account=test_account,
        amount=Amount(Decimal("100")),
        direction=1,
        journal=JournalEntry(
            date=datetime.date(2023, 6, 15),
            description="Test transaction",
            postings=[]
        )
    )
    posting.journal.postings = [posting]
    
    journal_entries = [posting.journal]
    
    # Define mock functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return journal_entries
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert test_account in result.ledgers
    assert len(result.ledgers) == 1
    assert result.ledgers[test_account].account == test_account
    assert result.ledgers[test_account].initial == initial_balances[test_account]
    assert len(result.ledgers[test_account].entries) == 1
    assert result.ledgers[test_account].entries[0].posting == posting


# LLM-generated content at query #21
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account(name="Account1", number="1000")
        return {
            account1: Balance(date=period.since, value=Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account(name="Account1", number="1000")
        account2 = Account(name="Account2", number="2000")
        
        from .journaling import Direction
        
        journal_entry = JournalEntry(
            date=date(2024, 1, 15),
            description="Test transaction",
            postings=[
                Posting(account=account1, amount=Amount(Decimal("100")), direction=Direction.DEBIT, journal=None),
                Posting(account=account2, amount=Amount(Decimal("100")), direction=Direction.CREDIT, journal=None),
            ]
        )
        journal_entry.postings[0].journal = journal_entry
        journal_entry.postings[1].journal = journal_entry
        
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries,
    )
    
    # Create a test period
    period = DateRange(since=date(2024, 1, 1), until=date(2024, 12, 31))
    
    # Call the program
    general_ledger = program(period)
    
    # Assert results
    assert isinstance(general_ledger, GeneralLedger)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) >= 1
    
    # Verify that initial balance was loaded
    account1 = Account(name="Account1", number="1000")
    assert account1 in general_ledger.ledgers or any(
        ledger.account.number == "1000" for ledger in general_ledger.ledgers.values()
    )
    
    # Verify that journal entries were processed
    ledgers_with_entries = [ledger for ledger in general_ledger.ledgers.values() if len(ledger.entries) > 0]
    assert len(ledgers_with_entries) > 0


# LLM-generated content at query #22
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    # Setup test data
    account1 = Account(name="Cash", number="1000")
    account2 = Account(name="Revenue", number="4000")
    account3 = Account(name="Expense", number="5000")
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    initial_balance = Balance(
        since=datetime.date(2022, 12, 31),
        value=Quantity(Decimal("1000"))
    )
    
    initial_balances = {account1: initial_balance}
    
    # Create journal entries with postings
    posting1 = Posting(
        account=account1,
        amount=Decimal("500"),
        direction=Direction.DEBIT,
        journal=None
    )
    posting2 = Posting(
        account=account2,
        amount=Decimal("500"),
        direction=Direction.CREDIT,
        journal=None
    )
    
    journal_entry1 = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction 1",
        postings=[posting1, posting2]
    )
    
    posting3 = Posting(
        account=account1,
        amount=Decimal("200"),
        direction=Direction.CREDIT,
        journal=None
    )
    posting4 = Posting(
        account=account3,
        amount=Decimal("200"),
        direction=Direction.DEBIT,
        journal=None
    )
    
    journal_entry2 = JournalEntry(
        date=datetime.date(2023, 2, 20),
        description="Test transaction 2",
        postings=[posting3, posting4]
    )
    
    journal = [journal_entry1, journal_entry2]
    
    # Execute function
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    assert account1 in general_ledger.ledgers
    assert account2 in general_ledger.ledgers
    assert account3 in general_ledger.ledgers
    
    # Check account1 ledger (initial balance 1000, +500 debit, -200 credit)
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balance
    assert len(ledger1.entries) == 2
    
    # Check account2 ledger (new account, -500 credit)
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("0"))
    assert len(ledger2.entries) == 1
    
    # Check account3 ledger (new account, +200 debit)
    ledger3 = general_ledger.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal("0"))
    assert len(ledger3.entries) == 1


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    account1 = Account(name="Cash", number="1000")
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    initial_balance = Balance(
        since=datetime.date(2022, 12, 31),
        value=Quantity(Decimal("5000"))
    )
    
    initial_balances = {account1: initial_balance}
    journal = []
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert account1 in general_ledger.ledgers
    assert len(general_ledger.ledgers[account1].entries) == 0
    assert general_ledger.ledgers[account1].initial == initial_balance


def test_build_general_ledger_filters_by_period():
    """Test that build_general_ledger filters entries by period."""
    account1 = Account(name="Cash", number="1000")
    account2 = Account(name="Revenue", number="4000")
    
    period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    initial_balances = {}
    
    # Entry within period
    posting1 = Posting(
        account=account1,
        amount=Decimal("100"),
        direction=Direction.DEBIT,
        journal=None
    )
    posting2 = Posting(
        account=account2,
        amount=Decimal("100"),
        direction=Direction.CREDIT,
        journal=None
    )
    
    entry_in_period = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Within period",
        postings=[posting1, posting2]
    )
    
    # Entry outside period (before)
    posting3 = Posting(
        account=account1,
        amount=Decimal("50"),
        direction=Direction.DEBIT,
        journal=None
    )
    posting4 = Posting(
        account=account2,
        amount=Decimal("50"),
        direction=Direction.CREDIT,
        journal=None
    )
    
    entry_before_period = JournalEntry(
        date=datetime.date(2022, 12, 31),
        description="Before period",
        postings=[posting3, posting4]
    )
    
    journal = [entry_before_period, entry_in_period]
    
    general_ledger = build_general_ledger(period, journal, initial_balances)
    
    # Only the entry within period should be included
    assert len(general_ledger.ledgers[account1].entries) == 1
    assert len(general_ledger.ledgers[account2].entries) == 1


# LLM-generated content at query #23
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test the __call__ method of ReadInitialBalances protocol."""
    from datetime import date
    from decimal import Decimal
    
    # Create a mock implementation of ReadInitialBalances
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        """Mock implementation that returns initial balances."""
        account1 = Account(code="1000", name="Cash")
        account2 = Account(code="2000", name="Payable")
        
        return {
            account1: Balance(date=period.since, value=Quantity(Decimal("1000.00"))),
            account2: Balance(date=period.since, value=Quantity(Decimal("500.00"))),
        }
    
    # Create a date range for testing
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    # Call the mock function
    result = mock_read_initial_balances(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure of returned balances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.date == period.since
        assert isinstance(balance.value, Quantity)
    
    # Verify specific values
    accounts_list = list(result.keys())
    assert result[accounts_list[0]].value == Quantity(Decimal("1000.00"))
    assert result[accounts_list[1]].value == Quantity(Decimal("500.00"))


# LLM-generated content at query #24
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    test_account = Account(code="1000", name="Test Account", account_type="Asset")
    test_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000")))
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        assert period == test_period
        return {test_account: test_balance}
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        assert period == test_period
        return []
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries,
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert test_account in result.ledgers
    assert result.ledgers[test_account].account == test_account
    assert result.ledgers[test_account].initial == test_balance
    assert len(result.ledgers[test_account].entries) == 0


def test_GeneralLedgerProgram___call___with_journal_entries():
    """Test the __call__ method of GeneralLedgerProgram with journal entries."""
    
    # Setup test data
    test_period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account(code="1000", name="Test Account 1", account_type="Asset")
    account2 = Account(code="2000", name="Test Account 2", account_type="Liability")
    test_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000")))
    
    # Create journal entry with postings
    from ..commons.numbers import Direction
    posting1 = Posting(account=account1, amount=Amount(Decimal("100")), direction=Direction.DEBIT, journal=None)
    posting2 = Posting(account=account2, amount=Amount(Decimal("100")), direction=Direction.CREDIT, journal=None)
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 6, 15),
        description="Test Journal Entry",
        postings=[posting1, posting2]
    )
    # Set back-reference
    posting1.journal = journal_entry
    posting2.journal = journal_entry
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return {account1: test_balance}
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return [journal_entry]
    
    # Compile and call the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries,
    )
    
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #25
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    from decimal import Decimal
    
    # Create test data
    test_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    period = DateRange(test_date, end_date)
    
    # Create mock accounts
    account1 = Account(number="1000", name="Cash")
    account2 = Account(number="2000", name="Payable")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(test_date, Quantity(Decimal("1000"))),
        account2: Balance(test_date, Quantity(Decimal("500"))),
    }
    
    # Create journal entries with postings
    posting1 = Posting(
        account=account1,
        amount=Amount(Decimal("100")),
        direction=Direction.DEBIT,
        journal=JournalEntry(
            date=date(2023, 6, 15),
            description="Test entry",
            postings=[]
        )
    )
    
    posting2 = Posting(
        account=account2,
        amount=Amount(Decimal("100")),
        direction=Direction.CREDIT,
        journal=JournalEntry(
            date=date(2023, 6, 15),
            description="Test entry",
            postings=[]
        )
    )
    
    journal_entry = JournalEntry(
        date=date(2023, 6, 15),
        description="Test entry",
        postings=[posting1, posting2]
    )
    
    # Create mock read functions
    def mock_read_initial_balances(p: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(p: DateRange) -> Iterable[JournalEntry]:
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Call the program with the period
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].account == account1
    assert result.ledgers[account2].account == account2
    assert result.ledgers[account1].initial == initial_balances[account1]
    assert result.ledgers[account2].initial == initial_balances[account2]
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1


# LLM-generated content at query #26
#--------------------------

```python
def test_ReadInitialBalances___call__():
    """Test ReadInitialBalances protocol __call__ method."""
    from datetime import date
    
    # Create a concrete implementation of ReadInitialBalances protocol
    def read_initial_balances_impl(period: DateRange) -> InitialBalances:
        """Concrete implementation of ReadInitialBalances."""
        account1 = Account(code="1000", name="Cash")
        account2 = Account(code="2000", name="Payable")
        
        return {
            account1: Balance(date=period.since, value=Quantity(Decimal("1000"))),
            account2: Balance(date=period.since, value=Quantity(Decimal("500"))),
        }
    
    # Create a DateRange for testing
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    period = DateRange(since=start_date, until=end_date)
    
    # Call the function through the protocol interface
    result = read_initial_balances_impl(period)
    
    # Assertions
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify the structure of returned balances
    for account, balance in result.items():
        assert isinstance(account, Account)
        assert isinstance(balance, Balance)
        assert balance.value > 0
    
    # Verify specific values
    accounts = list(result.keys())
    assert accounts[0].code == "1000"
    assert accounts[1].code == "2000"
    assert result[accounts[0]].value == Quantity(Decimal("1000"))
    assert result[accounts[1]].value == Quantity(Decimal("500"))


# LLM-generated content at query #27
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    # Setup test data
    test_account = Account(code="1000", name="Test Account")
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create mock initial balances
    initial_balances = {
        test_account: Balance(
            date=datetime.date(2022, 12, 31),
            value=Quantity(Decimal("1000"))
        )
    }
    
    # Create mock journal entries
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 6, 15),
            description="Test transaction",
            postings=[
                Posting(
                    account=test_account,
                    amount=Amount(Decimal("500")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        )
    ]
    
    # Create mock read functions
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        return journal_entries
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries
    )
    
    # Call the program
    result = program(test_period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert test_account in result.ledgers
    assert result.ledgers[test_account].account == test_account
    assert result.ledgers[test_account].initial == initial_balances[test_account]
    assert len(result.ledgers[test_account].entries) == 1


# LLM-generated content at query #28
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    from decimal import Decimal
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1000", "Cash", None)
        return {
            account1: Balance(period.since, Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account("1000", "Cash", None)
        account2 = Account("2000", "Accounts Payable", None)
        
        journal_entry = JournalEntry(
            date=period.since,
            description="Test entry",
            postings=[
                Posting(account1, Decimal("100"), Direction.DEBIT, journal_entry_ref=None),
                Posting(account2, Decimal("100"), Direction.CREDIT, journal_entry_ref=None),
            ]
        )
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        read_initial_balances=mock_read_initial_balances,
        read_journal_entries=mock_read_journal_entries,
    )
    
    # Create a test period
    period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    
    # Call the program
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) >= 1
    
    # Verify that the ledgers contain the expected accounts
    account1 = Account("1000", "Cash", None)
    assert account1 in result.ledgers or any(
        ledger.account.code == "1000" for ledger in result.ledgers.values()
    )


# LLM-generated content at query #29
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    """Test the __call__ method of GeneralLedgerProgram."""
    from datetime import date
    from decimal import Decimal
    
    # Create mock implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        account1 = Account("1000", "Cash")
        return {
            account1: Balance(period.since, Quantity(Decimal("1000")))
        }
    
    def mock_read_journal_entries(period: DateRange) -> Iterable[JournalEntry]:
        account1 = Account("1000", "Cash")
        account2 = Account("2000", "Accounts Payable")
        
        posting1 = Posting(account1, Decimal("500"), Direction.DEBIT)
        posting2 = Posting(account2, Decimal("500"), Direction.CREDIT)
        journal_entry = JournalEntry(
            date=period.since,
            description="Test transaction",
            postings=[posting1, posting2]
        )
        return [journal_entry]
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Test the program
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(period)
    
    # Assertions
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) >= 1
    
    # Verify the ledgers contain expected accounts
    account1 = Account("1000", "Cash")
    assert account1 in result.ledgers
    
    # Verify the ledger has entries
    ledger1 = result.ledgers[account1]
    assert len(ledger1.entries) > 0
    assert ledger1.initial.value == Quantity(Decimal("1000"))


# LLM-generated content at query #30
#--------------------------

```python
def test_build_general_ledger():
    """Test build_general_ledger function."""
    from datetime import date
    from decimal import Decimal
    
    # Setup test data
    account1 = Account("1000", "Cash", "asset")
    account2 = Account("2000", "Accounts Payable", "liability")
    account3 = Account("3000", "Revenue", "income")
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Create initial balances
    initial_balances = {
        account1: Balance(date(2023, 1, 1), Quantity(Decimal("1000"))),
        account2: Balance(date(2023, 1, 1), Quantity(Decimal("500"))),
    }
    
    # Create journal entries with postings
    posting1 = Posting(account1, Amount(Decimal("100")), Direction.DEBIT, JournalEntry("test"))
    posting2 = Posting(account2, Amount(Decimal("100")), Direction.CREDIT, JournalEntry("test"))
    
    journal_entry1 = JournalEntry(
        date=date(2023, 6, 15),
        description="Test entry 1",
        postings=[posting1, posting2]
    )
    
    posting3 = Posting(account3, Amount(Decimal("500")), Direction.CREDIT, JournalEntry("test2"))
    posting4 = Posting(account1, Amount(Decimal("500")), Direction.DEBIT, JournalEntry("test2"))
    
    journal_entry2 = JournalEntry(
        date=date(2023, 7, 20),
        description="Test entry 2",
        postings=[posting3, posting4]
    )
    
    journal_entries = [journal_entry1, journal_entry2]
    
    # Build general ledger
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    # Assertions
    assert result.period == period
    assert len(result.ledgers) == 3
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert account3 in result.ledgers
    
    # Check account1 ledger
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial.value == Quantity(Decimal("1000"))
    assert len(ledger1.entries) == 2
    
    # Check account2 ledger
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial.value == Quantity(Decimal("500"))
    assert len(ledger2.entries) == 1
    
    # Check account3 ledger
    ledger3 = result.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial.value == Quantity(Decimal("0"))
    assert len(ledger3.entries) == 1


def test_build_general_ledger_empty_journal():
    """Test build_general_ledger with empty journal."""
    from datetime import date
    from decimal import Decimal
    
    account1 = Account("1000", "Cash", "asset")
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    initial_balances = {
        account1: Balance(date(2023, 1, 1), Quantity(Decimal("5000"))),
    }
    
    result = build_general_ledger(period, [], initial_balances)
    
    assert result.period == period
    assert len(result.ledgers) == 1
    assert result.ledgers[account1].initial.value == Quantity(Decimal("5000"))
    assert len(result.ledgers[account1].entries) == 0


def test_build_general_ledger_outside_period():
    """Test build_general_ledger filters entries outside period."""
    from datetime import date
    from decimal import Decimal
    
    account1 = Account("1000", "Cash", "asset")
    account2 = Account("2000", "Accounts Payable", "liability")
    
    period = DateRange(date(2023, 6, 1), date(2023, 6, 30))
    
    initial_balances = {
        account1: Balance(date(2023, 1, 1), Quantity(Decimal("1000"))),
    }
    
    # Entry outside period (before)
    posting1 = Posting(account1, Amount(Decimal("100")), Direction.DEBIT, JournalEntry("test"))
    posting2 = Posting(account2, Amount(Decimal("100")), Direction.CREDIT, JournalEntry("test"))
    
    journal_entry_before = JournalEntry(
        date=date(2023, 5, 15),
        description="Before period",
        postings=[posting1, posting2]
    )
    
    # Entry inside period
    posting3 = Posting(account1, Amount(Decimal("200")), Direction.DEBIT, JournalEntry("test2"))
    posting4 = Posting(account2, Amount(Decimal("200")), Direction.CREDIT, JournalEntry("test2"))
    
    journal_entry_inside = JournalEntry(
        date=date(2023, 6, 15),
        description="Inside period",
        postings=[posting3, posting4]
    )
    
    # Entry outside period (after)
    posting5 = Posting(account1, Amount(Decimal("300")), Direction.DEBIT, JournalEntry("test3"))
    posting6 = Posting(account2, Amount(Decimal("300")), Direction.CREDIT, JournalEntry("test3"))
    
    journal_entry_after = JournalEntry(
        date=date(2023, 7, 15),
        description="After period",
        postings=[posting5, posting6]
    )
    
    journal_entries = [journal_entry_before, journal_entry_inside, journal_entry_after]
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert len(result.ledgers[account1].entries) == 1
    assert result.ledgers[account1].entries[0].date == date(2023, 6, 15)


