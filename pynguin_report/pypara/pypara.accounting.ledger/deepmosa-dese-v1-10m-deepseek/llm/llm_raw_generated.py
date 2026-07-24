####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_call_returns_general_ledger_for_given_period():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    mock_ledger = GeneralLedger()
    program = MagicMock(spec=GeneralLedgerProgram)
    program.return_value = mock_ledger
    result = program(mock_period)
    assert result is mock_ledger
    program.assert_called_once_with(mock_period)


# LLM-generated content at query #2
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Cash"), currency=Currency(code="USD"))
    mock_posting = Posting(account=Account(name="Cash"), amount=Amount(quantity=Quantity(number=Decimal("100"), unit=Unit(code="USD")), date=datetime.date(2023, 1, 1)), direction=Direction.DEBIT, journal=Journal(description="Test", date=datetime.date(2023, 1, 1), postings=[]))
    balance = Quantity(number=Decimal("100"), unit=Unit(code="USD"))
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == balance


# LLM-generated content at query #3
#--------------------------

def test_ledger_constructor_with_initial_balance():
    mock_account = Account()
    mock_initial_balance = Balance(Quantity(100.0))
    ledger = Ledger(account=mock_account, initial=mock_initial_balance)
    assert ledger.account is mock_account
    assert ledger.initial is mock_initial_balance
    assert ledger.entries == []

def test_ledger_constructor_entries_is_empty_list_by_default():
    mock_account = Account()
    mock_initial = Balance(Quantity(0.0))
    ledger = Ledger(account=mock_account, initial=mock_initial)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0


# LLM-generated content at query #4
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Test Account"))
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(value=100, currency="USD"), direction=Direction.DEBIT, journal=Journal(description="Test Journal", postings=[]))
    mock_balance = Quantity(value=100, currency="USD")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #5
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #6
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #7
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #8
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #9
#--------------------------

def test_read_initial_balances_returns_initial_balances():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    mock_balances = InitialBalances(balances={"account1": 1000.0, "account2": 2000.0})
    reader = ReadInitialBalances(lambda period: mock_balances)
    result = reader(mock_period)
    assert result == mock_balances

def test_read_initial_balances_called_with_correct_period():
    captured_period = None
    def mock_call(period):
        nonlocal captured_period
        captured_period = period
        return InitialBalances(balances={})
    reader = ReadInitialBalances(mock_call)
    expected_period = DateRange(start=date(2023, 5, 1), end=date(2023, 5, 31))
    reader(expected_period)
    assert captured_period == expected_period

def test_read_initial_balances_returns_empty_balances():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    reader = ReadInitialBalances(lambda period: InitialBalances(balances={}))
    result = reader(mock_period)
    assert result.balances == {}

def test_read_initial_balances_handles_single_account():
    mock_period = DateRange(start=date(2023, 6, 1), end=date(2023, 6, 30))
    expected_balances = InitialBalances(balances={"cash": 500.0})
    reader = ReadInitialBalances(lambda period: expected_balances)
    result = reader(mock_period)
    assert result.balances == {"cash": 500.0}

def test_read_initial_balances_handles_multiple_period_calls():
    call_count = 0
    def mock_call(period):
        nonlocal call_count
        call_count += 1
        return InitialBalances(balances={"count": float(call_count)})
    reader = ReadInitialBalances(mock_call)
    period1 = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    period2 = DateRange(start=2023, 2, 1, end=date(2023, 2, 28))
    result1 = reader(period1)
    result2 = reader(period2)
    assert result1.balances["count"] == 1.0
    assert result2.balances["count"] == 2.0
    assert call_count == 2


# LLM-generated content at query #10
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #11
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100.00"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100.00")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #12
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #13
#--------------------------

def test_build_general_ledger_with_no_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    account = Account("1000", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000")))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_build_general_ledger_with_journal_entry_inside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    account = Account("1000", "Cash")
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("500")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial.date == period.since
    assert ledger.initial.value == Quantity(Decimal(0))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == account
    assert ledger_entry.balance == Quantity(Decimal("500"))


def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test entry", source)
    account = Account("1000", "Cash")
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal("500")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.ledgers == {}


def test_build_general_ledger_with_multiple_postings_to_same_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry1 = JournalEntry(datetime.date(2023, 6, 15), "Test entry 1", source)
    journal_entry2 = JournalEntry(datetime.date(2023, 6, 20), "Test entry 2", source)
    account = Account("1000", "Cash")
    journal_entry1.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("500")))
    journal_entry2.post(datetime.date(2023, 6, 20), account, Quantity(Decimal("-200")))
    journal = [journal_entry1, journal_entry2]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 2
    assert ledger.entries[0].balance == Quantity(Decimal("500"))
    assert ledger.entries[1].balance == Quantity(Decimal("300"))


def test_build_general_ledger_with_initial_balance_and_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    account = Account("1000", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000")))
    initial = {account: initial_balance}
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("500")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    ledger = result.ledgers[account]
    assert ledger.initial == initial_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal("1500"))


def test_build_general_ledger_with_postings_to_different_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    journal_entry.post(datetime.date(2023, 6, 15), account1, Quantity(Decimal("500")))
    journal_entry.post(datetime.date(2023, 6, 15), account2, Quantity(Decimal("-500")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    ledger1 = result.ledgers[account1]
    ledger2 = result.ledgers[account2]
    assert ledger1.entries[0].balance == Quantity(Decimal("500"))
    assert ledger2.entries[0].balance == Quantity(Decimal("-500"))


# LLM-generated content at query #14
#--------------------------

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

def test___call___returns_general_ledger_for_given_period():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    mock_ledger = GeneralLedger()
    program_instance = GeneralLedgerProgram()
    program_instance.__call__ = Mock(return_value=mock_ledger)
    result = program_instance.__call__(mock_period)
    assert result is mock_ledger

def test___call___invoked_with_correct_date_range():
    mock_period = DateRange(start=date(2023, 5, 1), end=date(2023, 5, 31))
    program_instance = GeneralLedgerProgram()
    program_instance.__call__ = Mock()
    program_instance.__call__(mock_period)
    program_instance.__call__.assert_called_once_with(mock_period)

def test___call___returns_general_ledger_with_correct_type_parameter():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    mock_ledger = GeneralLedger[int]()
    program_instance = GeneralLedgerProgram[int]()
    program_instance.__call__ = Mock(return_value=mock_ledger)
    result = program_instance.__call__(mock_period)
    assert isinstance(result, GeneralLedger)
    assert result is mock_ledger


# LLM-generated content at query #16
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), journal=Journal(description="Test", postings=[]), amount=Amount(100), direction=Direction.DEBIT)
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #17
#--------------------------

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

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #19
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), journal=Journal(description="Test", postings=[]), amount=Amount(100), direction=Direction.DEBIT)
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #20
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100.00"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100.00")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #21
#--------------------------

def test___call___returns_initial_balances_for_given_period():
    mock_initial_balances = {"account1": 100.0, "account2": 200.0}
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    mock_reader = Mock(spec=ReadInitialBalances)
    mock_reader.return_value = mock_initial_balances
    result = mock_reader(mock_period)
    assert result == mock_initial_balances

def test___call___receives_correct_period_argument():
    mock_initial_balances = {"account1": 50.0}
    mock_period = DateRange(start=date(2023, 5, 1), end=date(2023, 5, 15))
    mock_reader = Mock(spec=ReadInitialBalances)
    mock_reader.return_value = mock_initial_balances
    mock_reader(mock_period)
    mock_reader.assert_called_once_with(mock_period)

def test___call___returns_empty_initial_balances():
    mock_initial_balances = {}
    mock_period = DateRange(start=date(2023, 10, 1), end=date(2023, 10, 10))
    mock_reader = Mock(spec=ReadInitialBalances)
    mock_reader.return_value = mock_initial_balances
    result = mock_reader(mock_period)
    assert result == {}

def test___call___handles_single_day_period():
    mock_initial_balances = {"accountA": 300.0}
    mock_period = DateRange(start=date(2023, 12, 25), end=date(2023, 12, 25))
    mock_reader = Mock(spec=ReadInitialBalances)
    mock_reader.return_value = mock_initial_balances
    result = mock_reader(mock_period)
    assert result == mock_initial_balances


# LLM-generated content at query #22
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #23
#--------------------------

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

def test_build_general_ledger_with_no_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    account = Account("1000", "Cash")
    initial_balance = Balance(period.since, Quantity(Decimal(1000)))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_build_general_ledger_with_journal_entry_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    account = Account("1000", "Cash")
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(500)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == account
    assert ledger_entry.balance == Quantity(Decimal(500))


def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2022, 12, 31), "Test", source)
    account = Account("1000", "Cash")
    entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(500)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_multiple_postings_to_same_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry1 = JournalEntry(datetime.date(2023, 6, 15), "Test1", source)
    entry2 = JournalEntry(datetime.date(2023, 6, 20), "Test2", source)
    account = Account("1000", "Cash")
    entry1.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(500)))
    entry2.post(datetime.date(2023, 6, 20), account, Quantity(Decimal(-200)))
    journal = [entry1, entry2]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 2
    assert ledger.entries[0].balance == Quantity(Decimal(500))
    assert ledger.entries[1].balance == Quantity(Decimal(300))


def test_build_general_ledger_with_postings_to_different_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    entry.post(datetime.date(2023, 6, 15), account1, Quantity(Decimal(500)))
    entry.post(datetime.date(2023, 6, 15), account2, Quantity(Decimal(-500)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    ledger1 = result.ledgers[account1]
    ledger2 = result.ledgers[account2]
    assert ledger1.initial == Balance(period.since, Quantity(Decimal(0)))
    assert ledger2.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger1.entries) == 1
    assert len(ledger2.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal(500))
    assert ledger2.entries[0].balance == Quantity(Decimal(-500))


def test_build_general_ledger_with_initial_balance_and_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    account = Account("1000", "Cash")
    initial_balance = Balance(period.since, Quantity(Decimal(1000)))
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(500)))
    journal = [entry]
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal(1500))


# LLM-generated content at query #25
#--------------------------

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

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #27
#--------------------------

def test_build_general_ledger_creates_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    posting = Posting(None, datetime.date(2023, 6, 15), Account("TestAccount"), Direction.INC, Amount(Decimal("100")))
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test", None)
    journal_entry.postings.append(posting)
    journal.append(journal_entry)
    general_ledger = build_general_ledger(period, journal, initial)
    assert posting.account in general_ledger.ledgers


# LLM-generated content at query #28
#--------------------------

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
def test_build_general_ledger_creates_ledger_for_new_account():
    from datetime import date
    from decimal import Decimal
    from pypara.accounting.generic import Balance, Quantity
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.ledger import build_general_ledger, DateRange
    from pypara.accounting.accounts import Account, AccountType
    from pypara.commons.numbers import Amount as Amt
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    initial = {}
    account = Account("123", "Test Account", AccountType.ASSETS)
    journal_entry = JournalEntry(date(2023, 6, 15), "Test Entry", None)
    journal_entry.post(date(2023, 6, 15), account, Quantity(Decimal("100.00")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert account in result.ledgers
    assert result.ledgers[account].account == account
    assert result.ledgers[account].initial == Balance(period.since, Quantity(Decimal("0")))
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].posting.account == account
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("100.00"))


# LLM-generated content at query #30
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #31
#--------------------------

def test_read_initial_balances_returns_initial_balances():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"account1": 100.0, "account2": 200.0}
    reader = MockReadInitialBalances()
    period = ("2023-01-01", "2023-01-31")
    result = reader(period)
    assert isinstance(result, dict)
    assert result["account1"] == 100.0
    assert result["account2"] == 200.0

def test_read_initial_balances_called_with_correct_period():
    captured_period = None
    class MockReadInitialBalances:
        def __call__(self, period):
            nonlocal captured_period
            captured_period = period
            return {}
    reader = MockReadInitialBalances()
    period = ("2023-02-01", "2023-02-28")
    reader(period)
    assert captured_period == period

def test_read_initial_balances_returns_empty_dict():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {}
    reader = MockReadInitialBalances()
    period = ("2023-03-01", "2023-03-31")
    result = reader(period)
    assert result == {}

def test_read_initial_balances_handles_single_account():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"account99": 999.99}
    reader = MockReadInitialBalances()
    period = ("2023-04-01", "2023-04-30")
    result = reader(period)
    assert len(result) == 1
    assert "account99" in result
    assert result["account99"] == 999.99

def test_read_initial_balances_with_zero_balance():
    class MockReadInitialBalances:
        def __call__(self, period):
            return {"zero_account": 0.0}
    reader = MockReadInitialBalances()
    period = ("2023-05-01", "2023-05-31")
    result = reader(period)
    assert result["zero_account"] == 0.0


# LLM-generated content at query #32
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #33
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #34
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100.00"), direction=Direction.DEBIT, account=Account("Cash"), journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100.00")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #35
#--------------------------

def test_build_general_ledger_with_no_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    account = Account("1234", "Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2022, 12, 31), "Outdated", source)
    entry.post(datetime.date(2022, 12, 31), Account("1234", "Test Account"), Quantity(Decimal("50.00")))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_single_posting_and_no_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    account = Account("1234", "Test Account")
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("50.00")))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == account
    assert ledger_entry.posting.amount == Amount(Decimal("50.00"))
    assert ledger_entry.posting.direction == Direction.INC
    assert ledger_entry.balance == Quantity(Decimal("50.00"))


def test_build_general_ledger_with_multiple_postings_to_same_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry1 = JournalEntry(datetime.date(2023, 6, 15), "Test 1", source)
    entry2 = JournalEntry(datetime.date(2023, 7, 20), "Test 2", source)
    account = Account("1234", "Test Account")
    entry1.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("50.00")))
    entry2.post(datetime.date(2023, 7, 20), account, Quantity(Decimal("-30.00")))
    journal = [entry1, entry2]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 2
    assert ledger.entries[0].balance == Quantity(Decimal("50.00"))
    assert ledger.entries[1].balance == Quantity(Decimal("20.00"))


def test_build_general_ledger_with_initial_balance_and_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    account = Account("1234", "Test Account")
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("50.00")))
    journal = [entry]
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("100.00")))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal("150.00"))


def test_build_general_ledger_with_postings_to_different_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    account1 = Account("1234", "Test Account 1")
    account2 = Account("5678", "Test Account 2")
    entry.post(datetime.date(2023, 6, 15), account1, Quantity(Decimal("50.00")))
    entry.post(datetime.date(2023, 6, 15), account2, Quantity(Decimal("-50.00")))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    ledger1 = result.ledgers[account1]
    ledger2 = result.ledgers[account2]
    assert ledger1.initial == Balance(period.since, Quantity(Decimal(0)))
    assert ledger2.initial == Balance(period.since, Quantity(Decimal(0)))
    assert ledger1.entries[0].balance == Quantity(Decimal("50.00"))
    assert ledger2.entries[0].balance == Quantity(Decimal("-50.00"))


def test_build_general_ledger_with_zero_quantity_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", source)
    account = Account("1234", "Test Account")
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("0.00")))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


# LLM-generated content at query #36
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #37
#--------------------------

def test___call___returns_general_ledger_for_given_period():
    from typing import Protocol, TypeVar
    from datetime import date
    _T = TypeVar("_T")
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    class GeneralLedger:
        def __init__(self, period: DateRange):
            self.period = period
    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger(period)
    program = MockGeneralLedgerProgram()
    test_period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(test_period)
    assert isinstance(result, GeneralLedger)
    assert result.period is test_period

def test___call___handles_empty_date_range():
    from typing import Protocol, TypeVar
    from datetime import date
    _T = TypeVar("_T")
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    class GeneralLedger:
        def __init__(self, period: DateRange):
            self.period = period
    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger(period)
    program = MockGeneralLedgerProgram()
    test_period = DateRange(date(2023, 1, 1), date(2023, 1, 1))
    result = program(test_period)
    assert isinstance(result, GeneralLedger)
    assert result.period.start == date(2023, 1, 1)
    assert result.period.end == date(2023, 1, 1)

def test___call___returns_general_ledger_with_correct_period():
    from typing import Protocol, TypeVar
    from datetime import date
    _T = TypeVar("_T")
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    class GeneralLedger:
        def __init__(self, period: DateRange):
            self.period = period
    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger(period)
    program = MockGeneralLedgerProgram()
    start_date = date(2024, 5, 10)
    end_date = date(2024, 5, 20)
    test_period = DateRange(start_date, end_date)
    result = program(test_period)
    assert result.period.start == start_date
    assert result.period.end == end_date


# LLM-generated content at query #38
#--------------------------

def test_ledger_constructor_with_default_entries():
    mock_account = Account()
    mock_initial = Balance(Quantity(Decimal('100.00')))
    ledger = Ledger(account=mock_account, initial=mock_initial)
    assert ledger.account == mock_account
    assert ledger.initial == mock_initial
    assert ledger.entries == []

def test_ledger_constructor_entries_not_provided():
    mock_account = Account()
    mock_initial = Balance(Quantity(Decimal('0.00')))
    ledger = Ledger(account=mock_account, initial=mock_initial)
    assert isinstance(ledger.entries, list)
    assert len(ledger.entries) == 0

def test_ledger_constructor_initial_balance_preserved():
    mock_account = Account()
    initial_quantity = Quantity(Decimal('50.50'))
    mock_initial = Balance(initial_quantity)
    ledger = Ledger(account=mock_account, initial=mock_initial)
    assert ledger.initial.value == initial_quantity


# LLM-generated content at query #39
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #40
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Cash"), currency="USD")
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, account=Account(name="Cash"), journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #41
#--------------------------

def test_build_general_ledger_predicate_at_line_16():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import GeneralLedger, Ledger, LedgerEntry
    from pypara.accounting.accounts import Account
    from pypara.currencies import Currency
    from pypara.commons.numbers import Quantity, Decimal
    from pypara.commons.zeitgeist import DateRange
    import datetime
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1234", "Test Account")
    currency = Currency("USD", 2)
    quantity = Quantity(Decimal("100"), currency)
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test", None)
    posting = Posting(journal_entry, datetime.date(2023, 6, 15), account, Direction.INC, Amount(quantity))
    journal_entry.postings.append(posting)
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert isinstance(ledger, Ledger)
    assert ledger.account == account
    assert ledger.initial.date == period.since
    assert ledger.initial.value == Quantity(Decimal(0), currency)
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert isinstance(ledger_entry, LedgerEntry)
    assert ledger_entry.posting == posting
    assert ledger_entry.balance == quantity


# LLM-generated content at query #42
#--------------------------

def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances_only():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    account = Account("1000", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00")))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_build_general_ledger_with_single_posting_and_no_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 1, 15), "Test", source)
    account = Account("1000", "Cash")
    entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("500.00")))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == account
    assert ledger_entry.posting.amount == Amount(Decimal("500.00"))
    assert ledger_entry.balance == Quantity(Decimal("500.00"))


def test_build_general_ledger_with_multiple_postings_and_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry1 = JournalEntry(datetime.date(2023, 1, 10), "Sale", source)
    entry2 = JournalEntry(datetime.date(2023, 1, 20), "Expense", source)
    cash_account = Account("1000", "Cash")
    revenue_account = Account("4000", "Revenue")
    expense_account = Account("5000", "Expense")
    entry1.post(datetime.date(2023, 1, 10), cash_account, Quantity(Decimal("1000.00")))
    entry1.post(datetime.date(2023, 1, 10), revenue_account, Quantity(Decimal("-1000.00")))
    entry2.post(datetime.date(2023, 1, 20), expense_account, Quantity(Decimal("200.00")))
    entry2.post(datetime.date(2023, 1, 20), cash_account, Quantity(Decimal("-200.00")))
    journal = [entry1, entry2]
    initial_cash_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("500.00")))
    initial = {cash_account: initial_cash_balance}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 3
    cash_ledger = result.ledgers[cash_account]
    assert cash_ledger.initial == initial_cash_balance
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    revenue_ledger = result.ledgers[revenue_account]
    assert revenue_ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-1000.00"))
    expense_ledger = result.ledgers[expense_account]
    assert expense_ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))


def test_build_general_ledger_with_posting_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry_inside = JournalEntry(datetime.date(2023, 6, 1), "Inside", source)
    entry_outside_before = JournalEntry(datetime.date(2022, 12, 31), "Before", source)
    entry_outside_after = JournalEntry(datetime.date(2024, 1, 1), "After", source)
    account = Account("1000", "Cash")
    entry_inside.post(datetime.date(2023, 6, 1), account, Quantity(Decimal("300.00")))
    entry_outside_before.post(datetime.date(2022, 12, 31), account, Quantity(Decimal("100.00")))
    entry_outside_after.post(datetime.date(2024, 1, 1), account, Quantity(Decimal("200.00")))
    journal = [entry_inside, entry_outside_before, entry_outside_after]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.date == datetime.date(2023, 6, 1)


def test_build_general_ledger_with_zero_quantity_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 1, 15), "Zero", source)
    account = Account("1000", "Cash")
    entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("0.00")))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.ledgers == {}


def test_build_general_ledger_balance_calculation_with_decrement():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 1, 15), "Withdrawal", source)
    account = Account("1000", "Cash")
    entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal("-300.00")))
    journal = [entry]
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000.00")))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal("700.00"))


# LLM-generated content at query #43
#--------------------------

def test_build_general_ledger_with_no_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000")))
    journal = []
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


def test_build_general_ledger_with_journal_entry_inside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial = {}
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("500")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == account
    assert ledger_entry.balance == Quantity(Decimal("500"))


def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial = {}
    source = object()
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test entry", source)
    journal_entry.post(datetime.date(2022, 12, 31), account, Quantity(Decimal("500")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


def test_build_general_ledger_with_multiple_postings_to_same_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial = {}
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("500")))
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("-200")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 2
    assert ledger.entries[0].balance == Quantity(Decimal("500"))
    assert ledger.entries[1].balance == Quantity(Decimal("300"))


def test_build_general_ledger_with_postings_to_different_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    initial = {}
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), account1, Quantity(Decimal("500")))
    journal_entry.post(datetime.date(2023, 6, 15), account2, Quantity(Decimal("-500")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    ledger1 = result.ledgers[account1]
    ledger2 = result.ledgers[account2]
    assert len(ledger1.entries) == 1
    assert len(ledger2.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal("500"))
    assert ledger2.entries[0].balance == Quantity(Decimal("-500"))


def test_build_general_ledger_with_initial_balance_and_journal_entry():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal("1000")))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("500")))
    journal = [journal_entry]
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert len(result.ledgers) == 1
    ledger = result.ledgers[account]
    assert ledger.initial == initial_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].balance == Quantity(Decimal("1500"))


def test_build_general_ledger_with_zero_quantity_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial = {}
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal("0")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert result.period == period
    assert result.ledgers == {}


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Cash"), currency=Currency(code="USD"))
    mock_posting = Posting(account=Account(name="Cash"), amount=Amount(quantity=Quantity(value=100, unit=Unit(currency=Currency(code="USD"))), date=datetime.date(2023, 1, 1)), direction=Direction.DEBIT, journal=Journal(description="Test", date=datetime.date(2023, 1, 1), postings=[]))
    balance = Quantity(value=100, unit=Unit(currency=Currency(code="USD")))
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == balance


# LLM-generated content at query #2
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #3
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Cash"), currency=Currency(code="USD"))
    mock_posting = Posting(account=Account(name="Cash"), amount=Amount(value=100, currency=Currency(code="USD")), direction=Direction.DEBIT, date=datetime.date(2023, 1, 1), journal=Journal(description="Test", postings=[]))
    balance = Quantity(value=100, currency=Currency(code="USD"))
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == balance


# LLM-generated content at query #4
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100.00"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100.00")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #5
#--------------------------

def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    ledger = build_general_ledger(period, journal, initial)
    assert ledger.period == period
    assert len(ledger.ledgers) == 1
    assert ledger.ledgers[Account("Cash")].account == Account("Cash")
    assert ledger.ledgers[Account("Cash")].initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    assert ledger.ledgers[Account("Cash")].entries == []

def test_build_general_ledger_with_journal_entry_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", object())
    entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(500)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 1
    assert ledger.ledgers[Account("Cash")].account == Account("Cash")
    assert ledger.ledgers[Account("Cash")].initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))
    assert len(ledger.ledgers[Account("Cash")].entries) == 1
    assert ledger.ledgers[Account("Cash")].entries[0].posting.account == Account("Cash")
    assert ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(500))

def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    entry = JournalEntry(datetime.date(2022, 12, 31), "Test", object())
    entry.post(datetime.date(2022, 12, 31), Account("Cash"), Quantity(Decimal(500)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 0

def test_build_general_ledger_with_multiple_postings_to_same_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", object())
    entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(500)))
    entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(-200)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 1
    assert len(ledger.ledgers[Account("Cash")].entries) == 2
    assert ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(500))
    assert ledger.ledgers[Account("Cash")].entries[1].balance == Quantity(Decimal(300))

def test_build_general_ledger_with_initial_balance_and_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", object())
    entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(500)))
    journal = [entry]
    initial = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 1
    assert ledger.ledgers[Account("Cash")].initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    assert len(ledger.ledgers[Account("Cash")].entries) == 1
    assert ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(1500))

def test_build_general_ledger_with_postings_to_different_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", object())
    entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(500)))
    entry.post(datetime.date(2023, 6, 15), Account("Revenue"), Quantity(Decimal(-500)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 2
    assert Account("Cash") in ledger.ledgers
    assert Account("Revenue") in ledger.ledgers
    assert ledger.ledgers[Account("Cash")].entries[0].balance == Quantity(Decimal(500))
    assert ledger.ledgers[Account("Revenue")].entries[0].balance == Quantity(Decimal(-500))

def test_build_general_ledger_with_zero_quantity_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test", object())
    entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(0)))
    journal = [entry]
    initial = {}
    ledger = build_general_ledger(period, journal, initial)
    assert len(ledger.ledgers) == 0


# LLM-generated content at query #6
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Test Account"))
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(value=100, currency="USD"), direction=Direction.DEBIT, journal=Journal(description="Test Journal", postings=[]))
    mock_balance = Quantity(value=100, currency="USD")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #7
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = object()
    mock_posting = object()
    mock_balance = object()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #8
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #9
#--------------------------

def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert Account("Cash") in general_ledger.ledgers
    ledger = general_ledger.ledgers[Account("Cash")]
    assert ledger.account == Account("Cash")
    assert ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    assert ledger.entries == []

def test_build_general_ledger_with_journal_entry_within_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(500)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 1
    ledger = general_ledger.ledgers[Account("Cash")]
    assert ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == Account("Cash")
    assert ledger_entry.balance == Quantity(Decimal(500))

def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Test entry", source)
    journal_entry.post(datetime.date(2022, 12, 31), Account("Cash"), Quantity(Decimal(500)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 0

def test_build_general_ledger_with_multiple_postings_to_same_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(500)))
    journal_entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(-200)))
    journal = [journal_entry]
    initial = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    general_ledger = build_general_ledger(period, journal, initial)
    ledger = general_ledger.ledgers[Account("Cash")]
    assert len(ledger.entries) == 2
    assert ledger.entries[0].balance == Quantity(Decimal(600))
    assert ledger.entries[1].balance == Quantity(Decimal(400))

def test_build_general_ledger_with_postings_to_different_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(500)))
    journal_entry.post(datetime.date(2023, 6, 15), Account("Revenue"), Quantity(Decimal(-500)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 2
    cash_ledger = general_ledger.ledgers[Account("Cash")]
    revenue_ledger = general_ledger.ledgers[Account("Revenue")]
    assert cash_ledger.entries[0].balance == Quantity(Decimal(500))
    assert revenue_ledger.entries[0].balance == Quantity(Decimal(-500))

def test_build_general_ledger_with_initial_balance_and_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(300)))
    journal = [journal_entry]
    initial = {Account("Cash"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(200)))}
    general_ledger = build_general_ledger(period, journal, initial)
    ledger = general_ledger.ledgers[Account("Cash")]
    assert ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(200)))
    assert ledger.entries[0].balance == Quantity(Decimal(500))

def test_build_general_ledger_with_zero_quantity_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", source)
    journal_entry.post(datetime.date(2023, 6, 15), Account("Cash"), Quantity(Decimal(0)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 0


# LLM-generated content at query #10
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #11
#--------------------------

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

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #13
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Cash"), currency=USD)
    mock_posting = Posting(account=Account(name="Cash"), amount=Amount(100, USD), direction=Direction.DEBIT, date=datetime.date(2023, 1, 1), journal=Journal(description="Sale", postings=[]))
    mock_balance = Quantity(100, USD)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #14
#--------------------------

def test_ledger_constructor():
    mock_account = Account()
    mock_initial = Balance(Quantity(Decimal("100.00")))
    ledger = Ledger(account=mock_account, initial=mock_initial)
    assert ledger.account is mock_account
    assert ledger.initial is mock_initial
    assert ledger.entries == []


# LLM-generated content at query #15
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Cash"), currency="USD")
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == balance


# LLM-generated content at query #16
#--------------------------

def test_build_general_ledger_creates_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.ledgers == {}

def test_build_general_ledger_uses_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    initial = {account: initial_balance}
    journal = []
    result = build_general_ledger(period, journal, initial)
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []

def test_build_general_ledger_adds_posting_to_existing_ledger():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))
    initial = {account: initial_balance}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", None)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(200)))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == account
    assert ledger_entry.balance == Quantity(Decimal(1200))

def test_build_general_ledger_creates_ledger_for_new_account_from_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("2000", "Revenue")
    initial = {}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Test", None)
    journal_entry.post(datetime.date(2023, 1, 15), account, Quantity(Decimal(500)))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == account
    assert ledger_entry.balance == Quantity(Decimal(500))

def test_build_general_ledger_filters_journal_entries_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("1000", "Cash")
    initial = {}
    journal_entry_before = JournalEntry(datetime.date(2022, 12, 31), "Before", None)
    journal_entry_before.post(datetime.date(2022, 12, 31), account, Quantity(Decimal(100)))
    journal_entry_during = JournalEntry(datetime.date(2023, 6, 15), "During", None)
    journal_entry_during.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(200)))
    journal_entry_after = JournalEntry(datetime.date(2024, 1, 1), "After", None)
    journal_entry_after.post(datetime.date(2024, 1, 1), account, Quantity(Decimal(300)))
    journal = [journal_entry_before, journal_entry_during, journal_entry_after]
    result = build_general_ledger(period, journal, initial)
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.journal_entry.description == "During"
    assert ledger_entry.balance == Quantity(Decimal(200))

def test_build_general_ledger_handles_multiple_accounts_and_postings():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    cash_account = Account("1000", "Cash")
    revenue_account = Account("2000", "Revenue")
    initial = {cash_account: Balance(datetime.date(2023, 1, 1), Quantity(Decimal(1000)))}
    journal_entry = JournalEntry(datetime.date(2023, 1, 15), "Sale", None)
    journal_entry.post(datetime.date(2023, 1, 15), cash_account, Quantity(Decimal(500)))
    journal_entry.post(datetime.date(2023, 1, 15), revenue_account, Quantity(Decimal(-500)))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    cash_ledger = result.ledgers[cash_account]
    revenue_ledger = result.ledgers[revenue_account]
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Quantity(Decimal(1500))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal(-500))


# LLM-generated content at query #17
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #18
#--------------------------

def test_ledger_constructor():
    mock_account = Account()
    mock_initial = Balance(Quantity(100))
    ledger = Ledger(account=mock_account, initial=mock_initial)
    assert ledger.account == mock_account
    assert ledger.initial == mock_initial
    assert ledger.entries == []


# LLM-generated content at query #19
#--------------------------

def test_call_returns_general_ledger():
    from typing import Protocol, TypeVar
    from datetime import date
    from dataclasses import dataclass

    _T = TypeVar("_T")

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        period: DateRange
        entries: list

    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...

    class ConcreteProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(period=period, entries=[])

    program = ConcreteProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = program(period)
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.entries == []

def test_call_with_different_period():
    from typing import Protocol, TypeVar
    from datetime import date
    from dataclasses import dataclass

    _T = TypeVar("_T")

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        period: DateRange
        entries: list

    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...

    class ConcreteProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(period=period, entries=["entry1", "entry2"])

    program = ConcreteProgram()
    period = DateRange(start=date(2024, 6, 1), end=date(2024, 6, 30))
    result = program(period)
    assert result.period == period
    assert result.entries == ["entry1", "entry2"]

def test_call_returns_correct_type():
    from typing import Protocol, TypeVar
    from datetime import date
    from dataclasses import dataclass

    _T = TypeVar("_T")

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        period: DateRange
        entries: list

    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...

    class ConcreteProgram:
        def __call__(self, period: DateRange) -> GeneralLedger:
            return GeneralLedger(period=period, entries=[])

    program = ConcreteProgram()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = program(period)
    assert type(result) is GeneralLedger


# LLM-generated content at query #20
#--------------------------

def test_read_initial_balances_returns_initial_balances():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    expected_balances = InitialBalances(balances={"account1": 100.0, "account2": 200.0})
    mock_reader = MagicMock(spec=ReadInitialBalances)
    mock_reader.return_value = expected_balances
    result = mock_reader(mock_period)
    assert result == expected_balances
    mock_reader.assert_called_once_with(mock_period)

def test_read_initial_balances_called_with_correct_period():
    mock_period = DateRange(start=date(2024, 5, 1), end=date(2024, 5, 31))
    mock_reader = MagicMock(spec=ReadInitialBalances)
    mock_reader.return_value = InitialBalances(balances={})
    _ = mock_reader(mock_period)
    mock_reader.assert_called_once_with(mock_period)

def test_read_initial_balances_returns_empty_initial_balances():
    mock_period = DateRange(start=date(2023, 6, 1), end=date(2023, 6, 30))
    expected_balances = InitialBalances(balances={})
    mock_reader = MagicMock(spec=ReadInitialBalances)
    mock_reader.return_value = expected_balances
    result = mock_reader(mock_period)
    assert result == expected_balances
    assert result.balances == {}

def test_read_initial_balances_handles_single_account():
    mock_period = DateRange(start=date(2023, 3, 1), end=date(2023, 3, 31))
    expected_balances = InitialBalances(balances={"savings": 5000.0})
    mock_reader = MagicMock(spec=ReadInitialBalances)
    mock_reader.return_value = expected_balances
    result = mock_reader(mock_period)
    assert result == expected_balances
    assert "savings" in result.balances
    assert result.balances["savings"] == 5000.0


# LLM-generated content at query #21
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Test Account"))
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(value=100, currency="USD"), direction=Direction.DEBIT, journal=Journal(description="Test Journal", postings=[]))
    mock_balance = Quantity(value=100, currency="USD")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #22
#--------------------------

def test_build_general_ledger_with_no_initial_balances_and_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 0

def test_build_general_ledger_with_initial_balances_but_no_journal_entries():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    account = Account("1234", "Test Account")
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2022, 12, 31), "Outdated Entry", source)
    entry.post(datetime.date(2022, 12, 31), Account("1234", "Test Account"), Quantity(Decimal(50)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 0

def test_build_general_ledger_with_journal_entry_inside_period_and_no_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", source)
    account = Account("1234", "Test Account")
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(50)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    ledger = result.ledgers[account]
    assert ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.account == account

def test_build_general_ledger_with_journal_entry_inside_period_and_existing_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 6, 15), "Test Entry", source)
    account = Account("1234", "Test Account")
    entry.post(datetime.date(2023, 6, 15), account, Quantity(Decimal(50)))
    journal = [entry]
    initial_balance = Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    initial = {account: initial_balance}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    ledger = result.ledgers[account]
    assert ledger.initial == initial_balance
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting.account == account

def test_build_general_ledger_with_multiple_journal_entries_and_accounts():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry1 = JournalEntry(datetime.date(2023, 6, 15), "Entry 1", source)
    account1 = Account("1234", "Account 1")
    entry1.post(datetime.date(2023, 6, 15), account1, Quantity(Decimal(50)))
    entry2 = JournalEntry(datetime.date(2023, 7, 20), "Entry 2", source)
    account2 = Account("5678", "Account 2")
    entry2.post(datetime.date(2023, 7, 20), account2, Quantity(Decimal(30)))
    journal = [entry1, entry2]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    ledger1 = result.ledgers[account1]
    ledger2 = result.ledgers[account2]
    assert ledger1.initial == Balance(period.since, Quantity(Decimal(0)))
    assert ledger2.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(ledger1.entries) == 1
    assert len(ledger2.entries) == 1

def test_build_general_ledger_with_journal_entry_on_period_boundary_since():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 1, 1), "Boundary Entry", source)
    account = Account("1234", "Test Account")
    entry.post(datetime.date(2023, 1, 1), account, Quantity(Decimal(50)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 1

def test_build_general_ledger_with_journal_entry_on_period_boundary_until():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    entry = JournalEntry(datetime.date(2023, 12, 31), "Boundary Entry", source)
    account = Account("1234", "Test Account")
    entry.post(datetime.date(2023, 12, 31), account, Quantity(Decimal(50)))
    journal = [entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert len(result.ledgers) == 1
    ledger = result.ledgers[account]
    assert len(ledger.entries) == 1


# LLM-generated content at query #23
#--------------------------

def test___call___returns_general_ledger_for_given_period():
    from typing import Protocol, TypeVar
    from datetime import date
    from dataclasses import dataclass

    _T = TypeVar("_T")

    @dataclass(frozen=True)
    class DateRange:
        start: date
        end: date

    @dataclass
    class GeneralLedger:
        period: DateRange
        entries: list

    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...

    class MockGeneralLedgerProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger(period=period, entries=[])

    program = MockGeneralLedgerProgram()
    test_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = program(test_period)
    assert isinstance(result, GeneralLedger)
    assert result.period == test_period
    assert result.entries == []


# LLM-generated content at query #24
#--------------------------

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

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100.00"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100.00")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #26
#--------------------------

def test_ledger_constructor():
    mock_account = Account()
    mock_initial = Balance(Quantity(Decimal("100.00")))
    ledger = Ledger(account=mock_account, initial=mock_initial)
    assert ledger.account == mock_account
    assert ledger.initial == mock_initial
    assert ledger.entries == []


# LLM-generated content at query #27
#--------------------------

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

def test_build_general_ledger_creates_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(date=datetime.date(2023, 6, 15), description="Test", source=None)
    journal_entry.post(date=datetime.date(2023, 6, 15), account=Account("TestAccount"), quantity=Quantity(Decimal("100")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert Account("TestAccount") in result.ledgers

def test_build_general_ledger_uses_existing_ledger_from_initial():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("ExistingAccount")
    initial_balance = Balance(date=period.since, value=Quantity(Decimal("50")))
    initial = {account: initial_balance}
    journal_entry = JournalEntry(date=datetime.date(2023, 6, 15), description="Test", source=None)
    journal_entry.post(date=datetime.date(2023, 6, 15), account=account, quantity=Quantity(Decimal("100")))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert account in result.ledgers
    assert result.ledgers[account].initial == initial_balance

def test_build_general_ledger_filters_postings_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry_inside = JournalEntry(date=datetime.date(2023, 6, 15), description="Inside", source=None)
    journal_entry_inside.post(date=datetime.date(2023, 6, 15), account=Account("Account1"), quantity=Quantity(Decimal("100")))
    journal_entry_before = JournalEntry(date=datetime.date(2022, 12, 31), description="Before", source=None)
    journal_entry_before.post(date=datetime.date(2022, 12, 31), account=Account("Account2"), quantity=Quantity(Decimal("200")))
    journal_entry_after = JournalEntry(date=datetime.date(2024, 1, 1), description="After", source=None)
    journal_entry_after.post(date=datetime.date(2024, 1, 1), account=Account("Account3"), quantity=Quantity(Decimal("300")))
    journal = [journal_entry_inside, journal_entry_before, journal_entry_after]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert Account("Account1") in result.ledgers
    assert Account("Account2") not in result.ledgers
    assert Account("Account3") not in result.ledgers

def test_build_general_ledger_handles_zero_quantity_posting():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal_entry = JournalEntry(date=datetime.date(2023, 6, 15), description="Zero", source=None)
    journal_entry.post(date=datetime.date(2023, 6, 15), account=Account("ZeroAccount"), quantity=Quantity(Decimal("0")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert Account("ZeroAccount") not in result.ledgers

def test_build_general_ledger_creates_ledger_with_zero_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    account = Account("NewAccount")
    journal_entry = JournalEntry(date=datetime.date(2023, 6, 15), description="Test", source=None)
    journal_entry.post(date=datetime.date(2023, 6, 15), account=account, quantity=Quantity(Decimal("150")))
    journal = [journal_entry]
    initial = {}
    result = build_general_ledger(period, journal, initial)
    assert account in result.ledgers
    assert result.ledgers[account].initial == Balance(period.since, Quantity(Decimal(0)))


# LLM-generated content at query #29
#--------------------------

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

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #31
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


# LLM-generated content at query #32
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Cash"))
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(value=100, currency="USD"), direction=Direction.DEBIT, journal=Journal(description="Sale", postings=[]))
    mock_balance = Quantity(value=100, currency="USD")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #33
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(100), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity(100)
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #34
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger(account=Account(name="Test Account"))
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount(value=100, currency="USD"), direction=Direction.DEBIT, journal=Journal(description="Test Journal", postings=[]))
    mock_balance = Quantity(value=100, currency="USD")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #35
#--------------------------

def test_ledger_constructor():
    account = Account("Test Account")
    initial_balance = Balance(Decimal("100.00"), Currency.USD)
    ledger = Ledger(account, initial_balance)
    assert ledger.account == account
    assert ledger.initial == initial_balance
    assert ledger.entries == []


# LLM-generated content at query #36
#--------------------------

def test_call_returns_general_ledger_for_given_period():
    from typing import Protocol, TypeVar
    from datetime import date
    _T = TypeVar('_T')
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    class GeneralLedger:
        def __init__(self, period: DateRange):
            self.period = period
    class GeneralLedgerProgram(Protocol[_T]):
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            ...
    class ConcreteProgram:
        def __call__(self, period: DateRange) -> GeneralLedger[_T]:
            return GeneralLedger(period)
    program = ConcreteProgram()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = program(period)
    assert isinstance(result, GeneralLedger)
    assert result.period is period


# LLM-generated content at query #37
#--------------------------

def test_build_general_ledger_creates_ledger_for_new_account():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    initial = {}
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Test entry", None)
    journal_entry.post(datetime.date(2023, 6, 15), Account("Assets", "Cash"), Quantity(Decimal(100)))
    journal = [journal_entry]
    result = build_general_ledger(period, journal, initial)
    assert journal_entry.postings[0].account in result.ledgers
    ledger = result.ledgers[journal_entry.postings[0].account]
    assert ledger.account == journal_entry.postings[0].account
    assert ledger.initial.date == period.since
    assert ledger.initial.value == Quantity(Decimal(0))
    assert len(ledger.entries) == 1
    assert ledger.entries[0].posting == journal_entry.postings[0]
    assert ledger.entries[0].balance == Quantity(Decimal(100))


# LLM-generated content at query #38
#--------------------------

def test_readinitialbalances_call():
    mock_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    expected_balances = InitialBalances(data={"account1": 1000.0, "account2": 2000.0})
    reader = ReadInitialBalances(lambda period: expected_balances)
    result = reader(mock_period)
    assert result == expected_balances


# LLM-generated content at query #39
#--------------------------

def test___call___returns_general_ledger_for_given_period():
    mock_ledger = GeneralLedger(period=DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31)), initial=None, entries=[])
    mock_program = MagicMock(spec=GeneralLedgerProgram)
    mock_program.return_value = mock_ledger
    test_period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = mock_program(test_period)
    assert result == mock_ledger
    mock_program.assert_called_once_with(test_period)


# LLM-generated content at query #40
#--------------------------

def test_read_initial_balances_returns_initial_balances():
    from typing import Protocol
    from datetime import date
    from dataclasses import dataclass

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class InitialBalances:
        balances: dict

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balances={"account1": 100.0, "account2": 200.0})

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = reader(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {"account1": 100.0, "account2": 200.0}

def test_read_initial_balances_with_empty_balances():
    from typing import Protocol
    from datetime import date
    from dataclasses import dataclass

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class InitialBalances:
        balances: dict

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return InitialBalances(balances={})

    reader = MockReadInitialBalances()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = reader(period)
    assert isinstance(result, InitialBalances)
    assert result.balances == {}

def test_read_initial_balances_uses_period_parameter():
    from typing import Protocol
    from datetime import date
    from dataclasses import dataclass

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class InitialBalances:
        balances: dict

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            if period.start == date(2023, 1, 1) and period.end == date(2023, 12, 31):
                return InitialBalances(balances={"account1": 100.0})
            else:
                return InitialBalances(balances={"account2": 200.0})

    reader = MockReadInitialBalances()
    period1 = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result1 = reader(period1)
    assert result1.balances == {"account1": 100.0}
    period2 = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
    result2 = reader(period2)
    assert result2.balances == {"account2": 200.0}


# LLM-generated content at query #41
#--------------------------

def test_build_general_ledger_initial_balances_not_in_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {Account("A1"): Balance(datetime.date(2022, 12, 31), Quantity(Decimal(100)))}
    general_ledger = build_general_ledger(period, journal, initial)
    ledger = general_ledger.ledgers[Account("A1")]
    assert ledger.initial.date == datetime.date(2022, 12, 31)
    assert ledger.initial.value == Quantity(Decimal(100))
    assert len(ledger.entries) == 0


# LLM-generated content at query #42
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting(date=datetime.date(2023, 1, 1), amount=Amount("100.00"), direction=Direction.DEBIT, journal=Journal(description="Test", postings=[]))
    mock_balance = Quantity("100.00")
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger == mock_ledger
    assert entry.posting == mock_posting
    assert entry.balance == mock_balance


# LLM-generated content at query #43
#--------------------------

def test_build_general_ledger_predicate_true_for_entries_in_period():
    from datetime import date
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.generic import Balance
    from pypara.accounting.ledger import build_general_ledger, DateRange, Quantity, Account, Decimal
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    source = object()
    entry = JournalEntry(date(2023, 6, 15), "Test Entry", source)
    account = Account("1234", "Test Account")
    quantity = Quantity(Decimal("100.00"))
    entry.post(date(2023, 6, 15), account, quantity)
    journal = [entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert account in general_ledger.ledgers
    assert len(general_ledger.ledgers[account].entries) == 1


# LLM-generated content at query #44
#--------------------------

def test_build_general_ledger_with_empty_journal_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    journal = []
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    general_ledger = build_general_ledger(period, journal, initial)
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    assert Account("A1") in general_ledger.ledgers
    ledger = general_ledger.ledgers[Account("A1")]
    assert ledger.account == Account("A1")
    assert ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    assert ledger.entries == []

def test_build_general_ledger_with_journal_entry_outside_period():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2022, 12, 31), "Outdated", source)
    journal_entry.post(datetime.date(2022, 12, 31), Account("A1"), Quantity(Decimal(50)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 0

def test_build_general_ledger_with_journal_entry_inside_period_and_no_initial_balance():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 6, 15), "Transaction", source)
    journal_entry.post(datetime.date(2023, 6, 15), Account("A2"), Quantity(Decimal(200)))
    journal = [journal_entry]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 1
    assert Account("A2") in general_ledger.ledgers
    ledger = general_ledger.ledgers[Account("A2")]
    assert ledger.account == Account("A2")
    assert ledger.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))
    assert len(ledger.entries) == 1
    ledger_entry = ledger.entries[0]
    assert ledger_entry.posting.account == Account("A2")
    assert ledger_entry.balance == Quantity(Decimal(200))

def test_build_general_ledger_with_multiple_postings_and_initial_balances():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 7, 1), "Complex", source)
    journal_entry.post(datetime.date(2023, 7, 1), Account("A1"), Quantity(Decimal(150)))
    journal_entry.post(datetime.date(2023, 7, 1), Account("A2"), Quantity(Decimal(-150)))
    journal = [journal_entry]
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 2
    ledger_a1 = general_ledger.ledgers[Account("A1")]
    assert ledger_a1.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(100)))
    assert len(ledger_a1.entries) == 1
    assert ledger_a1.entries[0].balance == Quantity(Decimal(250))
    ledger_a2 = general_ledger.ledgers[Account("A2")]
    assert ledger_a2.initial == Balance(datetime.date(2023, 1, 1), Quantity(Decimal(0)))
    assert len(ledger_a2.entries) == 1
    assert ledger_a2.entries[0].balance == Quantity(Decimal(-150))

def test_build_general_ledger_with_journal_entry_on_period_boundary():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry_since = JournalEntry(period.since, "First day", source)
    journal_entry_since.post(period.since, Account("A1"), Quantity(Decimal(10)))
    journal_entry_until = JournalEntry(period.until, "Last day", source)
    journal_entry_until.post(period.until, Account("A2"), Quantity(Decimal(20)))
    journal = [journal_entry_since, journal_entry_until]
    initial = {}
    general_ledger = build_general_ledger(period, journal, initial)
    assert len(general_ledger.ledgers) == 2
    assert Account("A1") in general_ledger.ledgers
    assert Account("A2") in general_ledger.ledgers

def test_build_general_ledger_verifies_ledger_entry_balance_calculation():
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    source = object()
    journal_entry = JournalEntry(datetime.date(2023, 5, 10), "Test", source)
    journal_entry.post(datetime.date(2023, 5, 10), Account("A1"), Quantity(Decimal(30)))
    journal_entry.post(datetime.date(2023, 5, 10), Account("A1"), Quantity(Decimal(-10)))
    journal = [journal_entry]
    initial = {Account("A1"): Balance(datetime.date(2023, 1, 1), Quantity(Decimal(5)))}
    general_ledger = build_general_ledger(period, journal, initial)
    ledger = general_ledger.ledgers[Account("A1")]
    assert len(ledger.entries) == 2
    assert ledger.entries[0].balance == Quantity(Decimal(35))
    assert ledger.entries[1].balance == Quantity(Decimal(25))


# LLM-generated content at query #45
#--------------------------

def test_ledger_entry_constructor():
    mock_ledger = Ledger()
    mock_posting = Posting()
    mock_balance = Quantity()
    entry = LedgerEntry(ledger=mock_ledger, posting=mock_posting, balance=mock_balance)
    assert entry.ledger is mock_ledger
    assert entry.posting is mock_posting
    assert entry.balance is mock_balance


