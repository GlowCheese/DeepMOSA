####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    class MockAccount(Account):
        def __init__(self, name):
            self.name = name
        
        def __hash__(self):
            return hash(self.name)
        
        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    account1 = MockAccount("Cash")
    account2 = MockAccount("Revenue")
    account3 = MockAccount("Expense")
    
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    account=account1,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=account2,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=datetime.date(2023, 2, 20),
            description="Expense",
            postings=[
                Posting(
                    account=account3,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=account1,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=datetime.date(2022, 12, 30),
            description="Out of period",
            postings=[
                Posting(
                    account=account1,
                    amount=Amount(Decimal("999.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=datetime.date(2024, 1, 1),
            description="Future period",
            postings=[
                Posting(
                    account=account1,
                    amount=Amount(Decimal("888.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        )
    ]
    
    for entry in journal_entries:
        for posting in entry.postings:
            posting.journal = entry
    
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    cash_ledger = general_ledger.ledgers[account1]
    assert cash_ledger.account == account1
    assert cash_ledger.initial == initial_balances[account1]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    
    revenue_ledger = general_ledger.ledgers[account2]
    assert revenue_ledger.account == account2
    assert revenue_ledger.initial == Balance(period.since, Quantity(Decimal("0")))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    
    expense_ledger = general_ledger.ledgers[account3]
    assert expense_ledger.account == account3
    assert expense_ledger.initial == Balance(period.since, Quantity(Decimal("0")))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))
    
    empty_ledger = build_general_ledger(period, [], {})
    assert empty_ledger.period == period
    assert len(empty_ledger.ledgers) == 0


# LLM-generated content at query #2
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    class MockT:
        def __init__(self, value):
            self.value = value

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    account3 = Account("3000", "Expense")
    
    initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        account2: Balance(date(2022, 12, 31), Quantity(Decimal("0.00")))
    }
    
    journal_entries = [
        JournalEntry(
            date=date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    account=account1,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=account2,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 2, 20),
            description="Expense",
            postings=[
                Posting(
                    account=account3,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=account1,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=date(2022, 12, 30),
            description="Out of period",
            postings=[
                Posting(
                    account=account1,
                    amount=Amount(Decimal("999.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=date(2024, 1, 1),
            description="Future transaction",
            postings=[
                Posting(
                    account=account1,
                    amount=Amount(Decimal("999.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        )
    ]
    
    for entry in journal_entries:
        for posting in entry.postings:
            posting.journal = entry
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert account3 in result.ledgers
    
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2
    
    assert ledger1.entries[0].posting.account == account1
    assert ledger1.entries[0].posting.amount == Amount(Decimal("500.00"))
    assert ledger1.entries[0].posting.direction == Direction.DEBIT
    assert ledger1.entries[0].balance == Quantity(Decimal("1500.00"))
    
    assert ledger1.entries[1].posting.account == account1
    assert ledger1.entries[1].posting.amount == Amount(Decimal("200.00"))
    assert ledger1.entries[1].posting.direction == Direction.CREDIT
    assert ledger1.entries[1].balance == Quantity(Decimal("1300.00"))
    
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    
    assert ledger2.entries[0].posting.account == account2
    assert ledger2.entries[0].posting.amount == Amount(Decimal("500.00"))
    assert ledger2.entries[0].posting.direction == Direction.CREDIT
    assert ledger2.entries[0].balance == Quantity(Decimal("500.00"))
    
    ledger3 = result.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert len(ledger3.entries) == 1
    
    assert ledger3.entries[0].posting.account == account3
    assert ledger3.entries[0].posting.amount == Amount(Decimal("200.00"))
    assert ledger3.entries[0].posting.direction == Direction.DEBIT
    assert ledger3.entries[0].balance == Quantity(Decimal("200.00"))
    
    empty_period = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    empty_result = build_general_ledger(empty_period, journal_entries, initial_balances)
    
    assert empty_result.period == empty_period
    assert len(empty_result.ledgers) == 2
    assert account1 in empty_result.ledgers
    assert account2 in empty_result.ledgers
    assert account3 not in empty_result.ledgers
    
    for ledger in empty_result.ledgers.values():
        assert len(ledger.entries) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockAccount(Account):
        def __init__(self, name: str):
            self.name = name
        
        def __hash__(self):
            return hash(self.name)
        
        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            return {
                MockAccount("Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
                MockAccount("Revenue"): Balance(period.since, Quantity(Decimal("0.00")))
            }

    class MockReadJournalEntries:
        def __init__(self, entries: List[JournalEntry]):
            self.entries = entries
        
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            return [e for e in self.entries if period.since <= e.date <= period.until]

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    
    journal_entries = [
        JournalEntry(
            date=date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    account=MockAccount("Cash"),
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=MockAccount("Revenue"),
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        )
    ]
    
    for je in journal_entries:
        for posting in je.postings:
            posting.journal = je

    read_initial_balances = MockReadInitialBalances()
    read_journal_entries = MockReadJournalEntries(journal_entries)
    
    program = compile_general_ledger_program(read_initial_balances, read_journal_entries)
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2
    
    cash_ledger = result.ledgers[MockAccount("Cash")]
    assert cash_ledger.account.name == "Cash"
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Decimal("1500.00")
    assert cash_ledger.entries[0].is_debit == True
    
    revenue_ledger = result.ledgers[MockAccount("Revenue")]
    assert revenue_ledger.account.name == "Revenue"
    assert revenue_ledger.initial.value == Decimal("0.00")
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Decimal("-500.00")
    assert revenue_ledger.entries[0].is_credit == True

    empty_period = DateRange(date(2023, 2, 1), date(2023, 2, 28))
    result_empty = program(empty_period)
    
    assert isinstance(result_empty, GeneralLedger)
    assert result_empty.period == empty_period
    assert len(result_empty.ledgers) == 2
    
    cash_ledger_empty = result_empty.ledgers[MockAccount("Cash")]
    assert cash_ledger_empty.initial.value == Decimal("1000.00")
    assert len(cash_ledger_empty.entries) == 0
    
    revenue_ledger_empty = result_empty.ledgers[MockAccount("Revenue")]
    assert revenue_ledger_empty.initial.value == Decimal("0.00")
    assert len(revenue_ledger_empty.entries) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    # Test data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    
    # Create test accounts
    cash_account = Account("101", "Cash")
    revenue_account = Account("401", "Revenue")
    expense_account = Account("501", "Expense")
    
    # Test 1: Empty journal with initial balances
    initial_balances = {
        cash_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        revenue_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("0.00")))
    }
    
    empty_journal = []
    gl = build_general_ledger(period, empty_journal, initial_balances)
    
    assert gl.period == period
    assert len(gl.ledgers) == 2
    assert cash_account in gl.ledgers
    assert revenue_account in gl.ledgers
    assert gl.ledgers[cash_account].initial == initial_balances[cash_account]
    assert gl.ledgers[cash_account].entries == []
    assert gl.ledgers[revenue_account].initial == initial_balances[revenue_account]
    
    # Test 2: Journal entries within period
    journal_entry1 = JournalEntry(
        datetime.date(2023, 1, 15),
        "Sale",
        [
            Posting(cash_account, Direction.DEBIT, Amount(Decimal("500.00")), None),
            Posting(revenue_account, Direction.CREDIT, Amount(Decimal("500.00")), None)
        ]
    )
    
    journal_entry2 = JournalEntry(
        datetime.date(2023, 1, 20),
        "Expense",
        [
            Posting(expense_account, Direction.DEBIT, Amount(Decimal("200.00")), None),
            Posting(cash_account, Direction.CREDIT, Amount(Decimal("200.00")), None)
        ]
    )
    
    journal = [journal_entry1, journal_entry2]
    gl = build_general_ledger(period, journal, initial_balances)
    
    assert len(gl.ledgers) == 3  # cash, revenue, expense
    assert expense_account in gl.ledgers
    assert gl.ledgers[expense_account].initial == Balance(period.since, Quantity(Decimal("0.00")))
    
    # Verify cash ledger entries and balances
    cash_ledger = gl.ledgers[cash_account]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))  # 1000 + 500
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))  # 1500 - 200
    
    # Verify revenue ledger entries
    revenue_ledger = gl.ledgers[revenue_account]
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    
    # Verify expense ledger entries
    expense_ledger = gl.ledgers[expense_account]
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))
    
    # Test 3: Journal entry outside period should be ignored
    journal_entry_outside = JournalEntry(
        datetime.date(2022, 12, 31),
        "Old transaction",
        [
            Posting(cash_account, Direction.DEBIT, Amount(Decimal("100.00")), None),
            Posting(revenue_account, Direction.CREDIT, Amount(Decimal("100.00")), None)
        ]
    )
    
    journal_with_outside = [journal_entry1, journal_entry_outside]
    gl = build_general_ledger(period, journal_with_outside, initial_balances)
    
    cash_ledger = gl.ledgers[cash_account]
    assert len(cash_ledger.entries) == 1  # Only the within-period entry
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    
    # Test 4: Multiple postings to same account in single journal entry
    complex_journal_entry = JournalEntry(
        datetime.date(2023, 1, 25),
        "Complex transaction",
        [
            Posting(cash_account, Direction.DEBIT, Amount(Decimal("300.00")), None),
            Posting(cash_account, Direction.CREDIT, Amount(Decimal("100.00")), None),
            Posting(revenue_account, Direction.CREDIT, Amount(Decimal("200.00")), None)
        ]
    )
    
    journal_complex = [complex_journal_entry]
    gl = build_general_ledger(period, journal_complex, initial_balances)
    
    cash_ledger = gl.ledgers[cash_account]
    assert len(cash_ledger.entries) == 2  # Two separate ledger entries
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1300.00"))  # 1000 + 300
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1200.00"))  # 1300 - 100
    
    # Test 5: Empty initial balances
    empty_initial = {}
    gl = build_general_ledger(period, journal, empty_initial)
    
    assert len(gl.ledgers) == 3  # All accounts should be created with zero initial balance
    assert gl.ledgers[cash_account].initial == Balance(period.since, Quantity(Decimal("0.00")))
    
    # Test 6: Verify ledger entry properties
    cash_ledger = gl.ledgers[cash_account]
    entry = cash_ledger.entries[0]
    
    assert entry.ledger == cash_ledger
    assert entry.posting == journal_entry1.postings[0]
    assert entry.date == datetime.date(2023, 1, 15)
    assert entry.description == "Sale"
    assert entry.amount == Amount(Decimal("500.00"))
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == Amount(Decimal("500.00"))
    assert entry.credit == None
    assert len(entry.cntraccts) == 1
    assert entry.cntraccts[0] == revenue_account


# LLM-generated content at query #5
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from ..commons.numbers import Amount, Quantity
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockAccount(Account):
        def __init__(self, name: str):
            self.name = name

        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

        def __hash__(self):
            return hash(self.name)

    class MockT:
        pass

    def create_posting(account, amount, direction, posting_date):
        journal = JournalEntry(
            date=posting_date,
            description="Test",
            postings=[]
        )
        posting = Posting(
            journal=journal,
            account=account,
            amount=Amount(Decimal(amount)),
            direction=direction
        )
        journal.postings.append(posting)
        return posting

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))

    # Test 1: Empty journal with initial balances
    cash_acc = MockAccount("Cash")
    equity_acc = MockAccount("Equity")
    initial = {
        cash_acc: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        equity_acc: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    journal = []
    gl = build_general_ledger(period, journal, initial)
    assert gl.period == period
    assert len(gl.ledgers) == 2
    assert cash_acc in gl.ledgers
    assert equity_acc in gl.ledgers
    assert gl.ledgers[cash_acc].initial.value == Decimal("1000.00")
    assert len(gl.ledgers[cash_acc].entries) == 0
    assert gl.ledgers[equity_acc].initial.value == Decimal("1000.00")

    # Test 2: Journal entries within period
    cash_acc = MockAccount("Cash")
    revenue_acc = MockAccount("Revenue")
    initial = {cash_acc: Balance(date(2022, 12, 31), Quantity(Decimal("500.00")))}
    
    journal = []
    je1 = JournalEntry(
        date=date(2023, 1, 15),
        description="Sale",
        postings=[]
    )
    p1 = Posting(
        journal=je1,
        account=cash_acc,
        amount=Amount(Decimal("200.00")),
        direction=Direction.DEBIT
    )
    p2 = Posting(
        journal=je1,
        account=revenue_acc,
        amount=Amount(Decimal("200.00")),
        direction=Direction.CREDIT
    )
    je1.postings.extend([p1, p2])
    journal.append(je1)

    gl = build_general_ledger(period, journal, initial)
    assert len(gl.ledgers) == 2
    assert cash_acc in gl.ledgers
    assert revenue_acc in gl.ledgers
    assert gl.ledgers[cash_acc].initial.value == Decimal("500.00")
    assert len(gl.ledgers[cash_acc].entries) == 1
    assert gl.ledgers[cash_acc].entries[0].balance == Decimal("700.00")
    assert gl.ledgers[revenue_acc].initial.value == Decimal("0.00")
    assert len(gl.ledgers[revenue_acc].entries) == 1
    assert gl.ledgers[revenue_acc].entries[0].balance == Decimal("-200.00")

    # Test 3: Journal entries outside period should be ignored
    cash_acc = MockAccount("Cash")
    initial = {cash_acc: Balance(date(2022, 12, 31), Quantity(Decimal("300.00")))}
    
    journal = []
    je1 = JournalEntry(
        date=date(2022, 12, 30),
        description="Old transaction",
        postings=[]
    )
    p1 = Posting(
        journal=je1,
        account=cash_acc,
        amount=Amount(Decimal("100.00")),
        direction=Direction.DEBIT
    )
    je1.postings.append(p1)
    journal.append(je1)

    je2 = JournalEntry(
        date=date(2024, 1, 1),
        description="Future transaction",
        postings=[]
    )
    p2 = Posting(
        journal=je2,
        account=cash_acc,
        amount=Amount(Decimal("50.00")),
        direction=Direction.CREDIT
    )
    je2.postings.append(p2)
    journal.append(je2)

    gl = build_general_ledger(period, journal, initial)
    assert len(gl.ledgers[cash_acc].entries) == 0
    assert gl.ledgers[cash_acc].initial.value == Decimal("300.00")

    # Test 4: Multiple postings to same account
    cash_acc = MockAccount("Cash")
    initial = {cash_acc: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00")))}
    
    journal = []
    for i in range(3):
        je = JournalEntry(
            date=date(2023, 1, i + 1),
            description=f"Transaction {i+1}",
            postings=[]
        )
        p = Posting(
            journal=je,
            account=cash_acc,
            amount=Amount(Decimal("100.00")),
            direction=Direction.DEBIT if i % 2 == 0 else Direction.CREDIT
        )
        je.postings.append(p)
        journal.append(je)

    gl = build_general_ledger(period, journal, initial)
    assert len(gl.ledgers[cash_acc].entries) == 3
    assert gl.ledgers[cash_acc].entries[0].balance == Decimal("1100.00")
    assert gl.ledgers[cash_acc].entries[1].balance == Decimal("1000.00")
    assert gl.ledgers[cash_acc].entries[2].balance == Decimal("1100.00")

    # Test 5: Account without initial balance gets created with zero balance
    cash_acc = MockAccount("Cash")
    expense_acc = MockAccount("Expense")
    initial = {cash_acc: Balance(date(2022, 12, 31), Quantity(Decimal("500.00")))}
    
    journal = []
    je = JournalEntry(
        date=date(2023, 1, 10),
        description="Expense",
        postings=[]
    )
    p1 = Posting(
        journal=je,
        account=cash_acc,
        amount=Amount(Decimal("100.00")),
        direction=Direction.CREDIT
    )
    p2 = Posting(
        journal=je,
        account=expense_acc,
        amount=Amount(Decimal("100.00")),
        direction=Direction.DEBIT
    )
    je.postings.extend([p1, p2])
    journal.append(je)

    gl = build_general_ledger(period, journal, initial)
    assert expense_acc in gl.ledgers
    assert gl.ledgers[expense_acc].initial.value == Decimal("0.00")
    assert len(gl.ledgers[expense_acc].entries) == 1
    assert gl.ledgers[expense_acc].entries[0].balance == Decimal("100.00")

    # Test 6: Verify ledger entry properties
    cash_acc = MockAccount("Cash")
    initial = {cash_acc: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00")))}
    
    journal = []
    je = JournalEntry(
        date=date(2023, 2, 1),
        description="Test Entry",
        postings=[]
    )
    p = Posting(
        journal=je,
        account=cash_acc,
        amount=Amount(Decimal("200.00")),
        direction=Direction.DEBIT
    )
    je.postings.append(p)
    journal.append(je)

    gl = build_general_ledger(period, journal, initial)
    entry = gl.ledgers[cash_acc].entries[0]
    assert entry.date == date(2023, 2, 1)
    assert entry.description == "Test Entry"
    assert entry.amount == Decimal("200.00")
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == Decimal("200.00")
    assert entry.credit == None
    assert entry.balance == Decimal("1200.00")


# LLM-generated content at query #6
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account = Account("1000", "Cash")
            return {account: Balance(period.since, Quantity(Decimal(100)))}

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            account = Account("1000", "Cash")
            journal_entry = JournalEntry(
                date=date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        journal=None,
                        account=account,
                        direction=Direction.DEBIT,
                        amount=Amount(Decimal(50))
                    )
                ]
            )
            journal_entry.postings[0].journal = journal_entry
            return [journal_entry]

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries()
    )
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 1
    
    account = Account("1000", "Cash")
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial == Balance(period.since, Quantity(Decimal(100)))
    assert len(ledger.entries) == 1
    
    entry = ledger.entries[0]
    assert entry.balance == Quantity(Decimal(150))
    assert entry.amount == Amount(Decimal(50))
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == Amount(Decimal(50))
    assert entry.credit is None
    assert entry.date == date(2023, 1, 15)
    assert entry.description == "Test transaction"


# LLM-generated content at query #7
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from ..commons.numbers import Quantity

    class MockReadInitialBalances:
        def __call__(self, period: DateRange):
            return {
                Account("1000"): Balance(period.since, Quantity(Decimal("1000.00"))),
                Account("2000"): Balance(period.since, Quantity(Decimal("2000.00"))),
            }

    reader = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    result = reader(period)
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("1000") in result
    assert Account("2000") in result
    assert result[Account("1000")].value == Quantity(Decimal("1000.00"))
    assert result[Account("2000")].value == Quantity(Decimal("2000.00"))
    assert result[Account("1000")].date == period.since
    assert result[Account("2000")].date == period.since


# LLM-generated content at query #8
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from datetime import date
    
    # Create a mock implementation of ReadInitialBalances protocol
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> Dict[Account, Balance]:
            # Return some dummy initial balances
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Accounts Payable")
            return {
                account1: Balance(period.since, Quantity(Decimal("1000.00"))),
                account2: Balance(period.since, Quantity(Decimal("-500.00")))
            }
    
    # Test that the protocol can be instantiated and called
    reader = MockReadInitialBalances()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    # Call the protocol implementation
    result = reader(period)
    
    # Verify the result
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Check account keys
    accounts = list(result.keys())
    assert accounts[0].number == "1000"
    assert accounts[0].name == "Cash"
    assert accounts[1].number == "2000"
    assert accounts[1].name == "Accounts Payable"
    
    # Check balance values
    assert result[accounts[0]].value == Quantity(Decimal("1000.00"))
    assert result[accounts[1]].value == Quantity(Decimal("-500.00"))
    
    # Verify date matches period start
    assert result[accounts[0]].date == period.since
    assert result[accounts[1]].date == period.since
    
    # Test with different period
    period2 = DateRange(date(2024, 1, 1), date(2024, 12, 31))
    result2 = reader(period2)
    assert result2[accounts[0]].date == period2.since


# LLM-generated content at query #9
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Revenue")
            return {
                account1: Balance(period.since, Quantity(Decimal("1000.00"))),
                account2: Balance(period.since, Quantity(Decimal("0.00"))),
            }

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry]:
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Revenue")
            
            journal_entry = JournalEntry(
                date=date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        journal=None,
                        account=account1,
                        direction=Direction.DEBIT,
                        amount=Amount(Decimal("500.00")),
                        metadata=None,
                    ),
                    Posting(
                        journal=None,
                        account=account2,
                        direction=Direction.CREDIT,
                        amount=Amount(Decimal("500.00")),
                        metadata=None,
                    ),
                ],
            )
            journal_entry.postings[0].journal = journal_entry
            journal_entry.postings[1].journal = journal_entry
            
            return [journal_entry]

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries(),
    )
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2
    
    cash_account = Account("1000", "Cash")
    revenue_account = Account("2000", "Revenue")
    
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    
    cash_ledger = result.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert cash_ledger.entries[0].is_debit == True
    
    revenue_ledger = result.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert revenue_ledger.initial.value == Quantity(Decimal("0.00"))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    assert revenue_ledger.entries[0].amount == Amount(Decimal("500.00"))
    assert revenue_ledger.entries[0].is_credit == True


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from ..commons.numbers import Amount, Quantity
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockAccount(Account):
        def __init__(self, name):
            self.name = name
        
        def __hash__(self):
            return hash(self.name)
        
        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    account1 = MockAccount("Cash")
    account2 = MockAccount("Revenue")
    account3 = MockAccount("Expense")
    
    initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    journal_entries = [
        JournalEntry(
            date=date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    date=date(2023, 1, 15),
                    account=account1,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.DEBIT
                ),
                Posting(
                    date=date(2023, 1, 15),
                    account=account2,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.CREDIT
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 2, 1),
            description="Expense",
            postings=[
                Posting(
                    date=date(2023, 2, 1),
                    account=account3,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.DEBIT
                ),
                Posting(
                    date=date(2023, 2, 1),
                    account=account1,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.CREDIT
                )
            ]
        ),
        JournalEntry(
            date=date(2022, 12, 15),
            description="Out of period",
            postings=[
                Posting(
                    date=date(2022, 12, 15),
                    account=account1,
                    amount=Amount(Decimal("999.00")),
                    direction=Direction.DEBIT
                )
            ]
        )
    ]
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert result.period == period
    assert len(result.ledgers) == 3
    
    cash_ledger = result.ledgers[account1]
    assert cash_ledger.account == account1
    assert cash_ledger.initial == initial_balances[account1]
    assert len(cash_ledger.entries) == 2
    
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[0].is_debit == True
    assert cash_ledger.entries[0].debit == Amount(Decimal("500.00"))
    assert cash_ledger.entries[0].credit == None
    
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    assert cash_ledger.entries[1].is_credit == True
    assert cash_ledger.entries[1].debit == None
    assert cash_ledger.entries[1].credit == Amount(Decimal("200.00"))
    
    revenue_ledger = result.ledgers[account2]
    assert revenue_ledger.account == account2
    assert revenue_ledger.initial == Balance(period.since, Quantity(Decimal("0")))
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    assert revenue_ledger.entries[0].is_credit == True
    
    expense_ledger = result.ledgers[account3]
    assert expense_ledger.account == account3
    assert expense_ledger.initial == Balance(period.since, Quantity(Decimal("0")))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))
    assert expense_ledger.entries[0].is_debit == True
    
    empty_initial = {}
    result_empty = build_general_ledger(period, journal_entries, empty_initial)
    assert len(result_empty.ledgers) == 3
    for ledger in result_empty.ledgers.values():
        assert ledger.initial == Balance(period.since, Quantity(Decimal("0")))
    
    empty_journal = []
    result_no_journal = build_general_ledger(period, empty_journal, initial_balances)
    assert len(result_no_journal.ledgers) == 1
    assert len(result_no_journal.ledgers[account1].entries) == 0


# LLM-generated content at query #11
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    # Test data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Create test accounts
    cash_account = Account("1000", "Cash")
    revenue_account = Account("4000", "Revenue")
    expense_account = Account("5000", "Expense")
    
    # Test 1: Empty journal with initial balances
    initial_balances = {
        cash_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        revenue_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("0.00")))
    }
    
    journal_entries = []
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert result.period == period
    assert len(result.ledgers) == 2
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    assert result.ledgers[cash_account].initial.value == Decimal("1000.00")
    assert len(result.ledgers[cash_account].entries) == 0
    assert len(result.ledgers[revenue_account].entries) == 0
    
    # Test 2: Journal entries within period
    journal_entry1 = JournalEntry(
        datetime.date(2023, 1, 15),
        "Test transaction 1",
        [
            Posting(cash_account, Direction.DEBIT, Amount(Decimal("500.00")), None),
            Posting(revenue_account, Direction.CREDIT, Amount(Decimal("500.00")), None)
        ]
    )
    
    journal_entry2 = JournalEntry(
        datetime.date(2023, 2, 20),
        "Test transaction 2",
        [
            Posting(expense_account, Direction.DEBIT, Amount(Decimal("200.00")), None),
            Posting(cash_account, Direction.CREDIT, Amount(Decimal("200.00")), None)
        ]
    )
    
    journal_entries = [journal_entry1, journal_entry2]
    initial_balances = {
        cash_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert len(result.ledgers) == 3  # cash, revenue, expense
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    assert expense_account in result.ledgers
    
    # Check cash ledger
    cash_ledger = result.ledgers[cash_account]
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Decimal("1500.00")  # 1000 + 500
    assert cash_ledger.entries[1].balance == Decimal("1300.00")  # 1500 - 200
    
    # Check revenue ledger (created on the fly)
    revenue_ledger = result.ledgers[revenue_account]
    assert revenue_ledger.initial.value == Decimal("0.00")
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Decimal("-500.00")  # 0 - 500 (credit)
    
    # Check expense ledger (created on the fly)
    expense_ledger = result.ledgers[expense_account]
    assert expense_ledger.initial.value == Decimal("0.00")
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Decimal("200.00")  # 0 + 200 (debit)
    
    # Test 3: Journal entry outside period should be ignored
    journal_entry_outside = JournalEntry(
        datetime.date(2022, 12, 15),  # Before period
        "Outside period",
        [
            Posting(cash_account, Direction.DEBIT, Amount(Decimal("100.00")), None),
            Posting(revenue_account, Direction.CREDIT, Amount(Decimal("100.00")), None)
        ]
    )
    
    journal_entries = [journal_entry_outside, journal_entry1]
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    # Only journal_entry1 should be processed
    assert len(result.ledgers[cash_account].entries) == 1
    assert result.ledgers[cash_account].entries[0].balance == Decimal("1500.00")
    
    # Test 4: Multiple postings to same account in one journal entry
    complex_journal_entry = JournalEntry(
        datetime.date(2023, 3, 10),
        "Complex transaction",
        [
            Posting(cash_account, Direction.DEBIT, Amount(Decimal("300.00")), None),
            Posting(cash_account, Direction.CREDIT, Amount(Decimal("100.00")), None),
            Posting(revenue_account, Direction.CREDIT, Amount(Decimal("200.00")), None)
        ]
    )
    
    journal_entries = [complex_journal_entry]
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    cash_ledger = result.ledgers[cash_account]
    assert len(cash_ledger.entries) == 2  # Two separate ledger entries
    assert cash_ledger.entries[0].balance == Decimal("1300.00")  # 1000 + 300
    assert cash_ledger.entries[1].balance == Decimal("1200.00")  # 1300 - 100
    
    # Test 5: Empty initial balances
    result = build_general_ledger(period, [journal_entry1], {})
    assert len(result.ledgers) == 2  # Both accounts created with zero initial balance
    assert result.ledgers[cash_account].initial.value == Decimal("0.00")
    assert result.ledgers[revenue_account].initial.value == Decimal("0.00")
    assert result.ledgers[cash_account].entries[0].balance == Decimal("500.00")
    assert result.ledgers[revenue_account].entries[0].balance == Decimal("-500.00")


# LLM-generated content at query #12
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account = Account("1000", "Cash")
            return {account: Balance(period.since, Quantity(Decimal("1000.00")))}
    
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            account = Account("1000", "Cash")
            journal = JournalEntry(
                date=date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        journal=None,
                        account=account,
                        direction=Direction.DEBIT,
                        amount=Amount(Decimal("500.00")),
                        metadata=None
                    )
                ]
            )
            journal.postings[0].journal = journal
            return [journal]
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries()
    )
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 1
    
    account = Account("1000", "Cash")
    assert account in result.ledgers
    ledger = result.ledgers[account]
    
    assert ledger.account == account
    assert ledger.initial.value == Decimal("1000.00")
    assert len(ledger.entries) == 1
    
    entry = ledger.entries[0]
    assert entry.balance == Decimal("1500.00")
    assert entry.amount == Decimal("500.00")
    assert entry.is_debit == True
    assert entry.debit == Decimal("500.00")
    assert entry.credit is None


# LLM-generated content at query #13
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    # Test data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Create test accounts
    cash_account = Account("1000", "Cash")
    revenue_account = Account("4000", "Revenue")
    expense_account = Account("5000", "Expense")
    
    # Test 1: Basic ledger creation with initial balances
    initial_balances = {
        cash_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        revenue_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("0.00")))
    }
    
    # Create test journal entries
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Sale",
            [
                Posting(cash_account, Amount(Decimal("500.00")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("500.00")), Direction.CREDIT, None)
            ]
        ),
        JournalEntry(
            datetime.date(2023, 2, 1),
            "Expense",
            [
                Posting(expense_account, Amount(Decimal("200.00")), Direction.DEBIT, None),
                Posting(cash_account, Amount(Decimal("200.00")), Direction.CREDIT, None)
            ]
        )
    ]
    
    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    # Verify period
    assert general_ledger.period == period
    
    # Verify all accounts are present
    assert len(general_ledger.ledgers) == 3
    assert cash_account in general_ledger.ledgers
    assert revenue_account in general_ledger.ledgers
    assert expense_account in general_ledger.ledgers
    
    # Verify cash ledger
    cash_ledger = general_ledger.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial == initial_balances[cash_account]
    assert len(cash_ledger.entries) == 2
    
    # Verify cash ledger entries
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    assert cash_ledger.entries[0].is_debit == True
    assert cash_ledger.entries[1].is_credit == True
    
    # Verify revenue ledger
    revenue_ledger = general_ledger.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert revenue_ledger.initial == initial_balances[revenue_account]
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))
    
    # Verify expense ledger (created automatically)
    expense_ledger = general_ledger.ledgers[expense_account]
    assert expense_ledger.account == expense_account
    assert expense_ledger.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))
    
    # Test 2: Journal entries outside period should be ignored
    journal_entries_outside = [
        JournalEntry(
            datetime.date(2022, 12, 15),
            "Old transaction",
            [
                Posting(cash_account, Amount(Decimal("100.00")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("100.00")), Direction.CREDIT, None)
            ]
        ),
        JournalEntry(
            datetime.date(2024, 1, 15),
            "Future transaction",
            [
                Posting(cash_account, Amount(Decimal("50.00")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("50.00")), Direction.CREDIT, None)
            ]
        )
    ]
    
    general_ledger2 = build_general_ledger(period, journal_entries_outside, initial_balances)
    
    # Verify no entries were added for accounts with initial balances
    assert len(general_ledger2.ledgers[cash_account].entries) == 0
    assert len(general_ledger2.ledgers[revenue_account].entries) == 0
    
    # Test 3: Empty journal entries
    general_ledger3 = build_general_ledger(period, [], initial_balances)
    assert len(general_ledger3.ledgers) == 2
    assert len(general_ledger3.ledgers[cash_account].entries) == 0
    assert len(general_ledger3.ledgers[revenue_account].entries) == 0
    
    # Test 4: Empty initial balances
    general_ledger4 = build_general_ledger(period, journal_entries, {})
    assert len(general_ledger4.ledgers) == 3
    assert cash_ledger.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert revenue_ledger.initial == Balance(period.since, Quantity(Decimal("0.00")))
    
    # Test 5: Complex transaction with multiple postings
    complex_journal = [
        JournalEntry(
            datetime.date(2023, 3, 1),
            "Complex transaction",
            [
                Posting(cash_account, Amount(Decimal("300.00")), Direction.DEBIT, None),
                Posting(expense_account, Amount(Decimal("100.00")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("400.00")), Direction.CREDIT, None)
            ]
        )
    ]
    
    general_ledger5 = build_general_ledger(period, complex_journal, initial_balances)
    
    # Verify all accounts have correct entry counts
    assert len(general_ledger5.ledgers[cash_account].entries) == 1
    assert len(general_ledger5.ledgers[expense_account].entries) == 1
    assert len(general_ledger5.ledgers[revenue_account].entries) == 1
    
    # Test 6: Verify ledger entry properties
    test_entry = cash_ledger.entries[0]
    assert test_entry.date == datetime.date(2023, 1, 15)
    assert test_entry.description == "Sale"
    assert test_entry.amount == Amount(Decimal("500.00"))
    assert test_entry.debit == Amount(Decimal("500.00"))
    assert test_entry.credit is None
    assert test_entry.is_debit == True
    assert test_entry.is_credit == False
    assert len(test_entry.cntraccts) == 1
    assert test_entry.cntraccts[0] == revenue_account


# LLM-generated content at query #14
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    # Mock data
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    
    # Mock initial balances
    mock_initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    # Mock journal entries
    mock_journal_entries = [
        JournalEntry(
            date(2023, 1, 15),
            "Sale",
            [
                Posting(account1, Amount(Decimal("500.00")), Direction.DEBIT, None),
                Posting(account2, Amount(Decimal("500.00")), Direction.CREDIT, None)
            ]
        ),
        JournalEntry(
            date(2023, 2, 20),
            "Expense",
            [
                Posting(account1, Amount(Decimal("200.00")), Direction.CREDIT, None),
                Posting(Account("3000", "Expense"), Amount(Decimal("200.00")), Direction.DEBIT, None)
            ]
        )
    ]
    
    # Mock protocol implementations
    def mock_read_initial_balances(period: DateRange) -> InitialBalances:
        return mock_initial_balances
    
    def mock_read_journal_entries(period: DateRange) -> List[JournalEntry]:
        return [je for je in mock_journal_entries if period.since <= je.date <= period.until]
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Execute the program
    general_ledger = program(period)
    
    # Assertions
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3  # account1, account2, and expense account
    
    # Check account1 ledger
    ledger1 = general_ledger.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == mock_initial_balances[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal("1500.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("1300.00"))
    
    # Check account2 ledger (no initial balance)
    ledger2 = general_ledger.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(period.since, Quantity(Decimal("0")))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("500.00"))
    
    # Check that entries outside period are filtered
    period2 = DateRange(date(2023, 2, 1), date(2023, 12, 31))
    general_ledger2 = program(period2)
    ledger1_2 = general_ledger2.ledgers[account1]
    assert len(ledger1_2.entries) == 1  # Only the expense posting from Feb 20
    
    # Test with empty journal entries
    def mock_empty_journal(period: DateRange) -> List[JournalEntry]:
        return []
    
    empty_program = compile_general_ledger_program(mock_read_initial_balances, mock_empty_journal)
    empty_ledger = empty_program(period)
    assert len(empty_ledger.ledgers) == 1  # Only account1 with initial balance
    assert len(empty_ledger.ledgers[account1].entries) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    
    class MockReadInitialBalances:
        def __init__(self, return_value):
            self.return_value = return_value
            self.called_with = None
        
        def __call__(self, period):
            self.called_with = period
            return self.return_value
    
    class MockReadJournalEntries:
        def __init__(self, return_value):
            self.return_value = return_value
            self.called_with = None
        
        def __call__(self, period):
            self.called_with = period
            return self.return_value
    
    # Test 1: Basic functionality with empty data
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    mock_initial_balances = MockReadInitialBalances({})
    mock_journal_entries = MockReadJournalEntries([])
    
    program = compile_general_ledger_program(mock_initial_balances, mock_journal_entries)
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert result.ledgers == {}
    assert mock_initial_balances.called_with == period
    assert mock_journal_entries.called_with == period
    
    # Test 2: With initial balances but no journal entries
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        account2: Balance(date(2022, 12, 31), Quantity(Decimal("-500.00")))
    }
    
    mock_initial_balances = MockReadInitialBalances(initial_balances)
    mock_journal_entries = MockReadJournalEntries([])
    
    program = compile_general_ledger_program(mock_initial_balances, mock_journal_entries)
    result = program(period)
    
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].initial == initial_balances[account1]
    assert result.ledgers[account2].initial == initial_balances[account2]
    assert len(result.ledgers[account1].entries) == 0
    assert len(result.ledgers[account2].entries) == 0
    
    # Test 3: With journal entries that create new ledgers
    journal_entry = JournalEntry(
        date=date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(
                date=date(2023, 1, 15),
                account=account1,
                amount=Amount(Decimal("100.00")),
                direction=Direction.DEBIT,
                journal=None
            ),
            Posting(
                date=date(2023, 1, 15),
                account=account2,
                amount=Amount(Decimal("100.00")),
                direction=Direction.CREDIT,
                journal=None
            )
        ]
    )
    # Attach journal reference to postings
    for posting in journal_entry.postings:
        posting.journal = journal_entry
    
    mock_initial_balances = MockReadInitialBalances({})
    mock_journal_entries = MockReadJournalEntries([journal_entry])
    
    program = compile_general_ledger_program(mock_initial_balances, mock_journal_entries)
    result = program(period)
    
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("100.00"))
    assert result.ledgers[account2].entries[0].balance == Quantity(Decimal("-100.00"))
    
    # Test 4: With both initial balances and journal entries
    mock_initial_balances = MockReadInitialBalances(initial_balances)
    mock_journal_entries = MockReadJournalEntries([journal_entry])
    
    program = compile_general_ledger_program(mock_initial_balances, mock_journal_entries)
    result = program(period)
    
    assert len(result.ledgers) == 2
    assert result.ledgers[account1].initial == initial_balances[account1]
    assert result.ledgers[account2].initial == initial_balances[account2]
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("1100.00"))
    assert result.ledgers[account2].entries[0].balance == Quantity(Decimal("-600.00"))
    
    # Test 5: Journal entries outside period should be ignored
    journal_entry_outside = JournalEntry(
        date=date(2022, 12, 30),
        description="Outside period",
        postings=[
            Posting(
                date=date(2022, 12, 30),
                account=account1,
                amount=Amount(Decimal("50.00")),
                direction=Direction.DEBIT,
                journal=None
            )
        ]
    )
    for posting in journal_entry_outside.postings:
        posting.journal = journal_entry_outside
    
    mock_initial_balances = MockReadInitialBalances({})
    mock_journal_entries = MockReadJournalEntries([journal_entry, journal_entry_outside])
    
    program = compile_general_ledger_program(mock_initial_balances, mock_journal_entries)
    result = program(period)
    
    assert len(result.ledgers) == 2
    assert len(result.ledgers[account1].entries) == 1
    assert result.ledgers[account1].entries[0].amount == Amount(Decimal("100.00"))
    
    # Test 6: Multiple journal entries for same account
    journal_entry2 = JournalEntry(
        date=date(2023, 1, 20),
        description="Second transaction",
        postings=[
            Posting(
                date=date(2023, 1, 20),
                account=account1,
                amount=Amount(Decimal("200.00")),
                direction=Direction.DEBIT,
                journal=None
            ),
            Posting(
                date=date(2023, 1, 20),
                account=Account("3000", "Revenue"),
                amount=Amount(Decimal("200.00")),
                direction=Direction.CREDIT,
                journal=None
            )
        ]
    )
    for posting in journal_entry2.postings:
        posting.journal = journal_entry2
    
    mock_initial_balances = MockReadInitialBalances(initial_balances)
    mock_journal_entries = MockReadJournalEntries([journal_entry, journal_entry2])
    
    program = compile_general_ledger_program(mock_initial_balances, mock_journal_entries)
    result = program(period)
    
    assert len(result.ledgers) == 3
    assert len(result.ledgers[account1].entries) == 2
    assert result.ledgers[account1].entries[0].balance == Quantity(Decimal("1100.00"))
    assert result.ledgers[account1].entries[1].balance == Quantity(Decimal("1300.00"))


# LLM-generated content at query #16
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    
    class MockReadInitialBalances:
        def __call__(self, period: DateRange):
            return {
                Account("1000"): Balance(period.since, Quantity(Decimal("1000.00"))),
                Account("2000"): Balance(period.since, Quantity(Decimal("-500.00")))
            }
    
    reader = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    result = reader(period)
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("1000") in result
    assert Account("2000") in result
    assert result[Account("1000")].value == Quantity(Decimal("1000.00"))
    assert result[Account("2000")].value == Quantity(Decimal("-500.00"))


# LLM-generated content at query #17
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Revenue")
            return {
                account1: Balance(period.since, Quantity(Decimal("1000.00"))),
                account2: Balance(period.since, Quantity(Decimal("0.00")))
            }

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Revenue")
            
            journal_entry = JournalEntry(
                date=date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        journal=None,
                        account=account1,
                        direction=Direction.DEBIT,
                        amount=Amount(Decimal("500.00"))
                    ),
                    Posting(
                        journal=None,
                        account=account2,
                        direction=Direction.CREDIT,
                        amount=Amount(Decimal("500.00"))
                    )
                ]
            )
            
            for posting in journal_entry.postings:
                posting.journal = journal_entry
            
            return [journal_entry]

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries()
    )
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 2
    
    cash_account = Account("1000", "Cash")
    revenue_account = Account("2000", "Revenue")
    
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    
    cash_ledger = result.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Decimal("1500.00")
    assert cash_ledger.entries[0].is_debit == True
    
    revenue_ledger = result.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert revenue_ledger.initial.value == Decimal("0.00")
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Decimal("500.00")
    assert revenue_ledger.entries[0].is_credit == True
    
    assert cash_ledger.entries[0].date == date(2023, 1, 15)
    assert revenue_ledger.entries[0].date == date(2023, 1, 15)
    assert cash_ledger.entries[0].description == "Test transaction"
    assert revenue_ledger.entries[0].description == "Test transaction"


# LLM-generated content at query #18
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    # Mock implementation of ReadInitialBalances
    class MockReadInitialBalances:
        def __call__(self, period: DateRange):
            # Return a simple initial balances dict
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Accounts Payable")
            return {
                account1: Balance(period.since, Quantity(Decimal("1000.00"))),
                account2: Balance(period.since, Quantity(Decimal("-500.00"))),
            }

    # Test the protocol implementation
    reader = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    result = reader(period)
    
    # Verify the result is a dict
    assert isinstance(result, dict)
    
    # Verify the keys are Account instances
    for account in result.keys():
        assert isinstance(account, Account)
    
    # Verify the values are Balance instances
    for balance in result.values():
        assert isinstance(balance, Balance)
    
    # Verify the balances have correct dates
    for balance in result.values():
        assert balance.date == period.since
    
    # Verify specific values
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    assert result[account1].value == Quantity(Decimal("1000.00"))
    assert result[account2].value == Quantity(Decimal("-500.00"))
    
    # Test with different period
    period2 = DateRange(datetime.date(2023, 6, 1), datetime.date(2023, 6, 30))
    result2 = reader(period2)
    
    # Verify balances adapt to new period
    for balance in result2.values():
        assert balance.date == period2.since


# LLM-generated content at query #19
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    # Create test accounts
    cash_account = Account("1000", "Cash")
    revenue_account = Account("4000", "Revenue")
    expense_account = Account("5000", "Expense")

    # Create test period
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))

    # Create initial balances
    initial_balances = {
        cash_account: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        revenue_account: Balance(date(2022, 12, 31), Quantity(Decimal("0.00")))
    }

    # Create test journal entries
    journal_entries = [
        JournalEntry(
            date=date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=revenue_account,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 1, 20),
            description="Expense",
            postings=[
                Posting(
                    account=expense_account,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        )
    ]

    # Build general ledger
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)

    # Test general ledger properties
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3  # cash, revenue, expense

    # Test cash ledger
    cash_ledger = general_ledger.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial == initial_balances[cash_account]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))

    # Test revenue ledger
    revenue_ledger = general_ledger.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert revenue_ledger.initial == initial_balances[revenue_account]
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("500.00"))

    # Test expense ledger (created automatically)
    expense_ledger = general_ledger.ledgers[expense_account]
    assert expense_ledger.account == expense_account
    assert expense_ledger.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))

    # Test ledger entry properties
    cash_entry = cash_ledger.entries[0]
    assert cash_entry.date == date(2023, 1, 15)
    assert cash_entry.description == "Sale"
    assert cash_entry.amount == Amount(Decimal("500.00"))
    assert cash_entry.is_debit == True
    assert cash_entry.debit == Amount(Decimal("500.00"))
    assert cash_entry.credit == None
    assert cash_entry.balance == Quantity(Decimal("1500.00"))

    # Test filtering by period
    journal_entry_outside_period = JournalEntry(
        date=date(2022, 12, 30),
        description="Old transaction",
        postings=[
            Posting(
                account=cash_account,
                amount=Amount(Decimal("100.00")),
                direction=Direction.DEBIT,
                journal=None
            )
        ]
    )
    
    journal_with_mixed_dates = journal_entries + [journal_entry_outside_period]
    filtered_ledger = build_general_ledger(period, journal_with_mixed_dates, initial_balances)
    
    # Should still have only 2 entries in cash ledger (old one filtered out)
    assert len(filtered_ledger.ledgers[cash_account].entries) == 2

    # Test with empty journal
    empty_ledger = build_general_ledger(period, [], initial_balances)
    assert len(empty_ledger.ledgers) == 2  # Only accounts with initial balances
    assert len(empty_ledger.ledgers[cash_account].entries) == 0

    # Test with no initial balances
    no_initial_ledger = build_general_ledger(period, journal_entries, {})
    assert len(no_initial_ledger.ledgers) == 3  # All accounts created
    assert no_initial_ledger.ledgers[cash_account].initial == Balance(period.since, Quantity(Decimal("0.00")))


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    # Create test accounts
    cash_account = Account("1000", "Cash")
    revenue_account = Account("4000", "Revenue")
    expense_account = Account("5000", "Expense")

    # Create test period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))

    # Test 1: Empty journal and empty initial balances
    initial_balances = {}
    journal_entries = []
    result = build_general_ledger(period, journal_entries, initial_balances)
    assert result.period == period
    assert len(result.ledgers) == 0

    # Test 2: Journal entries with no initial balances
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 10),
            "Sale",
            [
                Posting(cash_account, Amount(Decimal("1000")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("1000")), Direction.CREDIT, None),
            ],
        ),
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Expense",
            [
                Posting(expense_account, Amount(Decimal("200")), Direction.DEBIT, None),
                Posting(cash_account, Amount(Decimal("200")), Direction.CREDIT, None),
            ],
        ),
    ]
    result = build_general_ledger(period, journal_entries, initial_balances)
    assert len(result.ledgers) == 3
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    assert expense_account in result.ledgers

    # Verify cash ledger
    cash_ledger = result.ledgers[cash_account]
    assert cash_ledger.initial == Balance(period.since, Quantity(Decimal(0)))
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1000"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("800"))

    # Test 3: With initial balances
    initial_balances = {
        cash_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("5000"))),
        revenue_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("0"))),
    }
    result = build_general_ledger(period, journal_entries, initial_balances)
    cash_ledger = result.ledgers[cash_account]
    assert cash_ledger.initial == Balance(datetime.date(2022, 12, 31), Quantity(Decimal("5000")))
    assert cash_ledger.entries[0].balance == Quantity(Decimal("6000"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("5800"))

    # Test 4: Journal entries outside period should be ignored
    journal_entries = [
        JournalEntry(
            datetime.date(2022, 12, 31),
            "Old entry",
            [
                Posting(cash_account, Amount(Decimal("100")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("100")), Direction.CREDIT, None),
            ],
        ),
        JournalEntry(
            datetime.date(2023, 1, 15),
            "In period",
            [
                Posting(cash_account, Amount(Decimal("200")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("200")), Direction.CREDIT, None),
            ],
        ),
        JournalEntry(
            datetime.date(2023, 2, 1),
            "Future entry",
            [
                Posting(cash_account, Amount(Decimal("300")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("300")), Direction.CREDIT, None),
            ],
        ),
    ]
    result = build_general_ledger(period, journal_entries, initial_balances)
    cash_ledger = result.ledgers[cash_account]
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Quantity(Decimal("5200"))

    # Test 5: Multiple postings to same account in one journal entry
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 20),
            "Complex entry",
            [
                Posting(cash_account, Amount(Decimal("100")), Direction.DEBIT, None),
                Posting(cash_account, Amount(Decimal("50")), Direction.CREDIT, None),
                Posting(revenue_account, Amount(Decimal("50")), Direction.CREDIT, None),
            ],
        ),
    ]
    result = build_general_ledger(period, journal_entries, {})
    cash_ledger = result.ledgers[cash_account]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("100"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("50"))

    # Test 6: Verify ledger entry properties
    entry = cash_ledger.entries[0]
    assert entry.date == datetime.date(2023, 1, 20)
    assert entry.description == "Complex entry"
    assert entry.amount == Amount(Decimal("100"))
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == Amount(Decimal("100"))
    assert entry.credit == None
    assert len(entry.cntraccts) == 1
    assert revenue_account in entry.cntraccts


# LLM-generated content at query #2
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    
    # Mock implementation of ReadInitialBalances protocol
    class MockReadInitialBalances:
        def __call__(self, period: DateRange):
            return {
                Account("cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
                Account("receivables"): Balance(period.since, Quantity(Decimal("500.00"))),
            }
    
    # Create instance and test
    reader = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    result = reader(period)
    
    # Verify result type and structure
    assert isinstance(result, dict)
    assert len(result) == 2
    
    # Verify account keys
    assert Account("cash") in result
    assert Account("receivables") in result
    
    # Verify balance values
    assert result[Account("cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("receivables")].value == Quantity(Decimal("500.00"))
    
    # Verify balance dates match period start
    assert result[Account("cash")].date == period.since
    assert result[Account("receivables")].date == period.since


# LLM-generated content at query #3
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting
    from ..commons.numbers import Amount, Quantity

    class MockAccount(Account):
        def __init__(self, name: str):
            self.name = name

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    class MockDirection:
        def __init__(self, value: int):
            self.value = value

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    account1 = MockAccount("Cash")
    account2 = MockAccount("Revenue")
    account3 = MockAccount("Expense")

    initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        account2: Balance(date(2022, 12, 31), Quantity(Decimal("0.00")))
    }

    journal_entries = [
        JournalEntry(
            date=date(2023, 1, 5),
            description="Sale",
            postings=[
                Posting(
                    journal=None,
                    account=account1,
                    amount=Amount(Decimal("500.00")),
                    direction=MockDirection(1)
                ),
                Posting(
                    journal=None,
                    account=account2,
                    amount=Amount(Decimal("500.00")),
                    direction=MockDirection(-1)
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 1, 10),
            description="Expense",
            postings=[
                Posting(
                    journal=None,
                    account=account1,
                    amount=Amount(Decimal("200.00")),
                    direction=MockDirection(-1)
                ),
                Posting(
                    journal=None,
                    account=account3,
                    amount=Amount(Decimal("200.00")),
                    direction=MockDirection(1)
                )
            ]
        ),
        JournalEntry(
            date=date(2022, 12, 30),
            description="Out of period",
            postings=[
                Posting(
                    journal=None,
                    account=account1,
                    amount=Amount(Decimal("999.00")),
                    direction=MockDirection(1)
                )
            ]
        )
    ]

    for je in journal_entries:
        for posting in je.postings:
            posting.journal = je

    result = build_general_ledger(period, journal_entries, initial_balances)

    assert result.period == period
    assert len(result.ledgers) == 3

    cash_ledger = result.ledgers[account1]
    assert cash_ledger.account == account1
    assert cash_ledger.initial == initial_balances[account1]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    assert cash_ledger.entries[0].posting.amount == Amount(Decimal("500.00"))
    assert cash_ledger.entries[1].posting.amount == Amount(Decimal("200.00"))

    revenue_ledger = result.ledgers[account2]
    assert revenue_ledger.account == account2
    assert revenue_ledger.initial == initial_balances[account2]
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500.00"))

    expense_ledger = result.ledgers[account3]
    assert expense_ledger.account == account3
    assert expense_ledger.initial == Balance(date(2023, 1, 1), Quantity(Decimal("0.00")))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))

    empty_initial = {}
    result_empty = build_general_ledger(period, journal_entries, empty_initial)
    assert len(result_empty.ledgers) == 3
    for ledger in result_empty.ledgers.values():
        assert ledger.initial.value == Quantity(Decimal("0.00"))


# LLM-generated content at query #4
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Revenue")
            return {
                account1: Balance(period.since, Quantity(Decimal("1000.00"))),
                account2: Balance(period.since, Quantity(Decimal("0.00")))
            }
    
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Revenue")
            
            journal_entry = JournalEntry(
                date=date(2023, 1, 15),
                description="Sale",
                postings=[
                    Posting(
                        journal=None,
                        account=account1,
                        direction=Direction.DEBIT,
                        amount=Decimal("500.00")
                    ),
                    Posting(
                        journal=None,
                        account=account2,
                        direction=Direction.CREDIT,
                        amount=Decimal("500.00")
                    )
                ]
            )
            
            for posting in journal_entry.postings:
                posting.journal = journal_entry
            
            return [journal_entry]
    
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries()
    )
    
    general_ledger = program(period)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 2
    
    cash_account = Account("1000", "Cash")
    revenue_account = Account("2000", "Revenue")
    
    assert cash_account in general_ledger.ledgers
    assert revenue_account in general_ledger.ledgers
    
    cash_ledger = general_ledger.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 1
    assert cash_ledger.entries[0].balance == Decimal("1500.00")
    assert cash_ledger.entries[0].is_debit == True
    
    revenue_ledger = general_ledger.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert revenue_ledger.initial.value == Decimal("0.00")
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Decimal("500.00")
    assert revenue_ledger.entries[0].is_credit == True


# LLM-generated content at query #5
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from ..commons.numbers import Quantity

    # Mock implementation of ReadInitialBalances protocol
    class MockReadInitialBalances:
        def __call__(self, period: DateRange):
            return {
                Account("Assets:Cash"): Balance(period.since, Quantity(Decimal("1000.00"))),
                Account("Liabilities:Loan"): Balance(period.since, Quantity(Decimal("-500.00"))),
            }

    # Create instance and test period
    reader = MockReadInitialBalances()
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Execute the call
    result = reader(period)
    
    # Verify the result
    assert isinstance(result, dict)
    assert len(result) == 2
    assert Account("Assets:Cash") in result
    assert Account("Liabilities:Loan") in result
    assert result[Account("Assets:Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("Liabilities:Loan")].value == Quantity(Decimal("-500.00"))
    assert all(isinstance(balance, Balance) for balance in result.values())
    assert all(balance.date == period.since for balance in result.values())


# LLM-generated content at query #6
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    class MockT:
        pass

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    account3 = Account("3000", "Expense")
    
    initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        account2: Balance(date(2022, 12, 31), Quantity(Decimal("0.00")))
    }
    
    journal_entries = [
        JournalEntry(
            date=date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    journal=None,
                    account=account1,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("500.00")),
                    metadata={}
                ),
                Posting(
                    journal=None,
                    account=account2,
                    direction=Direction.CREDIT,
                    amount=Amount(Decimal("500.00")),
                    metadata={}
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 2, 1),
            description="Expense",
            postings=[
                Posting(
                    journal=None,
                    account=account3,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("200.00")),
                    metadata={}
                ),
                Posting(
                    journal=None,
                    account=account1,
                    direction=Direction.CREDIT,
                    amount=Amount(Decimal("200.00")),
                    metadata={}
                )
            ]
        ),
        JournalEntry(
            date=date(2022, 12, 31),
            description="Out of period",
            postings=[
                Posting(
                    journal=None,
                    account=account1,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("999.00")),
                    metadata={}
                )
            ]
        ),
        JournalEntry(
            date=date(2024, 1, 1),
            description="Future transaction",
            postings=[
                Posting(
                    journal=None,
                    account=account1,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("888.00")),
                    metadata={}
                )
            ]
        )
    ]
    
    for entry in journal_entries:
        for posting in entry.postings:
            posting.journal = entry
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert account3 in result.ledgers
    
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2
    
    entry1 = ledger1.entries[0]
    assert entry1.date == date(2023, 1, 15)
    assert entry1.amount == Amount(Decimal("500.00"))
    assert entry1.is_debit == True
    assert entry1.balance == Quantity(Decimal("1500.00"))
    
    entry2 = ledger1.entries[1]
    assert entry2.date == date(2023, 2, 1)
    assert entry2.amount == Amount(Decimal("200.00"))
    assert entry2.is_credit == True
    assert entry2.balance == Quantity(Decimal("1300.00"))
    
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    
    entry3 = ledger2.entries[0]
    assert entry3.date == date(2023, 1, 15)
    assert entry3.amount == Amount(Decimal("500.00"))
    assert entry3.is_credit == True
    assert entry3.balance == Quantity(Decimal("500.00"))
    
    ledger3 = result.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial == Balance(date(2023, 1, 1), Quantity(Decimal("0.00")))
    assert len(ledger3.entries) == 1
    
    entry4 = ledger3.entries[0]
    assert entry4.date == date(2023, 2, 1)
    assert entry4.amount == Amount(Decimal("200.00"))
    assert entry4.is_debit == True
    assert entry4.balance == Quantity(Decimal("200.00"))
    
    empty_period = DateRange(date(2023, 6, 1), date(2023, 6, 30))
    empty_result = build_general_ledger(empty_period, journal_entries, initial_balances)
    
    assert empty_result.period == empty_period
    assert len(empty_result.ledgers) == 2
    assert account1 in empty_result.ledgers
    assert account2 in empty_result.ledgers
    assert account3 not in empty_result.ledgers
    
    empty_ledger1 = empty_result.ledgers[account1]
    assert empty_ledger1.initial == initial_balances[account1]
    assert len(empty_ledger1.entries) == 0
    
    empty_initial_balances = {}
    no_initial_result = build_general_ledger(period, journal_entries, empty_initial_balances)
    
    assert len(no_initial_result.ledgers) == 3
    assert account1 in no_initial_result.ledgers
    assert account2 in no_initial_result.ledgers
    assert account3 in no_initial_result.ledgers
    
    no_initial_ledger1 = no_initial_result.ledgers[account1]
    assert no_initial_ledger1.initial == Balance(date(2023, 1, 1), Quantity(Decimal("0.00")))
    assert len(no_initial_ledger1.entries) == 2
    assert no_initial_ledger1.entries[0].balance == Quantity(Decimal("500.00"))
    assert no_initial_ledger1.entries[1].balance == Quantity(Decimal("300.00"))


# LLM-generated content at query #7
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting
    from ..commons.numbers import Amount, Quantity
    
    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account = Account("1000", "Cash")
            return {account: Balance(period.since, Quantity(Decimal("1000.00")))}
    
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            account = Account("1000", "Cash")
            journal_entry = JournalEntry(
                date=date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        journal=None,
                        account=account,
                        direction=1,
                        amount=Amount(Decimal("500.00"))
                    )
                ]
            )
            journal_entry.postings[0].journal = journal_entry
            return [journal_entry]
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    read_initial_balances = MockReadInitialBalances()
    read_journal_entries = MockReadJournalEntries()
    
    program = compile_general_ledger_program(read_initial_balances, read_journal_entries)
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 1
    
    account = Account("1000", "Cash")
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial.value == Decimal("1000.00")
    assert len(ledger.entries) == 1
    
    entry = ledger.entries[0]
    assert entry.balance == Decimal("1500.00")
    assert entry.date == date(2023, 1, 15)
    assert entry.description == "Test transaction"
    assert entry.amount == Decimal("500.00")
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == Decimal("500.00")
    assert entry.credit is None


# LLM-generated content at query #8
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from unittest.mock import Mock
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    # Mock data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Create mock accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    
    # Mock initial balances
    mock_initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    # Mock journal entries
    mock_posting1 = Posting(
        journal=None,
        date=datetime.date(2023, 1, 15),
        account=account1,
        amount=Amount(Decimal("500.00")),
        direction=Direction.DEBIT
    )
    
    mock_posting2 = Posting(
        journal=None,
        date=datetime.date(2023, 1, 15),
        account=account2,
        amount=Amount(Decimal("500.00")),
        direction=Direction.CREDIT
    )
    
    mock_journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[mock_posting1, mock_posting2]
    )
    
    # Attach journal back to postings
    mock_posting1.journal = mock_journal_entry
    mock_posting2.journal = mock_journal_entry
    
    # Mock the algebra implementations
    mock_read_initial_balances = Mock(return_value=mock_initial_balances)
    mock_read_journal_entries = Mock(return_value=[mock_journal_entry])
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Execute the program
    result = program(period)
    
    # Assertions
    assert result.period == period
    assert len(result.ledgers) == 2  # Both accounts should have ledgers
    
    # Check account1 ledger
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == mock_initial_balances[account1]
    assert len(ledger1.entries) == 1
    assert ledger1.entries[0].balance == Quantity(Decimal("1500.00"))
    
    # Check account2 ledger
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("500.00"))
    
    # Verify mocks were called correctly
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)


# LLM-generated content at query #9
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account = Account("1000", "Cash")
            return {account: Balance(period.since, Quantity(Decimal("1000.00")))}

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry]:
            account = Account("1000", "Cash")
            journal = JournalEntry(
                date=date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        journal=None,
                        account=account,
                        direction=Direction.DEBIT,
                        amount=Amount(Decimal("500.00")),
                    )
                ],
            )
            journal.postings[0].journal = journal
            return [journal]

    read_initial = MockReadInitialBalances()
    read_journal = MockReadJournalEntries()
    
    program = compile_general_ledger_program(read_initial, read_journal)
    
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    general_ledger = program(period)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 1
    
    account = Account("1000", "Cash")
    ledger = general_ledger.ledgers[account]
    assert ledger.account == account
    assert ledger.initial.value == Decimal("1000.00")
    assert len(ledger.entries) == 1
    
    entry = ledger.entries[0]
    assert entry.amount == Decimal("500.00")
    assert entry.balance == Decimal("1500.00")
    assert entry.is_debit == True
    assert entry.debit == Decimal("500.00")
    assert entry.credit == None


# LLM-generated content at query #10
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.numbers import Amount, Quantity
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockAccount(Account):
        def __init__(self, name):
            self.name = name
        
        def __hash__(self):
            return hash(self.name)
        
        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    
    cash_account = MockAccount("Cash")
    revenue_account = MockAccount("Revenue")
    expense_account = MockAccount("Expense")
    
    initial_balances = {
        cash_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        revenue_account: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("0.00")))
    }
    
    journal_entries = [
        JournalEntry(
            datetime.date(2023, 1, 15),
            "Sale",
            [
                Posting(cash_account, Amount(Decimal("500.00")), Direction.DEBIT, None),
                Posting(revenue_account, Amount(Decimal("500.00")), Direction.CREDIT, None)
            ]
        ),
        JournalEntry(
            datetime.date(2023, 1, 20),
            "Expense",
            [
                Posting(expense_account, Amount(Decimal("200.00")), Direction.DEBIT, None),
                Posting(cash_account, Amount(Decimal("200.00")), Direction.CREDIT, None)
            ]
        )
    ]
    
    general_ledger = build_general_ledger(period, journal_entries, initial_balances)
    
    assert general_ledger.period == period
    assert len(general_ledger.ledgers) == 3
    
    cash_ledger = general_ledger.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial.value == Decimal("1000.00")
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Decimal("1500.00")
    assert cash_ledger.entries[1].balance == Decimal("1300.00")
    
    revenue_ledger = general_ledger.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert revenue_ledger.initial.value == Decimal("0.00")
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Decimal("500.00")
    
    expense_ledger = general_ledger.ledgers[expense_account]
    assert expense_ledger.account == expense_account
    assert expense_ledger.initial.value == Decimal("0.00")
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Decimal("200.00")
    
    out_of_period_entry = JournalEntry(
        datetime.date(2023, 2, 1),
        "Out of period",
        [
            Posting(cash_account, Amount(Decimal("100.00")), Direction.DEBIT, None),
            Posting(revenue_account, Amount(Decimal("100.00")), Direction.CREDIT, None)
        ]
    )
    
    journal_with_out_of_period = list(journal_entries) + [out_of_period_entry]
    filtered_ledger = build_general_ledger(period, journal_with_out_of_period, initial_balances)
    
    cash_ledger_filtered = filtered_ledger.ledgers[cash_account]
    assert len(cash_ledger_filtered.entries) == 2
    
    empty_initial = {}
    ledger_no_initial = build_general_ledger(period, journal_entries, empty_initial)
    assert len(ledger_no_initial.ledgers) == 3
    
    for account in [cash_account, revenue_account, expense_account]:
        assert ledger_no_initial.ledgers[account].initial.value == Decimal("0.00")


# LLM-generated content at query #11
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from unittest.mock import Mock, call
    from ..commons.zeitgeist import DateRange
    
    # Create a mock implementation of ReadInitialBalances protocol
    mock_reader = Mock(spec=ReadInitialBalances)
    
    # Setup test data
    test_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 12, 31)
    )
    
    # Create expected initial balances
    from .accounts import Account
    from .generic import Balance
    from ..commons.numbers import Quantity
    from decimal import Decimal
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    
    expected_balances = {
        account1: Balance(
            date=datetime.date(2022, 12, 31),
            value=Quantity(Decimal("10000.00"))
        ),
        account2: Balance(
            date=datetime.date(2022, 12, 31),
            value=Quantity(Decimal("5000.00"))
        )
    }
    
    # Configure the mock to return expected balances
    mock_reader.return_value = expected_balances
    
    # Call the protocol implementation
    result = mock_reader(test_period)
    
    # Verify the call was made with correct parameter
    mock_reader.assert_called_once_with(test_period)
    
    # Verify the result matches expected balances
    assert result == expected_balances
    assert isinstance(result, dict)
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    
    # Verify balance values
    assert result[account1].value == Quantity(Decimal("10000.00"))
    assert result[account2].value == Quantity(Decimal("5000.00"))
    
    # Test with empty period
    empty_period = DateRange(
        since=datetime.date(2023, 1, 1),
        until=datetime.date(2023, 1, 1)
    )
    
    # Reset mock and test with empty result
    mock_reader.reset_mock()
    mock_reader.return_value = {}
    
    empty_result = mock_reader(empty_period)
    
    mock_reader.assert_called_once_with(empty_period)
    assert empty_result == {}
    assert isinstance(empty_result, dict)
    assert len(empty_result) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from datetime import date

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> Dict[Account, Balance]:
            account1 = Account("1000", "Cash")
            account2 = Account("2000", "Accounts Payable")
            return {
                account1: Balance(period.since, Quantity(Decimal("1000.00"))),
                account2: Balance(period.since, Quantity(Decimal("-500.00"))),
            }

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    reader = MockReadInitialBalances()
    result = reader(period)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert all(isinstance(k, Account) for k in result.keys())
    assert all(isinstance(v, Balance) for v in result.values())
    assert result[Account("1000", "Cash")].value == Quantity(Decimal("1000.00"))
    assert result[Account("2000", "Accounts Payable")].value == Quantity(Decimal("-500.00"))
    for balance in result.values():
        assert balance.date == period.since


# LLM-generated content at query #13
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockReadInitialBalances:
        def __call__(self, period: DateRange) -> InitialBalances:
            account = Account("1000", "Cash")
            return {account: Balance(period.since, Quantity(Decimal("1000.00")))}

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> List[JournalEntry]:
            account = Account("1000", "Cash")
            entry = JournalEntry(
                date=date(2023, 1, 15),
                description="Test transaction",
                postings=[
                    Posting(
                        account=account,
                        amount=Amount(Decimal("500.00")),
                        direction=Direction.DEBIT,
                        journal=None
                    )
                ]
            )
            entry.postings[0].journal = entry
            return [entry]

    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    program = compile_general_ledger_program(
        MockReadInitialBalances(),
        MockReadJournalEntries()
    )
    
    result = program(period)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 1
    
    account = Account("1000", "Cash")
    assert account in result.ledgers
    ledger = result.ledgers[account]
    assert ledger.account == account
    assert ledger.initial.value == Quantity(Decimal("1000.00"))
    assert len(ledger.entries) == 1
    
    entry = ledger.entries[0]
    assert entry.ledger == ledger
    assert entry.posting.account == account
    assert entry.posting.amount == Amount(Decimal("500.00"))
    assert entry.balance == Quantity(Decimal("1500.00"))
    assert entry.is_debit == True
    assert entry.is_credit == False
    assert entry.debit == Amount(Decimal("500.00"))
    assert entry.credit == None


# LLM-generated content at query #14
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.numbers import Amount, Quantity
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockAccount(Account):
        def __init__(self, name):
            self.name = name

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    class MockDirection:
        def __init__(self, value):
            self.value = value

    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    
    cash_account = MockAccount("Cash")
    revenue_account = MockAccount("Revenue")
    expense_account = MockAccount("Expense")
    
    initial_balances = {
        cash_account: Balance(period.since, Quantity(Decimal("1000.00"))),
        revenue_account: Balance(period.since, Quantity(Decimal("0.00")))
    }
    
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("500.00")),
                    direction=MockDirection(1),
                    journal=None
                ),
                Posting(
                    account=revenue_account,
                    amount=Amount(Decimal("500.00")),
                    direction=MockDirection(-1),
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=datetime.date(2023, 1, 20),
            description="Expense",
            postings=[
                Posting(
                    account=expense_account,
                    amount=Amount(Decimal("200.00")),
                    direction=MockDirection(1),
                    journal=None
                ),
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("200.00")),
                    direction=MockDirection(-1),
                    journal=None
                )
            ]
        )
    ]
    
    for entry in journal_entries:
        for posting in entry.postings:
            posting.journal = entry
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert result.period == period
    assert len(result.ledgers) == 3
    
    cash_ledger = result.ledgers[cash_account]
    assert cash_ledger.account == cash_account
    assert cash_ledger.initial == initial_balances[cash_account]
    assert len(cash_ledger.entries) == 2
    assert cash_ledger.entries[0].balance == Quantity(Decimal("1500.00"))
    assert cash_ledger.entries[1].balance == Quantity(Decimal("1300.00"))
    
    revenue_ledger = result.ledgers[revenue_account]
    assert revenue_ledger.account == revenue_account
    assert revenue_ledger.initial == initial_balances[revenue_account]
    assert len(revenue_ledger.entries) == 1
    assert revenue_ledger.entries[0].balance == Quantity(Decimal("-500.00"))
    
    expense_ledger = result.ledgers[expense_account]
    assert expense_ledger.account == expense_account
    assert expense_ledger.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert len(expense_ledger.entries) == 1
    assert expense_ledger.entries[0].balance == Quantity(Decimal("200.00"))
    
    journal_entry_outside_period = JournalEntry(
        date=datetime.date(2023, 2, 1),
        description="Outside period",
        postings=[
            Posting(
                account=cash_account,
                amount=Amount(Decimal("100.00")),
                direction=MockDirection(1),
                journal=None
            )
        ]
    )
    journal_entry_outside_period.postings[0].journal = journal_entry_outside_period
    
    result_with_filtered = build_general_ledger(
        period, 
        [journal_entries[0], journal_entry_outside_period], 
        initial_balances
    )
    assert len(result_with_filtered.ledgers[cash_account].entries) == 1
    
    empty_result = build_general_ledger(period, [], initial_balances)
    assert len(empty_result.ledgers) == 2
    assert len(empty_result.ledgers[cash_account].entries) == 0
    assert len(empty_result.ledgers[revenue_account].entries) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    import datetime
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    # Test 1: Empty journal with initial balances
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Accounts Payable")
    initial = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        account2: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("-500.00")))
    }
    journal = []
    
    result = build_general_ledger(period, journal, initial)
    
    assert result.period == period
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].initial.value == Decimal("1000.00")
    assert result.ledgers[account2].initial.value == Decimal("-500.00")
    assert len(result.ledgers[account1].entries) == 0
    assert len(result.ledgers[account2].entries) == 0

    # Test 2: Journal entries within period
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("4000", "Revenue")
    initial = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 15),
        "Test transaction",
        [
            Posting(account1, Amount(Decimal("200.00")), Direction.DEBIT, journal_entry),
            Posting(account2, Amount(Decimal("200.00")), Direction.CREDIT, journal_entry)
        ]
    )
    journal = [journal_entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert len(result.ledgers[account1].entries) == 1
    assert len(result.ledgers[account2].entries) == 1
    assert result.ledgers[account1].entries[0].balance == Decimal("1200.00")
    assert result.ledgers[account2].entries[0].balance == Decimal("200.00")

    # Test 3: Journal entries outside period should be ignored
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("1000", "Cash")
    initial = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    journal_entry1 = JournalEntry(
        datetime.date(2022, 12, 31),
        "Before period",
        [Posting(account1, Amount(Decimal("100.00")), Direction.DEBIT, journal_entry1)]
    )
    journal_entry2 = JournalEntry(
        datetime.date(2023, 2, 1),
        "After period",
        [Posting(account1, Amount(Decimal("50.00")), Direction.CREDIT, journal_entry2)]
    )
    journal = [journal_entry1, journal_entry2]
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 1
    assert len(result.ledgers[account1].entries) == 0
    assert result.ledgers[account1].initial.value == Decimal("1000.00")

    # Test 4: Multiple postings to same account
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("1000", "Cash")
    initial = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    journal_entry1 = JournalEntry(
        datetime.date(2023, 1, 10),
        "Transaction 1",
        [Posting(account1, Amount(Decimal("200.00")), Direction.DEBIT, journal_entry1)]
    )
    journal_entry2 = JournalEntry(
        datetime.date(2023, 1, 20),
        "Transaction 2",
        [Posting(account1, Amount(Decimal("100.00")), Direction.CREDIT, journal_entry2)]
    )
    journal = [journal_entry1, journal_entry2]
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers[account1].entries) == 2
    assert result.ledgers[account1].entries[0].balance == Decimal("1200.00")
    assert result.ledgers[account1].entries[1].balance == Decimal("1100.00")

    # Test 5: Account without initial balance gets created with zero balance
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    account1 = Account("1000", "Cash")
    account2 = Account("4000", "Revenue")
    initial = {}
    
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 15),
        "Test transaction",
        [
            Posting(account1, Amount(Decimal("200.00")), Direction.DEBIT, journal_entry),
            Posting(account2, Amount(Decimal("200.00")), Direction.CREDIT, journal_entry)
        ]
    )
    journal = [journal_entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 2
    assert account1 in result.ledgers
    assert account2 in result.ledgers
    assert result.ledgers[account1].initial.value == Decimal("0.00")
    assert result.ledgers[account2].initial.value == Decimal("0.00")
    assert result.ledgers[account1].entries[0].balance == Decimal("200.00")
    assert result.ledgers[account2].entries[0].balance == Decimal("200.00")

    # Test 6: Complex transaction with multiple accounts
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    cash = Account("1000", "Cash")
    ar = Account("1100", "Accounts Receivable")
    revenue = Account("4000", "Revenue")
    expense = Account("5000", "Expense")
    
    initial = {
        cash: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("5000.00"))),
        ar: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("2000.00")))
    }
    
    journal_entry = JournalEntry(
        datetime.date(2023, 1, 15),
        "Complex transaction",
        [
            Posting(cash, Amount(Decimal("1000.00")), Direction.DEBIT, journal_entry),
            Posting(ar, Amount(Decimal("500.00")), Direction.CREDIT, journal_entry),
            Posting(revenue, Amount(Decimal("1200.00")), Direction.CREDIT, journal_entry),
            Posting(expense, Amount(Decimal("300.00")), Direction.DEBIT, journal_entry)
        ]
    )
    journal = [journal_entry]
    
    result = build_general_ledger(period, journal, initial)
    
    assert len(result.ledgers) == 4
    assert result.ledgers[cash].entries[0].balance == Decimal("6000.00")
    assert result.ledgers[ar].entries[0].balance == Decimal("1500.00")
    assert result.ledgers[revenue].entries[0].balance == Decimal("1200.00")
    assert result.ledgers[expense].entries[0].balance == Decimal("300.00")


# LLM-generated content at query #16
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from ..commons.numbers import Amount, Quantity
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    class MockAccount(Account):
        def __init__(self, name):
            self.name = name
        
        def __hash__(self):
            return hash(self.name)
        
        def __eq__(self, other):
            return isinstance(other, MockAccount) and self.name == other.name

    class MockT:
        pass

    # Test data setup
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
    
    # Create test accounts
    cash_account = MockAccount("Cash")
    revenue_account = MockAccount("Revenue")
    expense_account = MockAccount("Expense")
    
    # Test 1: Empty journal with initial balances
    initial_balances = {
        cash_account: Balance(period.since, Quantity(Decimal("1000.00"))),
        revenue_account: Balance(period.since, Quantity(Decimal("0.00")))
    }
    
    empty_journal = []
    result = build_general_ledger(period, empty_journal, initial_balances)
    
    assert result.period == period
    assert len(result.ledgers) == 2
    assert cash_account in result.ledgers
    assert revenue_account in result.ledgers
    assert result.ledgers[cash_account].initial.value == Decimal("1000.00")
    assert len(result.ledgers[cash_account].entries) == 0
    assert len(result.ledgers[revenue_account].entries) == 0
    
    # Test 2: Journal entries within period
    journal_entries = [
        JournalEntry(
            date=datetime.date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=revenue_account,
                    amount=Amount(Decimal("500.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        )
    ]
    
    # Link postings back to their journal entries
    for je in journal_entries:
        for posting in je.postings:
            posting.journal = je
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert len(result.ledgers) == 2
    assert len(result.ledgers[cash_account].entries) == 1
    assert len(result.ledgers[revenue_account].entries) == 1
    
    cash_entry = result.ledgers[cash_account].entries[0]
    revenue_entry = result.ledgers[revenue_account].entries[0]
    
    assert cash_entry.balance == Decimal("1500.00")
    assert revenue_entry.balance == Decimal("500.00")
    assert cash_entry.is_debit
    assert revenue_entry.is_credit
    
    # Test 3: Journal entries outside period should be ignored
    journal_entries_outside = [
        JournalEntry(
            date=datetime.date(2022, 12, 31),
            description="Old transaction",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("100.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=datetime.date(2023, 2, 1),
            description="Future transaction",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        )
    ]
    
    for je in journal_entries_outside:
        for posting in je.postings:
            posting.journal = je
    
    result = build_general_ledger(period, journal_entries_outside, initial_balances)
    
    assert len(result.ledgers[cash_account].entries) == 0
    
    # Test 4: Account without initial balance gets created with zero balance
    journal_entries_new_account = [
        JournalEntry(
            date=datetime.date(2023, 1, 20),
            description="Expense",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("100.00")),
                    direction=Direction.CREDIT,
                    journal=None
                ),
                Posting(
                    account=expense_account,
                    amount=Amount(Decimal("100.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        )
    ]
    
    for je in journal_entries_new_account:
        for posting in je.postings:
            posting.journal = je
    
    result = build_general_ledger(period, journal_entries_new_account, initial_balances)
    
    assert expense_account in result.ledgers
    assert result.ledgers[expense_account].initial.value == Decimal("0.00")
    assert len(result.ledgers[expense_account].entries) == 1
    assert result.ledgers[expense_account].entries[0].balance == Decimal("100.00")
    
    # Test 5: Multiple transactions affecting same account
    journal_entries_multiple = [
        JournalEntry(
            date=datetime.date(2023, 1, 10),
            description="Transaction 1",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("300.00")),
                    direction=Direction.DEBIT,
                    journal=None
                ),
                Posting(
                    account=revenue_account,
                    amount=Amount(Decimal("300.00")),
                    direction=Direction.CREDIT,
                    journal=None
                )
            ]
        ),
        JournalEntry(
            date=datetime.date(2023, 1, 20),
            description="Transaction 2",
            postings=[
                Posting(
                    account=cash_account,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.CREDIT,
                    journal=None
                ),
                Posting(
                    account=expense_account,
                    amount=Amount(Decimal("200.00")),
                    direction=Direction.DEBIT,
                    journal=None
                )
            ]
        )
    ]
    
    for je in journal_entries_multiple:
        for posting in je.postings:
            posting.journal = je
    
    result = build_general_ledger(period, journal_entries_multiple, initial_balances)
    
    assert len(result.ledgers[cash_account].entries) == 2
    assert result.ledgers[cash_account].entries[0].balance == Decimal("1300.00")
    assert result.ledgers[cash_account].entries[1].balance == Decimal("1100.00")
    
    # Test 6: Empty initial balances
    empty_initial = {}
    result = build_general_ledger(period, journal_entries_multiple, empty_initial)
    
    assert cash_account in result.ledgers
    assert result.ledgers[cash_account].initial.value == Decimal("0.00")
    assert len(result.ledgers[cash_account].entries) == 2
    assert result.ledgers[cash_account].entries[0].balance == Decimal("300.00")
    assert result.ledgers[cash_account].entries[1].balance == Decimal("100.00")


# LLM-generated content at query #17
#--------------------------

```python
def test_build_general_ledger():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from ..commons.numbers import Amount, Quantity

    class MockT:
        pass

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    account3 = Account("3000", "Expense")
    
    initial_balances = {
        account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        account2: Balance(date(2022, 12, 31), Quantity(Decimal("0.00")))
    }
    
    journal_entries = [
        JournalEntry(
            date=date(2023, 1, 15),
            description="Sale",
            postings=[
                Posting(
                    journal=None,
                    account=account1,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("500.00")),
                    metadata={}
                ),
                Posting(
                    journal=None,
                    account=account2,
                    direction=Direction.CREDIT,
                    amount=Amount(Decimal("500.00")),
                    metadata={}
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 1, 20),
            description="Expense",
            postings=[
                Posting(
                    journal=None,
                    account=account3,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("200.00")),
                    metadata={}
                ),
                Posting(
                    journal=None,
                    account=account1,
                    direction=Direction.CREDIT,
                    amount=Amount(Decimal("200.00")),
                    metadata={}
                )
            ]
        )
    ]
    
    for je in journal_entries:
        for posting in je.postings:
            posting.journal = je
    
    result = build_general_ledger(period, journal_entries, initial_balances)
    
    assert isinstance(result, GeneralLedger)
    assert result.period == period
    assert len(result.ledgers) == 3
    
    assert account1 in result.ledgers
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 2
    assert ledger1.entries[0].balance == Quantity(Decimal("1500.00"))
    assert ledger1.entries[1].balance == Quantity(Decimal("1300.00"))
    
    assert account2 in result.ledgers
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == initial_balances[account2]
    assert len(ledger2.entries) == 1
    assert ledger2.entries[0].balance == Quantity(Decimal("500.00"))
    
    assert account3 in result.ledgers
    ledger3 = result.ledgers[account3]
    assert ledger3.account == account3
    assert ledger3.initial == Balance(period.since, Quantity(Decimal("0.00")))
    assert len(ledger3.entries) == 1
    assert ledger3.entries[0].balance == Quantity(Decimal("200.00"))
    
    journal_entry_outside_period = JournalEntry(
        date=date(2022, 12, 30),
        description="Old transaction",
        postings=[
            Posting(
                journal=None,
                account=account1,
                direction=Direction.DEBIT,
                amount=Amount(Decimal("100.00")),
                metadata={}
            )
        ]
    )
    for posting in journal_entry_outside_period.postings:
        posting.journal = journal_entry_outside_period
    
    result2 = build_general_ledger(period, [journal_entry_outside_period], initial_balances)
    assert account1 in result2.ledgers
    assert len(result2.ledgers[account1].entries) == 0
    
    result3 = build_general_ledger(period, [], {})
    assert len(result3.ledgers) == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from unittest.mock import Mock, call
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction

    # Create test data
    period = DateRange(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))
    
    # Create mock accounts
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    
    # Create initial balances
    initial_balances = {
        account1: Balance(datetime.date(2022, 12, 31), Quantity(Decimal("1000.00")))
    }
    
    # Create journal entries
    posting1 = Posting(
        journal=None,  # Will be set below
        account=account1,
        direction=Direction.DEBIT,
        amount=Amount(Decimal("500.00")),
        date=datetime.date(2023, 1, 15)
    )
    
    posting2 = Posting(
        journal=None,  # Will be set below
        account=account2,
        direction=Direction.CREDIT,
        amount=Amount(Decimal("500.00")),
        date=datetime.date(2023, 1, 15)
    )
    
    journal_entry = JournalEntry(
        date=datetime.date(2023, 1, 15),
        description="Test transaction",
        postings=[posting1, posting2]
    )
    
    # Set the journal reference on postings
    posting1.journal = journal_entry
    posting2.journal = journal_entry
    
    # Create mock functions
    mock_read_initial_balances = Mock(return_value=initial_balances)
    mock_read_journal_entries = Mock(return_value=[journal_entry])
    
    # Compile the program
    program = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    # Execute the program
    result = program(period)
    
    # Verify mocks were called correctly
    mock_read_initial_balances.assert_called_once_with(period)
    mock_read_journal_entries.assert_called_once_with(period)
    
    # Verify result structure
    assert result.period == period
    assert isinstance(result.ledgers, dict)
    assert len(result.ledgers) == 2  # Both accounts should have ledgers
    
    # Verify account1 ledger
    ledger1 = result.ledgers[account1]
    assert ledger1.account == account1
    assert ledger1.initial == initial_balances[account1]
    assert len(ledger1.entries) == 1
    
    entry1 = ledger1.entries[0]
    assert entry1.posting == posting1
    assert entry1.balance == Quantity(Decimal("1500.00"))  # 1000 + 500
    assert entry1.is_debit == True
    assert entry1.debit == Amount(Decimal("500.00"))
    assert entry1.credit == None
    
    # Verify account2 ledger (created automatically since no initial balance)
    ledger2 = result.ledgers[account2]
    assert ledger2.account == account2
    assert ledger2.initial == Balance(period.since, Quantity(Decimal("0")))
    assert len(ledger2.entries) == 1
    
    entry2 = ledger2.entries[0]
    assert entry2.posting == posting2
    assert entry2.balance == Quantity(Decimal("-500.00"))  # 0 - 500
    assert entry2.is_credit == True
    assert entry2.debit == None
    assert entry2.credit == Amount(Decimal("500.00"))
    
    # Test with empty journal entries
    mock_read_initial_balances.reset_mock()
    mock_read_journal_entries.reset_mock()
    mock_read_journal_entries.return_value = []
    
    program2 = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    result2 = program2(period)
    assert result2.period == period
    assert len(result2.ledgers) == 1  # Only account1 with initial balance
    assert len(result2.ledgers[account1].entries) == 0
    
    # Test with journal entry outside period
    mock_read_initial_balances.reset_mock()
    mock_read_journal_entries.reset_mock()
    
    # Create journal entry outside period
    posting_outside = Posting(
        journal=JournalEntry(
            date=datetime.date(2022, 12, 15),
            description="Outside period",
            postings=[]
        ),
        account=account1,
        direction=Direction.DEBIT,
        amount=Amount(Decimal("100.00")),
        date=datetime.date(2022, 12, 15)
    )
    
    mock_read_journal_entries.return_value = [
        JournalEntry(
            date=datetime.date(2022, 12, 15),
            description="Outside period",
            postings=[posting_outside]
        )
    ]
    
    program3 = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    result3 = program3(period)
    assert len(result3.ledgers[account1].entries) == 0  # No entries added
    
    # Test with multiple postings to same account
    mock_read_initial_balances.reset_mock()
    mock_read_journal_entries.reset_mock()
    
    posting3 = Posting(
        journal=JournalEntry(
            date=datetime.date(2023, 2, 1),
            description="Second transaction",
            postings=[]
        ),
        account=account1,
        direction=Direction.CREDIT,
        amount=Amount(Decimal("200.00")),
        date=datetime.date(2023, 2, 1)
    )
    
    mock_read_journal_entries.return_value = [
        journal_entry,
        JournalEntry(
            date=datetime.date(2023, 2, 1),
            description="Second transaction",
            postings=[posting3]
        )
    ]
    
    program4 = compile_general_ledger_program(
        mock_read_initial_balances,
        mock_read_journal_entries
    )
    
    result4 = program4(period)
    ledger1_multi = result4.ledgers[account1]
    assert len(ledger1_multi.entries) == 2
    assert ledger1_multi.entries[0].balance == Quantity(Decimal("1500.00"))  # 1000 + 500
    assert ledger1_multi.entries[1].balance == Quantity(Decimal("1300.00"))  # 1500 - 200


# LLM-generated content at query #19
#--------------------------

```python
def test_GeneralLedgerProgram___call__():
    from decimal import Decimal
    from datetime import date
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    
    class MockReadInitialBalances:
        def __init__(self, balances):
            self.balances = balances
        
        def __call__(self, period):
            return self.balances
    
    class MockReadJournalEntries:
        def __init__(self, entries):
            self.entries = entries
        
        def __call__(self, period):
            return self.entries
    
    # Test 1: Basic functionality with single posting
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    account = Account("1000", "Cash")
    initial_balances = {account: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00")))}
    
    journal_entry = JournalEntry(
        date=date(2023, 1, 15),
        description="Test transaction",
        postings=[
            Posting(
                journal=None,
                account=account,
                direction=Direction.DEBIT,
                amount=Amount(Decimal("500.00"))
            )
        ]
    )
    
    read_initial = MockReadInitialBalances(initial_balances)
    read_journal = MockReadJournalEntries([journal_entry])
    
    program = compile_general_ledger_program(read_initial, read_journal)
    result = program(period)
    
    assert result.period == period
    assert len(result.ledgers) == 1
    assert account in result.ledgers
    assert result.ledgers[account].initial == initial_balances[account]
    assert len(result.ledgers[account].entries) == 1
    assert result.ledgers[account].entries[0].balance == Quantity(Decimal("1500.00"))
    
    # Test 2: Multiple accounts and postings
    period2 = DateRange(date(2023, 2, 1), date(2023, 2, 28))
    account1 = Account("1000", "Cash")
    account2 = Account("2000", "Revenue")
    
    initial_balances2 = {
        account1: Balance(date(2023, 1, 31), Quantity(Decimal("1500.00"))),
        account2: Balance(date(2023, 1, 31), Quantity(Decimal("0.00")))
    }
    
    journal_entry2 = JournalEntry(
        date=date(2023, 2, 10),
        description="Revenue transaction",
        postings=[
            Posting(
                journal=None,
                account=account1,
                direction=Direction.DEBIT,
                amount=Amount(Decimal("200.00"))
            ),
            Posting(
                journal=None,
                account=account2,
                direction=Direction.CREDIT,
                amount=Amount(Decimal("200.00"))
            )
        ]
    )
    
    read_initial2 = MockReadInitialBalances(initial_balances2)
    read_journal2 = MockReadJournalEntries([journal_entry2])
    
    program2 = compile_general_ledger_program(read_initial2, read_journal2)
    result2 = program2(period2)
    
    assert len(result2.ledgers) == 2
    assert result2.ledgers[account1].entries[0].balance == Quantity(Decimal("1700.00"))
    assert result2.ledgers[account2].entries[0].balance == Quantity(Decimal("-200.00"))
    
    # Test 3: Posting outside period should be ignored
    period3 = DateRange(date(2023, 3, 1), date(2023, 3, 31))
    account3 = Account("3000", "Expenses")
    
    initial_balances3 = {account3: Balance(date(2023, 2, 28), Quantity(Decimal("0.00")))}
    
    journal_entries3 = [
        JournalEntry(
            date=date(2023, 2, 28),  # Before period
            description="Old transaction",
            postings=[
                Posting(
                    journal=None,
                    account=account3,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("100.00"))
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 3, 15),  # Within period
            description="Current transaction",
            postings=[
                Posting(
                    journal=None,
                    account=account3,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("50.00"))
                )
            ]
        ),
        JournalEntry(
            date=date(2023, 4, 1),  # After period
            description="Future transaction",
            postings=[
                Posting(
                    journal=None,
                    account=account3,
                    direction=Direction.DEBIT,
                    amount=Amount(Decimal("75.00"))
                )
            ]
        )
    ]
    
    read_initial3 = MockReadInitialBalances(initial_balances3)
    read_journal3 = MockReadJournalEntries(journal_entries3)
    
    program3 = compile_general_ledger_program(read_initial3, read_journal3)
    result3 = program3(period3)
    
    assert len(result3.ledgers[account3].entries) == 1
    assert result3.ledgers[account3].entries[0].balance == Quantity(Decimal("50.00"))
    
    # Test 4: Account without initial balance gets created with zero balance
    period4 = DateRange(date(2023, 4, 1), date(2023, 4, 30))
    account4 = Account("4000", "New Account")
    
    initial_balances4 = {}
    
    journal_entry4 = JournalEntry(
        date=date(2023, 4, 15),
        description="New account transaction",
        postings=[
            Posting(
                journal=None,
                account=account4,
                direction=Direction.DEBIT,
                amount=Amount(Decimal("300.00"))
            )
        ]
    )
    
    read_initial4 = MockReadInitialBalances(initial_balances4)
    read_journal4 = MockReadJournalEntries([journal_entry4])
    
    program4 = compile_general_ledger_program(read_initial4, read_journal4)
    result4 = program4(period4)
    
    assert account4 in result4.ledgers
    assert result4.ledgers[account4].initial.value == Quantity(Decimal("0.00"))
    assert result4.ledgers[account4].entries[0].balance == Quantity(Decimal("300.00"))
    
    # Test 5: Empty period with no transactions
    period5 = DateRange(date(2023, 5, 1), date(2023, 5, 31))
    account5 = Account("5000", "Test Account")
    
    initial_balances5 = {account5: Balance(date(2023, 4, 30), Quantity(Decimal("1000.00")))}
    
    read_initial5 = MockReadInitialBalances(initial_balances5)
    read_journal5 = MockReadJournalEntries([])
    
    program5 = compile_general_ledger_program(read_initial5, read_journal5)
    result5 = program5(period5)
    
    assert account5 in result5.ledgers
    assert len(result5.ledgers[account5].entries) == 0
    assert result5.ledgers[account5].initial == initial_balances5[account5]


# LLM-generated content at query #20
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from unittest.mock import Mock
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from ..commons.numbers import Quantity
    from decimal import Decimal
    import datetime

    # Create mock period
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    period = DateRange(start_date, end_date)

    # Create mock accounts and balances
    account1 = Mock(spec=Account)
    account2 = Mock(spec=Account)
    balance1 = Balance(start_date, Quantity(Decimal("1000.00")))
    balance2 = Balance(start_date, Quantity(Decimal("2000.00")))

    # Create expected initial balances
    expected_balances = {account1: balance1, account2: balance2}

    # Create a concrete implementation of ReadInitialBalances
    def mock_read_initial_balances(period_arg: DateRange) -> Dict[Account, Balance]:
        assert period_arg == period
        return expected_balances

    # Test the protocol implementation
    result = mock_read_initial_balances(period)
    
    # Verify the result
    assert result == expected_balances
    assert len(result) == 2
    assert account1 in result
    assert account2 in result
    assert result[account1] == balance1
    assert result[account2] == balance2


# LLM-generated content at query #21
#--------------------------

```python
def test_ReadInitialBalances___call__():
    from decimal import Decimal
    from ..commons.zeitgeist import DateRange
    from .accounts import Account
    from .generic import Balance
    from .journaling import JournalEntry, Posting, Direction
    from datetime import date
    
    # Create mock implementation of ReadInitialBalances protocol
    class MockReadInitialBalances:
        def __init__(self, return_value):
            self.return_value = return_value
            self.called_with = None
            
        def __call__(self, period: DateRange) -> InitialBalances:
            self.called_with = period
            return self.return_value
    
    # Test 1: Protocol implementation returns correct balances
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    mock_account1 = Account("1000", "Cash")
    mock_account2 = Account("2000", "Accounts Payable")
    expected_balances = {
        mock_account1: Balance(date(2022, 12, 31), Quantity(Decimal("1000.00"))),
        mock_account2: Balance(date(2022, 12, 31), Quantity(Decimal("-500.00")))
    }
    
    reader = MockReadInitialBalances(expected_balances)
    result = reader(period)
    
    assert result == expected_balances
    assert reader.called_with == period
    
    # Test 2: Protocol implementation returns empty balances
    reader2 = MockReadInitialBalances({})
    result2 = reader2(period)
    
    assert result2 == {}
    assert reader2.called_with == period
    
    # Test 3: Protocol implementation can be used as callback
    def test_callback_function(callback: ReadInitialBalances):
        test_period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
        return callback(test_period)
    
    mock_balances = {mock_account1: Balance(date(2022, 12, 31), Quantity(Decimal("500.00")))}
    callback_reader = MockReadInitialBalances(mock_balances)
    callback_result = test_callback_function(callback_reader)
    
    assert callback_result == mock_balances
    assert callback_reader.called_with == DateRange(date(2023, 1, 1), date(2023, 1, 31))


