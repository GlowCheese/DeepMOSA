####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #2
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    reader = ReadJournalEntries()
    result = reader(period)
    assert isinstance(result, Iterable)


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()
    assert len(entry.postings) == 2
    assert isum(i.amount for i in entry.debits) == isum(i.amount for i in entry.credits)

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-50))
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_with_zero_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()
    assert len(entry.postings) == 0
    assert isum(i.amount for i in entry.debits) == isum(i.amount for i in entry.credits) == ZERO

def test_validate_with_multiple_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("C"), Quantity(-150))
    entry.post(datetime.date(2023, 1, 1), Account("D"), Quantity(-150))
    entry.validate()
    assert len(entry.postings) == 4
    assert isum(i.amount for i in entry.debits) == isum(i.amount for i in entry.credits)


# LLM-generated content at query #4
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")
    posting = Posting(journal, date, account, direction, amount)
    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #5
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #6
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #7
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #9
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #10
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #11
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #12
#--------------------------

```python
def test_journal_entry_constructor_with_defaults():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)

def test_journal_entry_constructor_with_postings():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    postings = [Posting(JournalEntry(date, description, source), date, Account("Test"), Direction.INC, Amount(100))]
    entry = JournalEntry(date, description, source)
    entry.postings = postings
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == postings
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #13
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_no_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_zero_quantity_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(0))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(0))
    entry.validate()


# LLM-generated content at query #15
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #16
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #17
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=object()
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #19
#--------------------------

```python
def test_posting_constructor_creates_frozen_instance():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #20
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #22
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #23
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #24
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_fails_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source=object()
    )
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(Decimal("10")))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(Decimal("-5")))
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #26
#--------------------------

```python
def test_JournalEntry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #27
#--------------------------

```python
def test_post_positive_quantity_adds_increment_posting():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal == entry
    assert posting.date == datetime.date(2023, 1, 1)
    assert posting.account == account
    assert posting.direction == Direction.INC
    assert posting.amount == Amount(100, "USD")
    assert result == entry

def test_post_negative_quantity_adds_decrement_posting():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(-100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal == entry
    assert posting.date == datetime.date(2023, 1, 1)
    assert posting.account == account
    assert posting.direction == Direction.DEC
    assert posting.amount == Amount(100, "USD")
    assert result == entry

def test_post_zero_quantity_adds_no_posting():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result == entry


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #29
#--------------------------

```python
def test_post_with_positive_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].journal == entry
    assert entry.postings[0].date == datetime.date(2023, 1, 1)
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.INC
    assert entry.postings[0].amount == Amount(100)
    assert result == entry

def test_post_with_negative_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(-50)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].journal == entry
    assert entry.postings[0].date == datetime.date(2023, 1, 1)
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.DEC
    assert entry.postings[0].amount == Amount(50)
    assert result == entry

def test_post_with_zero_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result == entry


# LLM-generated content at query #30
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry("test_journal")
    date = datetime.date(2023, 1, 1)
    account = Account("test_account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #31
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #32
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #33
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    journal_entries = [
        JournalEntry(date=date(2023, 1, 1), amount=100.0),
        JournalEntry(date=date(2023, 1, 2), amount=200.0),
    ]
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))

    read_entries = ReadJournalEntries()
    result = read_entries(period)

    assert isinstance(result, Iterable)
    assert list(result) == journal_entries


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(datetime.date.today(), "Test", object())
    entry.post(datetime.date.today(), Account("A"), Quantity(10))
    entry.post(datetime.date.today(), Account("B"), Quantity(-5))
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #35
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source is source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", object())
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #37
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=None
    )
    entry.post(datetime.date(2023, 1, 1), Account("Asset"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Liability"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #39
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #40
#--------------------------

```python
def test_posting_constructor():
    journal = "test_journal"
    date = datetime.date(2023, 1, 1)
    account = Account("test_account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #41
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period):
            return [JournalEntry("test")]

    reader = MockReadJournalEntries()
    entries = reader(DateRange("2023-01-01", "2023-01-31"))
    assert isinstance(entries, Iterable)
    assert len(list(entries)) == 1
    assert isinstance(entries[0], JournalEntry)


# LLM-generated content at query #42
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(Decimal('10')))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(Decimal('-5')))
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #43
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date.today(), "Test", None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(datetime.date.today(), "Test", None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(datetime.date.today(), "Test", None)
    entry.validate()

def test_validate_with_multiple_postings():
    entry = JournalEntry(datetime.date.today(), "Test", None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(200))
    entry.post(datetime.date.today(), Account("C"), Quantity(-300))
    entry.validate()


# LLM-generated content at query #44
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #45
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 0
    assert result is journal_entry


# LLM-generated content at query #46
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #47
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #48
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #49
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #50
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #51
#--------------------------

```python
def test_post_with_positive_quantity_adds_increment_posting():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal == entry
    assert posting.date == datetime.date(2023, 1, 1)
    assert posting.account == account
    assert posting.direction == Direction.INC
    assert posting.amount == Amount(100, "USD")
    assert result == entry

def test_post_with_negative_quantity_adds_decrement_posting():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(-100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal == entry
    assert posting.date == datetime.date(2023, 1, 1)
    assert posting.account == account
    assert posting.direction == Direction.DEC
    assert posting.amount == Amount(100, "USD")
    assert result == entry

def test_post_with_zero_quantity_adds_no_posting():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result == entry

def test_post_adds_multiple_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account1 = Account("Test Account 1", AccountType.ASSET)
    account2 = Account("Test Account 2", AccountType.LIABILITY)
    quantity1 = Quantity(100, "USD")
    quantity2 = Quantity(-50, "USD")
    entry.post(datetime.date(2023, 1, 1), account1, quantity1)
    entry.post(datetime.date(2023, 1, 2), account2, quantity2)
    assert len(entry.postings) == 2
    posting1 = entry.postings[0]
    assert posting1.journal == entry
    assert posting1.date == datetime.date(2023, 1, 1)
    assert posting1.account == account1
    assert posting1.direction == Direction.INC
    assert posting1.amount == Amount(100, "USD")
    posting2 = entry.postings[1]
    assert posting2.journal == entry
    assert posting2.date == datetime.date(2023, 1, 2)
    assert posting2.account == account2
    assert posting2.direction == Direction.DEC
    assert posting2.amount == Amount(50, "USD")


# LLM-generated content at query #52
#--------------------------

```python
def test_post_with_zero_quantity_does_not_add_posting():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 0


# LLM-generated content at query #53
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #54
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable_of_journal_entries():
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry]:
            return [JournalEntry(date=period.start, amount=100)]

    reader = MockReadJournalEntries()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    entries = reader(period)

    assert isinstance(entries, Iterable)
    assert all(isinstance(entry, JournalEntry) for entry in entries)


# LLM-generated content at query #55
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"

    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #56
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #57
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #58
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #59
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #60
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #61
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #62
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #63
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"

    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #64
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #65
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #66
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("Test1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test2"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("Test1"), Quantity(-100))
    entry.post(datetime.date(2023, 1, 1), Account("Test2"), Quantity(-200))
    entry.validate()


# LLM-generated content at query #67
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #68
#--------------------------

```python
def test_journal_entry_constructor():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    assert entry.date == datetime.date(2023, 1, 1)
    assert entry.description == "Test entry"
    assert entry.source == "Test source"
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #69
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=None
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #70
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #2
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("C"), Quantity(-300))
    entry.validate()


# LLM-generated content at query #4
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #6
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #7
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source=None
    )
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date.today(), "Test", None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #10
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #11
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #12
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #13
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return [JournalEntry("test")]

    reader = MockReadJournalEntries()
    result = reader(DateRange("2023-01-01", "2023-01-31"))
    assert isinstance(result, Iterable)
    assert list(result) == [JournalEntry("test")]


# LLM-generated content at query #14
#--------------------------

```python
def test_post_debit():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].date == datetime.date(2023, 1, 1)
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.INC
    assert entry.postings[0].amount == Amount(100)
    assert result is entry

def test_post_credit():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.LIABILITY)
    quantity = Quantity(-100)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].date == datetime.date(2023, 1, 1)
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.DEC
    assert entry.postings[0].amount == Amount(100)
    assert result is entry

def test_post_zero_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result is entry


# LLM-generated content at query #15
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #16
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry(date=datetime.date(2023, 1, 1), description="Test")
    account = Account(name="Test Account", type=AccountType.ASSET)
    posting = Posting(
        journal=journal,
        date=datetime.date(2023, 1, 1),
        account=account,
        direction=Direction.DEBIT,
        amount=Amount(100, "USD")
    )
    assert posting.journal == journal
    assert posting.date == datetime.date(2023, 1, 1)
    assert posting.account == account
    assert posting.direction == Direction.DEBIT
    assert posting.amount == Amount(100, "USD")


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="Test")
    entry.post(datetime.date(2023, 1, 1), Account("Asset"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Liability"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="Test")
    entry.post(datetime.date(2023, 1, 1), Account("Asset"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Liability"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="Test")
    entry.validate()

def test_validate_with_multiple_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="Test")
    entry.post(datetime.date(2023, 1, 1), Account("Asset1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Asset2"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("Liability1"), Quantity(-150))
    entry.post(datetime.date(2023, 1, 1), Account("Liability2"), Quantity(-150))
    entry.validate()


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_quantity_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(0))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(0))
    entry.validate()

def test_validate_with_multiple_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("C"), Quantity(-300))
    entry.validate()

def test_validate_with_no_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.validate()


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source=object()
    )
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-50))
    with pytest.raises(AssertionError, match="Total Debits and Credits are not equal"):
        entry.validate()


# LLM-generated content at query #21
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #22
#--------------------------

```python
def test_journal_entry_constructor_with_defaults():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    assert entry.date == datetime.date(2023, 1, 1)
    assert entry.description == "Test"
    assert entry.source == "TestSource"
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #23
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period):
            return [JournalEntry("test"), JournalEntry("test2")]

    reader = MockReadJournalEntries()
    result = reader(DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31)))
    assert isinstance(result, Iterable)
    assert len(list(result)) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(datetime.date.today(), "Test", object())
    entry.post(datetime.date.today(), Account("Test"), Quantity(10))
    entry.post(datetime.date.today(), Account("Test"), Quantity(-5))
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #26
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #27
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source="Test"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("10")))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("-5")))
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEC, Amount(Decimal("100"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.INC, Amount(Decimal("100")))
    ]
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEC, Amount(Decimal("100"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.INC, Amount(Decimal("50")))
    ]
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings = []
    entry.validate()

def test_validate_with_multiple_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEC, Amount(Decimal("100"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.DEC, Amount(Decimal("50"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("C"), Direction.INC, Amount(Decimal("150")))
    ]
    entry.validate()


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #31
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #32
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[str]]:
            return [JournalEntry("test", "data")]

    reader = MockReadJournalEntries()
    result = reader(DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31)))
    assert isinstance(result, Iterable)
    assert list(result) == [JournalEntry("test", "data")]


# LLM-generated content at query #33
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    period = DateRange(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    reader = ReadJournalEntries()
    result = reader(period)
    assert isinstance(result, Iterable)


# LLM-generated content at query #34
#--------------------------

```python
def test_post_with_non_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100)
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 1
    assert journal_entry.postings[0].journal == journal_entry
    assert journal_entry.postings[0].date == date(2023, 1, 1)
    assert journal_entry.postings[0].account == account
    assert journal_entry.postings[0].direction == Direction.INC
    assert journal_entry.postings[0].amount == Amount(100)
    assert result == journal_entry


# LLM-generated content at query #35
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #36
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"

    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #37
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account(type=AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #38
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #39
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #40
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #41
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry("Test Journal")
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #42
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #43
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #44
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #45
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #46
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #47
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #48
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #49
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #50
#--------------------------

```python
def test_post_with_non_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100)
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 1
    assert result.postings[0].amount.value == 100


# LLM-generated content at query #51
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #52
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #53
#--------------------------

```python
def test_post_debit():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].account == account
    assert entry.postings[0].amount == Amount(100, "USD")
    assert entry.postings[0].direction == Direction.INC
    assert result is entry

def test_post_credit():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.LIABILITY)
    quantity = Quantity(-100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].account == account
    assert entry.postings[0].amount == Amount(100, "USD")
    assert entry.postings[0].direction == Direction.DEC
    assert result is entry

def test_post_zero_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result is entry


# LLM-generated content at query #54
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #55
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #56
#--------------------------

```python
def test_post_with_zero_quantity_does_not_add_posting():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", object())
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 0


# LLM-generated content at query #57
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()

    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #58
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Amount(0)
    result = journal_entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #59
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #60
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #61
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #62
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #63
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #64
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #65
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #66
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", object())
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(Decimal("10")))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(Decimal("-5")))
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #67
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #68
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date.today(), description="Test", source=None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date.today(), description="Test", source=None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(date=datetime.date.today(), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_equal_postings():
    entry = JournalEntry(date=datetime.date.today(), description="Test", source=None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(50))
    entry.post(datetime.date.today(), Account("C"), Quantity(-150))
    entry.validate()


# LLM-generated content at query #69
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test Entry", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #70
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries(ReadJournalEntries[str]):
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[str]]:
            return [JournalEntry(date=period.start, value="test")]

    reader = MockReadJournalEntries()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = reader(period)

    assert isinstance(result, Iterable)
    assert len(list(result)) == 1
    assert list(result)[0].date == period.start
    assert list(result)[0].value == "test"


# LLM-generated content at query #71
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #72
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_not_equal_to_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEB, Amount(Decimal("10"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.CRED, Amount(Decimal("5")))
    ]
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #73
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #74
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #75
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


