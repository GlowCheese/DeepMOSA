####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_journal_entry_constructor_with_defaults():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()

    entry = JournalEntry(date, description, source)

    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)

def test_journal_entry_constructor_immutability():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()

    entry = JournalEntry(date, description, source)

    with pytest.raises(AttributeError):
        entry.date = datetime.date(2023, 1, 2)
    with pytest.raises(AttributeError):
        entry.description = "New description"
    with pytest.raises(AttributeError):
        entry.source = object()
    with pytest.raises(AttributeError):
        entry.postings = []
    with pytest.raises(AttributeError):
        entry.guid = Guid("new-guid")


# LLM-generated content at query #2
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

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=object()
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=object()
    )
    entry.validate()

def test_validate_with_multiple_equal_postings():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=object()
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 3"), Quantity(-300))
    entry.validate()


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEC, Amount(Decimal("10"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.INC, Amount(Decimal("20"))),
    ]
    with pytest.raises(AssertionError, match="Total Debits and Credits are not equal: 20 != 10"):
        entry.validate()


# LLM-generated content at query #4
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

def test_validate_with_no_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.validate()

def test_validate_with_zero_quantity_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(0))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(0))
    entry.validate()


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test Entry",
        source=object()
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(Decimal("-50")))
    with pytest.raises(AssertionError, match="Total Debits and Credits are not equal: 100 != 50"):
        entry.validate()


# LLM-generated content at query #6
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


# LLM-generated content at query #9
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("TestAccount", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #10
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
def test_post_debit():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.INC
    assert entry.postings[0].amount == Amount(100, "USD")
    assert result == entry

def test_post_credit():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.LIABILITY)
    quantity = Quantity(-100, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.DEC
    assert entry.postings[0].amount == Amount(100, "USD")
    assert result == entry

def test_post_zero_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result == entry


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
def test_validate_fails_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source=object()
    )
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEC, Amount(Decimal("10"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.INC, Amount(Decimal("5"))),
    ]
    with pytest.raises(AssertionError, match="Total Debits and Credits are not equal: 10 != 5"):
        entry.validate()


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
def test_validate_with_balanced_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source=None
    )
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEC, Amount(Decimal('100'))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.INC, Amount(Decimal('100')))
    ]
    entry.validate()


# LLM-generated content at query #18
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


# LLM-generated content at query #19
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


# LLM-generated content at query #20
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
def test_validate_with_balanced_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(-100))
    entry.validate()


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
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[str]]:
            return [JournalEntry("2023-01-01", "Test entry")]

    reader = MockReadJournalEntries()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    entries = reader(period)

    assert isinstance(entries, Iterable)
    assert len(list(entries)) == 1
    assert list(entries)[0].content == "Test entry"


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date.today(), description="Test", source=None)
    entry.post(datetime.date.today(), Account("Test"), Quantity(100))
    entry.post(datetime.date.today(), Account("Test"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #27
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    period = DateRange(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
    reader = ReadJournalEntries()
    result = reader(period)
    assert isinstance(result, Iterable)


# LLM-generated content at query #28
#--------------------------

```python
def test_journal_entry_constructor():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test entry", source="Test source")
    assert entry.date == datetime.date(2023, 1, 1)
    assert entry.description == "Test entry"
    assert entry.source == "Test source"
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #29
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


# LLM-generated content at query #30
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


# LLM-generated content at query #31
#--------------------------

```python
def test_post_with_zero_quantity_does_not_add_posting():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 0


# LLM-generated content at query #32
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


# LLM-generated content at query #33
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


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEB, Amount(Decimal("100"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.CRED, Amount(Decimal("100")))
    ]
    entry.validate()


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_with_balanced_postings():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=None
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #38
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
def test_read_journal_entries_call_returns_iterable():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    reader = ReadJournalEntries()
    result = reader(period)
    assert isinstance(result, Iterable)


# LLM-generated content at query #41
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
def test_validate_with_balanced_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(Decimal("-100")))
    entry.validate()

def test_validate_with_unbalanced_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(Decimal("-50")))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.validate()

def test_validate_with_multiple_balanced_postings():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(Decimal("50")))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 3"), Quantity(Decimal("-150")))
    entry.validate()


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
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account(type="asset")
    direction = Direction.DEBIT
    amount = Amount(100, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #46
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[str]]:
            return [JournalEntry("test", period.start, "Test entry")]

    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = reader(period)

    assert isinstance(result, Iterable)
    assert len(list(result)) == 1
    assert isinstance(result[0], JournalEntry)


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
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #2
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account(type=AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")
    posting = Posting(journal, date, account, direction, amount)
    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #3
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


# LLM-generated content at query #4
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

def test_validate_with_multiple_equal_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("C"), Quantity(-300))
    entry.validate()


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
    assert entry.source is source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_post_positive_quantity():
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

def test_post_negative_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(-100)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].journal == entry
    assert entry.postings[0].date == datetime.date(2023, 1, 1)
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.DEC
    assert entry.postings[0].amount == Amount(100)
    assert result == entry

def test_post_zero_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0)
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result == entry


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("-100")))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("-50")))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_with_zero_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_equal_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("Test1"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test2"), Quantity(Decimal("200")))
    entry.post(datetime.date(2023, 1, 1), Account("Test1"), Quantity(Decimal("-100")))
    entry.post(datetime.date(2023, 1, 1), Account("Test2"), Quantity(Decimal("-200")))
    entry.validate()


# LLM-generated content at query #10
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account("TestAccount", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0, "USD")

    posting = Posting(journal, date, account, direction, amount)

    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #11
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    def mock_read(period: DateRange) -> Iterable[JournalEntry[str]]:
        return [JournalEntry("entry1"), JournalEntry("entry2")]

    reader = ReadJournalEntries[str](mock_read)
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = reader(period)

    assert isinstance(result, Iterable)
    assert list(result) == [JournalEntry("entry1"), JournalEntry("entry2")]


# LLM-generated content at query #12
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[str]]:
            return [JournalEntry(date=date(2023, 1, 1), content="Test entry")]

    reader = MockReadJournalEntries()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = reader(period)

    assert isinstance(result, Iterable)
    assert len(list(result)) == 1
    assert list(result)[0].content == "Test entry"


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", object())
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(Decimal("10")))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(Decimal("-5")))
    assert_raises(AssertionError, entry.validate)


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


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_post_with_positive_quantity():
    entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100, "USD")
    result = entry.post(date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].journal == entry
    assert entry.postings[0].date == date(2023, 1, 1)
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.INC
    assert entry.postings[0].amount == Amount(100)
    assert result == entry

def test_post_with_negative_quantity():
    entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(-100, "USD")
    result = entry.post(date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    assert entry.postings[0].journal == entry
    assert entry.postings[0].date == date(2023, 1, 1)
    assert entry.postings[0].account == account
    assert entry.postings[0].direction == Direction.DEC
    assert entry.postings[0].amount == Amount(100)
    assert result == entry

def test_post_with_zero_quantity():
    entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = entry.post(date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result == entry


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("A"), Direction.DEC, Amount(Decimal("100"))),
        Posting(entry, datetime.date(2023, 1, 1), Account("B"), Direction.INC, Amount(Decimal("50"))),
    ]
    with pytest.raises(AssertionError, match="Total Debits and Credits are not equal: 100 != 50"):
        entry.validate()


# LLM-generated content at query #22
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

def test_validate_with_no_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_postings():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("Test1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test2"), Quantity(200))
    entry.post(datetime.date(2023, 1, 1), Account("Test3"), Quantity(-300))
    entry.validate()


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_posting_constructor_creates_immutable_instance():
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

    with pytest.raises(AttributeError):
        posting.journal = JournalEntry()

    with pytest.raises(AttributeError):
        posting.date = datetime.date(2023, 1, 2)

    with pytest.raises(AttributeError):
        posting.account = Account("Another Account", AccountType.LIABILITY)

    with pytest.raises(AttributeError):
        posting.direction = Direction.CREDIT

    with pytest.raises(AttributeError):
        posting.amount = Amount(200, "USD")


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity("10"))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity("-10"))
    entry.validate()


# LLM-generated content at query #26
#--------------------------

```python
def test_posting_constructor_creates_immutable_instance():
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

    with pytest.raises(FrozenInstanceError):
        posting.journal = JournalEntry()


# LLM-generated content at query #27
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


# LLM-generated content at query #28
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


# LLM-generated content at query #29
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


# LLM-generated content at query #30
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


# LLM-generated content at query #31
#--------------------------

```python
def test_post_with_zero_quantity_does_not_add_posting():
    journal_entry = JournalEntry(date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    reader = ReadJournalEntries()
    result = reader(period)
    assert isinstance(result, Iterable)


# LLM-generated content at query #34
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


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_with_balanced_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source="Test source"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 1"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test Account 2"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #36
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
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("Asset"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Liability"), Quantity(-100))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("Asset"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Liability"), Quantity(-50))
    try:
        entry.validate()
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 100 != 50"
    else:
        assert False, "Expected AssertionError"

def test_validate_with_no_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.validate()

def test_validate_with_zero_quantity_posting():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("Asset"), Quantity(0))
    entry.validate()


# LLM-generated content at query #40
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(date=datetime.date.today(), description="Test", source=None)
    entry.post(datetime.date.today(), Account("Test"), Quantity(Decimal("10")))
    entry.post(datetime.date.today(), Account("Test"), Quantity(Decimal("-5")))
    with pytest.raises(AssertionError):
        entry.validate()


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #43
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_are_not_equal():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", object())
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("10")))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(Decimal("-5")))
    with pytest.raises(AssertionError):
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
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(Decimal("-100")))
    entry.validate()


# LLM-generated content at query #46
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry("Test Journal")
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


# LLM-generated content at query #47
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


# LLM-generated content at query #48
#--------------------------

```python
def test_read_journal_entries_call_returns_iterable():
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    reader = ReadJournalEntries()
    result = reader(period)
    assert isinstance(result, Iterable)


# LLM-generated content at query #49
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


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
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", "Source")
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


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
def test_constructor_defaults():
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


# LLM-generated content at query #55
#--------------------------

```python
def test_post_with_zero_quantity():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", "Test Source")
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = journal_entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(result.postings) == 0


# LLM-generated content at query #56
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


# LLM-generated content at query #57
#--------------------------

```python
def test_post_with_zero_quantity():
    date = datetime.date(2023, 1, 1)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    journal_entry = JournalEntry(date, "Test Entry", None)
    result = journal_entry.post(date, account, quantity)
    assert len(result.postings) == 0


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
    source = object()
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)


# LLM-generated content at query #62
#--------------------------

```python
def test_post_with_positive_quantity():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(100)
    result = journal_entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 1
    assert journal_entry.postings[0].date == datetime.date(2023, 1, 1)
    assert journal_entry.postings[0].account == account
    assert journal_entry.postings[0].direction == Direction.INC
    assert journal_entry.postings[0].amount == Amount(100)
    assert result == journal_entry

def test_post_with_negative_quantity():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(-100)
    result = journal_entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 1
    assert journal_entry.postings[0].date == datetime.date(2023, 1, 1)
    assert journal_entry.postings[0].account == account
    assert journal_entry.postings[0].direction == Direction.DEC
    assert journal_entry.postings[0].amount == Amount(100)
    assert result == journal_entry

def test_post_with_zero_quantity():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0)
    result = journal_entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == 0
    assert result == journal_entry


# LLM-generated content at query #63
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


# LLM-generated content at query #64
#--------------------------

```python
def test_constructor_creates_empty_postings_list():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    assert entry.postings == []

def test_constructor_generates_guid():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    assert isinstance(entry.guid, Guid)

def test_constructor_sets_date():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    assert entry.date == datetime.date(2023, 1, 1)

def test_constructor_sets_description():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    assert entry.description == "Test"

def test_constructor_sets_source():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    assert entry.source == "TestSource"


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
def test_read_journal_entries_call_returns_iterable():
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[str]]:
            return [JournalEntry("entry1"), JournalEntry("entry2")]

    reader = MockReadJournalEntries()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))
    result = reader(period)

    assert isinstance(result, Iterable)
    assert len(list(result)) == 2


# LLM-generated content at query #69
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test entry",
        source=None
    )
    entry.postings = [
        Posting(entry, datetime.date(2023, 1, 1), Account("Test Account 1"), Direction.DEC, Amount(Decimal('100'))),
        Posting(entry, datetime.date(2023, 1, 1), Account("Test Account 2"), Direction.INC, Amount(Decimal('100')))
    ]
    entry.validate()


# LLM-generated content at query #70
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date.today(), "Test", None)
    entry.post(datetime.date.today(), Account("A"), Quantity(100))
    entry.post(datetime.date.today(), Account("B"), Quantity(-100))
    entry.validate()


# LLM-generated content at query #71
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


# LLM-generated content at query #72
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


# LLM-generated content at query #73
#--------------------------

```python
def test_validate_balanced_journal_entry():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source="Test"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(-100))
    entry.validate()

def test_validate_unbalanced_journal_entry():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source="Test"
    )
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(100))
    entry.post(datetime.date(2023, 1, 1), Account("Test"), Quantity(-50))
    with pytest.raises(AssertionError):
        entry.validate()

def test_validate_empty_journal_entry():
    entry = JournalEntry(
        date=datetime.date(2023, 1, 1),
        description="Test",
        source="Test"
    )
    entry.validate()


# LLM-generated content at query #74
#--------------------------

```python
def test_post_positive_quantity():
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

def test_post_negative_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(-50, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal == entry
    assert posting.date == datetime.date(2023, 1, 1)
    assert posting.account == account
    assert posting.direction == Direction.DEC
    assert posting.amount == Amount(50, "USD")
    assert result == entry

def test_post_zero_quantity():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("Test Account", AccountType.ASSET)
    quantity = Quantity(0, "USD")
    result = entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(entry.postings) == 0
    assert result == entry


# LLM-generated content at query #75
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


