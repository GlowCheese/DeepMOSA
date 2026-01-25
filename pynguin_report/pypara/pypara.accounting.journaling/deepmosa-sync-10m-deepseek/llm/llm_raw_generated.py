####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_journal_entry_constructor():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #2
#--------------------------

```python
def test_posting_constructor():
    journal = object()
    date = datetime.date(2023, 1, 1)
    account = Account("Cash", AccountType.ASSET)
    direction = Direction.INCREASE
    amount = Amount(100, "USD")
    
    posting = Posting(
        journal=journal,
        date=date,
        account=account,
        direction=direction,
        amount=amount
    )
    
    assert posting.journal is journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #3
#--------------------------

```python
def test_post_adds_posting_to_journal_entry():
    account = Account("Cash", AccountType.ASSET)
    journal_entry = JournalEntry(datetime.date(2023, 10, 1), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 10, 1), account, Quantity(100))
    assert len(journal_entry.postings) == 1
    posting = journal_entry.postings[0]
    assert posting.journal == journal_entry
    assert posting.date == datetime.date(2023, 10, 1)
    assert posting.account == account
    assert posting.direction == Direction.INC
    assert posting.amount == Amount(100)

def test_post_does_not_add_posting_for_zero_quantity():
    account = Account("Cash", AccountType.ASSET)
    journal_entry = JournalEntry(datetime.date(2023, 10, 1), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 10, 1), account, Quantity(0))
    assert len(journal_entry.postings) == 0

def test_post_adds_debit_posting_for_negative_quantity():
    account = Account("Cash", AccountType.ASSET)
    journal_entry = JournalEntry(datetime.date(2023, 10, 1), "Test Entry", "Source")
    journal_entry.post(datetime.date(2023, 10, 1), account, Quantity(-100))
    assert len(journal_entry.postings) == 1
    posting = journal_entry.postings[0]
    assert posting.journal == journal_entry
    assert posting.date == datetime.date(2023, 10, 1)
    assert posting.account == account
    assert posting.direction == Direction.DEC
    assert posting.amount == Amount(100)


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_balanced_journal_entry():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "Test Source"
    account1 = Account("Account1")
    account2 = Account("Account2")
    journal_entry = JournalEntry(date, description, source)
    journal_entry.post(date, account1, Quantity(Decimal("100")))
    journal_entry.post(date, account2, Quantity(Decimal("-100")))
    journal_entry.validate()

def test_validate_unbalanced_journal_entry_raises_assertion_error():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "Test Source"
    account1 = Account("Account1")
    account2 = Account("Account2")
    journal_entry = JournalEntry(date, description, source)
    journal_entry.post(date, account1, Quantity(Decimal("100")))
    journal_entry.post(date, account2, Quantity(Decimal("-50")))
    try:
        journal_entry.validate()
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for unbalanced journal entry"


# LLM-generated content at query #5
#--------------------------

```python
def test_posting_constructor():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account("Assets:Cash", AccountType.ASSETS)
    test_direction = Direction.INCREASE
    test_amount = Amount(100, "USD")

    posting = Posting(
        journal=mock_journal,
        date=test_date,
        account=test_account,
        direction=test_direction,
        amount=test_amount
    )

    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount


# LLM-generated content at query #6
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 10, 1)
    description = "Test Entry"
    source = "Test Source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #7
#--------------------------

```
def test_journal_entry_constructor_initializes_fields_correctly():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test description"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_journal_entry_constructor_with_empty_description_raises_error():
    test_date = datetime.date(2023, 1, 1)
    test_source = "Test source"
    try:
        JournalEntry(date=test_date, description="", source=test_source)
        assert False, "Should raise ValueError for empty description"
    except ValueError:
        pass

def test_journal_entry_is_immutable():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test description"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    
    try:
        entry.date = datetime.date(2023, 1, 2)
        assert False, "Should not be able to modify date"
    except dataclasses.FrozenInstanceError:
        pass
    
    try:
        entry.description = "New description"
        assert False, "Should not be able to modify description"
    except dataclasses.FrozenInstanceError:
        pass
    
    try:
        entry.source = "New source"
        assert False, "Should not be able to modify source"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #8
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
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #9
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account(AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100, "USD")
    
    posting = Posting(
        journal=journal,
        date=date,
        account=account,
        direction=direction,
        amount=amount
    )
    
    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 1, 1), Account("Assets:Bank"), Quantity(100))
    journal_entry.post(datetime.date(2023, 1, 1), Account("Expenses:Tax"), Quantity(-100))
    journal_entry.validate()

def test_validate_with_unequal_debits_and_credits():
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", "Test Source")
    journal_entry.post(datetime.date(2023, 1, 1), Account("Assets:Bank"), Quantity(100))
    journal_entry.post(datetime.date(2023, 1, 1), Account("Expenses:Tax"), Quantity(-50))
    try:
        journal_entry.validate()
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 100 != 50"
    else:
        assert False, "Expected AssertionError was not raised"


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_equal_debits_and_credits():
    date = datetime.date(2023, 1, 1)
    source = "source"
    account = Account("account")
    journal_entry = JournalEntry(date, "description", source)
    journal_entry.post(date, account, Quantity(Decimal("100.00")))
    journal_entry.post(date, account, Quantity(Decimal("-100.00")))
    journal_entry.validate()

def test_validate_with_unequal_debits_and_credits():
    date = datetime.date(2023, 1, 1)
    source = "source"
    account = Account("account")
    journal_entry = JournalEntry(date, "description", source)
    journal_entry.post(date, account, Quantity(Decimal("100.00")))
    journal_entry.post(date, account, Quantity(Decimal("-50.00")))
    try:
        journal_entry.validate()
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError but no exception was raised")


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_debits_and_credits_equal():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    entry.post(date=datetime.date(2023, 1, 1), account=Account("A1"), quantity=Quantity(Decimal(100)))
    entry.post(date=datetime.date(2023, 1, 1), account=Account("A2"), quantity=Quantity(Decimal(-100)))
    entry.validate()


# LLM-generated content at query #13
#--------------------------

```python
def test_ReadJournalEntries___call__():
    from datetime import date
    from typing import Iterable, Generic, TypeVar

    _T = TypeVar('_T')

    class JournalEntry(Generic[_T]):
        def __init__(self, content: _T):
            self.content = content

    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return [JournalEntry("entry1"), JournalEntry("entry2")]

    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    reader = MockReadJournalEntries()
    entries = reader(period)
    assert len(list(entries)) == 2
    assert all(isinstance(entry, JournalEntry) for entry in entries)


# LLM-generated content at query #14
#--------------------------

```
def test_journal_entry_constructor():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #15
#--------------------------

```python
def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 10, 1)
    account = Account(type=AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(value=100, currency="USD")
    posting = Posting(journal=journal, date=date, account=account, direction=direction, amount=amount)
    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #16
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 10, 1)
    description = "Test Entry"
    source = "Test Source"
    entry = JournalEntry(date, description, source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #17
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
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #18
#--------------------------

```
def test_validate_asserts_when_debits_and_credits_are_equal():
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime

    entry = JournalEntry(datetime.date(2023, 1, 1), "Test entry", None)
    account = Account("123", "Test Account")
    entry.post(datetime.date(2023, 1, 1), account, Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), account, Quantity(Decimal("-100")))
    entry.validate()

def test_validate_asserts_when_debits_and_credits_are_equal_with_multiple_postings():
    from decimal import Decimal
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime

    entry = JournalEntry(datetime.date(2023, 1, 1), "Test entry", None)
    account1 = Account("123", "Test Account 1")
    account2 = Account("456", "Test Account 2")
    entry.post(datetime.date(2023, 1, 1), account1, Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), account2, Quantity(Decimal("50")))
    entry.post(datetime.date(2023, 1, 1), account1, Quantity(Decimal("-100")))
    entry.post(datetime.date(2023, 1, 1), account2, Quantity(Decimal("-50")))
    entry.validate()


# LLM-generated content at query #19
#--------------------------

def test_validate_with_equal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(Decimal("-100")))
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(Decimal("100")))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(Decimal("-50")))
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass

def test_validate_with_zero_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", None)
    entry.validate()

def test_validate_with_multiple_equal_postings():
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test Entry", None)
    entry.post(datetime.date(2023, 1, 1), Account("A"), Quantity(Decimal("50")))
    entry.post(datetime.date(2023, 1, 1), Account("B"), Quantity(Decimal("50")))
    entry.post(datetime.date(2023, 1, 1), Account("C"), Quantity(Decimal("-75")))
    entry.post(datetime.date(2023, 1, 1), Account("D"), Quantity(Decimal("-25")))
    entry.validate()


# LLM-generated content at query #20
#--------------------------

def test_validate_balanced_journal_entry():
    from decimal import Decimal
    from pypara.accounting import Account, Amount, Direction, JournalEntry, Posting, Quantity
    import datetime

    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    source = object()
    date = datetime.date.today()
    journal_entry = JournalEntry(date, "Test Entry", source)
    journal_entry.post(date, account1, Quantity(Decimal("100")))
    journal_entry.post(date, account2, Quantity(Decimal("-100")))
    journal_entry.validate()

def test_validate_unbalanced_journal_entry_raises_assertion_error():
    from decimal import Decimal
    from pypara.accounting import Account, Amount, Direction, JournalEntry, Posting, Quantity
    import datetime

    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    source = object()
    date = datetime.date.today()
    journal_entry = JournalEntry(date, "Test Entry", source)
    journal_entry.post(date, account1, Quantity(Decimal("100")))
    journal_entry.post(date, account2, Quantity(Decimal("-50")))
    try:
        journal_entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_journal_entry_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from decimal import Decimal
    from datetime import date
    from pypara.accounting import Account, Amount, Direction, JournalEntry, Posting, Quantity

    # Create a journal entry with unequal debits and credits
    entry = JournalEntry(date(2023, 1, 1), "Test Entry", None)
    entry.postings.append(Posting(entry, date(2023, 1, 1), Account("1"), Direction.INC, Amount(Decimal("100"))))
    entry.postings.append(Posting(entry, date(2023, 1, 1), Account("2"), Direction.DEC, Amount(Decimal("50"))))
    
    # This should raise AssertionError because debits (100) != credits (50)
    try:
        entry.validate()
        assert False, "Expected AssertionError but none was raised"
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 100 != 50"


# LLM-generated content at query #22
#--------------------------

```python
def test_posting_constructor():
    journal_entry = "JournalEntry"
    date = datetime.date(2023, 10, 1)
    account = Account("Cash", AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(100.0)
    posting = Posting(journal_entry, date, account, direction, amount)
    assert posting.journal == journal_entry
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #23
#--------------------------

```python
def test_journal_entry_constructor():
    from datetime import date
    source = object()
    journal_entry = JournalEntry(date(2023, 10, 1), "Test Description", source)
    assert journal_entry.date == date(2023, 10, 1)
    assert journal_entry.description == "Test Description"
    assert journal_entry.source is source
    assert len(journal_entry.postings) == 0
    assert isinstance(journal_entry.guid, str)


# LLM-generated content at query #24
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 10, 1)
    description = "Test Journal Entry"
    source = "Test Source"
    journal_entry = JournalEntry(date=date, description=description, source=source)
    assert journal_entry.date == date
    assert journal_entry.description == description
    assert journal_entry.source == source
    assert len(journal_entry.postings) == 0
    assert isinstance(journal_entry.guid, str)


# LLM-generated content at query #25
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
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #26
#--------------------------

```python
def test_post_quantity_zero_does_not_add_posting():
    journal_entry = JournalEntry(date=datetime.date(2023, 10, 1), description="Test Entry", source="Test Source")
    account = Account(name="Test Account")
    quantity = Quantity(0)
    journal_entry.post(date=datetime.date(2023, 10, 1), account=account, quantity=quantity)
    assert len(journal_entry.postings) == 0


# LLM-generated content at query #27
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 10, 1)
    description = "Test Entry"
    source = object()
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #28
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 10, 1)
    description = "Test Entry"
    source = "Test Source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)


# LLM-generated content at query #29
#--------------------------

```python
def test_journal_entry_constructor():
    date = datetime.date(2023, 10, 1)
    description = "Test Journal Entry"
    source = "Test Source"
    journal_entry = JournalEntry(date=date, description=description, source=source)
    assert journal_entry.date == date
    assert journal_entry.description == description
    assert journal_entry.source == source
    assert len(journal_entry.postings) == 0
    assert isinstance(journal_entry.guid, str)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_journal_entry_constructor_with_minimal_arguments():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test Description"
    test_source = "Test Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test Description"
    test_source = "Test Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2023, 2, 1)
        assert False, "Should not be able to assign to frozen field 'date'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to assign to frozen field 'description'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to assign to frozen field 'source'"
    except dataclasses.FrozenInstanceError:
        pass

def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test Description"
    test_source = "Test Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])
        assert False, "Should not be able to pass 'postings' to constructor"
    except TypeError:
        pass

def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test Description"
    test_source = "Test Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid=Guid())
        assert False, "Should not be able to pass 'guid' to constructor"
    except TypeError:
        pass

def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test Description"
    source_int = 123
    entry_int = JournalEntry(date=test_date, description=test_description, source=source_int)
    assert entry_int.source == source_int
    source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=source_dict)
    assert entry_dict.source == source_dict
    class CustomSource:
        pass
    custom_source = CustomSource()
    entry_custom = JournalEntry(date=test_date, description=test_description, source=custom_source)
    assert entry_custom.source is custom_source


# LLM-generated content at query #2
#--------------------------

def test_journal_entry_constructor_with_default_values():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)

def test_journal_entry_constructor_with_custom_source_type():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = 12345
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source

def test_journal_entry_constructor_frozen_immutability():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date=date, description=description, source=source)
    try:
        entry.date = datetime.date(2023, 2, 1)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.description = "New description"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.source = "New source"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


# LLM-generated content at query #3
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    from pypara.commons.numbers import ONE
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    acc1 = Account("A1", "Asset")
    acc2 = Account("A2", "Liability")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount(ONE)))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc2, Direction.DEC, Amount(ONE)))
    je.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    from pypara.commons.numbers import ONE
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    acc1 = Account("A1", "Asset")
    acc2 = Account("A2", "Liability")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount(ONE)))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc2, Direction.DEC, Amount(ONE * 2)))
    try:
        je.validate()
        assert False
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_with_no_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    je.validate()

def test_validate_with_multiple_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    from pypara.commons.numbers import ONE
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    acc1 = Account("A1", "Asset")
    acc2 = Account("A2", "Liability")
    acc3 = Account("A3", "Equity")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount(ONE)))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc2, Direction.DEC, Amount(ONE)))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc3, Direction.INC, Amount(ONE)))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc3, Direction.DEC, Amount(ONE)))
    je.validate()

def test_validate_with_zero_amount_postings():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    from pypara.commons.numbers import ZERO
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    acc1 = Account("A1", "Asset")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount(ZERO)))
    je.validate()


# LLM-generated content at query #4
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    from decimal import Decimal
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(Decimal("100")))
    je.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(Decimal("-50")))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #5
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    from decimal import Decimal
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account 1")
    account2 = Account(code="A2", name="Account 2")
    je.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(Decimal("100")))
    je.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(Decimal("50")))
    try:
        je.validate()
        raised = False
    except AssertionError as e:
        raised = True
        message = str(e)
    assert raised is True
    assert "Total Debits and Credits are not equal:" in message


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    from decimal import Decimal
    from pypara.commons.numbers import isum

    class MockSource:
        pass

    source = MockSource()
    date = datetime.date(2023, 1, 1)
    je = JournalEntry(date, "Test", source)
    account1 = Account("A1", "Account 1")
    account2 = Account("A2", "Account 2")
    je.post(date, account1, Quantity(Decimal("100")))
    je.post(date, account2, Quantity(Decimal("50")))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #7
#--------------------------

def test_journal_entry_constructor_initializes_fields_correctly():
    mock_source = object()
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source is mock_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_journal_entry_constructor_with_different_date():
    mock_source = object()
    test_date = datetime.date(2024, 12, 31)
    test_description = "Year-end entry"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source is mock_source

def test_journal_entry_constructor_with_different_description():
    mock_source = object()
    test_date = datetime.date(2023, 5, 15)
    test_description = ""
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source is mock_source

def test_journal_entry_constructor_with_different_source():
    test_source = "A string source"
    test_date = datetime.date(2023, 7, 4)
    test_description = "String source entry"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source

def test_journal_entry_is_immutable():
    mock_source = object()
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=mock_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2024, 1, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "Modified"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = object()


# LLM-generated content at query #8
#--------------------------

def test_journal_entry_constructor_with_default_values():
    from datetime import date
    from dataclasses import FrozenInstanceError
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source="BusinessObject")
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test"
    assert entry.source == "BusinessObject"
    assert entry.postings == []
    assert isinstance(entry.guid, str)
    try:
        entry.date = date(2023, 1, 2)
        assert False
    except FrozenInstanceError:
        assert True

def test_journal_entry_constructor_with_custom_source_type():
    from datetime import date
    class CustomSource:
        pass
    custom_source = CustomSource()
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source=custom_source)
    assert entry.source is custom_source

def test_journal_entry_constructor_ensures_frozen():
    from datetime import date
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source=123)
    try:
        entry.description = "Modified"
        assert False
    except FrozenInstanceError:
        assert True


# LLM-generated content at query #9
#--------------------------

def test_post_positive_quantity_increment():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account = Account("123", "Test Account", "ASSET")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), account, Quantity(100))
    posting = entry.postings[0]
    assert posting.date == datetime.date(2023, 1, 1)
    assert posting.account == account
    assert posting.direction.name == "INC"
    assert posting.amount == Amount(100)

def test_post_negative_quantity_decrement():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account = Account("123", "Test Account", "ASSET")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 2), account, Quantity(-50))
    posting = entry.postings[0]
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account == account
    assert posting.direction.name == "DEC"
    assert posting.amount == Amount(50)

def test_post_zero_quantity_no_posting():
    from pypara.accounting.journaling import JournalEntry, Account, Quantity
    import datetime
    account = Account("123", "Test Account", "ASSET")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), account, Quantity(0))
    assert len(entry.postings) == 0

def test_post_multiple_postings():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account1 = Account("123", "Test Account 1", "ASSET")
    account2 = Account("456", "Test Account 2", "LIABILITY")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 1), account1, Quantity(100))
    entry.post(datetime.date(2023, 1, 1), account2, Quantity(-100))
    assert len(entry.postings) == 2
    assert entry.postings[0].amount == Amount(100)
    assert entry.postings[1].amount == Amount(100)

def test_post_chainable():
    from pypara.accounting.journaling import JournalEntry, Account, Quantity
    import datetime
    account = Account("123", "Test Account", "ASSET")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    result = entry.post(datetime.date(2023, 1, 1), account, Quantity(100))
    assert result is entry
    assert len(entry.postings) == 1


# LLM-generated content at query #10
#--------------------------

def test_constructor_initializes_fields_correctly():
    mock_source = object()
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source is mock_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_constructor_with_different_source_types():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    source_str = "string_source"
    entry_str = JournalEntry(date=test_date, description=test_description, source=source_str)
    assert entry_str.source == source_str
    source_int = 123
    entry_int = JournalEntry(date=test_date, description=test_description, source=source_int)
    assert entry_int.source == source_int

def test_postings_list_is_empty_and_mutable():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    assert entry.postings == []
    entry.postings.append("test_posting")
    assert len(entry.postings) == 1
    assert entry.postings[0] == "test_posting"

def test_guid_is_unique_for_each_instance():
    entry1 = JournalEntry(date=datetime.date(2023, 1, 1), description="Test1", source=None)
    entry2 = JournalEntry(date=datetime.date(2023, 1, 1), description="Test2", source=None)
    assert entry1.guid != entry2.guid

def test_constructor_with_frozen_dataclass_behavior():
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=object())
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to frozen field"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New description"
        assert False, "Should not be able to assign to frozen field"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = object()
        assert False, "Should not be able to assign to frozen field"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #11
#--------------------------

def test_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Source object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert entry.postings == []
    assert isinstance(entry.guid, Guid)

def test_constructor_with_frozen_dataclass():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = 123
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "Modified"

def test_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = {"key": "value"}
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []

def test_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = None
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #12
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "TestSourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "TestSourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 10, 6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "New description"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = "NewSource"


def test_journal_entry_constructor_creates_unique_guids():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "TestSourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #13
#--------------------------

def test_posting_constructor():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency="USD")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount


# LLM-generated content at query #14
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 10, 6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "New Description"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = "New Source"


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    with pytest.raises(TypeError):
        JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    with pytest.raises(TypeError):
        JournalEntry(date=test_date, description=test_description, source=test_source, guid=makeguid())


# LLM-generated content at query #15
#--------------------------

def test_post_positive_quantity_increment():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account = Account("123", "Test Account", "ASSET")
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    quantity = Quantity(100)
    result = journal_entry.post(datetime.date(2023, 1, 2), account, quantity)
    assert result is journal_entry
    assert len(journal_entry.postings) == 1
    posting = journal_entry.postings[0]
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account == account
    assert posting.direction.name == "INC"
    assert posting.amount == Amount(100)

def test_post_negative_quantity_decrement():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account = Account("123", "Test Account", "ASSET")
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    quantity = Quantity(-50)
    result = journal_entry.post(datetime.date(2023, 1, 3), account, quantity)
    assert result is journal_entry
    assert len(journal_entry.postings) == 1
    posting = journal_entry.postings[0]
    assert posting.date == datetime.date(2023, 1, 3)
    assert posting.account == account
    assert posting.direction.name == "DEC"
    assert posting.amount == Amount(50)

def test_post_zero_quantity_no_posting():
    from pypara.accounting.journaling import JournalEntry, Account, Quantity
    import datetime
    account = Account("123", "Test Account", "ASSET")
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    quantity = Quantity(0)
    result = journal_entry.post(datetime.date(2023, 1, 4), account, quantity)
    assert result is journal_entry
    assert len(journal_entry.postings) == 0

def test_post_multiple_postings():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account1 = Account("123", "Test Account 1", "ASSET")
    account2 = Account("456", "Test Account 2", "LIABILITY")
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    journal_entry.post(datetime.date(2023, 1, 2), account1, Quantity(100))
    journal_entry.post(datetime.date(2023, 1, 3), account2, Quantity(-50))
    assert len(journal_entry.postings) == 2
    posting1 = journal_entry.postings[0]
    posting2 = journal_entry.postings[1]
    assert posting1.date == datetime.date(2023, 1, 2)
    assert posting1.account == account1
    assert posting1.direction.name == "INC"
    assert posting1.amount == Amount(100)
    assert posting2.date == datetime.date(2023, 1, 3)
    assert posting2.account == account2
    assert posting2.direction.name == "DEC"
    assert posting2.amount == Amount(50)

def test_post_chaining():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account1 = Account("123", "Test Account 1", "ASSET")
    account2 = Account("456", "Test Account 2", "LIABILITY")
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    result = journal_entry.post(datetime.date(2023, 1, 2), account1, Quantity(100)).post(datetime.date(2023, 1, 3), account2, Quantity(-50))
    assert result is journal_entry
    assert len(journal_entry.postings) == 2
    posting1 = journal_entry.postings[0]
    posting2 = journal_entry.postings[1]
    assert posting1.date == datetime.date(2023, 1, 2)
    assert posting1.account == account1
    assert posting1.direction.name == "INC"
    assert posting1.amount == Amount(100)
    assert posting2.date == datetime.date(2023, 1, 3)
    assert posting2.account == account2
    assert posting2.direction.name == "DEC"
    assert posting2.amount == Amount(50)


# LLM-generated content at query #16
#--------------------------

def test_post_with_zero_quantity_does_not_add_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.generic import Account, AccountType
    from pypara.currencies import Currency
    from decimal import Decimal
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("123", "Test Account", AccountType.ASSETS, Currency("USD"))
    quantity = Decimal("0")
    je.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(je.postings) == 0


# LLM-generated content at query #17
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2023, 10, 6)
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])
        assert False, "Should not be able to pass 'postings' to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid=Guid())
        assert False, "Should not be able to pass 'guid' to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=source_int)
    assert entry_int.source == source_int
    source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=source_dict)
    assert entry_dict.source == source_dict
    class CustomSource:
        pass
    custom_source = CustomSource()
    entry_custom = JournalEntry(date=test_date, description=test_description, source=custom_source)
    assert entry_custom.source is custom_source


# LLM-generated content at query #18
#--------------------------

def test_read_journal_entries_call():
    from typing import Iterable
    from datetime import date
    from typing import Protocol, TypeVar
    _T = TypeVar('_T')
    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end
    class JournalEntry:
        def __init__(self, data: _T):
            self.data = data
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry]:
            return [JournalEntry("test1"), JournalEntry("test2")]
    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = list(reader(period))
    assert len(result) == 2
    assert result[0].data == "test1"
    assert result[1].data == "test2"


# LLM-generated content at query #19
#--------------------------

def test___call___returns_iterable_of_journal_entries():
    class MockReadJournalEntries:
        def __call__(self, period):
            return [JournalEntry(period.start), JournalEntry(period.end)]

    mock_reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = mock_reader(period)
    assert isinstance(result, Iterable)
    entries = list(result)
    assert len(entries) == 2
    assert all(isinstance(entry, JournalEntry) for entry in entries)
    assert entries[0].date == period.start
    assert entries[1].date == period.end

def test___call___handles_empty_period():
    class MockReadJournalEntries:
        def __call__(self, period):
            return []

    mock_reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 1))
    result = mock_reader(period)
    entries = list(result)
    assert len(entries) == 0

def test___call___propagates_exceptions():
    class MockReadJournalEntries:
        def __call__(self, period):
            raise ValueError("Invalid period")

    mock_reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    try:
        mock_reader(period)
        assert False
    except ValueError as e:
        assert str(e) == "Invalid period"

def test___call___supports_lazy_iteration():
    call_count = 0
    def generator_func(period):
        nonlocal call_count
        call_count += 1
        yield JournalEntry(period.start)
        call_count += 1
        yield JournalEntry(period.end)

    class MockReadJournalEntries:
        def __call__(self, period):
            return generator_func(period)

    mock_reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = mock_reader(period)
    assert call_count == 0
    iterator = iter(result)
    assert call_count == 0
    first_entry = next(iterator)
    assert call_count == 1
    assert isinstance(first_entry, JournalEntry)
    assert first_entry.date == period.start
    second_entry = next(iterator)
    assert call_count == 2
    assert isinstance(second_entry, JournalEntry)
    assert second_entry.date == period.end
    try:
        next(iterator)
        assert False
    except StopIteration:
        pass


# LLM-generated content at query #20
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2023, 10, 6)
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New description"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New source"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])
        assert False, "Should not be able to pass postings to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid=makeguid())
        assert False, "Should not be able to pass guid to constructor"
    except TypeError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    class DummySource:
        pass
    source = DummySource()
    date = datetime.date(2023, 1, 1)
    account1 = Account.of("1234")
    account2 = Account.of("5678")
    journal_entry = JournalEntry(date, "Test entry", source)
    journal_entry.post(date, account1, Quantity(ONE))
    journal_entry.post(date, account2, Quantity(ONE))
    try:
        journal_entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #22
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    from pypara.commons.zeitgeist import Date
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currency
    from pypara.commons.numbers import Quantity
    dummy_date = datetime.date(2023, 1, 1)
    dummy_account = Account.of("1234", Currency("USD", 2))
    dummy_source = object()
    journal = JournalEntry(date=dummy_date, description="Test", source=dummy_source)
    posting_debit = Posting(journal, dummy_date, dummy_account, Direction.INC, Amount(Decimal("100")))
    posting_credit = Posting(journal, dummy_date, dummy_account, Direction.DEC, Amount(Decimal("50")))
    journal.postings.append(posting_debit)
    journal.postings.append(posting_credit)
    try:
        journal.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #23
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2023, 10, 6)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.description = "New Description"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.source = "New Source"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=source_int)
    assert entry_int.source == source_int
    source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=source_dict)
    assert entry_dict.source == source_dict
    class CustomSource:
        pass
    custom_source = CustomSource()
    entry_custom = JournalEntry(date=test_date, description=test_description, source=custom_source)
    assert entry_custom.source is custom_source


# LLM-generated content at query #24
#--------------------------

def test_journal_entry_constructor_with_default_values():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = "Test source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_journal_entry_constructor_with_custom_source_type():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = 12345
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, str)

def test_journal_entry_constructor_immutability_check():
    date = datetime.date(2023, 1, 1)
    description = "Test entry"
    source = object()
    entry = JournalEntry(date=date, description=description, source=source)
    try:
        entry.date = datetime.date(2023, 2, 1)
        assert False, "Should not be able to modify date"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "Modified"
        assert False, "Should not be able to modify description"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "Modified"
        assert False, "Should not be able to modify source"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #25
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    dummy_date = datetime.date(2023, 1, 1)
    dummy_source = object()
    dummy_account = Account("123", "Test Account")
    journal_entry = JournalEntry(dummy_date, "Test Entry", dummy_source)
    posting_debit = Posting(journal_entry, dummy_date, dummy_account, Direction.INC, Amount(ONE))
    posting_credit = Posting(journal_entry, dummy_date, dummy_account, Direction.DEC, Amount(ONE * 2))
    journal_entry.postings.append(posting_debit)
    journal_entry.postings.append(posting_credit)
    try:
        journal_entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #26
#--------------------------

```python
def test_post_with_zero_quantity_does_not_add_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.accounting.generic import Account, AccountType
    from pypara.currencies import Currency
    from decimal import Decimal
    import datetime

    currency = Currency("USD", 2)
    zero_quantity = Decimal("0.00")
    quantity = type("Quantity", (), {"is_zero": lambda self: True, "__abs__": lambda self: self})(zero_quantity)
    account = Account("cash", AccountType.ASSET, currency)
    journal_entry = JournalEntry(datetime.date(2023, 1, 1), "Test entry", None)
    original_postings_count = len(journal_entry.postings)
    journal_entry.post(datetime.date(2023, 1, 1), account, quantity)
    assert len(journal_entry.postings) == original_postings_count


# LLM-generated content at query #27
#--------------------------

def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account(name="Cash", type=AccountType.ASSET)
    direction = Direction.INCREASE
    amount = Amount(value=100, currency="USD")
    posting = Posting(journal=journal, date=date, account=account, direction=direction, amount=amount)
    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #28
#--------------------------

def test_validate_asserts_when_total_debits_and_credits_are_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.numbers import ZERO, ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Test Account")
    entry.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(-ONE))
    entry.validate()


# LLM-generated content at query #29
#--------------------------

def test_journal_entry_constructor_with_default_values():
    from datetime import date
    from dataclasses import FrozenInstanceError
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source="BusinessObject")
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test"
    assert entry.source == "BusinessObject"
    assert entry.postings == []
    assert isinstance(entry.guid, str)
    try:
        entry.date = date(2023, 1, 2)
        assert False
    except FrozenInstanceError:
        assert True

def test_journal_entry_constructor_with_custom_source_type():
    from datetime import date
    class CustomSource:
        pass
    custom_source = CustomSource()
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source=custom_source)
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test"
    assert entry.source is custom_source
    assert entry.postings == []
    assert isinstance(entry.guid, str)

def test_journal_entry_constructor_with_different_date_and_description():
    from datetime import date
    entry = JournalEntry(date=date(2024, 12, 31), description="Another Description", source=123)
    assert entry.date == date(2024, 12, 31)
    assert entry.description == "Another Description"
    assert entry.source == 123
    assert entry.postings == []
    assert isinstance(entry.guid, str)


# LLM-generated content at query #30
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account1")
    account2 = Account(code="A2", name="Account2")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(-ONE))
    entry.validate()

def test_validate_with_zero_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE, TWO
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account1")
    account2 = Account(code="A2", name="Account2")
    account3 = Account(code="A3", name="Account3")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account3, quantity=Quantity(-TWO))
    entry.validate()

def test_validate_raises_assertion_error_on_inequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE, TWO
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account1")
    account2 = Account(code="A2", name="Account2")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(-TWO))
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_journal_entry_constructor_with_minimal_fields():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "TestSource"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_journal_entry_constructor_with_different_source_types():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source_int = 123
    entry_int = JournalEntry(date=date, description=description, source=source_int)
    assert entry_int.source == source_int
    source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=date, description=description, source=source_dict)
    assert entry_dict.source == source_dict

def test_journal_entry_is_frozen():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "TestSource"
    entry = JournalEntry(date=date, description=description, source=source)
    try:
        entry.date = datetime.date(2023, 2, 1)
        assert False, "Should not be able to modify frozen instance"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to modify frozen instance"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to modify frozen instance"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #2
#--------------------------

def test_journalentry_constructor_with_default_values():
    from datetime import date
    from dataclasses import FrozenInstanceError
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source="BusinessObject")
    assert entry.date == date(2023, 1, 1)
    assert entry.description == "Test"
    assert entry.source == "BusinessObject"
    assert entry.postings == []
    assert isinstance(entry.guid, str)
    try:
        entry.date = date(2023, 1, 2)
        assert False
    except FrozenInstanceError:
        assert True

def test_journalentry_constructor_with_custom_source_type():
    from datetime import date
    class CustomSource:
        pass
    custom_source = CustomSource()
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source=custom_source)
    assert entry.source is custom_source

def test_journalentry_constructor_date_must_be_date():
    from datetime import date
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    assert isinstance(entry.date, date)

def test_journalentry_constructor_description_must_be_string():
    from datetime import date
    entry = JournalEntry(date=date(2023, 1, 1), description="", source=None)
    assert isinstance(entry.description, str)

def test_journalentry_constructor_guid_is_unique():
    from datetime import date
    entry1 = JournalEntry(date=date(2023, 1, 1), description="Test1", source="Source1")
    entry2 = JournalEntry(date=date(2023, 1, 2), description="Test2", source="Source2")
    assert entry1.guid != entry2.guid

def test_journalentry_constructor_postings_initialized_empty():
    from datetime import date
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source="Source")
    assert len(entry.postings) == 0


# LLM-generated content at query #3
#--------------------------

def test_post_positive_quantity_appends_increment_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime
    from decimal import Decimal
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("123", "Test Account")
    quantity = Decimal("100.00")
    result = je.post(datetime.date(2023, 1, 2), account, quantity)
    assert result is je
    assert len(je.postings) == 1
    posting = je.postings[0]
    assert posting.journal is je
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account is account
    assert posting.direction == Direction.INC
    assert posting.amount == Amount(quantity)

def test_post_negative_quantity_appends_decrement_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime
    from decimal import Decimal
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("123", "Test Account")
    quantity = Decimal("-100.00")
    result = je.post(datetime.date(2023, 1, 2), account, quantity)
    assert result is je
    assert len(je.postings) == 1
    posting = je.postings[0]
    assert posting.journal is je
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account is account
    assert posting.direction == Direction.DEC
    assert posting.amount == Amount(abs(quantity))

def test_post_zero_quantity_does_nothing():
    from pypara.accounting.journaling import JournalEntry, Account
    import datetime
    from decimal import Decimal
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("123", "Test Account")
    quantity = Decimal("0.00")
    result = je.post(datetime.date(2023, 1, 2), account, quantity)
    assert result is je
    assert len(je.postings) == 0

def test_post_multiple_postings_accumulate():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime
    from decimal import Decimal
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account1 = Account("123", "Test Account 1")
    account2 = Account("456", "Test Account 2")
    quantity1 = Decimal("50.00")
    quantity2 = Decimal("-30.00")
    je.post(datetime.date(2023, 1, 2), account1, quantity1)
    je.post(datetime.date(2023, 1, 3), account2, quantity2)
    assert len(je.postings) == 2
    posting1 = je.postings[0]
    posting2 = je.postings[1]
    assert posting1.account is account1
    assert posting1.direction == Direction.INC
    assert posting1.amount == Amount(quantity1)
    assert posting2.account is account2
    assert posting2.direction == Direction.DEC
    assert posting2.amount == Amount(abs(quantity2))

def test_post_same_date_and_account_multiple_times():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime
    from decimal import Decimal
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    account = Account("123", "Test Account")
    quantity1 = Decimal("100.00")
    quantity2 = Decimal("200.00")
    je.post(datetime.date(2023, 1, 2), account, quantity1)
    je.post(datetime.date(2023, 1, 2), account, quantity2)
    assert len(je.postings) == 2
    posting1 = je.postings[0]
    posting2 = je.postings[1]
    assert posting1.date == datetime.date(2023, 1, 2)
    assert posting1.account is account
    assert posting1.direction == Direction.INC
    assert posting1.amount == Amount(quantity1)
    assert posting2.date == datetime.date(2023, 1, 2)
    assert posting2.account is account
    assert posting2.direction == Direction.INC
    assert posting2.amount == Amount(quantity2)


# LLM-generated content at query #4
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zeitgeist import Date
    from decimal import Decimal
    je = JournalEntry(Date(2023, 1, 1), "Test", None)
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.INC, Amount(Decimal('100'))))
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('100'))))
    je.validate()

def test_validate_with_zero_postings():
    from pypara.accounting.journaling import JournalEntry
    from pypara.commons.zeitgeist import Date
    je = JournalEntry(Date(2023, 1, 1), "Test", None)
    je.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zeitgeist import Date
    from decimal import Decimal
    je = JournalEntry(Date(2023, 1, 1), "Test", None)
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.INC, Amount(Decimal('100'))))
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('50'))))
    try:
        je.validate()
        assert False
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 100 != 50"

def test_validate_with_multiple_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zeitgeist import Date
    from decimal import Decimal
    je = JournalEntry(Date(2023, 1, 1), "Test", None)
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.INC, Amount(Decimal('30'))))
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.INC, Amount(Decimal('70'))))
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('40'))))
    je.postings.append(Posting(je, Date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('60'))))
    je.validate()


# LLM-generated content at query #5
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account.of("A1")
    account2 = Account.of("A2")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(ZERO))
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #6
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "New Description"


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == 42
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == {"key": "value"}


# LLM-generated content at query #7
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "TestSourceObject"
    entry = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source_int = 123
    entry_int = JournalEntry[int](date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry[dict](date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict


def test_journal_entry_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "TestSource"
    entry = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.description = "New Description"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.source = "New Source"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


# LLM-generated content at query #8
#--------------------------

def test_validate_assertion_true_when_debits_equal_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account 1")
    account2 = Account(code="A2", name="Account 2")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(Decimal("100")))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(Decimal("-100")))
    entry.validate()


# LLM-generated content at query #9
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2023, 10, 6)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.description = "New Description"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.source = "New Source"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=source_int)
    assert entry_int.source == source_int
    source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=source_dict)
    assert entry_dict.source == source_dict
    class CustomSource:
        pass
    custom_source = CustomSource()
    entry_custom = JournalEntry(date=test_date, description=test_description, source=custom_source)
    assert entry_custom.source is custom_source


def test_journal_entry_constructor_guid_is_unique():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


def test_journal_entry_constructor_postings_list_is_independent():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.postings is not entry2.postings
    entry1.postings.append("test")
    assert len(entry1.postings) == 1
    assert len(entry2.postings) == 0


# LLM-generated content at query #10
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])
        assert False, "Should not be able to pass postings to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid=makeguid())
        assert False, "Should not be able to pass guid to constructor"
    except TypeError:
        pass


# LLM-generated content at query #11
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    import decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.postings.append(Posting(je, date(2023, 1, 1), account, Direction.INC, Amount(decimal.Decimal("100"))))
    je.postings.append(Posting(je, date(2023, 1, 1), account, Direction.DEC, Amount(decimal.Decimal("50"))))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_passes_when_debits_and_credits_are_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    import decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.postings.append(Posting(je, date(2023, 1, 1), account, Direction.INC, Amount(decimal.Decimal("100"))))
    je.postings.append(Posting(je, date(2023, 1, 1), account, Direction.DEC, Amount(decimal.Decimal("100"))))
    je.validate()

def test_validate_passes_with_multiple_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    import decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="1000", name="Cash")
    account2 = Account(code="2000", name="Revenue")
    je.postings.append(Posting(je, date(2023, 1, 1), account1, Direction.INC, Amount(decimal.Decimal("150"))))
    je.postings.append(Posting(je, date(2023, 1, 1), account2, Direction.DEC, Amount(decimal.Decimal("75"))))
    je.postings.append(Posting(je, date(2023, 1, 1), account2, Direction.DEC, Amount(decimal.Decimal("75"))))
    je.validate()

def test_validate_passes_when_no_postings():
    from pypara.accounting.journaling import JournalEntry
    from datetime import date
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    je.validate()

def test_validate_raises_assertion_error_with_only_debits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    import decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.postings.append(Posting(je, date(2023, 1, 1), account, Direction.INC, Amount(decimal.Decimal("100"))))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_raises_assertion_error_with_only_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    import decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.postings.append(Posting(je, date(2023, 1, 1), account, Direction.DEC, Amount(decimal.Decimal("100"))))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #12
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 10, 6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "New description"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = "New source"


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    with pytest.raises(TypeError):
        JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    with pytest.raises(TypeError):
        JournalEntry(date=test_date, description=test_description, source=test_source, guid=makeguid())


# LLM-generated content at query #13
#--------------------------

def test_journal_entry_constructor_with_minimal_fields():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "Test Source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_journal_entry_constructor_with_different_source_types():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source_int = 123
    entry_int = JournalEntry(date=date, description=description, source=source_int)
    assert entry_int.source == source_int
    source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=date, description=description, source=source_dict)
    assert entry_dict.source == source_dict

def test_journal_entry_is_immutable():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "Test Source"
    entry = JournalEntry(date=date, description=description, source=source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 2, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "New Description"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = "New Source"

def test_journal_entry_guid_is_unique():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "Test Source"
    entry1 = JournalEntry(date=date, description=description, source=source)
    entry2 = JournalEntry(date=date, description=description, source=source)
    assert entry1.guid != entry2.guid

def test_journal_entry_postings_list_is_initially_empty():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "Test Source"
    entry = JournalEntry(date=date, description=description, source=source)
    assert len(entry.postings) == 0
    assert isinstance(entry.postings, list)


# LLM-generated content at query #14
#--------------------------

def test_posting_constructor_initializes_fields_correctly():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 15)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency="USD")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount

def test_posting_constructor_with_different_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 2, 20)
    test_account = Account(name="Revenue", type=AccountType.REVENUE)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("50.00"), currency="EUR")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount

def test_posting_is_frozen_and_immutable():
    mock_journal = object()
    test_date = datetime.date(2023, 3, 10)
    test_account = Account(name="Expense", type=AccountType.EXPENSE)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("75.00"), currency="GBP")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    try:
        posting.date = datetime.date(2023, 4, 1)
        assert False, "Should not be able to modify attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        posting.account = Account(name="New", type=AccountType.LIABILITY)
        assert False, "Should not be able to modify attribute"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #15
#--------------------------

def test_validate_assertion_true_when_debits_equal_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.numbers import ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="1000", name="Asset")
    account2 = Account(code="2000", name="Liability")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(-ONE))
    entry.validate()


# LLM-generated content at query #16
#--------------------------

def test_posting_constructor():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency="USD")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount


# LLM-generated content at query #17
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to frozen instance attribute"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to assign to frozen instance attribute"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to assign to frozen instance attribute"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, postings=[Posting(None, test_date, Account("A"), Direction.INC, Amount(Quantity(10)))])
        assert False, "Should not be able to pass postings to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid=Guid())
        assert False, "Should not be able to pass guid to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    entry_int = JournalEntry(date=test_date, description=test_description, source=123)
    assert entry_int.source == 123
    entry_dict = JournalEntry(date=test_date, description=test_description, source={"key": "value"})
    assert entry_dict.source == {"key": "value"}
    class CustomSource:
        pass
    custom_obj = CustomSource()
    entry_custom = JournalEntry(date=test_date, description=test_description, source=custom_obj)
    assert entry_custom.source is custom_obj


# LLM-generated content at query #18
#--------------------------

def test_constructor_initializes_fields_correctly():
    from datetime import date
    from dataclasses import FrozenInstanceError
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_constructor_raises_error_when_missing_required_fields():
    from datetime import date
    try:
        JournalEntry()
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_constructor_is_immutable():
    from datetime import date
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source="Source")
    try:
        entry.date = date(2023, 2, 1)
    except FrozenInstanceError:
        pass
    else:
        assert False, "Expected FrozenInstanceError"

def test_constructor_with_different_source_types():
    from datetime import date
    source_int = 123
    entry_int = JournalEntry(date=date(2023, 1, 1), description="Int source", source=source_int)
    assert entry_int.source == source_int
    source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=date(2023, 1, 1), description="Dict source", source=source_dict)
    assert entry_dict.source == source_dict

def test_constructor_postings_list_is_empty_by_default():
    from datetime import date
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source="Source")
    assert entry.postings == []

def test_constructor_guid_is_unique():
    from datetime import date
    entry1 = JournalEntry(date=date(2023, 1, 1), description="Test1", source="Source1")
    entry2 = JournalEntry(date=date(2023, 1, 1), description="Test2", source="Source2")
    assert entry1.guid != entry2.guid


# LLM-generated content at query #19
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    from pypara.commons.numbers import Quantity
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    entry.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(Decimal("100")))
    entry.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(Decimal("-50")))
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #20
#--------------------------

def test_posting_constructor_initializes_fields_correctly():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency="USD")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount

def test_posting_constructor_with_debit_account_and_increase_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 2, 15)
    test_account = Account(name="Equipment", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("500.00"), currency="EUR")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_constructor_with_credit_account_and_increase_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 3, 10)
    test_account = Account(name="Loan", type=AccountType.LIABILITY)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("200.00"), currency="GBP")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_constructor_with_debit_account_and_decrease_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 4, 20)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("50.00"), currency="USD")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_constructor_with_credit_account_and_decrease_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 5, 5)
    test_account = Account(name="Revenue", type=AccountType.REVENUE)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("300.00"), currency="JPY")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_is_frozen_and_immutable():
    mock_journal = object()
    test_date = datetime.date(2023, 6, 30)
    test_account = Account(name="Test", type=AccountType.EQUITY)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("10.00"), currency="CAD")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    try:
        posting.date = datetime.date(2024, 1, 1)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        posting.account = Account(name="New", type=AccountType.EXPENSE)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


# LLM-generated content at query #21
#--------------------------

def test_posting_constructor():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency="USD")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount


# LLM-generated content at query #22
#--------------------------

def test___call___returns_iterable_of_journal_entries():
    from typing import Iterable
    from datetime import date
    from beancount.core.data import JournalEntry, Transaction
    from beancount.core.number import D
    from beancount.core.amount import Amount
    from beancount.core.position import Cost
    from beancount.core.inventory import Inventory
    from beancount.core.account_types import AccountTypes
    from beancount.core.account import Account
    from beancount.core.flags import FLAG_OKAY
    from beancount.core.position import Position
    from beancount.core.data import Posting
    from beancount.core.data import Open, Close, Commodity, Pad, Balance, Note, Document, Event, Query, Price, Custom
    from beancount.core.data import new_metadata
    from beancount.core.data import Booking
    from beancount.core.data import CostSpec
    from beancount.core.data import TxnPosting
    from beancount.core.data import EMPTY_SET
    from beancount.core.data import create_simple_posting
    from beancount.core.data import create_simple_transaction
    from beancount.core.getters import get_accounts
    from beancount.core.getters import get_account_open_close
    from beancount.core.getters import get_commodity_directives
    from beancount.core.getters import get_pad_directives
    from beancount.core.getters import get_balance_directives
    from beancount.core.getters import get_note_directives
    from beancount.core.getters import get_document_directives
    from beancount.core.getters import get_event_directives
    from beancount.core.getters import get_query_directives
    from beancount.core.getters import get_price_directives
    from beancount.core.getters import get_custom_directives
    from beancount.core.getters import get_entries_by_type
    from beancount.core.getters import get_entry
    from beancount.core.getters import get_min_max_dates
    from beancount.core.getters import get_active_years
    from beancount.core.getters import get_all_entries
    from beancount.core.getters import get_filtered_entries
    from beancount.core.getters import get_sorted_entries
    from beancount.core.getters import get_reversed_entries
    from beancount.core.getters import get_entries_with_links
    from beancount.core.getters import get_entries_with_tags
    from beancount.core.getters import get_entries_with_type
    from beancount.core.getters import get_entries_with_account
    from beancount.core.getters import get_entries_with_commodity
    from beancount.core.getters import get_entries_with_payee
    from beancount.core.getters import get_entries_with_narration
    from beancount.core.getters import get_entries_with_flag
    from beancount.core.getters import get_entries_with_metadata
    from beancount.core.getters import get_entries_with_booking
    from beancount.core.getters import get_entries_with_cost
    from beancount.core.getters import get_entries_with_price
    from beancount.core.getters import get_entries_with_position
    from beancount.core.getters import get_entries_with_inventory
    from beancount.core.getters import get_entries_with_amount
    from beancount.core.getters import get_entries_with_number
    from beancount.core.getters import get_entries_with_date
    from beancount.core.getters import get_entries_with_date_range
    from beancount.core.getters import get_entries_with_month
    from beancount.core.getters import get_entries_with_year
    from beancount.core.getters import get_entries_with_week
    from beancount.core.getters import get_entries_with_quarter
    from beancount.core.getters import get_entries_with_period
    from beancount.core.getters import get_entries_with_interval
    from beancount.core.getters import get_entries_with_frequency
    from beancount.core.getters import get_entries_with_recurrence
    from beancount.core.getters import get_entries_with_reminder
    from beancount.core.getters import get_entries_with_scheduled
    from beancount.core.getters import get_entries_with_pending
    from beancount.core.getters import get_entries_with_cleared
    from beancount.core.getters import get_entries_with_void
    from beancount.core.getters import get_entries_with_balanced
    from beancount.core.getters import get_entries_with_unbalanced
    from beancount.core.getters import get_entries_with_unknown
    from beancount.core.getters import get_entries_with_other
    from beancount.core.getters import get_entries_with_all
    from beancount.core.getters import get_entries_with_none
    from beancount.core.getters import get_entries_with_any
    from beancount.core.getters import get_entries_with_not
    from beancount.core.getters import get_entries_with_and
    from beancount.core.getters import get_entries_with_or
    from beancount.core.getters import get_entries_with_xor
    from beancount.core.getters import get_entries_with_add
    from beancount.core.getters import get_entries_with_sub
    from beancount.core.getters import get_entries_with_mul
    from beancount.core.getters import get_entries_with_div
    from beancount.core.getters import get_entries_with_mod
    from beancount.core.getters import get_entries_with_pow
    from beancount.core.getters import get_entries_with_neg
    from beancount.core.getters import get_entries_with_pos
    from beancount.core.getters import get_entries_with_abs
    from beancount.core.getters import get_entries_with_round
    from beancount.core.getters import get_entries_with_floor
    from beancount.core.getters import get_entries_with_ceil
    from beancount.core.getters import get_entries_with_trunc
    from beancount.core.getters import get_entries_with_sqrt
    from beancount.core.getters import get_entries_with_exp
    from beancount.core.getters import get_entries_with_log
    from beancount.core.getters import get_entries_with_log10
    from beancount.core.getters import get_entries_with_sin
    from beancount.core.getters import get_entries_with_cos
    from beancount.core.getters import get_entries_with_tan
    from beancount.core.getters import get_entries_with_asin
    from beancount.core.getters import get_entries_with_acos
    from beancount.core.getters import get_entries_with_atan
    from beancount.core.getters import get_entries_with_atan2
    from beancount.core.getters import get_entries_with_sinh
    from beancount.core.getters import get_entries_with_cosh
    from beancount.core.getters import get_entries_with_tanh
    from beancount.core.getters import get_entries_with_asinh
    from beancount.core.getters import get_entries_with_acosh
    from beancount.core.getters import get_entries_with_atanh
    from beancount.core.getters import get_entries_with_degrees
    from beancount.core.getters import get_entries_with_radians
    from beancount.core.getters import get_entries_with_pi
    from beancount.core.getters import get_entries_with_e
    from beancount.core.getters import get_entries_with_inf
    from beancount.core.getters import get_entries_with_nan
    from beancount.core.getters import get_entries_with_true
    from beancount.core.getters import get_entries_with_false
    from beancount.core.getters import get_entries_with_none
    from beancount.core.getters import get_entries_with_bool
    from beancount.core.getters import get_entries_with_int
    from beancount.core.getters import get_entries_with_float
    from beancount.core.getters import get_entries_with_str
    from beancount.core.getters import get_entries_with_bytes
    from beancount.core.getters import get_entries_with_list
    from beancount.core.getters import get_entries_with_tuple
    from beancount.core.getters import get_entries_with_set
    from beancount.core.getters import get_entries_with_dict
    from beancount.core.getters import get_entries_with_object
    from beancount.core.getters import get_entries_with_type
    from beancount.core.getters import get_entries_with_class
    from beancount.core.getters import get_entries_with_function
    from beancount.core.getters import get_entries_with_module
    from beancount.core.getters import get_entries_with_package



# LLM-generated content at query #23
#--------------------------

def test_journal_entry_constructor_initializes_fields_correctly():
    mock_source = object()
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source is mock_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_journal_entry_constructor_with_different_date():
    mock_source = "source"
    test_date = datetime.date(2022, 12, 31)
    test_description = "Year end entry"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == mock_source

def test_journal_entry_constructor_with_empty_description():
    mock_source = 123
    test_date = datetime.date(2023, 2, 15)
    test_description = ""
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == mock_source

def test_journal_entry_constructor_postings_is_empty_list():
    mock_source = None
    test_date = datetime.date(2023, 3, 10)
    test_description = "No postings"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.postings == []

def test_journal_entry_constructor_guid_is_unique():
    mock_source = object()
    test_date = datetime.date(2023, 4, 1)
    test_description = "Guid test"
    entry1 = JournalEntry(date=test_date, description=test_description, source=mock_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry1.guid != entry2.guid

def test_journal_entry_is_immutable():
    mock_source = object()
    test_date = datetime.date(2023, 5, 20)
    test_description = "Immutable test"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 5, 21)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "Changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = object()


# LLM-generated content at query #24
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account1")
    account2 = Account(code="A2", name="Account2")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(-ONE))
    entry.validate()

def test_validate_with_zero_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Account, Quantity
    from pypara.commons.zed import ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account1")
    account2 = Account(code="A2", name="Account2")
    account3 = Account(code="A3", name="Account3")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account3, quantity=Quantity(-ONE * 2))
    entry.validate()

def test_validate_raises_assertion_error_on_imbalance():
    from pypara.accounting.journaling import JournalEntry, Account, Quantity
    from pypara.commons.zed import ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account1")
    account2 = Account(code="A2", name="Account2")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(-ONE * 2))
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #25
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New description"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New source"
        assert False, "Should not be able to assign to frozen instance attribute"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_postings_field_is_init_false():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []
    entry.postings.append("test")
    assert entry.postings == ["test"]


def test_journal_entry_constructor_guid_field_is_init_false_and_unique():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert isinstance(entry1.guid, Guid)
    assert isinstance(entry2.guid, Guid)
    assert entry1.guid != entry2.guid


