####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
        entry.date = datetime.date(2024, 1, 1)


def test_journal_entry_constructor_uses_default_factory_for_postings():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.postings is not entry2.postings
    assert entry1.postings == entry2.postings


def test_journal_entry_constructor_uses_default_factory_for_guid():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #2
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


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])
        assert False
    except TypeError:
        assert True


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid=makeguid())
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #3
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
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict


def test_journal_entry_is_frozen():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to frozen attribute 'date'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New description"
        assert False, "Should not be able to assign to frozen attribute 'description'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New source"
        assert False, "Should not be able to assign to frozen attribute 'source'"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_initializes_postings_as_empty_list():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []
    assert isinstance(entry.postings, list)


def test_journal_entry_constructor_initializes_unique_guid():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid
    assert isinstance(entry1.guid, str)
    assert isinstance(entry2.guid, str)


# LLM-generated content at query #4
#--------------------------

def test_read_journal_entries_call():
    from typing import Iterable
    from datetime import date
    from typing import Protocol, TypeVar

    _T = TypeVar("_T")

    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class JournalEntry:
        def __init__(self, data: _T):
            self.data = data

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return [JournalEntry("entry1"), JournalEntry("entry2")]

    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = list(reader(period))
    assert len(result) == 2
    assert result[0].data == "entry1"
    assert result[1].data == "entry2"


# LLM-generated content at query #5
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ZERO, ONE
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE)))
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE)))
    entry.validate()

def test_validate_with_zero_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ONE, TWO
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE)))
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(TWO)))
    try:
        entry.validate()
        assert False
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_with_multiple_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ONE, TWO
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE)))
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(TWO)))
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE)))
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(TWO)))
    entry.validate()

def test_validate_with_only_debits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE)))
    try:
        entry.validate()
        assert False
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_with_only_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ONE
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE)))
    try:
        entry.validate()
        assert False
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #6
#--------------------------

def test_post_positive_quantity_increment():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account = Account("123", "Test Account")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 2), account, Quantity(100))
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account == account
    assert posting.direction.name == "INC"
    assert posting.amount == Amount(100)

def test_post_negative_quantity_decrement():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account = Account("123", "Test Account")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 2), account, Quantity(-50))
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account == account
    assert posting.direction.name == "DEC"
    assert posting.amount == Amount(50)

def test_post_zero_quantity_no_posting():
    from pypara.accounting.journaling import JournalEntry, Account, Quantity
    import datetime
    account = Account("123", "Test Account")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 2), account, Quantity(0))
    assert len(entry.postings) == 0

def test_post_multiple_postings():
    from pypara.accounting.journaling import JournalEntry, Account, Amount, Quantity
    import datetime
    account1 = Account("123", "Test Account 1")
    account2 = Account("456", "Test Account 2")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 2), account1, Quantity(100))
    entry.post(datetime.date(2023, 1, 3), account2, Quantity(-200))
    assert len(entry.postings) == 2
    posting1 = entry.postings[0]
    posting2 = entry.postings[1]
    assert posting1.date == datetime.date(2023, 1, 2)
    assert posting1.account == account1
    assert posting1.direction.name == "INC"
    assert posting1.amount == Amount(100)
    assert posting2.date == datetime.date(2023, 1, 3)
    assert posting2.account == account2
    assert posting2.direction.name == "DEC"
    assert posting2.amount == Amount(200)

def test_post_returns_self_for_chaining():
    from pypara.accounting.journaling import JournalEntry, Account, Quantity
    import datetime
    account = Account("123", "Test Account")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    result = entry.post(datetime.date(2023, 1, 2), account, Quantity(100))
    assert result is entry


# LLM-generated content at query #7
#--------------------------

def test_validate_assertion_succeeds_when_debits_equal_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.numbers import ONE
    import datetime
    from decimal import Decimal
    source = object()
    date = datetime.date(2023, 1, 1)
    account1 = Account("A1", "Account 1")
    account2 = Account("A2", "Account 2")
    je = JournalEntry(date, "Test", source)
    je.post(date, account1, Quantity(ONE))
    je.post(date, account2, Quantity(-ONE))
    je.validate()


# LLM-generated content at query #8
#--------------------------

def test_journal_entry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_with_different_source_type():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = 12345
    entry = JournalEntry[int](date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journal_entry_constructor_frozen_immutability_check():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source"
    entry = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to date"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to assign to description"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to assign to source"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_ensures_guid_uniqueness():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source"
    entry1 = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


def test_journal_entry_constructor_postings_list_is_independent():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source"
    entry1 = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry[str](date=test_date, description=test_description, source=test_source)
    dummy_posting = Posting(entry1, test_date, Account("A"), Direction.INC, Amount(Quantity(10)))
    entry1.postings.append(dummy_posting)
    assert len(entry1.postings) == 1
    assert len(entry2.postings) == 0


# LLM-generated content at query #9
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

def test_posting_constructor_with_different_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 2, 1)
    test_account = Account(name="Revenue", type=AccountType.REVENUE)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("50.00"), currency="EUR")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount

def test_posting_constructor_with_zero_amount():
    mock_journal = object()
    test_date = datetime.date(2023, 3, 1)
    test_account = Account(name="Expense", type=AccountType.EXPENSE)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("0.00"), currency="GBP")
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount


# LLM-generated content at query #10
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ZERO
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ZERO)))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ZERO)))
    je.validate()

def test_validate_with_zero_total():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ZERO
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ZERO)))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ZERO)))
    je.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from decimal import Decimal
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal("10"))))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal("5"))))
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

def test_validate_with_multiple_equal_postings():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from decimal import Decimal
    import datetime
    je = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal("10"))))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal("20"))))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal("15"))))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal("15"))))
    je.validate()


# LLM-generated content at query #11
#--------------------------

def test_read_journal_entries_call():
    from typing import Iterable
    from datetime import date
    from typing import Protocol, TypeVar

    _T = TypeVar("_T")

    class DateRange:
        def __init__(self, start: date, end: date):
            self.start = start
            self.end = end

    class JournalEntry:
        def __init__(self, data: _T):
            self.data = data

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry]:
            return [JournalEntry("entry1"), JournalEntry("entry2")]

    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 31))
    result = list(reader(period))
    assert len(result) == 2
    assert result[0].data == "entry1"
    assert result[1].data == "entry2"


# LLM-generated content at query #12
#--------------------------

def test_constructor_initializes_fields_correctly():
    mock_source = object()
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source is mock_source
    assert entry.postings == []
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_constructor_with_different_date_and_description():
    mock_source = "source_object"
    test_date = datetime.date(2022, 12, 31)
    test_description = "Year-end adjustment"
    entry = JournalEntry(date=test_date, description=test_description, source=mock_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == mock_source

def test_constructor_postings_is_empty_list_by_default():
    mock_source = 123
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Desc", source=mock_source)
    assert entry.postings == []
    assert len(entry.postings) == 0

def test_constructor_guid_is_unique_and_non_empty():
    mock_source = None
    entry1 = JournalEntry(date=datetime.date(2023, 1, 1), description="E1", source=mock_source)
    entry2 = JournalEntry(date=datetime.date(2023, 1, 1), description="E2", source=mock_source)
    assert isinstance(entry1.guid, str)
    assert isinstance(entry2.guid, str)
    assert len(entry1.guid) > 0
    assert len(entry2.guid) > 0
    assert entry1.guid != entry2.guid

def test_constructor_with_frozen_dataclass_prevents_field_modification():
    mock_source = ["a", "b"]
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=mock_source)
    try:
        entry.date = datetime.date(2023, 2, 1)
        assert False, "Should not be able to modify frozen dataclass field"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New"
        assert False, "Should not be able to modify frozen dataclass field"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "new source"
        assert False, "Should not be able to modify frozen dataclass field"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #13
#--------------------------

def test_journal_entry_constructor_with_default_values():
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

def test_journal_entry_constructor_with_custom_source_type():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = 12345
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.date == date
    assert entry.description == description
    assert entry.source == source
    assert entry.postings == []
    assert isinstance(entry.guid, str)

def test_journal_entry_constructor_frozen_immutability():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = object()
    entry = JournalEntry(date=date, description=description, source=source)
    try:
        entry.date = datetime.date(2023, 2, 1)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.description = "Modified"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.source = None
        assert False
    except dataclasses.FrozenInstanceError:
        assert True

def test_journal_entry_constructor_postings_field_not_in_init():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = None
    entry = JournalEntry(date=date, description=description, source=source)
    assert entry.postings == []
    entry.postings.append("test")
    assert entry.postings == ["test"]

def test_journal_entry_constructor_guid_unique_per_instance():
    date = datetime.date(2023, 1, 1)
    description = "Test Entry"
    source = "Source"
    entry1 = JournalEntry(date=date, description=description, source=source)
    entry2 = JournalEntry(date=date, description=description, source=source)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #14
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="A1", name="Account 1")
    account2 = Account(code="A2", name="Account 2")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(Decimal("100")))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(Decimal("50")))
    try:
        entry.validate()
        assert False
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting1 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal('100')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('100')))
    entry.postings.extend([posting1, posting2])
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting1 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal('100')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('50')))
    entry.postings.extend([posting1, posting2])
    try:
        entry.validate()
        assert False
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 100 != 50"

def test_validate_with_no_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting1 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal('30')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal('70')))
    posting3 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('40')))
    posting4 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('60')))
    entry.postings.extend([posting1, posting2, posting3, posting4])
    entry.validate()

def test_validate_with_zero_amount_postings():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting1 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal('0')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('0')))
    entry.postings.extend([posting1, posting2])
    entry.validate()


# LLM-generated content at query #17
#--------------------------

def test_read_journal_entries_call():
    from typing import Iterable, Protocol, TypeVar
    from datetime import date
    _T = TypeVar("_T")
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


# LLM-generated content at query #18
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    posting1 = Posting(entry, datetime.date(2023, 1, 1), account="Asset", direction=Direction.INC, amount=Amount(Decimal('100')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), account="Equity", direction=Direction.DEC, amount=Amount(Decimal('100')))
    entry.postings.extend([posting1, posting2])
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    posting1 = Posting(entry, datetime.date(2023, 1, 1), account="Asset", direction=Direction.INC, amount=Amount(Decimal('100')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), account="Equity", direction=Direction.DEC, amount=Amount(Decimal('50')))
    entry.postings.extend([posting1, posting2])
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 100 != 50"

def test_validate_with_no_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    entry.validate()

def test_validate_with_multiple_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    posting1 = Posting(entry, datetime.date(2023, 1, 1), account="Asset1", direction=Direction.INC, amount=Amount(Decimal('75')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), account="Asset2", direction=Direction.INC, amount=Amount(Decimal('25')))
    posting3 = Posting(entry, datetime.date(2023, 1, 1), account="Equity1", direction=Direction.DEC, amount=Amount(Decimal('50')))
    posting4 = Posting(entry, datetime.date(2023, 1, 1), account="Equity2", direction=Direction.DEC, amount=Amount(Decimal('50')))
    entry.postings.extend([posting1, posting2, posting3, posting4])
    entry.validate()

def test_validate_with_zero_amount_postings():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import isum
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source="TestSource")
    posting1 = Posting(entry, datetime.date(2023, 1, 1), account="Asset", direction=Direction.INC, amount=Amount(Decimal('0')))
    posting2 = Posting(entry, datetime.date(2023, 1, 1), account="Equity", direction=Direction.DEC, amount=Amount(Decimal('0')))
    entry.postings.extend([posting1, posting2])
    entry.validate()


# LLM-generated content at query #19
#--------------------------

def test_post_positive_quantity_appends_increment_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime
    from decimal import Decimal
    account = Account("123", "Test Account")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    result = entry.post(datetime.date(2023, 1, 2), account, Decimal("100"))
    assert result is entry
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal is entry
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account is account
    assert posting.direction == Direction.INC
    assert posting.amount == Amount(Decimal("100"))

def test_post_negative_quantity_appends_decrement_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime
    from decimal import Decimal
    account = Account("123", "Test Account")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    result = entry.post(datetime.date(2023, 1, 2), account, Decimal("-100"))
    assert result is entry
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal is entry
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account is account
    assert posting.direction == Direction.DEC
    assert posting.amount == Amount(Decimal("100"))

def test_post_zero_quantity_does_nothing():
    from pypara.accounting.journaling import JournalEntry, Account
    import datetime
    from decimal import Decimal
    account = Account("123", "Test Account")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    result = entry.post(datetime.date(2023, 1, 2), account, Decimal("0"))
    assert result is entry
    assert len(entry.postings) == 0

def test_post_multiple_postings_appends_all():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    import datetime
    from decimal import Decimal
    account1 = Account("123", "Test Account 1")
    account2 = Account("456", "Test Account 2")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    entry.post(datetime.date(2023, 1, 2), account1, Decimal("100"))
    entry.post(datetime.date(2023, 1, 3), account2, Decimal("-50"))
    assert len(entry.postings) == 2
    posting1 = entry.postings[0]
    posting2 = entry.postings[1]
    assert posting1.direction == Direction.INC
    assert posting1.amount == Amount(Decimal("100"))
    assert posting2.direction == Direction.DEC
    assert posting2.amount == Amount(Decimal("50"))

def test_post_chaining_returns_same_entry():
    from pypara.accounting.journaling import JournalEntry, Account
    import datetime
    from decimal import Decimal
    account1 = Account("123", "Test Account 1")
    account2 = Account("456", "Test Account 2")
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", None)
    result = entry.post(datetime.date(2023, 1, 2), account1, Decimal("100")).post(datetime.date(2023, 1, 3), account2, Decimal("-50"))
    assert result is entry
    assert len(entry.postings) == 2


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount
    from pypara.commons.numbers import Quantity
    import datetime
    from decimal import Decimal
    journal = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="1000", name="Asset")
    account2 = Account(code="2000", name="Liability")
    posting1 = Posting(journal, datetime.date(2023, 1, 1), account1, Direction.INC, Amount(Decimal("100")))
    posting2 = Posting(journal, datetime.date(2023, 1, 1), account2, Direction.DEC, Amount(Decimal("50")))
    journal.postings.append(posting1)
    journal.postings.append(posting2)
    try:
        journal.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    from decimal import Decimal

    class MockSource:
        pass

    source = MockSource()
    date = datetime.date(2023, 1, 1)
    account1 = Account("A1", "Account 1")
    account2 = Account("A2", "Account 2")

    journal_entry = JournalEntry(date, "Test Entry", source)
    journal_entry.postings.append(Posting(journal_entry, date, account1, Direction.INC, Amount(Decimal("100"))))
    journal_entry.postings.append(Posting(journal_entry, date, account2, Direction.DEC, Amount(Decimal("50"))))

    try:
        journal_entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #23
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ZERO, ONE
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting_debit = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE))
    posting_credit = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE))
    entry.postings.append(posting_debit)
    entry.postings.append(posting_credit)
    entry.validate()

def test_validate_with_unequal_debits_and_credits_raises_assertion():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ZERO, ONE
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting_debit = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE))
    posting_credit = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal('2')))
    entry.postings.append(posting_debit)
    entry.postings.append(posting_credit)
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_with_zero_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    entry.validate()

def test_validate_with_multiple_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zed import ZERO, ONE
    import datetime
    from decimal import Decimal
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting_debit1 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE))
    posting_debit2 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('2')))
    posting_credit1 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE))
    posting_credit2 = Posting(entry, datetime.date(2023, 1, 1), None, Direction.INC, Amount(Decimal('2')))
    entry.postings.append(posting_debit1)
    entry.postings.append(posting_debit2)
    entry.postings.append(posting_credit1)
    entry.postings.append(posting_credit2)
    entry.validate()


# LLM-generated content at query #24
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
    test_source_int = 123
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict
    class CustomSource:
        pass
    test_source_custom = CustomSource()
    entry_custom = JournalEntry(date=test_date, description=test_description, source=test_source_custom)
    assert entry_custom.source == test_source_custom


# LLM-generated content at query #25
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    from decimal import Decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.post(date(2023, 1, 1), account, Quantity(Decimal("100")))
    je.post(date(2023, 1, 1), account, Quantity(Decimal("-100")))
    je.validate()

def test_validate_with_zero_postings():
    from pypara.accounting.journaling import JournalEntry
    from datetime import date
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    je.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    from decimal import Decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.post(date(2023, 1, 1), account, Quantity(Decimal("100")))
    je.post(date(2023, 1, 1), account, Quantity(Decimal("-50")))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_with_multiple_postings():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    from decimal import Decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account1 = Account(code="1000", name="Cash")
    account2 = Account(code="2000", name="Revenue")
    je.post(date(2023, 1, 1), account1, Quantity(Decimal("150")))
    je.post(date(2023, 1, 1), account2, Quantity(Decimal("-150")))
    je.validate()

def test_validate_with_zero_quantity_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    from decimal import Decimal
    je = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.post(date(2023, 1, 1), account, Quantity(Decimal("0")))
    je.validate()


# LLM-generated content at query #26
#--------------------------

def test_journalentry_constructor_with_minimal_parameters():
    mock_source = object()
    test_date = datetime.date(2023, 10, 5)
    entry = JournalEntry(date=test_date, description="Test Entry", source=mock_source)
    assert entry.date == test_date
    assert entry.description == "Test Entry"
    assert entry.source is mock_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)


def test_journalentry_constructor_date_is_assigned_correctly():
    mock_source = object()
    test_date = datetime.date(2024, 1, 15)
    entry = JournalEntry(date=test_date, description="Another Entry", source=mock_source)
    assert entry.date == test_date


def test_journalentry_constructor_description_is_assigned_correctly():
    mock_source = object()
    entry = JournalEntry(date=datetime.date(2023, 12, 25), description="Christmas Transaction", source=mock_source)
    assert entry.description == "Christmas Transaction"


def test_journalentry_constructor_source_is_assigned_correctly():
    specific_source = "A String Source"
    entry = JournalEntry(date=datetime.date(2023, 7, 4), description="Independence Day", source=specific_source)
    assert entry.source == "A String Source"


def test_journalentry_constructor_postings_is_empty_list_by_default():
    mock_source = object()
    entry = JournalEntry(date=datetime.date(2023, 5, 1), description="May Day", source=mock_source)
    assert entry.postings == []
    assert isinstance(entry.postings, list)


def test_journalentry_constructor_guid_is_generated_and_unique():
    mock_source = object()
    entry1 = JournalEntry(date=datetime.date(2023, 1, 1), description="First", source=mock_source)
    entry2 = JournalEntry(date=datetime.date(2023, 1, 2), description="Second", source=mock_source)
    assert isinstance(entry1.guid, Guid)
    assert isinstance(entry2.guid, Guid)
    assert entry1.guid != entry2.guid


def test_journalentry_is_immutable_dataclass():
    mock_source = object()
    entry = JournalEntry(date=datetime.date(2023, 3, 8), description="Immutable Test", source=mock_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to frozen field 'date'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "Changed"
        assert False, "Should not be able to assign to frozen field 'description'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = None
        assert False, "Should not be able to assign to frozen field 'source'"
    except dataclasses.FrozenInstanceError:
        pass


# LLM-generated content at query #27
#--------------------------

def test_posting_constructor_initializes_fields_correctly():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency=Currency.USD)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount

def test_posting_constructor_with_different_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 2, 1)
    test_account = Account(name="Revenue", type=AccountType.REVENUE)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("50.00"), currency=Currency.EUR)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.direction == test_direction

def test_posting_is_debit_property_for_asset_increase():
    mock_journal = object()
    test_account = Account(name="Cash", type=AccountType.ASSET)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.INCREASE, amount=Amount(value=Decimal("100.00"), currency=Currency.USD))
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_is_credit_property_for_asset_decrease():
    mock_journal = object()
    test_account = Account(name="Cash", type=AccountType.ASSET)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.DECREASE, amount=Amount(value=Decimal("50.00"), currency=Currency.USD))
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_is_debit_property_for_liability_decrease():
    mock_journal = object()
    test_account = Account(name="Loan", type=AccountType.LIABILITY)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.DECREASE, amount=Amount(value=Decimal("200.00"), currency=Currency.USD))
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_is_credit_property_for_liability_increase():
    mock_journal = object()
    test_account = Account(name="Loan", type=AccountType.LIABILITY)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.INCREASE, amount=Amount(value=Decimal("300.00"), currency=Currency.USD))
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_is_debit_property_for_equity_decrease():
    mock_journal = object()
    test_account = Account(name="Retained Earnings", type=AccountType.EQUITY)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.DECREASE, amount=Amount(value=Decimal("150.00"), currency=Currency.USD))
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_is_credit_property_for_equity_increase():
    mock_journal = object()
    test_account = Account(name="Retained Earnings", type=AccountType.EQUITY)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.INCREASE, amount=Amount(value=Decimal("250.00"), currency=Currency.USD))
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_is_debit_property_for_revenue_decrease():
    mock_journal = object()
    test_account = Account(name="Sales", type=AccountType.REVENUE)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.DECREASE, amount=Amount(value=Decimal("75.00"), currency=Currency.USD))
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_is_credit_property_for_revenue_increase():
    mock_journal = object()
    test_account = Account(name="Sales", type=AccountType.REVENUE)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.INCREASE, amount=Amount(value=Decimal("125.00"), currency=Currency.USD))
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_is_debit_property_for_expense_increase():
    mock_journal = object()
    test_account = Account(name="Rent", type=AccountType.EXPENSE)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.INCREASE, amount=Amount(value=Decimal("500.00"), currency=Currency.USD))
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_is_credit_property_for_expense_decrease():
    mock_journal = object()
    test_account = Account(name="Rent", type=AccountType.EXPENSE)
    posting = Posting(journal=mock_journal, date=datetime.date(2023, 1, 1), account=test_account, direction=Direction.DECREASE, amount=Amount(value=Decimal("100.00"), currency=Currency.USD))
    assert posting.is_debit is False
    assert posting.is_credit is True


# LLM-generated content at query #28
#--------------------------

def test_posting_constructor_initializes_fields_correctly():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency=Currency.USD)
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
    test_amount = Amount(value=Decimal("500.00"), currency=Currency.EUR)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_constructor_with_credit_account_and_increase_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 3, 10)
    test_account = Account(name="Loan", type=AccountType.LIABILITY)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("200.00"), currency=Currency.GBP)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_constructor_with_debit_account_and_decrease_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 4, 5)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("50.00"), currency=Currency.USD)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is False
    assert posting.is_credit is True

def test_posting_constructor_with_credit_account_and_decrease_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 5, 20)
    test_account = Account(name="Revenue", type=AccountType.EQUITY)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("300.00"), currency=Currency.JPY)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.is_debit is True
    assert posting.is_credit is False

def test_posting_is_immutable():
    mock_journal = object()
    test_date = datetime.date(2023, 6, 1)
    test_account = Account(name="Test", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("10.00"), currency=Currency.USD)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    with pytest.raises(dataclasses.FrozenInstanceError):
        posting.date = datetime.date(2023, 6, 2)


# LLM-generated content at query #29
#--------------------------

def test_journalentry_constructor_with_minimal_parameters():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Description"
    test_source = "Test Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_journalentry_constructor_creates_frozen_instance():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Description"
    test_source = "Test Source Object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 10, 6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "New Description"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = "New Source"

def test_journalentry_constructor_initializes_unique_guid():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Description"
    test_source = "Test Source Object"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid

def test_journalentry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Description"
    entry_int = JournalEntry(date=test_date, description=test_description, source=42)
    assert entry_int.source == 42
    entry_dict = JournalEntry(date=test_date, description=test_description, source={"key": "value"})
    assert entry_dict.source == {"key": "value"}
    entry_none = JournalEntry(date=test_date, description=test_description, source=None)
    assert entry_none.source is None


# LLM-generated content at query #30
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


def test_journal_entry_constructor_postings_field_is_init_false():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []
    try:
        JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])
        assert False, "Should not be able to pass postings to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_guid_field_is_init_false():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert isinstance(entry.guid, Guid)
    try:
        JournalEntry(date=test_date, description=test_description, source=test_source, guid=makeguid())
        assert False, "Should not be able to pass guid to constructor"
    except TypeError:
        pass


# LLM-generated content at query #31
#--------------------------

def test_validate_asserts_when_debits_and_credits_are_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    from pypara.commons.numbers import isum
    from pypara.commons.zeitgeist import makeguid
    from decimal import Decimal
    source = object()
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=source)
    account1 = Account(code="1000", name="Asset")
    account2 = Account(code="2000", name="Liability")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(Decimal("100")))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(Decimal("-100")))
    entry.validate()


# LLM-generated content at query #32
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
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.description = "New Description"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.source = "NewSource"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


def test_journal_entry_constructor_postings_field_is_init_false():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []
    entry.postings.append("test")
    assert entry.postings == ["test"]


def test_journal_entry_constructor_guid_field_is_init_false():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid
    assert isinstance(entry1.guid, Guid)
    assert isinstance(entry2.guid, Guid)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import ONE
    from decimal import Decimal
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting1 = Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE))
    posting2 = Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE))
    je.postings.append(posting1)
    je.postings.append(posting2)
    je.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import ONE
    from decimal import Decimal
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting1 = Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE))
    posting2 = Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(Decimal('2')))
    je.postings.append(posting1)
    je.postings.append(posting2)
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)

def test_validate_with_zero_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    je.validate()

def test_validate_with_multiple_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.numbers import ONE
    from decimal import Decimal
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    posting1 = Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE))
    posting2 = Posting(je, datetime.date(2023, 1, 1), None, Direction.INC, Amount(ONE))
    posting3 = Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE))
    posting4 = Posting(je, datetime.date(2023, 1, 1), None, Direction.DEC, Amount(ONE))
    je.postings.append(posting1)
    je.postings.append(posting2)
    je.postings.append(posting3)
    je.postings.append(posting4)
    je.validate()


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

def test_constructor_initializes_fields_correctly():
    from datetime import date
    from dataclasses import FrozenInstanceError
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Source object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_constructor_raises_error_when_missing_required_arguments():
    from datetime import date
    try:
        JournalEntry()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

def test_constructor_creates_frozen_instance():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Source object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = date(2023, 2, 1)
        assert False, "Should have raised FrozenInstanceError"
    except FrozenInstanceError:
        pass

def test_constructor_with_different_source_types():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source_int = 123
    test_source_dict = {"key": "value"}
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_int.source == test_source_int
    assert entry_dict.source == test_source_dict

def test_constructor_ensures_postings_list_is_empty_by_default():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Source object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []


# LLM-generated content at query #5
#--------------------------

def test_constructor_initializes_fields_correctly():
    from datetime import date
    from dataclasses import FrozenInstanceError
    from typing import List
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, List)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_constructor_is_frozen():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = date(2023, 1, 2)
        assert False, "Should not be able to modify frozen instance"
    except FrozenInstanceError:
        pass

def test_constructor_with_different_source_types():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    entry_int = JournalEntry(date=test_date, description=test_description, source=123)
    assert entry_int.source == 123
    entry_dict = JournalEntry(date=test_date, description=test_description, source={"key": "value"})
    assert entry_dict.source == {"key": "value"}
    entry_none = JournalEntry(date=test_date, description=test_description, source=None)
    assert entry_none.source is None

def test_constructor_postings_is_empty_list_by_default():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []
    assert isinstance(entry.postings, list)

def test_constructor_guid_is_unique():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #6
#--------------------------

def test_validate_asserts_when_debits_and_credits_are_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    source = object()
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=source)
    account1 = Account.of("123")
    account2 = Account.of("456")
    entry.post(date=datetime.date(2023, 1, 1), account=account1, quantity=Quantity(ONE))
    entry.post(date=datetime.date(2023, 1, 1), account=account2, quantity=Quantity(-ONE))
    entry.validate()


# LLM-generated content at query #7
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
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 10, 6)


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict


def test_journal_entry_constructor_guid_is_unique():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


def test_journal_entry_constructor_postings_list_is_initially_empty():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []


# LLM-generated content at query #8
#--------------------------

def test_constructor_initializes_fields_correctly():
    from datetime import date
    from dataclasses import FrozenInstanceError
    from typing import List
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, List)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_constructor_is_frozen():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = date(2023, 1, 2)
        assert False, "Should not be able to modify frozen instance"
    except FrozenInstanceError:
        pass
    try:
        entry.description = "Modified"
        assert False, "Should not be able to modify frozen instance"
    except FrozenInstanceError:
        pass
    try:
        entry.source = "Modified source"
        assert False, "Should not be able to modify frozen instance"
    except FrozenInstanceError:
        pass

def test_constructor_with_different_source_types():
    from datetime import date
    test_date = date(2023, 1, 1)
    test_description = "Test entry"
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


# LLM-generated content at query #9
#--------------------------

def test_posting_constructor():
    journal = JournalEntry()
    date = datetime.date(2023, 1, 1)
    account = Account(name="Cash", type=AccountType.ASSET)
    direction = Direction.DEBIT
    amount = Amount(value=100, currency="USD")
    posting = Posting(journal=journal, date=date, account=account, direction=direction, amount=amount)
    assert posting.journal == journal
    assert posting.date == date
    assert posting.account == account
    assert posting.direction == direction
    assert posting.amount == amount


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.numbers import ONE, ZERO
    import datetime

    account1 = Account("1", "Account 1")
    account2 = Account("2", "Account 2")
    source = object()
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test entry", source)
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), account1, Direction.INC, Amount(ONE)))
    entry.postings.append(Posting(entry, datetime.date(2023, 1, 1), account2, Direction.DEC, Amount(ZERO)))
    try:
        entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #11
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
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 10, 6)


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict
    class CustomSource:
        pass
    test_source_custom = CustomSource()
    entry_custom = JournalEntry(date=test_date, description=test_description, source=test_source_custom)
    assert entry_custom.source == test_source_custom


# LLM-generated content at query #12
#--------------------------

def test_validate_assertion_succeeds_when_debits_equal_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    from decimal import Decimal
    journal = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    journal.post(date(2023, 1, 1), account, Quantity(Decimal("100")))
    journal.post(date(2023, 1, 1), account, Quantity(Decimal("-100")))
    journal.validate()


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

def test_journalentry_constructor_with_minimal_parameters():
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


def test_journalentry_constructor_is_frozen():
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


# LLM-generated content at query #15
#--------------------------

def test_read_journal_entries_call():
    from typing import Iterable, Protocol
    from datetime import date
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class JournalEntry(Generic[_T]):
        data: _T
        entry_date: date

    class MockReadJournalEntries(Protocol[_T]):
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            ...

    class TestReader:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[str]]:
            return [JournalEntry(data="test", entry_date=date(2023, 1, 1))]

    reader = TestReader()
    period = DateRange(start=date(2023, 1, 1), end=date(2023, 12, 31))
    result = list(reader(period))
    assert len(result) == 1
    assert result[0].data == "test"
    assert result[0].entry_date == date(2023, 1, 1)


# LLM-generated content at query #16
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    acc1 = Account(code="1000", name="Asset")
    acc2 = Account(code="2000", name="Liability")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount("100")))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc2, Direction.DEC, Amount("100")))
    je.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    acc1 = Account(code="1000", name="Asset")
    acc2 = Account(code="2000", name="Liability")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount("100")))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc2, Direction.DEC, Amount("50")))
    try:
        je.validate()
        assert False
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 100 != 50"

def test_validate_with_no_postings():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    je.validate()

def test_validate_with_multiple_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    acc1 = Account(code="1000", name="Asset")
    acc2 = Account(code="2000", name="Liability")
    acc3 = Account(code="3000", name="Equity")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount("75")))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc2, Direction.DEC, Amount("25")))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc3, Direction.DEC, Amount("50")))
    je.validate()

def test_validate_with_zero_amount_postings():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    acc1 = Account(code="1000", name="Asset")
    acc2 = Account(code="2000", name="Liability")
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc1, Direction.INC, Amount("0")))
    je.postings.append(Posting(je, datetime.date(2023, 1, 1), acc2, Direction.DEC, Amount("0")))
    je.validate()


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
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2024, 1, 1)


def test_journal_entry_constructor_guid_is_unique():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


def test_journal_entry_constructor_postings_list_is_independent():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry1.postings.append("dummy_posting")
    assert len(entry1.postings) == 1
    assert len(entry2.postings) == 0


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

def test_validate_with_equal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zeitgeist import Date
    from decimal import Decimal
    dummy_date = Date(2023, 1, 1)
    dummy_source = "source"
    entry = JournalEntry(dummy_date, dummy_source)
    posting_debit = Posting(entry, dummy_date, "account1", Direction.INC, Amount(Decimal("100")))
    posting_credit = Posting(entry, dummy_date, "account2", Direction.DEC, Amount(Decimal("100")))
    entry.postings.append(posting_debit)
    entry.postings.append(posting_credit)
    entry.validate()

def test_validate_with_unequal_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zeitgeist import Date
    from decimal import Decimal
    dummy_date = Date(2023, 1, 1)
    dummy_source = "source"
    entry = JournalEntry(dummy_date, dummy_source)
    posting_debit = Posting(entry, dummy_date, "account1", Direction.INC, Amount(Decimal("150")))
    posting_credit = Posting(entry, dummy_date, "account2", Direction.DEC, Amount(Decimal("100")))
    entry.postings.append(posting_debit)
    entry.postings.append(posting_credit)
    try:
        entry.validate()
        assert False
    except AssertionError as e:
        assert str(e) == "Total Debits and Credits are not equal: 150 != 100"

def test_validate_with_no_postings():
    from pypara.accounting.journaling import JournalEntry
    from pypara.commons.zeitgeist import Date
    dummy_date = Date(2023, 1, 1)
    dummy_source = "source"
    entry = JournalEntry(dummy_date, dummy_source)
    entry.validate()

def test_validate_with_multiple_debits_and_credits():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zeitgeist import Date
    from decimal import Decimal
    dummy_date = Date(2023, 1, 1)
    dummy_source = "source"
    entry = JournalEntry(dummy_date, dummy_source)
    posting_debit1 = Posting(entry, dummy_date, "account1", Direction.INC, Amount(Decimal("50")))
    posting_debit2 = Posting(entry, dummy_date, "account2", Direction.INC, Amount(Decimal("75")))
    posting_credit1 = Posting(entry, dummy_date, "account3", Direction.DEC, Amount(Decimal("100")))
    posting_credit2 = Posting(entry, dummy_date, "account4", Direction.DEC, Amount(Decimal("25")))
    entry.postings.extend([posting_debit1, posting_debit2, posting_credit1, posting_credit2])
    entry.validate()

def test_validate_with_zero_amount_postings():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    from pypara.commons.zeitgeist import Date
    from decimal import Decimal
    dummy_date = Date(2023, 1, 1)
    dummy_source = "source"
    entry = JournalEntry(dummy_date, dummy_source)
    posting_debit = Posting(entry, dummy_date, "account1", Direction.INC, Amount(Decimal("0")))
    posting_credit = Posting(entry, dummy_date, "account2", Direction.DEC, Amount(Decimal("0")))
    entry.postings.append(posting_debit)
    entry.postings.append(posting_credit)
    entry.validate()


# LLM-generated content at query #20
#--------------------------

def test_validate_asserts_total_debit_equals_total_credit():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from datetime import date
    from decimal import Decimal
    entry = JournalEntry(date=date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    entry.post(date(2023, 1, 1), account, Quantity(Decimal("100")))
    entry.post(date(2023, 1, 1), account, Quantity(Decimal("-100")))
    entry.validate()


# LLM-generated content at query #21
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
    test_description = "Test entry"
    entry_with_int_source = JournalEntry(date=test_date, description=test_description, source=123)
    assert entry_with_int_source.source == 123
    entry_with_dict_source = JournalEntry(date=test_date, description=test_description, source={"id": 456})
    assert entry_with_dict_source.source == {"id": 456}
    entry_with_none_source = JournalEntry(date=test_date, description=test_description, source=None)
    assert entry_with_none_source.source is None


def test_journal_entry_constructor_postings_list_is_initially_empty_and_mutable():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []
    mock_posting = Posting(entry, test_date, Account("Cash"), Direction.INC, Amount(Quantity(100)))
    entry.postings.append(mock_posting)
    assert len(entry.postings) == 1
    assert entry.postings[0] == mock_posting


def test_journal_entry_constructor_guid_is_unique_and_auto_generated():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "SourceObject"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert isinstance(entry1.guid, Guid)
    assert isinstance(entry2.guid, Guid)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #22
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


def test_journal_entry_constructor_guid_is_unique():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test entry"
    test_source = "Source Object"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #23
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.zed import ZERO, ONE
    import datetime
    source = object()
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=source)
    account = Account(code="1000", name="Test Account")
    je.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(ONE))
    je.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(ONE))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

def test_journalentry_constructor_with_minimal_fields():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Source object"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_journalentry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 2, 15)
    test_description = "Another entry"
    test_source = 12345
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source

def test_journalentry_constructor_verifies_frozen_dataclass():
    test_date = datetime.date(2023, 3, 10)
    test_description = "Frozen test"
    test_source = object()
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 3, 11)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "Modified"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = None


# LLM-generated content at query #26
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
    assert isinstance(entry.guid, str)
    assert len(entry.guid) > 0

def test_journal_entry_constructor_with_different_date():
    mock_source = object()
    test_date = datetime.date(2022, 12, 31)
    entry = JournalEntry(date=test_date, description="Year-end", source=mock_source)
    assert entry.date == test_date
    assert entry.description == "Year-end"
    assert entry.source is mock_source

def test_journal_entry_constructor_with_empty_description():
    mock_source = object()
    test_date = datetime.date(2023, 1, 1)
    entry = JournalEntry(date=test_date, description="", source=mock_source)
    assert entry.date == test_date
    assert entry.description == ""
    assert entry.source is mock_source

def test_journal_entry_constructor_source_can_be_any_type():
    test_date = datetime.date(2023, 1, 1)
    int_source = 42
    entry1 = JournalEntry(date=test_date, description="Int source", source=int_source)
    assert entry1.source == 42
    str_source = "source"
    entry2 = JournalEntry(date=test_date, description="Str source", source=str_source)
    assert entry2.source == "source"
    dict_source = {"key": "value"}
    entry3 = JournalEntry(date=test_date, description="Dict source", source=dict_source)
    assert entry3.source == {"key": "value"}

def test_journal_entry_is_immutable():
    mock_source = object()
    entry = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=mock_source)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.date = datetime.date(2023, 1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.description = "Modified"
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.source = object()


# LLM-generated content at query #27
#--------------------------

def test_constructor_initializes_fields_correctly():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.date == test_date
    assert entry.description == test_description
    assert entry.source == test_source
    assert isinstance(entry.postings, list)
    assert len(entry.postings) == 0
    assert isinstance(entry.guid, Guid)

def test_constructor_with_different_source_types():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source_int = 123
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict

def test_constructor_postings_is_empty_list_by_default():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []

def test_constructor_guid_is_unique():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid

def test_constructor_is_frozen():
    test_date = datetime.date(2023, 1, 1)
    test_description = "Test entry"
    test_source = "Test source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
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


# LLM-generated content at query #28
#--------------------------

def test_posting_constructor_initializes_fields_correctly():
    mock_journal = object()
    test_date = datetime.date(2023, 1, 1)
    test_account = Account(name="Cash", type=AccountType.ASSET)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("100.00"), currency=Currency.USD)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount

def test_posting_constructor_with_different_direction():
    mock_journal = object()
    test_date = datetime.date(2023, 2, 15)
    test_account = Account(name="Revenue", type=AccountType.REVENUE)
    test_direction = Direction.DECREASE
    test_amount = Amount(value=Decimal("50.00"), currency=Currency.EUR)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    assert posting.journal is mock_journal
    assert posting.date == test_date
    assert posting.account == test_account
    assert posting.direction == test_direction
    assert posting.amount == test_amount

def test_posting_is_immutable():
    mock_journal = object()
    test_date = datetime.date(2023, 3, 10)
    test_account = Account(name="Expense", type=AccountType.EXPENSE)
    test_direction = Direction.INCREASE
    test_amount = Amount(value=Decimal("75.00"), currency=Currency.GBP)
    posting = Posting(journal=mock_journal, date=test_date, account=test_account, direction=test_direction, amount=test_amount)
    try:
        posting.date = datetime.date(2023, 4, 1)
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


# LLM-generated content at query #29
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
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to frozen attribute 'date'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to assign to frozen attribute 'description'"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to assign to frozen attribute 'source'"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_postings_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, postings=[])
        assert False, "Should not be able to pass 'postings' to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_guid_field_not_in_init():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    try:
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid="test-guid")
        assert False, "Should not be able to pass 'guid' to constructor"
    except TypeError:
        pass


def test_journal_entry_constructor_guid_is_unique():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source Object"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.numbers import ZERO
    import datetime
    from decimal import Decimal

    class MockSource:
        pass

    source = MockSource()
    date = datetime.date(2023, 1, 1)
    account1 = Account("A1", "Account 1")
    account2 = Account("A2", "Account 2")

    journal_entry = JournalEntry(date, "Test Entry", source)
    journal_entry.post(date, account1, Quantity(Decimal("100")))
    journal_entry.post(date, account2, Quantity(Decimal("-50")))

    try:
        journal_entry.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #31
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
        entry.description = "Modified"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True
    try:
        entry.source = "Modified source"
        assert False
    except dataclasses.FrozenInstanceError:
        assert True


# LLM-generated content at query #32
#--------------------------

def test___call___returns_iterable_of_journal_entries():
    from typing import Iterable
    from datetime import date
    from dataclasses import dataclass
    from typing import Protocol, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class JournalEntry:
        content: _T

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return [JournalEntry("entry1"), JournalEntry("entry2")]

    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    result_list = list(result)
    assert len(result_list) == 2
    assert isinstance(result_list[0], JournalEntry)
    assert result_list[0].content == "entry1"
    assert isinstance(result_list[1], JournalEntry)
    assert result_list[1].content == "entry2"

def test___call___handles_empty_period():
    from typing import Iterable
    from datetime import date
    from dataclasses import dataclass
    from typing import Protocol, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class JournalEntry:
        content: _T

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return []

    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 1, 1))
    result = reader(period)
    result_list = list(result)
    assert len(result_list) == 0

def test___call___returns_correct_generic_type():
    from typing import Iterable
    from datetime import date
    from dataclasses import dataclass
    from typing import Protocol, TypeVar

    _T = TypeVar('_T')

    @dataclass
    class DateRange:
        start: date
        end: date

    @dataclass
    class JournalEntry:
        content: _T

    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[int]]:
            return [JournalEntry(1), JournalEntry(2)]

    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = reader(period)
    result_list = list(result)
    assert result_list[0].content == 1
    assert result_list[1].content == 2
    assert isinstance(result_list[0].content, int)


# LLM-generated content at query #33
#--------------------------

def test_post_positive_quantity_appends_increment_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currency
    from pypara.accounts import Account, AccountType
    from pypara.commons.numbers import Quantity
    dummy_source = object()
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", dummy_source)
    account = Account("123", AccountType.ASSETS, Currency("USD"))
    quantity = Quantity(Decimal("100.00"))
    result = entry.post(datetime.date(2023, 1, 2), account, quantity)
    assert result is entry
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal is entry
    assert posting.date == datetime.date(2023, 1, 2)
    assert posting.account is account
    assert posting.direction == Direction.INC
    assert posting.amount == Amount(quantity)

def test_post_negative_quantity_appends_decrement_posting():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currency
    from pypara.accounts import Account, AccountType
    from pypara.commons.numbers import Quantity
    dummy_source = object()
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", dummy_source)
    account = Account("123", AccountType.ASSETS, Currency("USD"))
    quantity = Quantity(Decimal("-50.00"))
    result = entry.post(datetime.date(2023, 1, 3), account, quantity)
    assert result is entry
    assert len(entry.postings) == 1
    posting = entry.postings[0]
    assert posting.journal is entry
    assert posting.date == datetime.date(2023, 1, 3)
    assert posting.account is account
    assert posting.direction == Direction.DEC
    assert posting.amount == Amount(abs(quantity))

def test_post_zero_quantity_does_nothing():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currency
    from pypara.accounts import Account, AccountType
    from pypara.commons.numbers import Quantity
    dummy_source = object()
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", dummy_source)
    account = Account("123", AccountType.ASSETS, Currency("USD"))
    quantity = Quantity(Decimal("0.00"))
    result = entry.post(datetime.date(2023, 1, 4), account, quantity)
    assert result is entry
    assert len(entry.postings) == 0

def test_post_multiple_postings_accumulate():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Amount
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currency
    from pypara.accounts import Account, AccountType
    from pypara.commons.numbers import Quantity
    dummy_source = object()
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", dummy_source)
    account1 = Account("123", AccountType.ASSETS, Currency("USD"))
    account2 = Account("456", AccountType.LIABILITIES, Currency("USD"))
    quantity1 = Quantity(Decimal("100.00"))
    quantity2 = Quantity(Decimal("-50.00"))
    entry.post(datetime.date(2023, 1, 2), account1, quantity1)
    entry.post(datetime.date(2023, 1, 3), account2, quantity2)
    assert len(entry.postings) == 2
    posting1 = entry.postings[0]
    posting2 = entry.postings[1]
    assert posting1.direction == Direction.INC
    assert posting1.account is account1
    assert posting2.direction == Direction.DEC
    assert posting2.account is account2

def test_post_returns_self_for_chaining():
    from pypara.accounting.journaling import JournalEntry
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currency
    from pypara.accounts import Account, AccountType
    from pypara.commons.numbers import Quantity
    dummy_source = object()
    entry = JournalEntry(datetime.date(2023, 1, 1), "Test", dummy_source)
    account = Account("123", AccountType.ASSETS, Currency("USD"))
    quantity = Quantity(Decimal("100.00"))
    chained = entry.post(datetime.date(2023, 1, 2), account, quantity).post(datetime.date(2023, 1, 3), account, quantity)
    assert chained is entry
    assert len(entry.postings) == 2


# LLM-generated content at query #34
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


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source_int = 123
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict


def test_journal_entry_constructor_immutability_check():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    original_guid = entry.guid
    original_postings = entry.postings
    assert entry.guid == original_guid
    assert entry.postings is original_postings


# LLM-generated content at query #37
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
        entry = JournalEntry(date=test_date, description=test_description, source=test_source, guid=Guid())
        assert False, "Should not be able to pass guid to constructor"
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


# LLM-generated content at query #38
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


def test_journal_entry_constructor_with_different_source_types():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source_int = 42
    entry_int = JournalEntry(date=test_date, description=test_description, source=test_source_int)
    assert entry_int.source == test_source_int
    test_source_dict = {"key": "value"}
    entry_dict = JournalEntry(date=test_date, description=test_description, source=test_source_dict)
    assert entry_dict.source == test_source_dict


def test_journal_entry_is_immutable_dataclass():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    try:
        entry.date = datetime.date(2024, 1, 1)
        assert False, "Should not be able to assign to date"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.description = "New Description"
        assert False, "Should not be able to assign to description"
    except dataclasses.FrozenInstanceError:
        pass
    try:
        entry.source = "New Source"
        assert False, "Should not be able to assign to source"
    except dataclasses.FrozenInstanceError:
        pass


def test_journal_entry_constructor_ensures_guid_uniqueness():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source"
    entry1 = JournalEntry(date=test_date, description=test_description, source=test_source)
    entry2 = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry1.guid != entry2.guid


def test_journal_entry_postings_list_is_initially_empty_and_mutable():
    test_date = datetime.date(2023, 10, 5)
    test_description = "Test Entry"
    test_source = "Source"
    entry = JournalEntry(date=test_date, description=test_description, source=test_source)
    assert entry.postings == []
    dummy_posting = Posting(entry, test_date, Account("Test"), Direction.INC, Amount(Quantity(10)))
    entry.postings.append(dummy_posting)
    assert len(entry.postings) == 1
    assert entry.postings[0] == dummy_posting


# LLM-generated content at query #39
#--------------------------

def test_validate_raises_assertion_error_when_debits_and_credits_not_equal():
    from pypara.accounting.journaling import JournalEntry, Posting, Direction, Account, Amount, Quantity
    from pypara.commons.numbers import ONE, ZERO
    import datetime
    je = JournalEntry(date=datetime.date(2023, 1, 1), description="Test", source=None)
    account = Account(code="1000", name="Cash")
    je.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(ONE))
    je.post(date=datetime.date(2023, 1, 1), account=account, quantity=Quantity(ZERO))
    try:
        je.validate()
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Total Debits and Credits are not equal" in str(e)


# LLM-generated content at query #40
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


# LLM-generated content at query #41
#--------------------------

def test_read_journal_entries_call_with_valid_period():
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
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return [JournalEntry("test1"), JournalEntry("test2")]
    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = list(reader(period))
    assert len(result) == 2
    assert result[0].data == "test1"
    assert result[1].data == "test2"

def test_read_journal_entries_call_returns_empty_iterable():
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
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return []
    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = list(reader(period))
    assert len(result) == 0

def test_read_journal_entries_call_uses_period_attributes():
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
    captured_period = None
    class MockReadJournalEntries:
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            nonlocal captured_period
            captured_period = period
            return [JournalEntry(period.start), JournalEntry(period.end)]
    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 5, 10), date(2023, 5, 20))
    result = list(reader(period))
    assert captured_period is period
    assert result[0].data == date(2023, 5, 10)
    assert result[1].data == date(2023, 5, 20)

def test_read_journal_entries_call_with_different_generic_type():
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
        def __call__(self, period: DateRange) -> Iterable[JournalEntry[_T]]:
            return [JournalEntry(123), JournalEntry(456)]
    reader = MockReadJournalEntries()
    period = DateRange(date(2023, 1, 1), date(2023, 12, 31))
    result = list(reader(period))
    assert result[0].data == 123
    assert result[1].data == 456


