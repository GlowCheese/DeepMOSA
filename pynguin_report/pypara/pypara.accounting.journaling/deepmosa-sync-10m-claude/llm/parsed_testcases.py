####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_guid_uniqueness. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_is_frozen. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2024
    var_7 = 2
    var_8 = 1
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 4/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry'
    var_4 = 'source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/30 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/31 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 12/38 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-100'
    var_14 = [var_13]

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-50'
    var_14 = [var_13]
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty entry'
    var_4 = 'test_source'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '1100'
    var_3 = 'Receivable'
    var_4 = '2000'
    var_5 = 'Payable'
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = 'Multiple postings'
    var_10 = 'test_source'
    var_11 = [var_6, var_7, var_7]
    var_12 = '50'
    var_13 = [var_12]
    var_14 = [var_6, var_7, var_7]
    var_15 = [var_12]
    var_16 = [var_6, var_7, var_7]
    var_17 = '-100'
    var_18 = [var_17]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 9/27 statements.


def test_case_0():
    var_0 = 'inbound'
    var_1 = 'outbound'
    var_2 = 'Cash'
    var_3 = 'asset'
    var_4 = 100.0
    var_5 = 'entry1'
    var_6 = 2024
    var_7 = 1
    var_8 = 15
    var_9 = [var_6, var_7, var_8]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 6/16 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 8/18 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 6/15 statements.
# Partially parsed test_post_multiple_postings. Retrieved 9/23 statements.
# Partially parsed test_post_returns_same_instance. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestAccount'
    var_6 = [var_1, var_2, var_2]
    var_7 = 100

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestAccount'
    var_6 = 15
    var_7 = [var_1, var_2, var_6]
    var_8 = -50
    var_9 = 50

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestAccount'
    var_6 = [var_1, var_2, var_2]
    var_7 = 0

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_1, var_2, var_2]
    var_8 = 2
    var_9 = [var_1, var_2, var_8]
    var_10 = 100
    var_11 = -100

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestAccount'
    var_6 = [var_1, var_2, var_2]
    var_7 = 100



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_is_frozen. Retrieved 3/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2024
    var_3 = 1
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2024
    var_3 = 1
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool('frozen' in str(type(e)).lower() or 'frozen' in str(e).lower())
    assert var_10 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 9/14 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = None

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2024
    var_7 = 2
    var_8 = 1
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/27 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/28 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_postings_balanced. Retrieved 13/35 statements.
# Partially parsed test_validate_zero_quantity_posting. Retrieved 11/32 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test entry'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-100'
    var_14 = [var_13]

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test entry'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-50'
    var_14 = [var_13]
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty entry'
    var_4 = 'test'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = '3000'
    var_5 = 'Revenue'
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = 'Multi-posting entry'
    var_10 = 'test'
    var_11 = [var_6, var_7, var_7]
    var_12 = '150'
    var_13 = [var_12]
    var_14 = [var_6, var_7, var_7]
    var_15 = '-100'
    var_16 = [var_15]
    var_17 = [var_6, var_7, var_7]
    var_18 = '-50'
    var_19 = [var_18]

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'With zero posting'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '0'
    var_14 = [var_13]
    var_15 = [var_4, var_5, var_5]
    var_16 = '-100'
    var_17 = [var_16]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'TestSource'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestAccount'
    var_6 = 'Asset'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-100'
    var_12 = [var_11]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 6/15 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 4/17 statements.


def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 2024
    var_9 = 2
    var_10 = 20
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2024
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 12
    var_6 = 31
    var_7 = [var_2, var_5, var_6]
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/19 statements.
# Partially parsed test_posting_is_frozen. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.0

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.0
    var_7 = 200.0
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/27 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'Source'
    var_6 = 2024
    var_7 = 2
    var_8 = 20
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Entry 1'
    var_5 = 'Source1'
    var_6 = 'Entry 2'
    var_7 = 'Source2'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 8/28 statements.


def test_case_0():
    var_0 = 'test_source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Asset'
    var_6 = 'Liability'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-100'
    var_12 = [var_11]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_post_with_zero_quantity. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_immutable. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '-50'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_predicate_total_debit_equals_total_credit. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/28 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 8/29 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 10/40 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test'
    var_5 = 'Asset'
    var_6 = 'Liability'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-100'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test'
    var_5 = 'Asset'
    var_6 = 'Liability'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty Entry'
    var_4 = 'test'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Multiple Postings'
    var_4 = 'test'
    var_5 = 'Asset'
    var_6 = 'Liability'
    var_7 = 'Equity'
    var_8 = [var_0, var_1, var_1]
    var_9 = '100'
    var_10 = [var_9]
    var_11 = [var_0, var_1, var_1]
    var_12 = '50'
    var_13 = [var_12]
    var_14 = [var_0, var_1, var_1]
    var_15 = '-75'
    var_16 = [var_15]
    var_17 = [var_0, var_1, var_1]
    var_18 = [var_15]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_balanced_entry. Retrieved 10/30 statements.
# Partially parsed test_validate_unbalanced_entry_raises_assertion. Retrieved 10/31 statements.
# Partially parsed test_validate_empty_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_postings_balanced. Retrieved 12/38 statements.
# Partially parsed test_validate_zero_quantity_posting. Retrieved 11/35 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test balanced entry'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-100'
    var_14 = [var_13]

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test unbalanced entry'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-50'
    var_14 = [var_13]
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test empty entry'
    var_4 = 'test'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '1100'
    var_3 = 'Receivable'
    var_4 = '2000'
    var_5 = 'Payable'
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = 'Test multiple postings balanced'
    var_10 = 'test'
    var_11 = [var_6, var_7, var_7]
    var_12 = '50'
    var_13 = [var_12]
    var_14 = [var_6, var_7, var_7]
    var_15 = [var_12]
    var_16 = [var_6, var_7, var_7]
    var_17 = '-100'
    var_18 = [var_17]

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test zero quantity posting'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '0'
    var_14 = [var_13]
    var_15 = [var_4, var_5, var_5]
    var_16 = '-100'
    var_17 = [var_16]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_balanced_entry. Retrieved 8/28 statements.
# Partially parsed test_validate_unbalanced_entry_raises. Retrieved 8/29 statements.
# Partially parsed test_validate_empty_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_postings. Retrieved 10/36 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test'
    var_5 = 'Cash'
    var_6 = 'Revenue'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-100'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test'
    var_5 = 'Cash'
    var_6 = 'Revenue'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test'
    var_5 = 'Cash'
    var_6 = 'Expense'
    var_7 = 'Revenue'
    var_8 = [var_0, var_1, var_1]
    var_9 = '100'
    var_10 = [var_9]
    var_11 = [var_0, var_1, var_1]
    var_12 = '50'
    var_13 = [var_12]
    var_14 = [var_0, var_1, var_1]
    var_15 = '-150'
    var_16 = [var_15]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2023
    var_7 = 2
    var_8 = 1
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 25/29 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 28/33 statements.


def test_case_0():
    var_0 = 'JournalEntry'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = var_3()
    var_5 = 'Account'
    var_6 = ()
    var_7 = 'type'
    var_8 = 'ASSET'
    var_9 = {var_7: var_8}
    var_10 = type(var_5, var_6, var_9)
    var_11 = var_10()
    var_12 = 'Direction'
    var_13 = ()
    var_14 = {}
    var_15 = type(var_12, var_13, var_14)
    var_16 = var_15()
    var_17 = 'Amount'
    var_18 = ()
    var_19 = {}
    var_20 = type(var_17, var_18, var_19)
    var_21 = var_20()
    var_22 = 2024
    var_23 = 1
    var_24 = 15
    var_25 = [var_22, var_23, var_24]

def test_case_0():
    var_0 = 'JournalEntry'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = var_3()
    var_5 = 'Account'
    var_6 = ()
    var_7 = 'type'
    var_8 = 'ASSET'
    var_9 = {var_7: var_8}
    var_10 = type(var_5, var_6, var_9)
    var_11 = var_10()
    var_12 = 'Direction'
    var_13 = ()
    var_14 = {}
    var_15 = type(var_12, var_13, var_14)
    var_16 = var_15()
    var_17 = 'Amount'
    var_18 = ()
    var_19 = {}
    var_20 = type(var_17, var_18, var_19)
    var_21 = var_20()
    var_22 = 2024
    var_23 = 1
    var_24 = 15
    var_25 = [var_22, var_23, var_24]
    var_26 = 2024
    var_27 = 2
    var_28 = 20
    var_29 = [var_26, var_27, var_28]
    var_30 = bool(False)
    assert var_30 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '-50'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 7/21 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 8/22 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 7/19 statements.
# Partially parsed test_post_multiple_postings_chaining. Retrieved 10/28 statements.
# Partially parsed test_post_preserves_journal_reference. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 'test_source'
    var_6 = 'Test entry'
    var_7 = 'TestAccount'
    var_8 = '100.00'
    var_9 = [var_8]
    var_10 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 'test_source'
    var_6 = 'Test entry'
    var_7 = 'TestAccount'
    var_8 = '-50.00'
    var_9 = [var_8]
    var_10 = '50.00'
    var_11 = [var_10]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 'test_source'
    var_6 = 'Test entry'
    var_7 = 'TestAccount'
    var_8 = '0.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 16
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_1]
    var_7 = 'test_source'
    var_8 = 'Test entry'
    var_9 = 'TestAccount1'
    var_10 = 'TestAccount2'
    var_11 = '100.00'
    var_12 = [var_11]
    var_13 = '-100.00'
    var_14 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_1]
    var_5 = 'test_source'
    var_6 = 'Test entry'
    var_7 = 'TestAccount'
    var_8 = '100.00'
    var_9 = [var_8]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_postings_default. Retrieved 7/14 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Frozen test'
    var_4 = 'immutable'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Default postings test'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 3
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'GUID uniqueness test'
    var_5 = 'test'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_post_with_non_zero_quantity. Retrieved 7/22 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Test Account'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 'Test entry'
    var_6 = 'test_source'
    var_7 = 100
    var_8 = [var_2, var_3, var_3]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 6/19 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 7/20 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 6/18 statements.
# Partially parsed test_post_multiple_postings. Retrieved 8/26 statements.
# Partially parsed test_post_returns_same_journal_entry. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = 100
    var_5 = 'TestSource'
    var_6 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = -50
    var_5 = 'TestSource'
    var_6 = 'Test Entry'
    var_7 = 50

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = 0
    var_5 = 'TestSource'
    var_6 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount1'
    var_4 = 'TestAccount2'
    var_5 = 100
    var_6 = -100
    var_7 = 'TestSource'
    var_8 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = 50
    var_5 = 'TestSource'
    var_6 = 'Test Entry'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Holiday transaction'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/30 statements.
# Partially parsed test_validate_unbalanced_journal_entry. Retrieved 10/31 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_postings_balanced. Retrieved 13/39 statements.
# Partially parsed test_validate_zero_quantity_posting. Retrieved 11/35 statements.


def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Test Account 1'
    var_2 = 'ACC002'
    var_3 = 'Test Account 2'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test Entry'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-100'
    var_14 = [var_13]

def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Test Account 1'
    var_2 = 'ACC002'
    var_3 = 'Test Account 2'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test Entry'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-50'
    var_14 = [var_13]
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty Entry'
    var_4 = 'test_source'

def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Test Account 1'
    var_2 = 'ACC002'
    var_3 = 'Test Account 2'
    var_4 = 'ACC003'
    var_5 = 'Test Account 3'
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = 'Multi Posting Entry'
    var_10 = 'test_source'
    var_11 = [var_6, var_7, var_7]
    var_12 = '150'
    var_13 = [var_12]
    var_14 = [var_6, var_7, var_7]
    var_15 = '-100'
    var_16 = [var_15]
    var_17 = [var_6, var_7, var_7]
    var_18 = '-50'
    var_19 = [var_18]

def test_case_0():
    var_0 = 'ACC001'
    var_1 = 'Test Account 1'
    var_2 = 'ACC002'
    var_3 = 'Test Account 2'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test Entry'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-100'
    var_14 = [var_13]
    var_15 = [var_4, var_5, var_5]
    var_16 = '0'
    var_17 = [var_16]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/23 statements.
# Partially parsed test_read_journal_entries_call_empty. Retrieved 3/22 statements.
# Partially parsed test_read_journal_entries_call_with_generator. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 30
    var_5 = [var_0, var_1, var_4]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_balanced_entry. Retrieved 10/27 statements.
# Partially parsed test_validate_unbalanced_entry_raises_assertion_error. Retrieved 10/28 statements.
# Partially parsed test_validate_empty_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 11/33 statements.
# Partially parsed test_validate_zero_quantity_not_posted. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'ASSET'
    var_7 = 'Account2'
    var_8 = 'LIABILITY'
    var_9 = [var_0, var_1, var_1]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_0, var_1, var_1]
    var_13 = '-100'
    var_14 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'ASSET'
    var_7 = 'Account2'
    var_8 = 'LIABILITY'
    var_9 = [var_0, var_1, var_1]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_0, var_1, var_1]
    var_13 = '-50'
    var_14 = [var_13]
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty entry'
    var_4 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Multi posting entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'ASSET'
    var_7 = 'Account2'
    var_8 = 'Account3'
    var_9 = 'LIABILITY'
    var_10 = [var_0, var_1, var_1]
    var_11 = '50'
    var_12 = [var_11]
    var_13 = [var_0, var_1, var_1]
    var_14 = [var_11]
    var_15 = [var_0, var_1, var_1]
    var_16 = '-100'
    var_17 = [var_16]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Zero quantity entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'ASSET'
    var_7 = [var_0, var_1, var_1]
    var_8 = '0'
    var_9 = [var_8]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/30 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/31 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_debits_and_credits. Retrieved 15/47 statements.
# Partially parsed test_validate_with_zero_quantity_postings. Retrieved 11/35 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test entry'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-100'
    var_14 = [var_13]

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test entry'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '-50'
    var_14 = [var_13]
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty entry'
    var_4 = 'test'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '1100'
    var_3 = 'Receivable'
    var_4 = '2000'
    var_5 = 'Payable'
    var_6 = '3000'
    var_7 = 'Revenue'
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = 'Complex entry'
    var_12 = 'test'
    var_13 = [var_8, var_9, var_9]
    var_14 = '60'
    var_15 = [var_14]
    var_16 = [var_8, var_9, var_9]
    var_17 = '40'
    var_18 = [var_17]
    var_19 = [var_8, var_9, var_9]
    var_20 = '-50'
    var_21 = [var_20]
    var_22 = [var_8, var_9, var_9]
    var_23 = [var_20]

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'With zero posting'
    var_8 = 'test'
    var_9 = [var_4, var_5, var_5]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = '0'
    var_14 = [var_13]
    var_15 = [var_4, var_5, var_5]
    var_16 = '-100'
    var_17 = [var_16]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 2
    var_8 = 1
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_equal_debits_and_credits. Retrieved 10/30 statements.


def test_case_0():
    var_0 = 'TestSource'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = 'Payable'
    var_9 = [var_1, var_2, var_2]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_1, var_2, var_2]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_equal_debits_and_credits. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'test_source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Asset'
    var_6 = 'Liability'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = [var_1, var_2, var_2]
    var_10 = [var_1, var_2, var_2]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_immutability. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Immutable test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'source'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_equal_debits_and_credits. Retrieved 8/28 statements.


def test_case_0():
    var_0 = 'Test Source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Cash'
    var_6 = 'Sales'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-100'
    var_12 = [var_11]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_post_with_non_zero_quantity. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/28 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/29 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 12/35 statements.
# Partially parsed test_validate_zero_quantity_not_posted. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '2000'
    var_6 = 'Payable'
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = '100'
    var_10 = [var_9]
    var_11 = '-100'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '2000'
    var_6 = 'Payable'
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = '100'
    var_10 = [var_9]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty entry'
    var_4 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '1100'
    var_6 = 'Receivable'
    var_7 = '2000'
    var_8 = 'Payable'
    var_9 = 'Multi-posting entry'
    var_10 = 'test_source'
    var_11 = '50'
    var_12 = [var_11]
    var_13 = [var_11]
    var_14 = '-100'
    var_15 = [var_14]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = 'Zero posting entry'
    var_6 = 'test_source'
    var_7 = '0'
    var_8 = [var_7]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 4/13 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 7/17 statements.


def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = '100.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = 2024
    var_10 = 1
    var_11 = 16
    var_12 = [var_9, var_10, var_11]
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_creates_unique_guids. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_default_empty. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = []

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '-50'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_type. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2023
    var_7 = 2
    var_8 = 1
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_postings_not_init_parameter. Retrieved 6/12 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 'postings'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'Source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Entry 1'
    var_5 = 'Source1'
    var_6 = 'Entry 2'
    var_7 = 'Source2'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_in_init. Retrieved 6/12 statements.
# Partially parsed test_journal_entry_constructor_guid_generated. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 42

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'
    var_6 = 2023
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'
    var_6 = 'postings'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/26 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'src'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/25 statements.
# Partially parsed test_read_journal_entries_call_with_entries. Retrieved 3/23 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = '__iter__'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_equal_debits_and_credits. Retrieved 10/30 statements.


def test_case_0():
    var_0 = 'Test Transaction'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry with balanced debits and credits'
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '4000'
    var_8 = 'Sales'
    var_9 = [var_1, var_2, var_2]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_1, var_2, var_2]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'asset'
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'debit'
    var_6 = 100.0



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '-50'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test'
    var_4 = 'src'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_append_posting. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_is_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'source'
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_constructor_with_keyword_arguments. Retrieved 3/10 statements.
# Partially parsed test_posting_is_frozen. Retrieved 6/15 statements.


def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 6
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 2024
    var_9 = 12
    var_10 = 31
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test'
    var_4 = 'src'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'src1'
    var_5 = [var_0, var_1, var_1]



