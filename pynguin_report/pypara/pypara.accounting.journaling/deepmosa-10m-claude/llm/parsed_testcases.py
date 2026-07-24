####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


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
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_balanced_entry. Retrieved 8/26 statements.
# Partially parsed test_validate_unbalanced_entry_raises_assertion_error. Retrieved 8/27 statements.
# Partially parsed test_validate_empty_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_zero_quantity_not_posted. Retrieved 6/18 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 10/32 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test'
    var_5 = 'Cash'
    var_6 = 'Revenue'
    var_7 = [var_0, var_1, var_1]
    var_8 = 100
    var_9 = [var_0, var_1, var_1]
    var_10 = -100

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test'
    var_5 = 'Cash'
    var_6 = 'Revenue'
    var_7 = [var_0, var_1, var_1]
    var_8 = 100
    var_9 = [var_0, var_1, var_1]
    var_10 = -50
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Empty Entry'
    var_4 = 'test'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Zero Quantity Entry'
    var_4 = 'test'
    var_5 = 'Cash'
    var_6 = [var_0, var_1, var_1]
    var_7 = 0

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Multiple Postings'
    var_4 = 'test'
    var_5 = 'Cash'
    var_6 = 'Expense'
    var_7 = 'Revenue'
    var_8 = [var_0, var_1, var_1]
    var_9 = 100
    var_10 = [var_0, var_1, var_1]
    var_11 = 50
    var_12 = [var_0, var_1, var_1]
    var_13 = -150



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 7/15 statements.


def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = 100
    var_7 = 'USD'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/30 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises. Retrieved 10/31 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 13/39 statements.


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
    var_3 = 'Test entry'
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
    var_9 = 'Test entry'
    var_10 = 'test'
    var_11 = [var_6, var_7, var_7]
    var_12 = '60'
    var_13 = [var_12]
    var_14 = [var_6, var_7, var_7]
    var_15 = '40'
    var_16 = [var_15]
    var_17 = [var_6, var_7, var_7]
    var_18 = '-100'
    var_19 = [var_18]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 2023
    var_7 = 6
    var_8 = 2
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry 1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source_object'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/17 statements.


def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Account'
    var_6 = 100
    var_7 = 'USD'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
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



# Parsed testcases at query #10
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
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

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
    var_4 = 'Source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 8/28 statements.


def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Test Debit Account'
    var_6 = 'Test Credit Account'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-100'
    var_12 = [var_11]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 9/14 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
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
    var_9 = None

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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 5/8 statements.
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
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 42

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
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
    var_5 = 'test_source'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 6/19 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 6/19 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 6/19 statements.
# Partially parsed test_post_multiple_postings. Retrieved 8/26 statements.
# Partially parsed test_post_chaining. Retrieved 8/26 statements.
# Partially parsed test_post_preserves_posting_date. Retrieved 8/19 statements.
# Partially parsed test_post_uses_absolute_value_for_amount. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Test Account'
    var_6 = 100
    var_7 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Test Account'
    var_6 = -50
    var_7 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Test Account'
    var_6 = 0
    var_7 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Account 1'
    var_6 = 'Account 2'
    var_7 = [var_0, var_1, var_1]
    var_8 = 100
    var_9 = [var_0, var_1, var_1]
    var_10 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Account 1'
    var_6 = 'Account 2'
    var_7 = [var_0, var_1, var_1]
    var_8 = 100
    var_9 = [var_0, var_1, var_1]
    var_10 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Test Account'
    var_6 = 6
    var_7 = 15
    var_8 = [var_0, var_6, var_7]
    var_9 = 100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Test Account'
    var_6 = [var_0, var_1, var_1]
    var_7 = -150
    var_8 = -150



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_read_journal_entries_protocol_call. Retrieved 5/17 statements.
# Partially parsed test_read_journal_entries_protocol_call_empty_result. Retrieved 5/16 statements.
# Partially parsed test_read_journal_entries_protocol_call_with_period_range. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'start'
    var_1 = 'end'
    var_2 = 2024
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 31
    var_6 = [var_2, var_3, var_5]

def test_case_0():
    var_0 = 'start'
    var_1 = 'end'
    var_2 = 2024
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 31
    var_6 = [var_2, var_3, var_5]

def test_case_0():
    var_0 = 'start'
    var_1 = 'end'
    var_2 = 2024
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 31
    var_6 = [var_2, var_3, var_5]
    var_7 = 15
    var_8 = [var_2, var_3, var_7]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_post_with_non_zero_quantity. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = 100
    var_5 = 'TestSource'
    var_6 = 'Test Entry'
    var_7 = -1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_guid_uniqueness. Retrieved 4/9 statements.


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
    var_10 = bool(True)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 7/11 statements.


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
    var_4 = 'Test'
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
    var_4 = 'Test 1'
    var_5 = 'source1'
    var_6 = 'Test 2'
    var_7 = 'source2'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_balanced_entry. Retrieved 10/28 statements.
# Partially parsed test_validate_unbalanced_entry_raises. Retrieved 10/29 statements.
# Partially parsed test_validate_zero_quantity_entry. Retrieved 7/19 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 13/35 statements.
# Partially parsed test_validate_empty_entry. Retrieved 4/9 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test Entry'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = 100
    var_11 = [var_4, var_5, var_5]
    var_12 = -100

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test Entry'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = 100
    var_11 = [var_4, var_5, var_5]
    var_12 = -50
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 'Test Entry'
    var_6 = 'test_source'
    var_7 = [var_2, var_3, var_3]
    var_8 = 0

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
    var_9 = 'Test Entry'
    var_10 = 'test_source'
    var_11 = [var_6, var_7, var_7]
    var_12 = 60
    var_13 = [var_6, var_7, var_7]
    var_14 = 40
    var_15 = [var_6, var_7, var_7]
    var_16 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'test_source'



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 6/15 statements.


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
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 'TestSource'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Assets'
    var_6 = 'ASSET'
    var_7 = 'Liabilities'
    var_8 = 'LIABILITY'
    var_9 = [var_1, var_2, var_2]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_1, var_2, var_2]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 4/8 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 3
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test transaction'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'

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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/16 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 6/16 statements.
# Partially parsed test_posting_is_frozen. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.5

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'liability'
    var_5 = 'credit'
    var_6 = 250.75

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.0
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


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
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 42

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 10
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Frozen test'
    var_5 = 'immutable'
    var_6 = 2023
    var_7 = 6
    var_8 = 11
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #27
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
    var_5 = 'TestSource'

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
    var_4 = 'Source'
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
    var_3 = 'Entry1'
    var_4 = 'Source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 6/15 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []
    var_8 = 2023
    var_9 = 2
    var_10 = 20
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'TestSource'
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



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_with_balanced_debits_and_credits. Retrieved 10/30 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = 'Payable'
    var_9 = [var_0, var_1, var_1]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_0, var_1, var_1]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/26 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/27 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_postings_balanced. Retrieved 13/32 statements.


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
    var_9 = 100
    var_10 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '2000'
    var_6 = 'Payable'
    var_7 = 'Unbalanced entry'
    var_8 = 'test_source'
    var_9 = 100
    var_10 = -50
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'

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
    var_9 = 'Multiple postings'
    var_10 = 'test_source'
    var_11 = 100
    var_12 = 50
    var_13 = -150



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 7/15 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 7/14 statements.
# Partially parsed test_posting_is_frozen. Retrieved 9/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Account'
    var_6 = 100
    var_7 = 'USD'

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 6
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Expense Account'
    var_6 = 250
    var_7 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Account'
    var_6 = 100
    var_7 = 'USD'
    var_8 = 200
    var_9 = 'USD'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/19 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 6/19 statements.
# Partially parsed test_posting_is_frozen. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.0

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'liability'
    var_5 = 'credit'
    var_6 = 250.5

def test_case_0():
    var_0 = 2023
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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 4/9 statements.


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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry 1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/14 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_guid_uniqueness. Retrieved 7/11 statements.


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
    var_4 = 'Test'
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
    var_4 = 'Entry 1'
    var_5 = 'source1'
    var_6 = 'Entry 2'
    var_7 = 'source2'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_in_init. Retrieved 6/12 statements.
# Partially parsed test_journal_entry_constructor_guid_generated. Retrieved 5/9 statements.


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
    var_6 = 'postings'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_guid_uniqueness. Retrieved 4/9 statements.


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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'GUID test'
    var_4 = 'test'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/28 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/29 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 13/36 statements.
# Partially parsed test_validate_zero_quantity_posting. Retrieved 11/32 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'ACC001'
    var_4 = 'Test Account 1'
    var_5 = 'ACC002'
    var_6 = 'Test Account 2'
    var_7 = 'Test Entry'
    var_8 = 'TestSource'
    var_9 = '100'
    var_10 = [var_9]
    var_11 = '-100'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'ACC001'
    var_4 = 'Test Account 1'
    var_5 = 'ACC002'
    var_6 = 'Test Account 2'
    var_7 = 'Test Entry'
    var_8 = 'TestSource'
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
    var_3 = 'Empty Entry'
    var_4 = 'TestSource'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'ACC001'
    var_4 = 'Test Account 1'
    var_5 = 'ACC002'
    var_6 = 'Test Account 2'
    var_7 = 'ACC003'
    var_8 = 'Test Account 3'
    var_9 = 'Test Entry'
    var_10 = 'TestSource'
    var_11 = '100'
    var_12 = [var_11]
    var_13 = '50'
    var_14 = [var_13]
    var_15 = '-150'
    var_16 = [var_15]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'ACC001'
    var_4 = 'Test Account 1'
    var_5 = 'ACC002'
    var_6 = 'Test Account 2'
    var_7 = 'Test Entry'
    var_8 = 'TestSource'
    var_9 = '100'
    var_10 = [var_9]
    var_11 = '0'
    var_12 = [var_11]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_postings_default. Retrieved 4/11 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 4/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

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
    var_3 = 'Test'
    var_4 = 'Source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry1'
    var_4 = 'Source1'
    var_5 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_guid_uniqueness. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2024
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
    var_4 = 'Test frozen'
    var_5 = 'Source'
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
    var_4 = 'Test uniqueness'
    var_5 = 'Source'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_post_with_zero_quantity. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

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



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 7/11 statements.


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
    var_4 = 'Test'
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
    var_4 = 'Test1'
    var_5 = 'source1'
    var_6 = 'Test2'
    var_7 = 'source2'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
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
    var_4 = 'Another test'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

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
    var_3 = 'Entry 1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/30 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/31 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 13/39 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test'
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
    var_7 = 'Test'
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
    var_3 = 'Test'
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
    var_9 = 'Test'
    var_10 = 'test_source'
    var_11 = [var_6, var_7, var_7]
    var_12 = '60'
    var_13 = [var_12]
    var_14 = [var_6, var_7, var_7]
    var_15 = '40'
    var_16 = [var_15]
    var_17 = [var_6, var_7, var_7]
    var_18 = '-100'
    var_19 = [var_18]



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 13/17 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 13/16 statements.
# Partially parsed test_posting_is_frozen. Retrieved 13/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Account'
    var_6 = ()
    var_7 = 'type'
    var_8 = 'ASSET'
    var_9 = {var_7: var_8}
    var_10 = type(var_5, var_6, var_9)
    var_11 = var_10()
    var_12 = 'DEBIT'
    var_13 = 100.0

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Account'
    var_5 = ()
    var_6 = 'type'
    var_7 = 'LIABILITY'
    var_8 = {var_6: var_7}
    var_9 = type(var_4, var_5, var_8)
    var_10 = var_9()
    var_11 = 'CREDIT'
    var_12 = 250.5
    var_13 = None

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Account'
    var_5 = ()
    var_6 = 'type'
    var_7 = 'ASSET'
    var_8 = {var_6: var_7}
    var_9 = type(var_4, var_5, var_8)
    var_10 = var_9()
    var_11 = None
    var_12 = 'DEBIT'
    var_13 = 100.0
    var_14 = bool(False)
    assert var_14 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_append. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = []
    var_5 = 0
    var_6 = 'Test'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_append_posting. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 8/22 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 9/23 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 8/21 statements.
# Partially parsed test_post_multiple_postings. Retrieved 12/30 statements.
# Partially parsed test_post_chaining. Retrieved 12/30 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 100
    var_8 = 'Test entry'
    var_9 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = '2000'
    var_6 = 'Liabilities'
    var_7 = -50
    var_8 = 'Test entry'
    var_9 = 'test_source'
    var_10 = 50

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 0
    var_8 = 'Test entry'
    var_9 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '3000'
    var_6 = 'Revenue'
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = 15
    var_10 = [var_0, var_1, var_9]
    var_11 = 100
    var_12 = 16
    var_13 = [var_0, var_1, var_12]
    var_14 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '3000'
    var_6 = 'Revenue'
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = 15
    var_10 = [var_0, var_1, var_9]
    var_11 = 100
    var_12 = 16
    var_13 = [var_0, var_1, var_12]
    var_14 = -100



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/26 statements.
# Partially parsed test_read_journal_entries_call_empty_period. Retrieved 4/24 statements.
# Partially parsed test_read_journal_entries_call_with_multiple_entries. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 30
    var_5 = [var_0, var_1, var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_post_with_nonzero_quantity. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = 100
    var_5 = []
    var_6 = 'Test Entry'
    var_7 = -1



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #58
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
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
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
    var_8 = 15
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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/14 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


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
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Frozen test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 7/28 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 7/28 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 7/26 statements.
# Partially parsed test_post_multiple_postings. Retrieved 10/36 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 'Test entry'
    var_8 = 100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = '2000'
    var_6 = 'Liabilities'
    var_7 = 'Test entry'
    var_8 = -50

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 'Test entry'
    var_8 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 15
    var_4 = [var_0, var_1, var_3]
    var_5 = 20
    var_6 = [var_0, var_1, var_5]
    var_7 = '1000'
    var_8 = 'Cash'
    var_9 = '3000'
    var_10 = 'Revenue'
    var_11 = 'Test entry'
    var_12 = 100



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 6/10 statements.


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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'entry1'
    var_4 = 'source1'
    var_5 = 'entry2'
    var_6 = 'source2'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 11/16 statements.
# Partially parsed test_journal_entry_constructor_creates_unique_guids. Retrieved 5/9 statements.
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
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 2
    var_10 = 3
    var_11 = [var_1, var_9, var_10]

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



# Parsed testcases at query #63
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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


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
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_init. Retrieved 6/10 statements.


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
    var_4 = 'Test'
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
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 'postings'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_date. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_creates_unique_guids. Retrieved 5/9 statements.
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
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Christmas entry'
    var_5 = 42

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



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 7/15 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 7/14 statements.
# Partially parsed test_posting_is_frozen. Retrieved 9/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Account'
    var_6 = 100
    var_7 = 'USD'

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 12
    var_3 = 31
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Another Account'
    var_6 = 250
    var_7 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test'
    var_6 = 100
    var_7 = 'USD'
    var_8 = 200
    var_9 = 'USD'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_not_equal_credits. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Assets'
    var_6 = 'Expenses'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 7/15 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 7/14 statements.
# Partially parsed test_posting_is_frozen. Retrieved 9/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = 100
    var_7 = 'USD'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Accounts Payable'
    var_5 = 500
    var_6 = 'EUR'
    var_7 = None

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = 100
    var_7 = 'USD'
    var_8 = 200
    var_9 = 'USD'
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/21 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 8/22 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/7 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 9/27 statements.
# Partially parsed test_validate_zero_quantity_not_posted. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_1, var_2, var_2]
    var_8 = 100
    var_9 = [var_1, var_2, var_2]
    var_10 = -100

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_1, var_2, var_2]
    var_8 = 100
    var_9 = [var_1, var_2, var_2]
    var_10 = -50
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = 'Account3'
    var_8 = [var_1, var_2, var_2]
    var_9 = 50
    var_10 = [var_1, var_2, var_2]
    var_11 = [var_1, var_2, var_2]
    var_12 = -100

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Account1'
    var_6 = [var_1, var_2, var_2]
    var_7 = 0



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_validate_with_balanced_debits_and_credits. Retrieved 10/26 statements.


def test_case_0():
    var_0 = 'TestSource'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Assets'
    var_6 = 'Asset'
    var_7 = 'Liabilities'
    var_8 = 'Liability'
    var_9 = [var_1, var_2, var_2]
    var_10 = 100
    var_11 = [var_1, var_2, var_2]
    var_12 = -100



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 9/14 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_postings_empty_by_default. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'string_source'
    var_6 = 42
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}

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
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2024
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
    var_5 = 'Source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'Source'
    var_6 = 2023
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 4/8 statements.


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
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Frozen test'
    var_4 = 'test'
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
    var_3 = 'Unique test'
    var_4 = 'test'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_postings_not_in_init. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_guid_generated. Retrieved 6/11 statements.


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
    var_4 = 'source'
    var_5 = 'postings'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry1'
    var_4 = 'src1'
    var_5 = [var_0, var_1, var_1]
    var_6 = 'Entry2'
    var_7 = 'src2'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/14 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guid. Retrieved 4/8 statements.
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
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry 1'
    var_4 = 'source1'

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



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 7/20 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 8/21 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 7/18 statements.
# Partially parsed test_post_multiple_postings. Retrieved 9/25 statements.
# Partially parsed test_post_returns_self_for_chaining. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'
    var_9 = [var_5]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '-50.00'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'
    var_9 = '50.00'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '0'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Account 1'
    var_5 = 'Account 2'
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = '-100.00'
    var_9 = [var_8]
    var_10 = 'test_source'
    var_11 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_postings_default. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source_object'

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
    var_3 = 'Entry 1'
    var_4 = 'src1'
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = 'Entry 2'
    var_8 = 'src2'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry 1'
    var_4 = 'src1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_post_with_non_zero_quantity_adds_posting. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 7/15 statements.
# Partially parsed test_posting_frozen. Retrieved 10/19 statements.


def test_case_0():
    var_0 = None
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = 100
    var_7 = 'USD'

def test_case_0():
    var_0 = None
    var_1 = 2024
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = 100
    var_7 = 'USD'
    var_8 = 2024
    var_9 = 1
    var_10 = 16
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #82
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
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
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
    var_5 = 'test_source'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_init. Retrieved 6/12 statements.


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
    var_4 = 'Another entry'
    var_5 = 12345

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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 'postings'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_immutability. Retrieved 7/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 6/10 statements.


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
    var_5 = 12345

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
    var_3 = 'Entry 1'
    var_4 = 'source1'
    var_5 = 'Entry 2'
    var_6 = 'source2'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_init. Retrieved 6/12 statements.


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
    var_4 = 'Test'
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
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 'postings'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/28 statements.


def test_case_0():
    var_0 = 'Test Transaction'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Cash'
    var_6 = 'Expense'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-100'
    var_12 = [var_11]



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_not_equal_credits. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 9/25 statements.


def test_case_0():
    var_0 = 'TestSource'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Asset'
    var_6 = 'ASSET'
    var_7 = 'Liability'
    var_8 = 'LIABILITY'
    var_9 = '100'
    var_10 = [var_9]
    var_11 = [var_1, var_2, var_2]
    var_12 = [var_1, var_2, var_2]



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 5/9 statements.


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
    var_4 = 'Another test'
    var_5 = 42

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



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/16 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 6/16 statements.
# Partially parsed test_posting_is_frozen. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.5

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'liability'
    var_5 = 'credit'
    var_6 = 250.75

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.0
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_post_with_non_zero_quantity. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_read_journal_entries_protocol_call. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/13 statements.
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
    var_4 = 'Another test'
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



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_append_posting. Retrieved 3/19 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #95
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
    var_4 = 'Test journal entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test frozen'
    var_5 = 'test'
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
    var_4 = 'Test'
    var_5 = 'test'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source_object'

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
    var_3 = 'Test'
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
    var_3 = 'Test1'
    var_4 = 'src1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 6/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'

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
    var_3 = 'Entry 1'
    var_4 = 'Source1'
    var_5 = 'Entry 2'
    var_6 = 'Source2'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/30 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/31 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 13/39 statements.


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
    var_9 = 'Test entry'
    var_10 = 'test_source'
    var_11 = [var_6, var_7, var_7]
    var_12 = '100'
    var_13 = [var_12]
    var_14 = [var_6, var_7, var_7]
    var_15 = '50'
    var_16 = [var_15]
    var_17 = [var_6, var_7, var_7]
    var_18 = '-150'
    var_19 = [var_18]



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 21/27 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 22/29 statements.


def test_case_0():
    var_0 = 'JournalEntry'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = var_3()
    var_5 = 2024
    var_6 = 1
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = 'Account'
    var_10 = ()
    var_11 = 'type'
    var_12 = 'asset'
    var_13 = {var_11: var_12}
    var_14 = type(var_9, var_10, var_13)
    var_15 = var_14()
    var_16 = 'Direction'
    var_17 = ()
    var_18 = {}
    var_19 = type(var_16, var_17, var_18)
    var_20 = var_19()
    var_21 = '100.00'
    var_22 = [var_21]

def test_case_0():
    var_0 = 'JournalEntry'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = var_3()
    var_5 = 2024
    var_6 = 1
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = 'Account'
    var_10 = ()
    var_11 = 'type'
    var_12 = 'asset'
    var_13 = {var_11: var_12}
    var_14 = type(var_9, var_10, var_13)
    var_15 = var_14()
    var_16 = 'Direction'
    var_17 = ()
    var_18 = {}
    var_19 = type(var_16, var_17, var_18)
    var_20 = var_19()
    var_21 = '100.00'
    var_22 = [var_21]
    var_23 = '200.00'
    var_24 = [var_23]
    var_25 = bool(False)
    assert var_25 is True



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_append_posting. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/13 statements.
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
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
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



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 'test_source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Account A'
    var_6 = 'asset'
    var_7 = 'Account B'
    var_8 = 'liability'
    var_9 = [var_1, var_2, var_2]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_1, var_2, var_2]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_post_with_non_zero_quantity. Retrieved 4/31 statements.


def test_case_0():
    var_0 = '100'
    var_1 = [var_0]
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 'Test'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_not_equal_credits. Retrieved 8/27 statements.


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



# Parsed testcases at query #105
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
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Frozen test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 5/9 statements.


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
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test frozen'
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
    var_4 = 'Test guid uniqueness'
    var_5 = 'source'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_read_journal_entries_protocol_call. Retrieved 4/26 statements.
# Partially parsed test_read_journal_entries_protocol_call_empty. Retrieved 3/23 statements.
# Partially parsed test_read_journal_entries_protocol_call_generator. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/24 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 5/12 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '_T'
    var_1 = []
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = 'asset'
    var_7 = 100.0

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'liability'
    var_5 = 250.5

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'equity'
    var_5 = 500.0
    var_6 = 600.0
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_unique_guid. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'
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
    var_5 = 'TestSource'



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/14 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 7/11 statements.


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
    var_4 = 'Another test'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Frozen test'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 6
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 3
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Entry 1'
    var_5 = 'source1'
    var_6 = 'Entry 2'
    var_7 = 'source2'



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_generic_type. Retrieved 8/11 statements.
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
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'id'
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_5: var_1, var_6: var_7}

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



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 5/9 statements.
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
    var_2 = 20
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



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_is_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'

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
    var_3 = 'Test'
    var_4 = 'Source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 3/12 statements.


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
    var_8 = []
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 4/9 statements.


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
    var_5 = 'type'
    var_6 = 'dict_source'
    var_7 = {var_5: var_6}

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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 13/17 statements.
# Partially parsed test_posting_immutability. Retrieved 13/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Account'
    var_6 = ()
    var_7 = 'type'
    var_8 = 'asset'
    var_9 = {var_7: var_8}
    var_10 = type(var_5, var_6, var_9)
    var_11 = var_10()
    var_12 = 'debit'
    var_13 = 100.5

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Account'
    var_6 = ()
    var_7 = 'type'
    var_8 = 'asset'
    var_9 = {var_7: var_8}
    var_10 = type(var_5, var_6, var_9)
    var_11 = var_10()
    var_12 = 'debit'
    var_13 = 100.5
    var_14 = bool(False)
    assert var_14 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/28 statements.
# Partially parsed test_validate_unbalanced_journal_entry. Retrieved 8/29 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 10/36 statements.


def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Revenue'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 'Test entry'
    var_6 = 'test_source'
    var_7 = [var_2, var_3, var_3]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_2, var_3, var_3]
    var_11 = '-100'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Revenue'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 'Test entry'
    var_6 = 'test_source'
    var_7 = [var_2, var_3, var_3]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_2, var_3, var_3]
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
    var_4 = 'test_source'

def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Receivables'
    var_2 = 'Revenue'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test entry'
    var_7 = 'test_source'
    var_8 = [var_3, var_4, var_4]
    var_9 = '60'
    var_10 = [var_9]
    var_11 = [var_3, var_4, var_4]
    var_12 = '40'
    var_13 = [var_12]
    var_14 = [var_3, var_4, var_4]
    var_15 = '-100'
    var_16 = [var_15]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 7/11 statements.


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
    var_8 = 20
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Entry 1'
    var_5 = 'source1'
    var_6 = 'Entry 2'
    var_7 = 'source2'



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/28 statements.


def test_case_0():
    var_0 = 'TestSource'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Cash'
    var_6 = 'Revenue'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-100'
    var_12 = [var_11]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 25/32 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 30/38 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 25/32 statements.
# Partially parsed test_post_multiple_postings. Retrieved 35/45 statements.
# Partially parsed test_post_returns_same_entry_for_chaining. Retrieved 28/37 statements.


def test_case_0():
    var_0 = 'datetime'
    var_1 = __import__(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 'test_source'
    var_5 = 'pypara.accounting.journaling'
    var_6 = 'JournalEntry'
    var_7 = [var_6]
    var_8 = __import__(var_5, fromlist=var_7)
    var_9 = 'Test Entry'
    var_10 = 'pypara.accounting.accounts'
    var_11 = 'Account'
    var_12 = [var_11]
    var_13 = __import__(var_10, fromlist=var_12)
    var_14 = '1000'
    var_15 = 'Cash'
    var_16 = 'AccountType'
    var_17 = [var_16]
    var_18 = __import__(var_10, fromlist=var_17)
    var_19 = var_18.AccountType.ASSET
    var_20 = 'pypara.accounting.quantity'
    var_21 = 'Quantity'
    var_22 = [var_21]
    var_23 = __import__(var_20, fromlist=var_22)
    var_24 = 100

def test_case_0():
    var_0 = 'datetime'
    var_1 = __import__(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 'test_source'
    var_5 = 'pypara.accounting.journaling'
    var_6 = 'JournalEntry'
    var_7 = [var_6]
    var_8 = __import__(var_5, fromlist=var_7)
    var_9 = 'Test Entry'
    var_10 = 'pypara.accounting.accounts'
    var_11 = 'Account'
    var_12 = [var_11]
    var_13 = __import__(var_10, fromlist=var_12)
    var_14 = '2000'
    var_15 = 'Liability'
    var_16 = 'AccountType'
    var_17 = [var_16]
    var_18 = __import__(var_10, fromlist=var_17)
    var_19 = var_18.AccountType.LIABILITY
    var_20 = 'pypara.accounting.quantity'
    var_21 = 'Quantity'
    var_22 = [var_21]
    var_23 = __import__(var_20, fromlist=var_22)
    var_24 = -50
    var_25 = 'pypara.accounting.amount'
    var_26 = 'Amount'
    var_27 = [var_26]
    var_28 = __import__(var_25, fromlist=var_27)
    var_29 = 50

def test_case_0():
    var_0 = 'datetime'
    var_1 = __import__(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 'test_source'
    var_5 = 'pypara.accounting.journaling'
    var_6 = 'JournalEntry'
    var_7 = [var_6]
    var_8 = __import__(var_5, fromlist=var_7)
    var_9 = 'Test Entry'
    var_10 = 'pypara.accounting.accounts'
    var_11 = 'Account'
    var_12 = [var_11]
    var_13 = __import__(var_10, fromlist=var_12)
    var_14 = '3000'
    var_15 = 'Equity'
    var_16 = 'AccountType'
    var_17 = [var_16]
    var_18 = __import__(var_10, fromlist=var_17)
    var_19 = var_18.AccountType.EQUITY
    var_20 = 'pypara.accounting.quantity'
    var_21 = 'Quantity'
    var_22 = [var_21]
    var_23 = __import__(var_20, fromlist=var_22)
    var_24 = 0

def test_case_0():
    var_0 = 'datetime'
    var_1 = __import__(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 'test_source'
    var_5 = 'pypara.accounting.journaling'
    var_6 = 'JournalEntry'
    var_7 = [var_6]
    var_8 = __import__(var_5, fromlist=var_7)
    var_9 = 'Test Entry'
    var_10 = 'pypara.accounting.accounts'
    var_11 = 'Account'
    var_12 = [var_11]
    var_13 = __import__(var_10, fromlist=var_12)
    var_14 = '1000'
    var_15 = 'Cash'
    var_16 = 'AccountType'
    var_17 = [var_16]
    var_18 = __import__(var_10, fromlist=var_17)
    var_19 = var_18.AccountType.ASSET
    var_20 = [var_11]
    var_21 = __import__(var_10, fromlist=var_20)
    var_22 = '2000'
    var_23 = 'Liability'
    var_24 = [var_16]
    var_25 = __import__(var_10, fromlist=var_24)
    var_26 = var_25.AccountType.LIABILITY
    var_27 = 'pypara.accounting.quantity'
    var_28 = 'Quantity'
    var_29 = [var_28]
    var_30 = __import__(var_27, fromlist=var_29)
    var_31 = 100
    var_32 = [var_28]
    var_33 = __import__(var_27, fromlist=var_32)
    var_34 = -100

def test_case_0():
    var_0 = 'datetime'
    var_1 = __import__(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 'test_source'
    var_5 = 'pypara.accounting.journaling'
    var_6 = 'JournalEntry'
    var_7 = [var_6]
    var_8 = __import__(var_5, fromlist=var_7)
    var_9 = 'Test Entry'
    var_10 = 'pypara.accounting.accounts'
    var_11 = 'Account'
    var_12 = [var_11]
    var_13 = __import__(var_10, fromlist=var_12)
    var_14 = '1000'
    var_15 = 'Cash'
    var_16 = 'AccountType'
    var_17 = [var_16]
    var_18 = __import__(var_10, fromlist=var_17)
    var_19 = var_18.AccountType.ASSET
    var_20 = 'pypara.accounting.quantity'
    var_21 = 'Quantity'
    var_22 = [var_21]
    var_23 = __import__(var_20, fromlist=var_22)
    var_24 = 100
    var_25 = [var_21]
    var_26 = __import__(var_20, fromlist=var_25)
    var_27 = 50



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_not_equal_credits. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #8
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
    var_7 = '5000'
    var_8 = 'Expense'
    var_9 = [var_1, var_2, var_2]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_1, var_2, var_2]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/14 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 4/9 statements.


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
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

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
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 5/28 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]
    var_8 = 3
    var_9 = [var_0, var_1, var_8]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/28 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 8/29 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 11/41 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
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
    var_3 = 'Test entry'
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
    var_3 = 'Empty entry'
    var_4 = 'test'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Multi posting'
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
    var_15 = '-100'
    var_16 = [var_15]
    var_17 = [var_0, var_1, var_1]
    var_18 = '-50'
    var_19 = [var_18]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
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
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 2024
    var_9 = 1
    var_10 = 16
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/14 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guid. Retrieved 4/9 statements.


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
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Frozen test'
    var_4 = 'test'
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
    var_3 = 'Entry 1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_debits_equal_credits. Retrieved 10/27 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = 100
    var_11 = [var_4, var_5, var_5]
    var_12 = -100



# Parsed testcases at query #17
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
    var_4 = 'Another entry'
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
    var_10 = bool(True)
    assert var_10 is True

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'src1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_post_with_nonzero_quantity. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 100
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/28 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 8/29 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 9/35 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
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
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
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
    var_3 = 'Empty entry'
    var_4 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = 'Account3'
    var_8 = [var_0, var_1, var_1]
    var_9 = '50'
    var_10 = [var_9]
    var_11 = [var_0, var_1, var_1]
    var_12 = [var_9]
    var_13 = [var_0, var_1, var_1]
    var_14 = '-100'
    var_15 = [var_14]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 3/23 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 9/14 statements.
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/12 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 5/8 statements.


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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_frozen. Retrieved 3/12 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 6/19 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 2024
    var_3 = 6
    var_4 = 30
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = 2022
    var_11 = 12
    var_12 = 25
    var_13 = [var_10, var_11, var_12]
    var_14 = []
    var_15 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_not_equal_credits. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 'test_business'
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Asset'
    var_6 = 'Liability'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #25
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
    var_4 = 'Another entry'
    var_5 = 12345

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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

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
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'Source'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 7/23 statements.


def test_case_0():
    var_0 = 'TestSource'
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 7/19 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 7/19 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 6/17 statements.
# Partially parsed test_post_multiple_times. Retrieved 9/24 statements.
# Partially parsed test_post_returns_same_instance. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'TestAccount'
    var_6 = 100
    var_7 = [var_0, var_1, var_1]
    var_8 = 100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'TestAccount'
    var_6 = -50
    var_7 = [var_0, var_1, var_1]
    var_8 = 50

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'TestAccount'
    var_6 = 0
    var_7 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_0, var_1, var_1]
    var_8 = 100
    var_9 = 2
    var_10 = [var_0, var_1, var_9]
    var_11 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'TestAccount'
    var_6 = [var_0, var_1, var_1]
    var_7 = 75



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.5
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 9/25 statements.


def test_case_0():
    var_0 = 'TestSource'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test Entry'
    var_5 = 'TestDebit'
    var_6 = 'Asset'
    var_7 = 'TestCredit'
    var_8 = 'Liability'
    var_9 = '100'
    var_10 = [var_9]
    var_11 = [var_1, var_2, var_2]
    var_12 = [var_1, var_2, var_2]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 9/14 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_in_init. Retrieved 6/12 statements.
# Partially parsed test_journal_entry_constructor_guid_generated. Retrieved 7/11 statements.


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
    var_4 = 'Test entry'
    var_5 = 'string_source'
    var_6 = 42
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}

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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 'postings'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test1'
    var_5 = 'source1'
    var_6 = 'Test2'
    var_7 = 'source2'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 3/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'frozen'
    var_11 = bool('frozen' in str(type(e).__name__).lower())
    assert var_11 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 9/12 statements.
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
    var_0 = 2024
    var_1 = 12
    var_2 = 25
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Holiday transaction'
    var_5 = 'type'
    var_6 = 'id'
    var_7 = 'dict_source'
    var_8 = 123
    var_9 = {var_5: var_7, var_6: var_8}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry 1'
    var_4 = 'source1'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Frozen test'
    var_4 = 'source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 7/20 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 8/21 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 7/18 statements.
# Partially parsed test_post_multiple_postings. Retrieved 9/25 statements.
# Partially parsed test_post_returns_same_instance. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'
    var_9 = [var_5]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '-50.00'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'
    var_9 = '50.00'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '0.00'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Account 1'
    var_5 = 'Account 2'
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = '-100.00'
    var_9 = [var_8]
    var_10 = 'test_source'
    var_11 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'test_source'
    var_8 = 'Test Entry'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/20 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 3/20 statements.
# Partially parsed test_posting_is_frozen. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 2023
    var_5 = 2
    var_6 = 1
    var_7 = [var_4, var_5, var_6]
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_in_init. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 20
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source'
    var_6 = 2023
    var_7 = 2
    var_8 = 20
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'test_source'
    var_6 = 'postings'



# Parsed testcases at query #37
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
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another test'
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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test1'
    var_4 = 'src1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/16 statements.
# Partially parsed test_posting_constructor_with_different_values. Retrieved 6/16 statements.
# Partially parsed test_posting_is_frozen. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.0

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'liability'
    var_5 = 'credit'
    var_6 = 250.5

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'asset'
    var_5 = 'debit'
    var_6 = 100.0
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

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
    var_5 = 'Source'

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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_post_with_non_zero_quantity. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestAccount'
    var_5 = 100
    var_6 = 'TestSource'
    var_7 = 'Test Entry'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_is_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

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
    var_3 = 'Entry 1'
    var_4 = 'source1'
    var_5 = 'source2'

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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_read_journal_entries_protocol_call. Retrieved 4/34 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guid. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'source_object'

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Entry 1'
    var_5 = 'source1'

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 10/28 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 10/29 statements.
# Partially parsed test_validate_empty_journal_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_debits_and_credits. Retrieved 16/43 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = 100
    var_11 = [var_4, var_5, var_5]
    var_12 = -100

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Test'
    var_8 = 'test_source'
    var_9 = [var_4, var_5, var_5]
    var_10 = 100
    var_11 = [var_4, var_5, var_5]
    var_12 = -50
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '1100'
    var_3 = 'Receivable'
    var_4 = '2000'
    var_5 = 'Payable'
    var_6 = '2100'
    var_7 = 'Accrued'
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = 'Test'
    var_12 = 'test_source'
    var_13 = [var_8, var_9, var_9]
    var_14 = 60
    var_15 = [var_8, var_9, var_9]
    var_16 = 40
    var_17 = [var_8, var_9, var_9]
    var_18 = -50
    var_19 = [var_8, var_9, var_9]
    var_20 = -50



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 4/18 statements.
# Partially parsed test_posting_constructor_frozen. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'DEBIT'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'DEBIT'
    var_5 = 2024
    var_6 = 1
    var_7 = 16
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/23 statements.
# Partially parsed test_read_journal_entries_call_empty. Retrieved 2/21 statements.
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
    var_3 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 12
    var_4 = 31
    var_5 = [var_0, var_3, var_4]



# Parsed testcases at query #47
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
    var_0 = 2024
    var_1 = 6
    var_2 = 20
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
    var_5 = 'test_source'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_not_equal_credits. Retrieved 8/27 statements.


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



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_dates. Retrieved 9/14 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_postings_not_in_init. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'Entry 1'
    var_8 = 'Source1'
    var_9 = 'Entry 2'
    var_10 = 'Source2'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Entry 1'
    var_5 = 'Source1'

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
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'Source'
    var_6 = 'postings'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_unique. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 2023
    var_7 = 1
    var_8 = 16
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #51
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
    var_4 = 'Test'
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
    var_4 = 'Test1'
    var_5 = 'source1'
    var_6 = 'Test2'
    var_7 = 'source2'



# Parsed testcases at query #52
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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source_object'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_append_posting. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_constructor_generates_unique_guids. Retrieved 4/8 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test journal entry'
    var_5 = 'TestSource'

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Entry 1'
    var_4 = 'Source1'
    var_5 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'
    var_5 = 2024
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.


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
    var_4 = 'Frozen test'
    var_5 = 'source'
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
    var_4 = 'Test'
    var_5 = 'source'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
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
    var_4 = 'Another entry'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

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
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_post_with_positive_quantity. Retrieved 7/20 statements.
# Partially parsed test_post_with_negative_quantity. Retrieved 8/21 statements.
# Partially parsed test_post_with_zero_quantity. Retrieved 7/18 statements.
# Partially parsed test_post_multiple_postings. Retrieved 9/25 statements.
# Partially parsed test_post_returns_same_instance. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = 'Test Source'
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = 'Test Entry'
    var_9 = [var_6]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = 'Test Source'
    var_6 = '-50.00'
    var_7 = [var_6]
    var_8 = 'Test Entry'
    var_9 = '50.00'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = 'Test Source'
    var_6 = '0.00'
    var_7 = [var_6]
    var_8 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Account 1'
    var_5 = 'Account 2'
    var_6 = 'Test Source'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = '-100.00'
    var_10 = [var_9]
    var_11 = 'Test Entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Account'
    var_5 = 'Test Source'
    var_6 = '75.50'
    var_7 = [var_6]
    var_8 = 'Test Entry'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_unique_guids. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 2023
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
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
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
    var_5 = 'test_source'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_postings_init_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
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
    var_5 = 'test_source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'
    var_6 = 'postings'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_post_with_zero_quantity. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'TestAccount'
    var_4 = 'Test Entry'
    var_5 = 'TestSource'
    var_6 = 0



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 2023
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
    var_4 = 'Another entry'
    var_5 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'test'
    var_4 = 'source'
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_postings_default. Retrieved 5/12 statements.


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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #64
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
    var_2 = 20
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
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #65
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
    var_5 = 'TestSource'

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
    var_3 = 'Test'
    var_4 = 'Source'
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
    var_4 = 'Source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
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

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'test_source'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_validate_balanced_entry. Retrieved 8/28 statements.
# Partially parsed test_validate_unbalanced_entry_raises_assertion. Retrieved 8/29 statements.
# Partially parsed test_validate_empty_entry. Retrieved 4/9 statements.
# Partially parsed test_validate_multiple_balanced_postings. Retrieved 9/35 statements.


def test_case_0():
    var_0 = 'test_source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Debit Account'
    var_6 = 'Credit Account'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-100'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Debit Account'
    var_6 = 'Credit Account'
    var_7 = [var_1, var_2, var_2]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_1, var_2, var_2]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'

def test_case_0():
    var_0 = 'test_source'
    var_1 = 2024
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'
    var_5 = 'Account 1'
    var_6 = 'Account 2'
    var_7 = 'Account 3'
    var_8 = [var_1, var_2, var_2]
    var_9 = '50'
    var_10 = [var_9]
    var_11 = [var_1, var_2, var_2]
    var_12 = [var_9]
    var_13 = [var_1, var_2, var_2]
    var_14 = '-100'
    var_15 = [var_14]



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 4/27 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_0, var_1, var_1]
    var_6 = 2
    var_7 = [var_0, var_1, var_6]
    var_8 = [var_0, var_1, var_3]



# Parsed testcases at query #69
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
    var_5 = 'TEST_SOURCE'

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another entry'
    var_5 = 42

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Frozen test'
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
    var_3 = 'Entry 1'
    var_4 = 'source1'
    var_5 = [var_0, var_1, var_1]



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_append_posting. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = [var_0, var_1, var_1]



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_creates_unique_guids. Retrieved 4/8 statements.
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
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
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



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 9/16 statements.
# Partially parsed test_journal_entry_guid_uniqueness. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_is_frozen. Retrieved 8/13 statements.


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
    var_5 = 'id'
    var_6 = 'name'
    var_7 = 123
    var_8 = 'source'
    var_9 = {var_5: var_7, var_6: var_8}

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



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_validate_passes_when_debits_equal_credits. Retrieved 8/26 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'test_source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '-100'
    var_10 = [var_9]



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 3/11 statements.
# Partially parsed test_posting_constructor_with_keyword_args. Retrieved 3/10 statements.
# Partially parsed test_posting_is_frozen. Retrieved 3/12 statements.


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
    var_1 = 2023
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
    var_8 = []
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 7/11 statements.


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
    var_2 = 25
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
    var_4 = 'Test'
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
    var_4 = 'Test 1'
    var_5 = 'source1'
    var_6 = 'Test 2'
    var_7 = 'source2'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.
# Partially parsed test_journal_entry_constructor_with_different_dates. Retrieved 10/15 statements.
# Partially parsed test_journal_entry_constructor_guid_uniqueness. Retrieved 7/11 statements.
# Partially parsed test_journal_entry_constructor_postings_default. Retrieved 5/12 statements.
# Partially parsed test_journal_entry_constructor_frozen. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'source_object'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 2024
    var_5 = 12
    var_6 = 31
    var_7 = [var_4, var_5, var_6]
    var_8 = 'Entry 1'
    var_9 = 'source1'
    var_10 = 'Entry 2'
    var_11 = 'source2'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Entry 1'
    var_5 = 'source1'
    var_6 = 'Entry 2'
    var_7 = 'source2'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test'
    var_5 = 'source'
    var_6 = 2024
    var_7 = 2
    var_8 = 1
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True



