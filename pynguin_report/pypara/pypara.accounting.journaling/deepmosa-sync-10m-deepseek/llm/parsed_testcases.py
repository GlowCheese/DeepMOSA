####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 5/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Cash'
    var_5 = 100
    var_6 = 'USD'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_post_adds_posting_to_journal_entry. Retrieved 8/20 statements.
# Partially parsed test_post_does_not_add_posting_for_zero_quantity. Retrieved 7/16 statements.
# Partially parsed test_post_adds_debit_posting_for_negative_quantity. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'Cash'
    var_1 = 2023
    var_2 = 10
    var_3 = 1
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Entry'
    var_6 = 'Source'
    var_7 = [var_1, var_2, var_3]
    var_8 = 100
    var_9 = 0
    var_10 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'Cash'
    var_1 = 2023
    var_2 = 10
    var_3 = 1
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Entry'
    var_6 = 'Source'
    var_7 = [var_1, var_2, var_3]
    var_8 = 0

def test_case_0():
    var_0 = 'Cash'
    var_1 = 2023
    var_2 = 10
    var_3 = 1
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Entry'
    var_6 = 'Source'
    var_7 = [var_1, var_2, var_3]
    var_8 = -100
    var_9 = 0
    var_10 = [var_1, var_2, var_3]
    var_11 = 100



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 8/19 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '-100'
    var_10 = [var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'
    var_5 = 'Account1'
    var_6 = 'Account2'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '-50'
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 5/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Assets:Cash'
    var_5 = 100
    var_6 = 'USD'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Test Source'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_journal_entry_constructor_initializes_fields_correctly. Retrieved 4/10 statements.
# Partially parsed test_journal_entry_constructor_with_empty_description_raises_error. Retrieved 4/7 statements.
# Partially parsed test_journal_entry_is_immutable. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test description'
    var_4 = 'Test source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test source'
    var_4 = ''
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test description'
    var_4 = 'Test source'
    var_5 = 2023
    var_6 = 1
    var_7 = 2
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 100
    var_4 = 'USD'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_equal_debits_and_credits. Retrieved 8/19 statements.
# Partially parsed test_validate_with_unequal_debits_and_credits. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'
    var_5 = [var_0, var_1, var_1]
    var_6 = 'Assets:Bank'
    var_7 = 100
    var_8 = [var_0, var_1, var_1]
    var_9 = 'Expenses:Tax'
    var_10 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'
    var_5 = [var_0, var_1, var_1]
    var_6 = 'Assets:Bank'
    var_7 = 100
    var_8 = [var_0, var_1, var_1]
    var_9 = 'Expenses:Tax'
    var_10 = -50
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_equal_debits_and_credits. Retrieved 7/17 statements.
# Partially parsed test_validate_with_unequal_debits_and_credits. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'source'
    var_4 = 'account'
    var_5 = 'description'
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = '-100.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'source'
    var_4 = 'account'
    var_5 = 'description'
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = '-50.00'
    var_9 = [var_8]
    var_10 = 'Expected AssertionError but no exception was raised'
    var_11 = AssertionError(var_10)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_debits_and_credits_equal. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'TestSource'
    var_5 = [var_0, var_1, var_1]
    var_6 = 'A1'
    var_7 = 100
    var_8 = [var_7]
    var_9 = [var_0, var_1, var_1]
    var_10 = 'A2'
    var_11 = -100
    var_12 = [var_11]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_ReadJournalEntries___call__. Retrieved 4/26 statements.


def test_case_0():
    var_0 = '_T'
    var_1 = []
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 31
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 100
    var_5 = 'USD'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Test Source'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_asserts_when_debits_and_credits_are_equal. Retrieved 8/23 statements.
# Partially parsed test_validate_asserts_when_debits_and_credits_are_equal_with_multiple_postings. Retrieved 12/36 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account'
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
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account 1'
    var_7 = '456'
    var_8 = 'Test Account 2'
    var_9 = [var_0, var_1, var_1]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_0, var_1, var_1]
    var_13 = '50'
    var_14 = [var_13]
    var_15 = [var_0, var_1, var_1]
    var_16 = '-100'
    var_17 = [var_16]
    var_18 = [var_0, var_1, var_1]
    var_19 = '-50'
    var_20 = [var_19]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_with_equal_debits_and_credits. Retrieved 8/21 statements.
# Partially parsed test_validate_with_unequal_debits_and_credits. Retrieved 8/22 statements.
# Partially parsed test_validate_with_zero_postings. Retrieved 4/7 statements.
# Partially parsed test_validate_with_multiple_equal_postings. Retrieved 11/34 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = None
    var_5 = [var_0, var_1, var_1]
    var_6 = 'A'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = [var_0, var_1, var_1]
    var_10 = 'B'
    var_11 = '-100'
    var_12 = [var_11]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = None
    var_5 = [var_0, var_1, var_1]
    var_6 = 'A'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = [var_0, var_1, var_1]
    var_10 = 'B'
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = None
    var_5 = [var_0, var_1, var_1]
    var_6 = 'A'
    var_7 = '50'
    var_8 = [var_7]
    var_9 = [var_0, var_1, var_1]
    var_10 = 'B'
    var_11 = [var_7]
    var_12 = [var_0, var_1, var_1]
    var_13 = 'C'
    var_14 = '-75'
    var_15 = [var_14]
    var_16 = [var_0, var_1, var_1]
    var_17 = 'D'
    var_18 = '-25'
    var_19 = [var_18]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_balanced_journal_entry. Retrieved 7/22 statements.
# Partially parsed test_validate_unbalanced_journal_entry_raises_assertion_error. Retrieved 7/23 statements.


def test_case_0():
    var_0 = '1'
    var_1 = 'Account 1'
    var_2 = '2'
    var_3 = 'Account 2'
    var_4 = []
    var_5 = 'Test Entry'
    var_6 = '100'
    var_7 = [var_6]
    var_8 = '-100'
    var_9 = [var_8]

def test_case_0():
    var_0 = '1'
    var_1 = 'Account 1'
    var_2 = '2'
    var_3 = 'Account 2'
    var_4 = []
    var_5 = 'Test Entry'
    var_6 = '100'
    var_7 = [var_6]
    var_8 = '-50'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_journal_entry_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = None
    var_5 = [var_0, var_1, var_1]
    var_6 = '1'
    var_7 = '100'
    var_8 = [var_7]
    var_9 = [var_0, var_1, var_1]
    var_10 = '2'
    var_11 = '50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'JournalEntry'
    var_1 = 2023
    var_2 = 10
    var_3 = 1
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = 100.0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 4/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 10
    var_3 = 1
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test Description'
    var_6 = [var_1, var_2, var_3]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Journal Entry'
    var_5 = 'Test Source'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_post_quantity_zero_does_not_add_posting. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Test Source'
    var_6 = 'Test Account'
    var_7 = 0
    var_8 = [var_0, var_1, var_2]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = []



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Test Source'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_journal_entry_constructor. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Journal Entry'
    var_5 = 'Test Source'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_arguments. Retrieved 4/12 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/15 statements.
# Partially parsed test_journal_entry_constructor_postings_field_not_in_init. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_guid_field_not_in_init. Retrieved 4/8 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Description'
    var_4 = 'Test Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Description'
    var_4 = 'Test Source Object'
    var_5 = 2023
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Description'
    var_4 = 'Test Source Object'
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Description'
    var_4 = 'Test Source Object'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Description'
    var_4 = 123
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_journal_entry_constructor_with_default_values. Retrieved 4/8 statements.
# Partially parsed test_journal_entry_constructor_with_custom_source_type. Retrieved 4/6 statements.
# Partially parsed test_journal_entry_constructor_frozen_immutability. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'
    var_5 = 2023
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True
    var_13 = bool(False)
    assert var_13 is True
    var_14 = bool(True)
    assert var_14 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_equal_debits_and_credits. Retrieved 8/26 statements.
# Partially parsed test_validate_with_unequal_debits_and_credits. Retrieved 9/29 statements.
# Partially parsed test_validate_with_no_postings. Retrieved 4/9 statements.
# Partially parsed test_validate_with_multiple_debits_and_credits. Retrieved 10/39 statements.
# Partially parsed test_validate_with_zero_amount_postings. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Asset'
    var_7 = 'A2'
    var_8 = 'Liability'
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Asset'
    var_7 = 'A2'
    var_8 = 'Liability'
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_0, var_1, var_1]
    var_11 = 2
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Asset'
    var_7 = 'A2'
    var_8 = 'Liability'
    var_9 = 'A3'
    var_10 = 'Equity'
    var_11 = [var_0, var_1, var_1]
    var_12 = [var_0, var_1, var_1]
    var_13 = [var_0, var_1, var_1]
    var_14 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Asset'
    var_7 = [var_0, var_1, var_1]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 12/28 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account 1'
    var_7 = 'A2'
    var_8 = 'Account 2'
    var_9 = [var_0, var_1, var_1]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_0, var_1, var_1]
    var_13 = '50'
    var_14 = [var_13]
    var_15 = False
    var_16 = True
    assert var_16 is True
    var_17 = 'Total Debits and Credits are not equal:'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'A1'
    var_5 = 'Account 1'
    var_6 = 'A2'
    var_7 = 'Account 2'
    var_8 = '100'
    var_9 = [var_8]
    var_10 = '50'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_journal_entry_constructor_initializes_fields_correctly. Retrieved 3/12 statements.
# Partially parsed test_journal_entry_constructor_with_different_date. Retrieved 4/7 statements.
# Partially parsed test_journal_entry_constructor_with_different_description. Retrieved 4/7 statements.
# Partially parsed test_journal_entry_constructor_with_different_source. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_is_immutable. Retrieved 5/14 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'

def test_case_0():
    var_0 = []
    var_1 = 2024
    var_2 = 12
    var_3 = 31
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Year-end entry'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 5
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = ''

def test_case_0():
    var_0 = 'A string source'
    var_1 = 2023
    var_2 = 7
    var_3 = 4
    var_4 = [var_1, var_2, var_3]
    var_5 = 'String source entry'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test'
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_journal_entry_constructor_with_default_values. Retrieved 7/16 statements.
# Partially parsed test_journal_entry_constructor_with_custom_source_type. Retrieved 3/9 statements.
# Partially parsed test_journal_entry_constructor_ensures_frozen. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'BusinessObject'
    var_5 = [var_0, var_1, var_1]
    var_6 = 2023
    var_7 = 1
    var_8 = 2
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 123
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_post_positive_quantity_increment. Retrieved 9/20 statements.
# Partially parsed test_post_negative_quantity_decrement. Retrieved 11/22 statements.
# Partially parsed test_post_zero_quantity_no_posting. Retrieved 8/18 statements.
# Partially parsed test_post_multiple_postings. Retrieved 12/28 statements.
# Partially parsed test_post_chainable. Retrieved 8/18 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = 'ASSET'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test'
    var_7 = None
    var_8 = [var_3, var_4, var_4]
    var_9 = 100
    var_10 = 0
    var_11 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = 'ASSET'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test'
    var_7 = None
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = -50
    var_11 = 0
    var_12 = [var_3, var_4, var_8]
    var_13 = 50

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = 'ASSET'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test'
    var_7 = None
    var_8 = [var_3, var_4, var_4]
    var_9 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account 1'
    var_2 = 'ASSET'
    var_3 = '456'
    var_4 = 'Test Account 2'
    var_5 = 'LIABILITY'
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = 'Test'
    var_10 = None
    var_11 = [var_6, var_7, var_7]
    var_12 = 100
    var_13 = [var_6, var_7, var_7]
    var_14 = -100

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = 'ASSET'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test'
    var_7 = None
    var_8 = [var_3, var_4, var_4]
    var_9 = 100



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_initializes_fields_correctly. Retrieved 3/12 statements.
# Partially parsed test_constructor_with_different_source_types. Retrieved 5/8 statements.
# Partially parsed test_postings_list_is_empty_and_mutable. Retrieved 5/10 statements.
# Partially parsed test_guid_is_unique_for_each_instance. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_frozen_dataclass_behavior. Retrieved 5/14 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'string_source'
    var_5 = 123

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'test_posting'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test1'
    var_4 = None
    var_5 = [var_0, var_1, var_1]
    var_6 = 'Test2'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = []
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(False)
    assert var_9 is True
    var_10 = []
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_with_minimal_parameters. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_frozen_dataclass. Retrieved 4/8 statements.
# Partially parsed test_constructor_postings_field_not_in_init. Retrieved 6/8 statements.
# Partially parsed test_constructor_guid_field_not_in_init. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Source object'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 123

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_creates_unique_guids. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSourceObject'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSourceObject'
    var_6 = 2023
    var_7 = 10
    var_8 = 6
    var_9 = [var_6, var_7, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSourceObject'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 5/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Cash'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'USD'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_postings_field_not_in_init. Retrieved 6/9 statements.
# Partially parsed test_journal_entry_constructor_guid_field_not_in_init. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'
    var_6 = 2023
    var_7 = 10
    var_8 = 6
    var_9 = [var_6, var_7, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'
    var_6 = []

import pypara.commons.guid as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'
    var_6 = module_0.makeguid()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_post_positive_quantity_increment. Retrieved 10/23 statements.
# Partially parsed test_post_negative_quantity_decrement. Retrieved 11/24 statements.
# Partially parsed test_post_zero_quantity_no_posting. Retrieved 9/19 statements.
# Partially parsed test_post_multiple_postings. Retrieved 16/36 statements.
# Partially parsed test_post_chaining. Retrieved 16/36 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = 'ASSET'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test'
    var_7 = None
    var_8 = 100
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = 0
    var_12 = [var_3, var_4, var_9]

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = 'ASSET'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test'
    var_7 = None
    var_8 = -50
    var_9 = 3
    var_10 = [var_3, var_4, var_9]
    var_11 = 0
    var_12 = [var_3, var_4, var_9]
    var_13 = 50

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account'
    var_2 = 'ASSET'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'Test'
    var_7 = None
    var_8 = 0
    var_9 = 4
    var_10 = [var_3, var_4, var_9]

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account 1'
    var_2 = 'ASSET'
    var_3 = '456'
    var_4 = 'Test Account 2'
    var_5 = 'LIABILITY'
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = 'Test'
    var_10 = None
    var_11 = 2
    var_12 = [var_6, var_7, var_11]
    var_13 = 100
    var_14 = 3
    var_15 = [var_6, var_7, var_14]
    var_16 = -50
    var_17 = 0
    var_18 = [var_6, var_7, var_11]
    var_19 = [var_6, var_7, var_14]
    var_20 = 50

def test_case_0():
    var_0 = '123'
    var_1 = 'Test Account 1'
    var_2 = 'ASSET'
    var_3 = '456'
    var_4 = 'Test Account 2'
    var_5 = 'LIABILITY'
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = 'Test'
    var_10 = None
    var_11 = 2
    var_12 = [var_6, var_7, var_11]
    var_13 = 100
    var_14 = 3
    var_15 = [var_6, var_7, var_14]
    var_16 = -50
    var_17 = 0
    var_18 = [var_6, var_7, var_11]
    var_19 = [var_6, var_7, var_14]
    var_20 = 50



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_add_posting. Retrieved 8/23 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account'
    var_7 = 'USD'
    var_8 = '0'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_postings_field_not_in_init. Retrieved 6/9 statements.
# Partially parsed test_journal_entry_constructor_guid_field_not_in_init. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = 2023
    var_7 = 10
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_read_journal_entries_call. Retrieved 5/26 statements.


def test_case_0():
    var_0 = '_T'
    var_1 = []
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 12
    var_6 = 31
    var_7 = [var_2, var_5, var_6]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test___call___returns_iterable_of_journal_entries. Retrieved 3/16 statements.
# Partially parsed test___call___handles_empty_period. Retrieved 2/12 statements.
# Partially parsed test___call___propagates_exceptions. Retrieved 3/12 statements.
# Partially parsed test___call___supports_lazy_iteration. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 31
    var_4 = [var_0, var_1, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 0
    assert var_0 == 0
    assert var_0 == 1
    assert var_0 == 2
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 31
    var_5 = [var_1, var_2, var_4]
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_postings_field_not_in_init. Retrieved 6/9 statements.
# Partially parsed test_journal_entry_constructor_guid_field_not_in_init. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = 2023
    var_7 = 10
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

import pypara.commons.guid as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = module_0.makeguid()
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1234'
    var_4 = '5678'
    var_5 = 'Test entry'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/31 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = '1234'
    var_4 = 'USD'
    var_5 = 2
    var_6 = []
    var_7 = 'Test'
    var_8 = '100'
    var_9 = [var_8]
    var_10 = '50'
    var_11 = [var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'
    var_6 = 2023
    var_7 = 10
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True
    var_14 = bool(False)
    assert var_14 is True
    var_15 = bool(True)
    assert var_15 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_journal_entry_constructor_with_default_values. Retrieved 4/10 statements.
# Partially parsed test_journal_entry_constructor_with_custom_source_type. Retrieved 4/8 statements.
# Partially parsed test_journal_entry_constructor_immutability_check. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 12345

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = []
    var_5 = 2023
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = []
    var_4 = '123'
    var_5 = 'Test Account'
    var_6 = 'Test Entry'
    var_7 = 2
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_add_posting. Retrieved 16/34 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '0.00'
    var_3 = [var_2]
    var_4 = 'Quantity'
    var_5 = ()
    var_6 = 'is_zero'
    var_7 = '__abs__'
    var_8 = True
    var_9 = lambda self: var_8
    var_10 = lambda self: self
    var_11 = {var_6: var_9, var_7: var_10}
    var_12 = type(var_4, var_5, var_11)
    var_13 = 'cash'
    var_14 = 2023
    var_15 = [var_14, var_8, var_8]
    var_16 = 'Test entry'
    var_17 = None
    var_18 = [var_14, var_8, var_8]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Cash'
    var_4 = 100
    var_5 = 'USD'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_asserts_when_total_debits_and_credits_are_equal. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Test Account'
    var_7 = [var_0, var_1, var_1]
    var_8 = [var_0, var_1, var_1]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_journal_entry_constructor_with_default_values. Retrieved 7/16 statements.
# Partially parsed test_journal_entry_constructor_with_custom_source_type. Retrieved 3/12 statements.
# Partially parsed test_journal_entry_constructor_with_different_date_and_description. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'BusinessObject'
    var_5 = [var_0, var_1, var_1]
    var_6 = 2023
    var_7 = 1
    var_8 = 2
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2024
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Another Description'
    var_5 = 123
    var_6 = [var_0, var_1, var_2]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_with_equal_debits_and_credits. Retrieved 8/23 statements.
# Partially parsed test_validate_with_zero_postings. Retrieved 4/9 statements.
# Partially parsed test_validate_with_multiple_equal_debits_and_credits. Retrieved 10/29 statements.
# Partially parsed test_validate_raises_assertion_error_on_inequal_debits_and_credits. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account1'
    var_7 = 'A2'
    var_8 = 'Account2'
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account1'
    var_7 = 'A2'
    var_8 = 'Account2'
    var_9 = 'A3'
    var_10 = 'Account3'
    var_11 = [var_0, var_1, var_1]
    var_12 = [var_0, var_1, var_1]
    var_13 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account1'
    var_7 = 'A2'
    var_8 = 'Account2'
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_0, var_1, var_1]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Total Debits and Credits are not equal'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_fields. Retrieved 4/10 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_is_frozen. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'TestSource'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 123
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'TestSource'
    var_5 = 2023
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_journalentry_constructor_with_default_values. Retrieved 7/16 statements.
# Partially parsed test_journalentry_constructor_with_custom_source_type. Retrieved 3/9 statements.
# Partially parsed test_journalentry_constructor_date_must_be_date. Retrieved 4/9 statements.
# Partially parsed test_journalentry_constructor_description_must_be_string. Retrieved 4/9 statements.
# Partially parsed test_journalentry_constructor_guid_is_unique. Retrieved 7/12 statements.
# Partially parsed test_journalentry_constructor_postings_initialized_empty. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'BusinessObject'
    var_5 = [var_0, var_1, var_1]
    var_6 = 2023
    var_7 = 1
    var_8 = 2
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = ''
    var_4 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test1'
    var_4 = 'Source1'
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = 'Test2'
    var_8 = 'Source2'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_post_positive_quantity_appends_increment_posting. Retrieved 9/23 statements.
# Partially parsed test_post_negative_quantity_appends_decrement_posting. Retrieved 9/24 statements.
# Partially parsed test_post_zero_quantity_does_nothing. Retrieved 8/19 statements.
# Partially parsed test_post_multiple_postings_accumulate. Retrieved 13/33 statements.
# Partially parsed test_post_same_date_and_account_multiple_times. Retrieved 10/30 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_0, var_1, var_9]
    var_11 = 0
    var_12 = [var_0, var_1, var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account'
    var_7 = '-100.00'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_0, var_1, var_9]
    var_11 = 0
    var_12 = [var_0, var_1, var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account'
    var_7 = '0.00'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_0, var_1, var_9]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account 1'
    var_7 = '456'
    var_8 = 'Test Account 2'
    var_9 = '50.00'
    var_10 = [var_9]
    var_11 = '-30.00'
    var_12 = [var_11]
    var_13 = 2
    var_14 = [var_0, var_1, var_13]
    var_15 = 3
    var_16 = [var_0, var_1, var_15]
    var_17 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '123'
    var_6 = 'Test Account'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = '200.00'
    var_10 = [var_9]
    var_11 = 2
    var_12 = [var_0, var_1, var_11]
    var_13 = [var_0, var_1, var_11]
    var_14 = 0
    var_15 = [var_0, var_1, var_11]
    var_16 = [var_0, var_1, var_11]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_equal_debits_and_credits. Retrieved 5/23 statements.
# Partially parsed test_validate_with_zero_postings. Retrieved 4/9 statements.
# Partially parsed test_validate_with_unequal_debits_and_credits. Retrieved 6/25 statements.
# Partially parsed test_validate_with_multiple_debits_and_credits. Retrieved 8/38 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'Test'
    var_3 = None
    var_4 = '100'
    var_5 = [var_4]
    var_6 = [var_4]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'Test'
    var_3 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'Test'
    var_3 = None
    var_4 = '100'
    var_5 = [var_4]
    var_6 = '50'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'Test'
    var_3 = None
    var_4 = '30'
    var_5 = [var_4]
    var_6 = '70'
    var_7 = [var_6]
    var_8 = '40'
    var_9 = [var_8]
    var_10 = '60'
    var_11 = [var_10]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'A2'
    var_7 = [var_0, var_1, var_1]
    var_8 = [var_0, var_1, var_1]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/14 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 8/13 statements.
# Partially parsed test_journal_entry_is_frozen. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSourceObject'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 123
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'TestSource'
    var_6 = 2024
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True
    var_13 = bool(False)
    assert var_13 is True
    var_14 = bool(True)
    assert var_14 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_assertion_true_when_debits_equal_credits. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account 1'
    var_7 = 'A2'
    var_8 = 'Account 2'
    var_9 = [var_0, var_1, var_1]
    var_10 = '100'
    var_11 = [var_10]
    var_12 = [var_0, var_1, var_1]
    var_13 = '-100'
    var_14 = [var_13]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/15 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 8/15 statements.
# Partially parsed test_journal_entry_constructor_guid_is_unique. Retrieved 5/8 statements.
# Partially parsed test_journal_entry_constructor_postings_list_is_independent. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'
    var_6 = 2023
    var_7 = 10
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True
    var_14 = bool(False)
    assert var_14 is True
    var_15 = bool(True)
    assert var_15 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 42
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Entry'
    var_5 = 'Source Object'
    var_6 = 'test'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/15 statements.
# Partially parsed test_journal_entry_constructor_postings_field_not_in_init. Retrieved 6/9 statements.
# Partially parsed test_journal_entry_constructor_guid_field_not_in_init. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = 2024
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

import pypara.commons.guid as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = module_0.makeguid()
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/28 statements.
# Partially parsed test_validate_passes_when_debits_and_credits_are_equal. Retrieved 7/26 statements.
# Partially parsed test_validate_passes_with_multiple_equal_debits_and_credits. Retrieved 10/36 statements.
# Partially parsed test_validate_passes_when_no_postings. Retrieved 4/9 statements.
# Partially parsed test_validate_raises_assertion_error_with_only_debits. Retrieved 7/21 statements.
# Partially parsed test_validate_raises_assertion_error_with_only_credits. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = [var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = 'Revenue'
    var_9 = [var_0, var_1, var_1]
    var_10 = '150'
    var_11 = [var_10]
    var_12 = [var_0, var_1, var_1]
    var_13 = '75'
    var_14 = [var_13]
    var_15 = [var_0, var_1, var_1]
    var_16 = [var_13]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Total Debits and Credits are not equal'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 8/16 statements.
# Partially parsed test_journal_entry_constructor_postings_field_not_in_init. Retrieved 6/9 statements.
# Partially parsed test_journal_entry_constructor_guid_field_not_in_init. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'
    var_6 = 2023
    var_7 = 10
    var_8 = 6
    var_9 = [var_6, var_7, var_8]

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'
    var_6 = []

import pypara.commons.guid as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'
    var_6 = module_0.makeguid()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_fields. Retrieved 4/10 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 7/10 statements.
# Partially parsed test_journal_entry_is_immutable. Retrieved 7/15 statements.
# Partially parsed test_journal_entry_guid_is_unique. Retrieved 4/7 statements.
# Partially parsed test_journal_entry_postings_list_is_initially_empty. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 123
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'
    var_5 = 2023
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test Entry'
    var_4 = 'Test Source'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_posting_constructor_initializes_fields_correctly. Retrieved 6/14 statements.
# Partially parsed test_posting_constructor_with_different_direction. Retrieved 6/14 statements.
# Partially parsed test_posting_is_frozen_and_immutable. Retrieved 10/23 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = 'USD'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 2
    var_3 = 20
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Revenue'
    var_6 = '50.00'
    var_7 = [var_6]
    var_8 = 'EUR'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 3
    var_3 = 10
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Expense'
    var_6 = '75.00'
    var_7 = [var_6]
    var_8 = 'GBP'
    var_9 = 2023
    var_10 = 4
    var_11 = 1
    var_12 = [var_9, var_10, var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'New'
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_assertion_true_when_debits_equal_credits. Retrieved 8/23 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Asset'
    var_7 = '2000'
    var_8 = 'Liability'
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_0, var_1, var_1]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 5/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Cash'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'USD'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/15 statements.
# Partially parsed test_journal_entry_constructor_postings_field_not_in_init. Retrieved 8/17 statements.
# Partially parsed test_journal_entry_constructor_guid_field_not_in_init. Retrieved 5/9 statements.
# Partially parsed test_journal_entry_constructor_with_different_source_types. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'
    var_6 = 2024
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'
    var_6 = None
    var_7 = 'A'
    var_8 = 10
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'Source Object'
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 123
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_constructor_initializes_fields_correctly. Retrieved 4/14 statements.
# Failed to parse test_constructor_raises_error_when_missing_required_fields.
# Partially parsed test_constructor_is_immutable. Retrieved 7/12 statements.
# Partially parsed test_constructor_with_different_source_types. Retrieved 8/13 statements.
# Partially parsed test_constructor_postings_list_is_empty_by_default. Retrieved 4/7 statements.
# Partially parsed test_constructor_guid_is_unique. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test entry'
    var_4 = 'Test source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'
    var_5 = 2023
    var_6 = 2
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 123
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Int source'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = [var_1, var_2, var_2]
    var_9 = 'Dict source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = 'Source'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test1'
    var_4 = 'Source1'
    var_5 = [var_0, var_1, var_1]
    var_6 = 'Test2'
    var_7 = 'Source2'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_raises_assertion_error_when_debits_and_credits_not_equal. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = [var_0, var_1, var_1]
    var_8 = '100'
    var_9 = [var_8]
    var_10 = [var_0, var_1, var_1]
    var_11 = '-50'
    var_12 = [var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_posting_constructor_initializes_fields_correctly. Retrieved 5/13 statements.
# Partially parsed test_posting_constructor_with_debit_account_and_increase_direction. Retrieved 6/14 statements.
# Partially parsed test_posting_constructor_with_credit_account_and_increase_direction. Retrieved 6/14 statements.
# Partially parsed test_posting_constructor_with_debit_account_and_decrease_direction. Retrieved 6/14 statements.
# Partially parsed test_posting_constructor_with_credit_account_and_decrease_direction. Retrieved 5/13 statements.
# Partially parsed test_posting_is_frozen_and_immutable. Retrieved 9/22 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Cash'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'USD'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 2
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Equipment'
    var_6 = '500.00'
    var_7 = [var_6]
    var_8 = 'EUR'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 3
    var_3 = 10
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Loan'
    var_6 = '200.00'
    var_7 = [var_6]
    var_8 = 'GBP'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 4
    var_3 = 20
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Cash'
    var_6 = '50.00'
    var_7 = [var_6]
    var_8 = 'USD'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 5
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Revenue'
    var_5 = '300.00'
    var_6 = [var_5]
    var_7 = 'JPY'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 6
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Test'
    var_6 = '10.00'
    var_7 = [var_6]
    var_8 = 'CAD'
    var_9 = 2024
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True
    var_14 = 'New'
    var_15 = bool(False)
    assert var_15 is True
    var_16 = bool(True)
    assert var_16 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_posting_constructor. Retrieved 5/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Cash'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 'USD'



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_journal_entry_constructor_initializes_fields_correctly. Retrieved 3/12 statements.
# Partially parsed test_journal_entry_constructor_with_different_date. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_constructor_with_empty_description. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_constructor_postings_is_empty_list. Retrieved 5/7 statements.
# Partially parsed test_journal_entry_constructor_guid_is_unique. Retrieved 4/8 statements.
# Partially parsed test_journal_entry_is_immutable. Retrieved 7/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 'Test entry'

def test_case_0():
    var_0 = 'source'
    var_1 = 2022
    var_2 = 12
    var_3 = 31
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Year end entry'

def test_case_0():
    var_0 = 123
    var_1 = 2023
    var_2 = 2
    var_3 = 15
    var_4 = [var_1, var_2, var_3]
    var_5 = ''

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 3
    var_3 = 10
    var_4 = [var_1, var_2, var_3]
    var_5 = 'No postings'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 4
    var_3 = 1
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Guid test'

def test_case_0():
    var_0 = []
    var_1 = 2023
    var_2 = 5
    var_3 = 20
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Immutable test'
    var_6 = 2023
    var_7 = 5
    var_8 = 21
    var_9 = [var_6, var_7, var_8]
    var_10 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_with_equal_debits_and_credits. Retrieved 8/23 statements.
# Partially parsed test_validate_with_zero_postings. Retrieved 4/9 statements.
# Partially parsed test_validate_with_multiple_equal_debits_and_credits. Retrieved 11/31 statements.
# Partially parsed test_validate_raises_assertion_error_on_imbalance. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account1'
    var_7 = 'A2'
    var_8 = 'Account2'
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_0, var_1, var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account1'
    var_7 = 'A2'
    var_8 = 'Account2'
    var_9 = 'A3'
    var_10 = 'Account3'
    var_11 = [var_0, var_1, var_1]
    var_12 = [var_0, var_1, var_1]
    var_13 = [var_0, var_1, var_1]
    var_14 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 'Test'
    var_4 = None
    var_5 = 'A1'
    var_6 = 'Account1'
    var_7 = 'A2'
    var_8 = 'Account2'
    var_9 = [var_0, var_1, var_1]
    var_10 = [var_0, var_1, var_1]
    var_11 = 2
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_journal_entry_constructor_with_minimal_parameters. Retrieved 5/13 statements.
# Partially parsed test_journal_entry_constructor_is_frozen. Retrieved 7/15 statements.
# Partially parsed test_journal_entry_constructor_postings_field_is_init_false. Retrieved 6/9 statements.
# Partially parsed test_journal_entry_constructor_guid_field_is_init_false_and_unique. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = 2024
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'
    var_6 = 'test'

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test entry'
    var_5 = 'SourceObject'



