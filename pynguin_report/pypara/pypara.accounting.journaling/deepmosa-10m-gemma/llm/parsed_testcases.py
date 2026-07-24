####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 4/8 statements.


import datetime as module_0

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 4/9 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 100



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_journal_entry_validate_success. Retrieved 13/19 statements.
# Partially parsed test_journal_entry_validate_failure_imbalance. Retrieved 15/23 statements.


import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Assets'
    var_6 = 'Cash'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = '-100.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = 'Test Entry'
    var_16 = 'TestSource'
    var_17 = module_2.JournalEntry(var_4, var_15, var_16)
    var_18 = var_17.validate()

import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Assets'
    var_6 = 'Cash'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = '-50.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = 'Imbalanced Entry'
    var_16 = 'TestSource'
    var_17 = module_2.JournalEntry(var_4, var_15, var_16)
    var_18 = var_17.validate()
    var_19 = 'Validation should have failed due to imbalance'
    var_20 = AssertionError(var_19)
    var_21 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_read_journal_entries_call_returns_expected_iterable. Retrieved 8/28 statements.
# Partially parsed test_read_journal_entries_call_with_empty_range. Retrieved 4/19 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 31
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 'Entry 1'
    var_10 = 2
    var_11 = 'Entry 2'

import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 4/8 statements.


import datetime as module_0

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_posting_constructor_valid_data. Retrieved 5/10 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 27
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 100.0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'Test Source'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_read_journal_entries_call_returns_expected_entries. Retrieved 5/22 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 31
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 5/9 statements.


import datetime as module_0

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 100



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_journal_entry_validate_success. Retrieved 15/21 statements.
# Partially parsed test_journal_entry_validate_failure_unbalanced. Retrieved 15/22 statements.
# Partially parsed test_journal_entry_validate_multiple_postings. Retrieved 19/28 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test'
    var_6 = 'TestSource'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = 'A'
    var_9 = 'B'
    var_10 = [var_0, var_1, var_1]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '100.00'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_2.Decimal(*var_14, **var_15)
    var_17 = [var_0, var_1, var_1]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = '-100.00'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = var_7.validate()

import datetime as module_0
import pypara.accounting.journaling as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test'
    var_6 = 'TestSource'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = 'A'
    var_9 = 'B'
    var_10 = [var_0, var_1, var_1]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '100.00'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_2.Decimal(*var_14, **var_15)
    var_17 = [var_0, var_1, var_1]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = '-50.00'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = var_7.validate()
    var_25 = bool(False)
    assert var_25 is True
    var_26 = 'Total Debits and Credits are not equal'

import pypara.accounting.journaling as module_0
import datetime as module_1

def test_case_0():
    var_0 = 0
    var_1 = int(var_0)
    var_2 = 'Empty'
    var_3 = 'Test'
    var_4 = module_0.JournalEntry(var_1, var_2, var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.date(*var_7, **var_8)
    var_10 = module_0.JournalEntry(var_9, var_2, var_3)
    var_11 = var_10.validate()

import datetime as module_0
import pypara.accounting.journaling as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Multi'
    var_6 = 'Test'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = 'A'
    var_9 = 'B'
    var_10 = 'C'
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = '50.00'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = [var_0, var_1, var_1]
    var_19 = {}
    var_20 = module_0.date(*var_18, **var_19)
    var_21 = '30.00'
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_2.Decimal(*var_22, **var_23)
    var_25 = [var_0, var_1, var_1]
    var_26 = {}
    var_27 = module_0.date(*var_25, **var_26)
    var_28 = '-80.00'
    var_29 = [var_28]
    var_30 = {}
    var_31 = module_2.Decimal(*var_29, **var_30)
    var_32 = var_7.validate()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_raises_assertion_error_on_imbalance. Retrieved 17/30 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1
import decimal as module_2
import builtins as module_3

def test_case_0():
    var_0 = 'Assets'
    var_1 = 'Equity'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = 'Imbalanced Entry'
    var_8 = None
    var_9 = module_1.JournalEntry(var_6, var_7, var_8)
    var_10 = [var_2, var_3, var_3]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '100.00'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_2.Decimal(*var_14, **var_15)
    var_17 = [var_2, var_3, var_3]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = '-50.00'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = var_9.validate()
    var_25 = 'AssertionError not raised'
    var_26 = [var_25]
    var_27 = {}
    var_28 = module_3.Exception(*var_26, **var_27)
    var_29 = 'Total Debits and Credits are not equal'
    var_30 = '100.00 != 50.00'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_success. Retrieved 11/20 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = 'B'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = '-100.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = 'Test'
    var_16 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'Test Source'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 5/9 statements.


import datetime as module_0

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 100



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 5/9 statements.


import datetime as module_0

def test_case_0():
    var_0 = None
    var_1 = 2023
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 100



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 8/9 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 27
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 'Test Entry'
    var_7 = 'SourceObject'
    var_8 = module_1.JournalEntry(var_5, var_6, var_7)
    var_9 = var_8.date
    var_10 = bool(var_8.date == var_5)
    assert var_10 is True
    var_11 = var_8.description
    var_12 = bool(var_8.description == var_6)
    assert var_12 is True
    var_13 = var_8.source
    var_14 = bool(var_8.source == var_7)
    assert var_14 is True
    var_15 = var_8.postings
    var_16 = bool(var_8.postings == [])
    assert var_16 is True
    var_17 = var_8.guid



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_journal_entry_post_adds_posting_when_quantity_is_nonzero. Retrieved 4/16 statements.
# Failed to parse test_journal_entry_post_does_not_add_posting_when_quantity_is_zero.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 5/10 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 27
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 100



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'TestSource'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_raises_assertion_error_on_unbalanced_entry. Retrieved 11/20 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Assets'
    var_6 = 'Liabilities'
    var_7 = 100
    var_8 = 50
    var_9 = 'Unbalanced Entry'
    var_10 = 'Test'
    var_11 = module_1.JournalEntry(var_4, var_9, var_10)
    var_12 = var_11.validate()
    var_13 = 'Total Debits and Credits are not equal'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_posting_constructor_valid_data. Retrieved 4/9 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 27
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_read_journal_entries_call_returns_iterable_of_entries. Retrieved 6/21 statements.
# Partially parsed test_read_journal_entries_call_with_empty_range_returns_empty_list. Retrieved 5/19 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 31
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = module_1.DateRange(var_4, var_8)

import datetime as module_0
import pypara.commons.zeitgeist as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = module_1.DateRange(var_4, var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_success_when_debits_equal_credits. Retrieved 15/27 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test'
    var_6 = 'TestSource'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = 'A'
    var_9 = 'B'
    var_10 = '100.00'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_2.Decimal(*var_11, **var_12)
    var_14 = [var_0, var_1, var_1]
    var_15 = {}
    var_16 = module_0.date(*var_14, **var_15)
    var_17 = [var_0, var_1, var_1]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = '-100.00'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = var_7.validate()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_journal_entry_post_adds_increment_posting. Retrieved 5/18 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test'
    var_6 = 100



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_journal_entry_validate_success. Retrieved 13/19 statements.
# Partially parsed test_journal_entry_validate_failure. Retrieved 13/20 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = 'B'
    var_7 = 'Test'
    var_8 = 'Source'
    var_9 = module_1.JournalEntry(var_4, var_7, var_8)
    var_10 = '100.00'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_2.Decimal(*var_11, **var_12)
    var_14 = '-100.00'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = var_9.validate()

import datetime as module_0
import pypara.accounting.journaling as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = 'B'
    var_7 = 'Test'
    var_8 = 'Source'
    var_9 = module_1.JournalEntry(var_4, var_7, var_8)
    var_10 = '100.00'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_2.Decimal(*var_11, **var_12)
    var_14 = '-50.00'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = var_9.validate()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_success_when_debits_equal_credits. Retrieved 13/24 statements.


import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = 'B'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = '-100.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = 'Test Entry'
    var_16 = 'TestSource'
    var_17 = module_2.JournalEntry(var_4, var_15, var_16)
    var_18 = var_17.validate()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_read_journal_entries_call_returns_iterable_of_entries. Retrieved 5/21 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 31
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'Source'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_posting_constructor_initialization. Retrieved 4/9 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 27
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'Source'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_post_adds_posting_when_quantity_is_non_zero. Retrieved 8/22 statements.
# Partially parsed test_post_does_nothing_when_quantity_is_zero. Retrieved 6/17 statements.
# Partially parsed test_post_creates_correct_amount_and_direction. Retrieved 6/19 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test'
    var_6 = 2
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 0
    var_11 = [var_0, var_1, var_6]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)

import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test'
    var_6 = 2
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)

import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test'
    var_6 = [var_0, var_1, var_1]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 100



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'Test Source'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_journal_entry_constructor_initialization. Retrieved 7/8 statements.


import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Test Entry'
    var_6 = 'SourceObject'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.date
    var_9 = bool(var_7.date == var_4)
    assert var_9 is True
    var_10 = var_7.description
    var_11 = bool(var_7.description == var_5)
    assert var_11 is True
    var_12 = var_7.source
    var_13 = bool(var_7.source == var_6)
    assert var_13 is True
    var_14 = var_7.postings
    var_15 = bool(var_7.postings == [])
    assert var_15 is True
    var_16 = var_7.guid



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_journal_entry_validate_success. Retrieved 13/19 statements.
# Partially parsed test_journal_entry_validate_failure_imbalance. Retrieved 11/24 statements.
# Partially parsed test_journal_entry_validate_single_zero_quantity. Retrieved 10/13 statements.


import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = 'B'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = '-100.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = 'Test'
    var_16 = 'Source'
    var_17 = module_2.JournalEntry(var_4, var_15, var_16)
    var_18 = var_17.validate()

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = 'B'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = '-50.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = 'Test'
    var_16 = 'Source'

import datetime as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Empty'
    var_6 = 'Source'
    var_7 = module_1.JournalEntry(var_4, var_5, var_6)
    var_8 = var_7.validate()

import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = '0.00'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Decimal(*var_7, **var_8)
    var_10 = 'Zero'
    var_11 = 'Source'
    var_12 = module_2.JournalEntry(var_4, var_10, var_11)
    var_13 = var_12.validate()



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_post_with_zero_quantity_does_not_add_posting. Retrieved 5/16 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'Zero quantity test'
    var_6 = [var_0, var_1, var_1]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_validate_success. Retrieved 11/20 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 'A'
    var_6 = 'B'
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = 'Test Entry'
    var_12 = None
    var_13 = '-100.00'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)



