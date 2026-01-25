####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_general_ledger_program_call_returns_general_ledger. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'Test'
    var_3 = 'Test Journal'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 7/23 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 'Test transaction'
    var_4 = 'Asset'
    var_5 = 100
    var_6 = 'Equity'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = module_0.ReadInitialBalances()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/17 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 'Test Account'
    var_4 = 100
    var_5 = 'USD'
    var_6 = 'Test Description'
    var_7 = []



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_general_ledger_program_call_returns_general_ledger. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.GeneralLedgerProgram()
    var_1 = 2023
    var_2 = 1
    var_3 = 31



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ledger_constructor_initializes_correctly. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/11 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1
    var_5 = 'Test entry'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/11 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Cash'
    var_4 = 'Test'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/6 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 'Test'
    var_2 = 100
    var_3 = module_1.Posting(var_0, var_1, var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/11 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 7/18 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 11/27 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 10/23 statements.
# Partially parsed test_build_general_ledger_multiple_postings. Retrieved 14/36 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = 100
    var_6 = []

import builtins as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = module_0.object()
    var_6 = 6
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 50
    var_10 = {}

import builtins as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = module_0.object()
    var_6 = 2022
    var_7 = 'Test Entry'
    var_8 = 50
    var_9 = {}

import builtins as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = module_0.object()
    var_6 = 6
    var_7 = 15
    var_8 = 'Test Entry 1'
    var_9 = 50
    var_10 = 7
    var_11 = 'Test Entry 2'
    var_12 = -30
    var_13 = {}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_ledger_constructor_with_default_entries. Retrieved 2/6 statements.
# Partially parsed test_ledger_constructor_with_non_default_entries. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100

def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100
    var_2 = 50
    var_3 = 150



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/12 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 11/38 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount'
    var_5 = 100
    var_6 = 6
    var_7 = 'Test Entry'
    var_8 = None
    var_9 = 50
    var_10 = 2024



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_LedgerEntry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Assets:Cash'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/10 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 7/19 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 10/30 statements.
# Partially parsed test_build_general_ledger_with_multiple_postings. Retrieved 12/43 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 11/26 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 15
    var_7 = 'Test Entry'
    var_8 = 'Test Source'
    var_9 = '50.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account 1'
    var_5 = 'Test Account 2'
    var_6 = {}
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 'Test Source'
    var_10 = '50.00'
    var_11 = '30.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 2022
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 'Test Source'
    var_10 = '50.00'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_initial_balances. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = module_0.ReadInitialBalances()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/12 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_ledger_constructor_with_default_entries. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledger_for_new_account. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 'Test Entry'
    var_7 = 'Test Source'
    var_8 = 100
    var_9 = 0



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/26 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test Account'
    var_4 = 2023
    var_5 = 1
    var_6 = 'Test Description'
    var_7 = 'Counter Account'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 12/39 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = 100
    var_6 = 6
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 'Test Source'
    var_10 = 50
    var_11 = 2024



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_build_general_ledger_predicate. Retrieved 11/30 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 6
    var_5 = 15
    var_6 = 'Test entry'
    var_7 = 'Test source'
    var_8 = 'Test Account'
    var_9 = '100'
    var_10 = '50'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/10 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 7/19 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 10/27 statements.
# Partially parsed test_build_general_ledger_with_postings_outside_period. Retrieved 11/23 statements.
# Partially parsed test_build_general_ledger_with_multiple_postings. Retrieved 12/37 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 15
    var_7 = 'Test Entry'
    var_8 = 'Source'
    var_9 = '50.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 2022
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 'Source'
    var_10 = '50.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account 1'
    var_5 = 'Test Account 2'
    var_6 = {}
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 'Source'
    var_10 = '50.00'
    var_11 = '-30.00'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 100



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = module_0.ReadInitialBalances()



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_build_general_ledger_with_empty_journal. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 11/34 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount'
    var_5 = 100
    var_6 = 6
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = None
    var_10 = 50



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_general_ledger_program_call_returns_general_ledger. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.GeneralLedgerProgram()
    var_1 = 2023
    var_2 = 1
    var_3 = 31



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_call_returns_initial_balances. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2023
    var_2 = 1
    var_3 = 31



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 16/51 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = 'TestAccount'
    var_4 = 0
    var_5 = 2022
    var_6 = 12
    var_7 = 'Old Entry'
    var_8 = None
    var_9 = 100
    var_10 = 15
    var_11 = 'Valid Entry'
    var_12 = 200
    var_13 = 2
    var_14 = 'Future Entry'
    var_15 = 300



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/10 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 7/19 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 11/30 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 10/22 statements.
# Partially parsed test_build_general_ledger_multiple_postings. Retrieved 16/47 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount'
    var_5 = '100.00'
    var_6 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount'
    var_5 = {}
    var_6 = 6
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 'TestSource'
    var_10 = '50.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount'
    var_5 = {}
    var_6 = 2024
    var_7 = 'Test Entry'
    var_8 = 'TestSource'
    var_9 = '50.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount1'
    var_5 = 'TestAccount2'
    var_6 = {}
    var_7 = 6
    var_8 = 15
    var_9 = 'Test Entry 1'
    var_10 = 'TestSource1'
    var_11 = '50.00'
    var_12 = 7
    var_13 = 'Test Entry 2'
    var_14 = 'TestSource2'
    var_15 = '100.00'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 7/15 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 'Test'
    var_4 = []
    var_5 = 100
    var_6 = 'USD'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 'Test'
    var_2 = 100
    var_3 = 'USD'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_build_general_ledger_with_empty_journal. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_general_ledger_program_call_returns_general_ledger. Retrieved 5/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.GeneralLedgerProgram()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/25 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 'Test transaction'
    var_4 = 'Test Account'
    var_5 = 100
    var_6 = 'USD'
    var_7 = 'Another Account'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/10 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 7/19 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 12/41 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 11/23 statements.
# Partially parsed test_build_general_ledger_with_initial_and_postings. Retrieved 11/35 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Assets:Cash'
    var_5 = '1000.00'
    var_6 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Assets:Cash'
    var_5 = 'Expenses:Rent'
    var_6 = 15
    var_7 = 'Rent payment'
    var_8 = None
    var_9 = '-500.00'
    var_10 = '500.00'
    var_11 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Assets:Cash'
    var_5 = 2022
    var_6 = 15
    var_7 = 'Out of period'
    var_8 = None
    var_9 = '100.00'
    var_10 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Assets:Cash'
    var_5 = '1000.00'
    var_6 = 15
    var_7 = 'Deposit'
    var_8 = None
    var_9 = '500.00'
    var_10 = '1500.00'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/11 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 2023
    var_3 = 1
    var_4 = 'Test'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/25 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 'Test transaction'
    var_4 = 'Assets:Cash'
    var_5 = 100
    var_6 = 'USD'
    var_7 = 'Income:Salary'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/11 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'Test'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/12 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test Account'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 7/23 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 'Test transaction'
    var_4 = 'Account1'
    var_5 = 100
    var_6 = 'Account2'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/21 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = 'Test Ledger'
    var_1 = module_0.Ledger(var_0)
    var_2 = 'Test Account'
    var_3 = 'Test Journal'
    var_4 = 2023
    var_5 = 1
    var_6 = 'Test Description'
    var_7 = 100
    var_8 = 'USD'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/12 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'Test'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/12 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledger_for_new_account. Retrieved 12/28 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = 6
    var_6 = 15
    var_7 = 'Test Entry'
    var_8 = None
    var_9 = '100.00'
    var_10 = {}
    var_11 = 0



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_general_ledger_program_call_returns_general_ledger. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 3/5 statements.


import pypara.accounting.ledger as module_0
import pypara.accounting.journaling as module_1

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = module_1.Posting()
    var_2 = 100



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 100



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_initial_balances. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2023
    var_2 = 1
    var_3 = 31



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 4/10 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 'TestAccount'
    var_2 = 100
    var_3 = 'USD'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 14/43 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount'
    var_5 = 0
    var_6 = 6
    var_7 = 15
    var_8 = 'Test entry in period'
    var_9 = None
    var_10 = 100
    var_11 = 2024
    var_12 = 'Test entry out of period'
    var_13 = 200



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 5/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_ledger_constructor_with_default_entries. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 'Test'
    var_2 = 100
    var_3 = 'USD'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 7/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 100
    var_2 = 'USD'
    var_3 = 'Test'
    var_4 = 2023
    var_5 = 1
    var_6 = 'Test Description'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_initial_balances. Retrieved 4/9 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2023
    var_2 = 1
    var_3 = 31



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/10 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 7/19 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 12/41 statements.
# Partially parsed test_build_general_ledger_with_initial_and_postings. Retrieved 11/35 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 10/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Account 1'
    var_5 = 'Account 2'
    var_6 = {}
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = None
    var_10 = '50.00'
    var_11 = '-50.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = '100.00'
    var_6 = 15
    var_7 = 'Test Entry'
    var_8 = None
    var_9 = '50.00'
    var_10 = '150.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 2022
    var_7 = 'Test Entry'
    var_8 = None
    var_9 = '50.00'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_build_general_ledger_includes_postings_within_period. Retrieved 11/31 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = 100
    var_6 = 6
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = None
    var_10 = 50



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/10 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 7/19 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 10/29 statements.
# Partially parsed test_build_general_ledger_with_multiple_postings. Retrieved 16/47 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = 100
    var_6 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 15
    var_7 = 'Test Entry'
    var_8 = 'Source'
    var_9 = 50

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account 1'
    var_5 = 'Test Account 2'
    var_6 = {}
    var_7 = 15
    var_8 = 'Test Entry 1'
    var_9 = 'Source1'
    var_10 = 50
    var_11 = 20
    var_12 = 'Test Entry 2'
    var_13 = 'Source2'
    var_14 = -30
    var_15 = -30

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Test Account'
    var_5 = {}
    var_6 = 2022
    var_7 = 15
    var_8 = 'Test Entry'
    var_9 = 'Source'
    var_10 = 50



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 6/13 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.Ledger()
    var_1 = 2023
    var_2 = 1
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'Test'



