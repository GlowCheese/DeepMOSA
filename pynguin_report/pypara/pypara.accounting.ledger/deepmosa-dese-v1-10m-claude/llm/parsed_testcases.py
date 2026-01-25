####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/29 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 500.0
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/43 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Description'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/7 statements.
# Partially parsed test_ledger_constructor_with_different_values. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '12345'
    var_2 = 1000

def test_case_0():
    var_0 = 'Savings'
    var_1 = '67890'
    var_2 = 5000



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 47/69 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Amount'
    var_5 = ()
    var_6 = '__repr__'
    var_7 = 'Amount(100)'
    var_8 = lambda self: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = 'Quantity'
    var_12 = ()
    var_13 = 'Quantity(100)'
    var_14 = lambda self: var_13
    var_15 = {var_6: var_14}
    var_16 = type(var_11, var_12, var_15)
    var_17 = 'Posting'
    var_18 = ()
    var_19 = 'date'
    var_20 = 'amount'
    var_21 = 'is_debit'
    var_22 = 'is_credit'
    var_23 = 'direction'
    var_24 = 'account'
    var_25 = 'journal'
    var_26 = 2024
    var_27 = 1
    var_28 = 15
    var_29 = True
    var_30 = False
    var_31 = 'debit'
    var_32 = 'Journal'
    var_33 = ()
    var_34 = 'description'
    var_35 = 'postings'
    var_36 = 'Test transaction'
    var_37 = ()
    var_38 = ()
    var_39 = 'credit'
    var_40 = ()
    var_41 = {}
    var_42 = type(var_0, var_40, var_41)
    var_43 = 'Ledger'
    var_44 = ()
    var_45 = {}
    var_46 = type(var_43, var_44, var_45)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/43 statements.


def test_case_0():
    var_0 = 'Cash'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test transaction'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 'General Ledger'
    var_12 = 500.0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 2023
    var_10 = 1
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 1000.0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Cash'
    var_4 = 100.0
    var_5 = 'USD'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 2024
    var_9 = 1
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_build_general_ledger_with_single_posting. Retrieved 14/50 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/28 statements.
# Partially parsed test_build_general_ledger_multiple_entries. Retrieved 20/66 statements.
# Partially parsed test_build_general_ledger_filters_by_period. Retrieved 20/66 statements.
# Partially parsed test_build_general_ledger_creates_missing_ledgers. Retrieved 16/47 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 1000
    var_7 = 15
    var_8 = 'Test entry'
    var_9 = 'source'
    var_10 = -100
    var_11 = 100
    var_12 = 12
    var_13 = 31

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 2024
    var_3 = 1
    var_4 = 5000
    var_5 = 12
    var_6 = 31
    var_7 = []

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 1000
    var_7 = 15
    var_8 = 'Entry 1'
    var_9 = 'source1'
    var_10 = -100
    var_11 = 100
    var_12 = 2
    var_13 = 20
    var_14 = 'Entry 2'
    var_15 = 'source2'
    var_16 = -200
    var_17 = 200
    var_18 = 12
    var_19 = 31

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 1000
    var_7 = 2
    var_8 = 15
    var_9 = 'In period'
    var_10 = 'source1'
    var_11 = -100
    var_12 = 100
    var_13 = 12
    var_14 = 25
    var_15 = 'Out of period'
    var_16 = 'source2'
    var_17 = -200
    var_18 = 200
    var_19 = 24

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = '3000'
    var_5 = 'Revenue'
    var_6 = 2024
    var_7 = 1
    var_8 = 1000
    var_9 = 15
    var_10 = 'Test entry'
    var_11 = 'source'
    var_12 = 100
    var_13 = -100
    var_14 = 12
    var_15 = 31



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 100.0
    var_3 = 'USD'
    var_4 = 'Cash'
    var_5 = 'debit'
    var_6 = True
    var_7 = False
    var_8 = 1000.0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_postings. Retrieved 15/57 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Payable'
    var_8 = 1000
    var_9 = 6
    var_10 = 15
    var_11 = 'Test entry'
    var_12 = 'source_obj'
    var_13 = -500
    var_14 = 500



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 10/40 statements.


def test_case_0():
    var_0 = 'Cash'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test transaction'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/21 statements.
# Partially parsed test_read_initial_balances_call_with_different_periods. Retrieved 5/27 statements.
# Partially parsed test_read_initial_balances_call_empty_balances. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 2024

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 'Test Journal'
    var_8 = []
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 4/8 statements.


import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Test Account'
    var_1 = 'asset'
    var_2 = 1000
    var_3 = module_0.Balance(var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/52 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2024
    var_6 = 1
    var_7 = 'debit'
    var_8 = None
    var_9 = 'Counter'
    var_10 = 'credit'
    var_11 = 'Test Journal'
    var_12 = 'Test Ledger'
    var_13 = module_1.Ledger()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 'Cash'
    var_4 = 100.0
    var_5 = 'USD'
    var_6 = 500.0
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 'Test transaction'
    var_11 = []



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2024
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'TestAccount'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_date_range. Retrieved 18/51 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Payable'
    var_8 = 1000
    var_9 = 6
    var_10 = 15
    var_11 = 'Entry within period'
    var_12 = 'source1'
    var_13 = 2024
    var_14 = 'Entry outside period'
    var_15 = 'source2'
    var_16 = 100
    var_17 = 50



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 23/81 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/27 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 25/73 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Accounts Payable'
    var_4 = '3000'
    var_5 = 'Revenue'
    var_6 = 2024
    var_7 = 1
    var_8 = 12
    var_9 = 31
    var_10 = 'Initial deposit'
    var_11 = 'source1'
    var_12 = '500'
    var_13 = '-500'
    var_14 = 6
    var_15 = 15
    var_16 = 'Payment'
    var_17 = 'source2'
    var_18 = '-300'
    var_19 = '300'
    var_20 = '1500'
    var_21 = '1200'
    var_22 = '0'

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 2024
    var_3 = 1
    var_4 = 12
    var_5 = 31
    var_6 = '5000'
    var_7 = []

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '3000'
    var_3 = 'Revenue'
    var_4 = 2024
    var_5 = 6
    var_6 = 1
    var_7 = 12
    var_8 = 31
    var_9 = {}
    var_10 = 15
    var_11 = 'Before period'
    var_12 = 'source1'
    var_13 = '100'
    var_14 = '-100'
    var_15 = 7
    var_16 = 'Within period'
    var_17 = 'source2'
    var_18 = '200'
    var_19 = '-200'
    var_20 = 2025
    var_21 = 'After period'
    var_22 = 'source3'
    var_23 = '300'
    var_24 = '-300'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/33 statements.


def test_case_0():
    var_0 = 'Test transaction'
    var_1 = []
    var_2 = 100.0
    var_3 = 2023
    var_4 = 1
    var_5 = 'debit'
    var_6 = True
    var_7 = False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 4/6 statements.
# Partially parsed test_ledger_constructor_with_account_and_balance. Retrieved 4/10 statements.
# Partially parsed test_ledger_constructor_entries_initialized_empty. Retrieved 5/10 statements.


import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Test Account'
    var_1 = 'ACC001'
    var_2 = 1000
    var_3 = module_0.Balance(var_2)

import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Savings'
    var_1 = 'SAV123'
    var_2 = 5000
    var_3 = module_0.Balance(var_2)

import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Checking'
    var_1 = 'CHK456'
    var_2 = 2500
    var_3 = module_0.Balance(var_2)
    var_4 = 'entries'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2024
    var_6 = 1
    var_7 = 'Test Journal'
    var_8 = []
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 37/61 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Amount'
    var_5 = ()
    var_6 = 'value'
    var_7 = '100.00'
    var_8 = 'Posting'
    var_9 = ()
    var_10 = 'date'
    var_11 = 'amount'
    var_12 = 'is_debit'
    var_13 = 'is_credit'
    var_14 = 'direction'
    var_15 = 'journal'
    var_16 = 2024
    var_17 = 1
    var_18 = 15
    var_19 = True
    var_20 = False
    var_21 = 'debit'
    var_22 = 'Journal'
    var_23 = ()
    var_24 = 'description'
    var_25 = 'postings'
    var_26 = 'Test transaction'
    var_27 = ()
    var_28 = 'account'
    var_29 = 'credit'
    var_30 = 'Ledger'
    var_31 = ()
    var_32 = {}
    var_33 = type(var_30, var_31, var_32)
    var_34 = 'Quantity'
    var_35 = ()
    var_36 = '500.00'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 44/63 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Account'
    var_3 = ()
    var_4 = {}
    var_5 = type(var_2, var_3, var_4)
    var_6 = 'Amount'
    var_7 = ()
    var_8 = '__init__'
    var_9 = None
    var_10 = lambda self: var_9
    var_11 = {var_8: var_10}
    var_12 = type(var_6, var_7, var_11)
    var_13 = 'Posting'
    var_14 = ()
    var_15 = 'date'
    var_16 = 'amount'
    var_17 = 'is_debit'
    var_18 = 'is_credit'
    var_19 = 'direction'
    var_20 = 'journal'
    var_21 = 2023
    var_22 = 1
    var_23 = 15
    var_24 = True
    var_25 = False
    var_26 = 'debit'
    var_27 = 'Journal'
    var_28 = ()
    var_29 = 'description'
    var_30 = 'postings'
    var_31 = 'Test transaction'
    var_32 = 'Post'
    var_33 = ()
    var_34 = 'account'
    var_35 = 'credit'
    var_36 = 'Ledger'
    var_37 = ()
    var_38 = {}
    var_39 = type(var_36, var_37, var_38)
    var_40 = 'Quantity'
    var_41 = ()
    var_42 = {}
    var_43 = type(var_40, var_41, var_42)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '12345'
    var_2 = 1000



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test Journal'
    var_8 = []
    var_9 = 2023
    var_10 = 1
    var_11 = 'debit'
    var_12 = 500.0



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/50 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 2024
    var_10 = 1
    var_11 = 'debit'
    var_12 = 500.0



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_postings. Retrieved 17/49 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = {}
    var_5 = 6
    var_6 = 15
    var_7 = 'Test entry'
    var_8 = 'source'
    var_9 = '1000'
    var_10 = 'Assets'
    var_11 = 'Cash'
    var_12 = '2000'
    var_13 = 'Liabilities'
    var_14 = 'Payables'
    var_15 = 100
    var_16 = -100



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/47 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Cash'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'debit'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_build_general_ledger_predicate_evaluates_to_false. Retrieved 11/38 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 2022
    var_5 = 'Test'
    var_6 = 'test_source'
    var_7 = '1000'
    var_8 = 'Test Account'
    var_9 = 100
    var_10 = {}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 25/80 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '2000'
    var_6 = 'Payable'
    var_7 = 15
    var_8 = 'Entry inside period'
    var_9 = 'source1'
    var_10 = 2023
    var_11 = 12
    var_12 = 'Entry before period'
    var_13 = 'source2'
    var_14 = 2
    var_15 = 'Entry after period'
    var_16 = 'source3'
    var_17 = 100
    var_18 = -100
    var_19 = 50
    var_20 = -50
    var_21 = 75
    var_22 = -75
    var_23 = {}
    var_24 = -100



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Cash'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/46 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Transaction'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 15
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 'General Ledger'
    var_14 = 500.0



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/21 statements.


import builtins as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_0.object()
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = module_0.object()
    var_8 = module_1.LedgerEntry(var_0, var_3, var_7)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 'Test Journal'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/41 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1
    var_5 = 'debit'
    var_6 = True
    var_7 = False
    var_8 = 'Test Journal'
    var_9 = []
    var_10 = 500.0



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 'Test Journal'
    var_8 = 'debit'
    var_9 = True
    var_10 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Description'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 500.0
    var_11 = module_1.Ledger()



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 500.0
    var_11 = module_1.Ledger()



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 20/58 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 1000
    var_7 = 6
    var_8 = 15
    var_9 = 'Inside period'
    var_10 = 'source1'
    var_11 = 2023
    var_12 = 'Before period'
    var_13 = 'source2'
    var_14 = 2025
    var_15 = 'After period'
    var_16 = 'source3'
    var_17 = 100
    var_18 = 50
    var_19 = 75



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2024
    var_4 = 1
    var_5 = 15
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 'Cash'
    var_9 = 'Test transaction'
    var_10 = []
    var_11 = 'debit'
    var_12 = 500.0



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/46 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Cash'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 500.0
    var_14 = 'General Ledger'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/40 statements.


def test_case_0():
    var_0 = 100.0
    var_1 = 'USD'
    var_2 = 'TestAccount'
    var_3 = 'Test Journal Entry'
    var_4 = []
    var_5 = 2024
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 'TestLedger'
    var_11 = 500.0



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()
    var_11 = 500.0



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 24/82 statements.
# Partially parsed test_build_general_ledger_with_account_not_in_initial_balances. Retrieved 15/50 statements.
# Partially parsed test_build_general_ledger_filters_by_period. Retrieved 17/53 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/28 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 1000
    var_9 = 500
    var_10 = 15
    var_11 = 'Test entry 1'
    var_12 = 'source1'
    var_13 = -100
    var_14 = 100
    var_15 = 2
    var_16 = 'Test entry 2'
    var_17 = 'source2'
    var_18 = 50
    var_19 = -50
    var_20 = 900
    var_21 = 950
    var_22 = 600
    var_23 = 550

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 1000
    var_9 = 15
    var_10 = 'Test entry'
    var_11 = 'source1'
    var_12 = -100
    var_13 = 100
    var_14 = 0

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 1
    var_3 = 12
    var_4 = 31
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 1000
    var_8 = 15
    var_9 = 'Entry outside period'
    var_10 = 'source1'
    var_11 = -100
    var_12 = 3
    var_13 = 'Entry inside period'
    var_14 = 'source2'
    var_15 = 50
    var_16 = 1050

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 1000
    var_7 = []



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 2024
    var_4 = 1
    var_5 = 15
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 'Test Journal Entry'
    var_9 = []
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 'Test Ledger'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_accounts. Retrieved 15/46 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 1000
    var_9 = 'test_source'
    var_10 = 6
    var_11 = 15
    var_12 = 'Test entry'
    var_13 = -100
    var_14 = 100



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/47 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = None
    var_9 = 'credit'
    var_10 = 'Test Transaction'
    var_11 = 'debit'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 500.0
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 22/81 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/31 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 11/38 statements.
# Partially parsed test_build_general_ledger_creates_new_accounts. Retrieved 14/50 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '4000'
    var_7 = 'Revenue'
    var_8 = 15
    var_9 = 'Initial deposit'
    var_10 = None
    var_11 = '500'
    var_12 = '-500'
    var_13 = 2
    var_14 = 10
    var_15 = 'Revenue earned'
    var_16 = '300'
    var_17 = '-300'
    var_18 = '1500'
    var_19 = '1800'
    var_20 = '0'
    var_21 = '-800'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '5000'
    var_7 = []

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 2023
    var_7 = 15
    var_8 = 'Before period'
    var_9 = None
    var_10 = '100'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '5000'
    var_7 = 'Expense'
    var_8 = 6
    var_9 = 15
    var_10 = None
    var_11 = '200'
    var_12 = '-200'
    var_13 = '0'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_build_general_ledger_posting_account_not_in_ledgers. Retrieved 11/30 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 12
    var_4 = 31
    var_5 = '1000'
    var_6 = 'Test Account'
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = '100'
    var_10 = {}



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 'Test Account'
    var_4 = 100.0
    var_5 = 'USD'
    var_6 = 1000.0
    var_7 = 'Test Entry'
    var_8 = []
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Assets'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/40 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 'Assets'
    var_4 = 100.0
    var_5 = 'USD'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 'Test Journal'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 33/43 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Journal'
    var_5 = ()
    var_6 = 'description'
    var_7 = 'postings'
    var_8 = 'Test Journal'
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = type(var_4, var_5, var_10)
    var_12 = 'Posting'
    var_13 = ()
    var_14 = 'date'
    var_15 = 'account'
    var_16 = 'journal'
    var_17 = 'amount'
    var_18 = 'direction'
    var_19 = 'is_debit'
    var_20 = 'is_credit'
    var_21 = 2024
    var_22 = 1
    var_23 = 15
    var_24 = 100.0
    var_25 = 'debit'
    var_26 = True
    var_27 = False
    var_28 = 'Ledger'
    var_29 = ()
    var_30 = {}
    var_31 = type(var_28, var_29, var_30)
    var_32 = 500.0



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 'Counter Account'
    var_8 = 'credit'
    var_9 = None
    var_10 = 'debit'
    var_11 = 'Test Journal'
    var_12 = module_1.Ledger()



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Description'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 1000.0



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 38/56 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Amount'
    var_5 = ()
    var_6 = 'value'
    var_7 = '100.00'
    var_8 = 'Posting'
    var_9 = ()
    var_10 = 'account'
    var_11 = 'amount'
    var_12 = 'date'
    var_13 = 'direction'
    var_14 = 'is_debit'
    var_15 = 'is_credit'
    var_16 = 'journal'
    var_17 = 2023
    var_18 = 1
    var_19 = 15
    var_20 = 'debit'
    var_21 = True
    var_22 = False
    var_23 = 'Journal'
    var_24 = ()
    var_25 = 'description'
    var_26 = 'postings'
    var_27 = 'Test transaction'
    var_28 = []
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = type(var_23, var_24, var_29)
    var_31 = 'Ledger'
    var_32 = ()
    var_33 = {}
    var_34 = type(var_31, var_32, var_33)
    var_35 = 'Quantity'
    var_36 = ()
    var_37 = '500.00'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_build_general_ledger_predicate_false. Retrieved 10/27 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 2023
    var_5 = 'TestAccount'
    var_6 = 'Test Entry'
    var_7 = 'source'
    var_8 = 100
    var_9 = {}



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/7 statements.
# Partially parsed test_ledger_constructor_with_different_values. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '12345'
    var_2 = 1000

def test_case_0():
    var_0 = 'Savings'
    var_1 = '67890'
    var_2 = 5000



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2024
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()
    var_12 = 500.0



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 14/59 statements.
# Partially parsed test_build_general_ledger_creates_missing_accounts. Retrieved 14/46 statements.
# Partially parsed test_build_general_ledger_filters_by_period. Retrieved 14/49 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = '1000.00'
    var_9 = '500.00'
    var_10 = 'Test transaction'
    var_11 = 'source_object'
    var_12 = '-100.00'
    var_13 = '100.00'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = '1000.00'
    var_9 = 'Test transaction'
    var_10 = 'source_object'
    var_11 = '-100.00'
    var_12 = '100.00'
    var_13 = '0.00'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '1000.00'
    var_6 = 15
    var_7 = 'In period'
    var_8 = 'source1'
    var_9 = '100.00'
    var_10 = 2
    var_11 = 'Out of period'
    var_12 = 'source2'
    var_13 = '200.00'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31
    var_3 = '1000'
    var_4 = 'Cash'
    var_5 = '1000.00'
    var_6 = []



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2024
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 5/22 statements.
# Partially parsed test_read_initial_balances_call_empty_balances. Retrieved 5/20 statements.
# Partially parsed test_read_initial_balances_call_multiple_accounts. Retrieved 5/21 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2023
    var_2 = 1
    var_3 = 12
    var_4 = 31

import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2023
    var_2 = 6
    var_3 = 1
    var_4 = 30

import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2024
    var_2 = 1
    var_3 = 3
    var_4 = 31



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/44 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Description'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = 500.0



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 10/45 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1
    var_5 = 'debit'
    var_6 = True
    var_7 = False
    var_8 = 'Test Description'
    var_9 = 500.0



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/51 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 'Test transaction'
    var_4 = []
    var_5 = 2024
    var_6 = 1
    var_7 = 15
    var_8 = 100.0
    var_9 = 'USD'
    var_10 = 'debit'
    var_11 = 'General Ledger'
    var_12 = module_1.Ledger()
    var_13 = 500.0



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 10/43 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '100.00'
    var_2 = 'USD'
    var_3 = 'Test Journal'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 24/89 statements.
# Partially parsed test_build_general_ledger_with_new_accounts. Retrieved 17/55 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 13/48 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/31 statements.


import builtins as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 1000
    var_9 = 500
    var_10 = module_0.object()
    var_11 = 15
    var_12 = 'Test entry 1'
    var_13 = -100
    var_14 = 100
    var_15 = 2
    var_16 = 20
    var_17 = 'Test entry 2'
    var_18 = 50
    var_19 = -50
    var_20 = 900
    var_21 = 950
    var_22 = 600
    var_23 = 550

import builtins as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '3000'
    var_7 = 'Revenue'
    var_8 = 5000
    var_9 = module_0.object()
    var_10 = 3
    var_11 = 10
    var_12 = 'Revenue entry'
    var_13 = 1000
    var_14 = -1000
    var_15 = 0
    var_16 = -1000

import builtins as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 1000
    var_7 = module_0.object()
    var_8 = 2022
    var_9 = 'Before period'
    var_10 = 100
    var_11 = 2024
    var_12 = 'After period'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 1000
    var_7 = []



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/46 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test entry'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 15
    var_10 = 'debit'
    var_11 = 'General Ledger'
    var_12 = module_1.Ledger()
    var_13 = 500.0



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 39/51 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Journal'
    var_5 = ()
    var_6 = 'description'
    var_7 = 'postings'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = type(var_4, var_5, var_10)
    var_12 = 'Posting'
    var_13 = ()
    var_14 = 'date'
    var_15 = 'account'
    var_16 = 'journal'
    var_17 = 'amount'
    var_18 = 'direction'
    var_19 = 'is_debit'
    var_20 = 'is_credit'
    var_21 = 2024
    var_22 = 1
    var_23 = 15
    var_24 = 'Amount'
    var_25 = ()
    var_26 = {}
    var_27 = type(var_24, var_25, var_26)
    var_28 = 'debit'
    var_29 = True
    var_30 = False
    var_31 = 'Ledger'
    var_32 = ()
    var_33 = {}
    var_34 = type(var_31, var_32, var_33)
    var_35 = 'Quantity'
    var_36 = ()
    var_37 = {}
    var_38 = type(var_35, var_36, var_37)



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 29/38 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_0.object()
    var_4 = module_0.object()
    var_5 = module_0.object()
    var_6 = 'Posting'
    var_7 = ()
    var_8 = 'date'
    var_9 = 'account'
    var_10 = 'journal'
    var_11 = 'amount'
    var_12 = 'direction'
    var_13 = 'is_debit'
    var_14 = 'is_credit'
    var_15 = 2024
    var_16 = 1
    var_17 = 15
    var_18 = 'debit'
    var_19 = True
    var_20 = False
    var_21 = 'Quantity'
    var_22 = ()
    var_23 = {}
    var_24 = type(var_21, var_22, var_23)
    var_25 = 'Ledger'
    var_26 = ()
    var_27 = {}
    var_28 = type(var_25, var_26, var_27)



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_build_general_ledger_posting_account_not_in_ledgers. Retrieved 15/45 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 6
    var_9 = 15
    var_10 = 'Test entry'
    var_11 = 'source'
    var_12 = 100
    var_13 = -100
    var_14 = 50



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #92
#--------------------------




import typing as module_0
import builtins as module_1
import pypara.accounting.ledger as module_2

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.object()
    var_3 = module_1.object()
    var_4 = module_1.object()
    var_5 = module_2.LedgerEntry(var_2, var_3, var_4)



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_build_general_ledger_predicate_filters_postings_within_period. Retrieved 24/81 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Test Account 1'
    var_2 = '2000'
    var_3 = 'Test Account 2'
    var_4 = 2024
    var_5 = 1
    var_6 = 31
    var_7 = 15
    var_8 = 'Inside period'
    var_9 = 'source1'
    var_10 = 2023
    var_11 = 12
    var_12 = 'Before period'
    var_13 = 'source2'
    var_14 = 2
    var_15 = 'After period'
    var_16 = 'source3'
    var_17 = 100
    var_18 = -100
    var_19 = 50
    var_20 = -50
    var_21 = 75
    var_22 = -75
    var_23 = 0



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2024
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 4/6 statements.
# Partially parsed test_ledger_constructor_with_different_values. Retrieved 4/10 statements.


import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Test Account'
    var_1 = 'ASSET'
    var_2 = 1000
    var_3 = module_0.Balance(var_2)

import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Savings'
    var_1 = 'LIABILITY'
    var_2 = 5000
    var_3 = module_0.Balance(var_2)



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2024
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'Cash'
    var_11 = 'debit'
    var_12 = 500.0



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/21 statements.


import builtins as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_0.object()
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = module_0.object()
    var_8 = module_1.LedgerEntry(var_0, var_3, var_7)



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/44 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Description'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 'Test Ledger'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 'Test Entry'
    var_8 = []
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_accounts. Retrieved 15/52 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 12
    var_4 = 31
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = 'Accounts Payable'
    var_9 = 1000
    var_10 = 'Test transaction'
    var_11 = 'source_obj'
    var_12 = -100
    var_13 = 100
    var_14 = 0



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 2023
    var_4 = 1
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = module_1.Ledger()
    var_11 = 500.0



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 10/41 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 'debit'
    var_7 = True
    var_8 = False
    var_9 = 500.0



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 'Test Description'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = 'Test Ledger'
    var_13 = module_1.Ledger()



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/40 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_read_initial_balances_protocol_call. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_date_range. Retrieved 19/58 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31
    var_3 = '1000'
    var_4 = 'Test Account'
    var_5 = 2023
    var_6 = 12
    var_7 = 'Before period'
    var_8 = 'source1'
    var_9 = 15
    var_10 = 'Within period'
    var_11 = 'source2'
    var_12 = 2
    var_13 = 'After period'
    var_14 = 'source3'
    var_15 = 100
    var_16 = 50
    var_17 = 75
    var_18 = 0



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 24/82 statements.
# Partially parsed test_build_general_ledger_with_new_accounts. Retrieved 19/61 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/31 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 15/37 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Account A'
    var_5 = '001'
    var_6 = 'Account B'
    var_7 = '002'
    var_8 = 1000
    var_9 = 500
    var_10 = 6
    var_11 = 15
    var_12 = 'Test entry 1'
    var_13 = 'test_source_1'
    var_14 = -100
    var_15 = 100
    var_16 = 7
    var_17 = 20
    var_18 = 'Test entry 2'
    var_19 = 'test_source_2'
    var_20 = 50
    var_21 = -50
    var_22 = 950
    var_23 = 550

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Account A'
    var_5 = '001'
    var_6 = 'Account B'
    var_7 = '002'
    var_8 = 'Account C'
    var_9 = '003'
    var_10 = 1000
    var_11 = 500
    var_12 = 6
    var_13 = 15
    var_14 = 'Test entry'
    var_15 = 'test_source'
    var_16 = -200
    var_17 = 200
    var_18 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Account A'
    var_5 = '001'
    var_6 = 1000
    var_7 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Account A'
    var_5 = '001'
    var_6 = 'Account B'
    var_7 = '002'
    var_8 = 1000
    var_9 = 500
    var_10 = 2024
    var_11 = 6
    var_12 = 15
    var_13 = 'Outside period'
    var_14 = 'test_source'



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Cash'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 500.0



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'Cash'
    var_3 = 100
    var_4 = 'USD'
    var_5 = 'debit'
    var_6 = True
    var_7 = False
    var_8 = 1000



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 4/8 statements.


import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Test Account'
    var_1 = 'ASSET'
    var_2 = 1000
    var_3 = module_0.Balance(var_2)



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/43 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal Entry'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #115
#--------------------------




import builtins as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_1.LedgerEntry(var_0, var_1, var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/19 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = '100.00'
    var_4 = 'USD'
    var_5 = 2024
    var_6 = 1
    var_7 = 15
    var_8 = '500.00'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/43 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 'Test Transaction'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0
    var_12 = 'General Ledger'



# Parsed testcases at query #3
#--------------------------




import typing as module_0
import builtins as module_1
import pypara.accounting.ledger as module_2

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.object()
    var_3 = module_1.object()
    var_4 = module_1.object()
    var_5 = module_2.LedgerEntry(var_2, var_3, var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 39/51 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Journal'
    var_5 = ()
    var_6 = 'description'
    var_7 = 'postings'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = type(var_4, var_5, var_10)
    var_12 = 'Amount'
    var_13 = ()
    var_14 = {}
    var_15 = type(var_12, var_13, var_14)
    var_16 = 'Quantity'
    var_17 = ()
    var_18 = {}
    var_19 = type(var_16, var_17, var_18)
    var_20 = 'Posting'
    var_21 = ()
    var_22 = 'date'
    var_23 = 'account'
    var_24 = 'journal'
    var_25 = 'amount'
    var_26 = 'direction'
    var_27 = 'is_debit'
    var_28 = 'is_credit'
    var_29 = 2024
    var_30 = 1
    var_31 = 15
    var_32 = 'debit'
    var_33 = True
    var_34 = False
    var_35 = 'Ledger'
    var_36 = ()
    var_37 = {}
    var_38 = type(var_35, var_36, var_37)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/16 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = 100
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 500



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 500.0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '123456'
    var_2 = 1000



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/15 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 10/36 statements.
# Partially parsed test_build_general_ledger_with_postings_in_period. Retrieved 14/50 statements.
# Partially parsed test_build_general_ledger_filters_postings_outside_period. Retrieved 17/59 statements.
# Partially parsed test_build_general_ledger_creates_ledger_for_new_accounts. Retrieved 13/35 statements.
# Partially parsed test_build_general_ledger_accumulates_multiple_postings. Retrieved 15/41 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = {}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = []
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = 'Liabilities'
    var_9 = '500'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '3000'
    var_7 = 'Revenue'
    var_8 = 6
    var_9 = 15
    var_10 = 'Test entry'
    var_11 = 'source'
    var_12 = '100'
    var_13 = {}

def test_case_0():
    var_0 = 2024
    var_1 = 6
    var_2 = 1
    var_3 = 30
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 5
    var_7 = 15
    var_8 = 'Before period'
    var_9 = 'source'
    var_10 = '50'
    var_11 = 'In period'
    var_12 = '100'
    var_13 = 7
    var_14 = 'After period'
    var_15 = '75'
    var_16 = {}

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 6
    var_7 = 15
    var_8 = 'Test entry'
    var_9 = 'source'
    var_10 = '100'
    var_11 = {}
    var_12 = 0

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 6
    var_7 = 15
    var_8 = 'Entry 1'
    var_9 = 'source1'
    var_10 = '100'
    var_11 = 20
    var_12 = 'Entry 2'
    var_13 = 'source2'
    var_14 = '50'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Test Journal'
    var_5 = []
    var_6 = 'Cash'
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 2024
    var_10 = 1
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 500.0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 'debit'
    var_10 = 500.0
    var_11 = module_1.Ledger()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/41 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Journal'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 500.0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/44 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Cash'
    var_8 = 'Test Transaction'
    var_9 = []
    var_10 = 'debit'
    var_11 = True
    var_12 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_accounts. Retrieved 12/45 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = 'Test entry'
    var_8 = 'test_source'
    var_9 = '-100'
    var_10 = '100'
    var_11 = 31



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'TestAccount'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_read_initial_balances_protocol_call. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '123456'
    var_2 = 1000



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_date_range. Retrieved 23/98 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 12
    var_7 = 31
    var_8 = 6
    var_9 = 15
    var_10 = 'Within period'
    var_11 = 'source1'
    var_12 = 100
    var_13 = 2023
    var_14 = 'Before period'
    var_15 = 'source2'
    var_16 = 50
    var_17 = 2025
    var_18 = 'After period'
    var_19 = 'source3'
    var_20 = 75
    var_21 = 1000
    var_22 = 500



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 'Test Journal'
    var_8 = []
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/7 statements.
# Partially parsed test_ledger_constructor_with_different_balance. Retrieved 3/10 statements.
# Partially parsed test_ledger_constructor_zero_balance. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '123456'
    var_2 = 1000

def test_case_0():
    var_0 = 'Savings'
    var_1 = '654321'
    var_2 = 5000

def test_case_0():
    var_0 = 'Empty Account'
    var_1 = '000000'
    var_2 = 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/21 statements.
# Partially parsed test_read_initial_balances_call_with_empty_balances. Retrieved 4/21 statements.
# Partially parsed test_read_initial_balances_call_returns_correct_type. Retrieved 5/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 6
    var_3 = 30
    var_4 = 'balances'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 'Test entry'
    var_9 = []
    var_10 = 'Cash'
    var_11 = 'debit'
    var_12 = 500.0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Cash'
    var_4 = 'Test transaction'
    var_5 = []
    var_6 = 2023
    var_7 = 1
    var_8 = 100.0
    var_9 = 'USD'
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 500.0



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/7 statements.
# Partially parsed test_ledger_constructor_with_different_values. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '12345'
    var_2 = 1000

def test_case_0():
    var_0 = 'Savings'
    var_1 = '67890'
    var_2 = 5000



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 'Cash'
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 'Test entry'
    var_12 = 500.0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = 'Cash'
    var_4 = 100.0
    var_5 = 'USD'
    var_6 = 'Test Transaction'
    var_7 = []
    var_8 = 2024
    var_9 = 1
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 500.0



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/35 statements.


def test_case_0():
    var_0 = 'Test transaction'
    var_1 = []
    var_2 = 100.0
    var_3 = 'USD'
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 500.0



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/18 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 8/26 statements.
# Partially parsed test_build_general_ledger_with_postings_within_period. Retrieved 15/56 statements.
# Partially parsed test_build_general_ledger_with_postings_outside_period. Retrieved 12/40 statements.
# Partially parsed test_build_general_ledger_creates_new_ledger_for_new_account. Retrieved 15/49 statements.
# Partially parsed test_build_general_ledger_multiple_entries. Retrieved 14/48 statements.


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
    var_4 = []
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 1000

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Payable'
    var_8 = 1000
    var_9 = 500
    var_10 = 15
    var_11 = 'Test entry'
    var_12 = None
    var_13 = -100
    var_14 = 100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 1000
    var_7 = 2024
    var_8 = 15
    var_9 = 'Test entry'
    var_10 = None
    var_11 = -100

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '3000'
    var_7 = 'Revenue'
    var_8 = 1000
    var_9 = 15
    var_10 = 'Test entry'
    var_11 = None
    var_12 = -100
    var_13 = 100
    var_14 = 0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 1000
    var_7 = 15
    var_8 = 'Entry 1'
    var_9 = None
    var_10 = 100
    var_11 = 2
    var_12 = 'Entry 2'
    var_13 = -50



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_accounts. Retrieved 15/45 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 1000
    var_9 = 6
    var_10 = 15
    var_11 = 'Test entry'
    var_12 = 'source_obj'
    var_13 = -100
    var_14 = 100



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = 'General Ledger'
    var_13 = module_1.Ledger()
    var_14 = 1000.0



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_within_period. Retrieved 23/83 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2023
    var_5 = 1
    var_6 = 12
    var_7 = 31
    var_8 = 'test_source'
    var_9 = 6
    var_10 = 15
    var_11 = 'Inside period'
    var_12 = 2022
    var_13 = 'Before period'
    var_14 = 2024
    var_15 = 'After period'
    var_16 = 100
    var_17 = -100
    var_18 = 50
    var_19 = -50
    var_20 = 75
    var_21 = -75
    var_22 = 0



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/47 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = module_1.Ledger()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 17/63 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 9/33 statements.
# Partially parsed test_build_general_ledger_multiple_entries. Retrieved 22/74 statements.
# Partially parsed test_build_general_ledger_out_of_period. Retrieved 13/41 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Cash'
    var_5 = '1000'
    var_6 = 'Asset'
    var_7 = 'Revenue'
    var_8 = '4000'
    var_9 = '1000.00'
    var_10 = 'Test Transaction'
    var_11 = 15
    var_12 = 'Test posting'
    var_13 = '500.00'
    var_14 = '-500.00'
    var_15 = '1500.00'
    var_16 = '0.00'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Cash'
    var_5 = '1000'
    var_6 = 'Asset'
    var_7 = '500.00'
    var_8 = []

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Cash'
    var_5 = '1000'
    var_6 = 'Asset'
    var_7 = 'Revenue'
    var_8 = '4000'
    var_9 = '1000.00'
    var_10 = 15
    var_11 = 'Entry 1'
    var_12 = 'Source1'
    var_13 = '100.00'
    var_14 = '-100.00'
    var_15 = 2
    var_16 = 'Entry 2'
    var_17 = 'Source2'
    var_18 = '200.00'
    var_19 = '-200.00'
    var_20 = '1100.00'
    var_21 = '1300.00'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'Cash'
    var_5 = '1000'
    var_6 = 'Asset'
    var_7 = '1000.00'
    var_8 = 2025
    var_9 = 15
    var_10 = 'Out of period'
    var_11 = 'Source'
    var_12 = '100.00'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_build_general_ledger_predicate_false. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 2024
    var_5 = 15
    var_6 = 'Test entry'
    var_7 = 'source'
    var_8 = '1000'
    var_9 = 'Test Account'
    var_10 = 100
    var_11 = {}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 38/65 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Amount'
    var_5 = ()
    var_6 = 'value'
    var_7 = '100.00'
    var_8 = 'Posting'
    var_9 = ()
    var_10 = 'date'
    var_11 = 'amount'
    var_12 = 'is_debit'
    var_13 = 'is_credit'
    var_14 = 'direction'
    var_15 = 'account'
    var_16 = 'journal'
    var_17 = 2024
    var_18 = 1
    var_19 = 15
    var_20 = True
    var_21 = False
    var_22 = 'debit'
    var_23 = 'Journal'
    var_24 = ()
    var_25 = 'description'
    var_26 = 'postings'
    var_27 = 'Test transaction'
    var_28 = ()
    var_29 = ()
    var_30 = 'credit'
    var_31 = 'Ledger'
    var_32 = ()
    var_33 = {}
    var_34 = type(var_31, var_32, var_33)
    var_35 = 'Quantity'
    var_36 = ()
    var_37 = '1000.00'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2024
    var_6 = 1
    var_7 = 15
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'debit'
    var_11 = 500.0
    var_12 = module_1.Ledger()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/41 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Account'
    var_6 = 'Test Description'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 'Test Ledger'
    var_12 = 500.0



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Test Account'
    var_4 = 'Test Journal'
    var_5 = []
    var_6 = 2023
    var_7 = 1
    var_8 = 100.0
    var_9 = 'USD'
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_build_general_ledger_posting_account_in_ledgers. Retrieved 10/40 statements.


def test_case_0():
    var_0 = 'TestAccount'
    var_1 = '1000'
    var_2 = 2024
    var_3 = 1
    var_4 = 12
    var_5 = 31
    var_6 = '100'
    var_7 = 'Test entry'
    var_8 = 'test'
    var_9 = '50'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_accounts. Retrieved 14/46 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 12
    var_4 = 31
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = 'Accounts Payable'
    var_9 = 1000
    var_10 = 'Test transaction'
    var_11 = 'source_data'
    var_12 = -100
    var_13 = 100



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/17 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 11/38 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 16/57 statements.
# Partially parsed test_build_general_ledger_creates_ledger_for_new_account. Retrieved 13/35 statements.
# Partially parsed test_build_general_ledger_filters_by_period. Retrieved 15/48 statements.
# Partially parsed test_build_general_ledger_accumulates_balances. Retrieved 16/53 statements.


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
    var_4 = []
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = '2000'
    var_8 = 'Accounts Payable'
    var_9 = '1000.00'
    var_10 = '500.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 6
    var_9 = 15
    var_10 = 'Test transaction'
    var_11 = 'source'
    var_12 = '-100.00'
    var_13 = '100.00'
    var_14 = '1000.00'
    var_15 = '0.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '3000'
    var_5 = 'Revenue'
    var_6 = 6
    var_7 = 15
    var_8 = 'Test transaction'
    var_9 = 'source'
    var_10 = '500.00'
    var_11 = {}
    var_12 = '0'

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = 12
    var_4 = 31
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 15
    var_8 = 'In period'
    var_9 = 'source'
    var_10 = '100.00'
    var_11 = 5
    var_12 = 'Out of period'
    var_13 = '200.00'
    var_14 = '0.00'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 3
    var_7 = 'Transaction 1'
    var_8 = 'source'
    var_9 = '100.00'
    var_10 = 6
    var_11 = 'Transaction 2'
    var_12 = '50.00'
    var_13 = '1000.00'
    var_14 = '1100.00'
    var_15 = '1150.00'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 27/38 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'Journal'
    var_2 = ()
    var_3 = 'description'
    var_4 = 'postings'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = type(var_1, var_2, var_7)
    var_9 = 'Posting'
    var_10 = ()
    var_11 = 'date'
    var_12 = 'journal'
    var_13 = 'amount'
    var_14 = 'direction'
    var_15 = 'is_debit'
    var_16 = 'is_credit'
    var_17 = 'account'
    var_18 = 2024
    var_19 = 1
    var_20 = 15
    var_21 = '100.00'
    var_22 = 'debit'
    var_23 = True
    var_24 = False
    var_25 = module_0.object()
    var_26 = '500.00'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Cash'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'debit'
    var_11 = 1000.0



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()
    var_12 = 500.0



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/41 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Description'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 500.0



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 2023
    var_4 = 1
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test entry'
    var_8 = []
    var_9 = 'debit'
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()
    var_12 = 500.0



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/29 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Cash'
    var_6 = 1000.0
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_accounts. Retrieved 16/58 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Payable'
    var_8 = 1000
    var_9 = 6
    var_10 = 15
    var_11 = 'Test entry'
    var_12 = 'test'
    var_13 = -100
    var_14 = 100
    var_15 = 0



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Entry'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()
    var_11 = 500.0



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Test Account'
    var_4 = 'Test Journal'
    var_5 = []
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 2023
    var_9 = 1
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/46 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '100.00'
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 'Test Entry'
    var_7 = []
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 'Test Ledger'
    var_12 = '500.00'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/50 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test Transaction'
    var_8 = []
    var_9 = 2023
    var_10 = 1
    var_11 = 15
    var_12 = 'debit'
    var_13 = 500.0



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()
    var_11 = 500.0



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 16/46 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2024
    var_3 = 1
    var_4 = 15
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 'Cash'
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 'General Ledger'
    var_14 = module_1.Ledger()
    var_15 = 1000.0



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 'Test Journal'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/50 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2024
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()
    var_12 = 500.0



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/47 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 'Test entry'
    var_6 = []
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'Cash'
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Entry'
    var_3 = []
    var_4 = 2023
    var_5 = 1
    var_6 = 15
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'Test Account'
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 500.0



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 100.0
    var_9 = 'USD'
    var_10 = 'Cash'
    var_11 = 'debit'
    var_12 = 500.0



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 2023
    var_10 = 1
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 500.0



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/38 statements.


def test_case_0():
    var_0 = 'Test Ledger'
    var_1 = 'Test Journal Entry'
    var_2 = []
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 'debit'
    var_9 = True
    var_10 = False
    var_11 = 500.0



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Test Account'
    var_4 = 'Test Journal'
    var_5 = []
    var_6 = 2023
    var_7 = 1
    var_8 = 100.0
    var_9 = 'USD'
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Cash'
    var_4 = 100.0
    var_5 = 'USD'
    var_6 = 'Test transaction'
    var_7 = []
    var_8 = 2024
    var_9 = 1
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/53 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Test Account'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 2024
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = 'Test Journal'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/42 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '100.00'
    var_2 = 'USD'
    var_3 = 'Test Description'
    var_4 = []
    var_5 = 2024
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = '500.00'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_build_general_ledger_predicate_filters_postings_within_period. Retrieved 21/89 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 31
    var_7 = 1000
    var_8 = 15
    var_9 = 'Inside period'
    var_10 = 'source1'
    var_11 = 100
    var_12 = 2023
    var_13 = 12
    var_14 = 'Before period'
    var_15 = 'source2'
    var_16 = 50
    var_17 = 2
    var_18 = 'After period'
    var_19 = 'source3'
    var_20 = 75



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/29 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test Entry'
    var_8 = []
    var_9 = 2024
    var_10 = 1
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 500.0



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_build_general_ledger_creates_ledgers_for_all_accounts. Retrieved 14/54 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Accounts Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 12
    var_7 = 31
    var_8 = 1000
    var_9 = 'Test entry'
    var_10 = 'source'
    var_11 = -100
    var_12 = 100
    var_13 = 0



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/51 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2024
    var_8 = 1
    var_9 = 15
    var_10 = 'debit'
    var_11 = 'General Ledger'
    var_12 = module_1.Ledger()
    var_13 = 500.0



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/7 statements.
# Partially parsed test_ledger_constructor_with_different_values. Retrieved 3/12 statements.
# Partially parsed test_ledger_constructor_initializes_empty_entries. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '12345'
    var_2 = 1000

def test_case_0():
    var_0 = 'Savings'
    var_1 = '98765'
    var_2 = 5000

def test_case_0():
    var_0 = 'Checking'
    var_1 = '11111'
    var_2 = 2500
    var_3 = 'entries'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_build_general_ledger_predicate_evaluates_to_false. Retrieved 11/40 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = 'TestAccount'
    var_5 = '1000'
    var_6 = 100
    var_7 = 2023
    var_8 = 'Test Entry'
    var_9 = 'test_source'
    var_10 = 50



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/44 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 1000.0



# Parsed testcases at query #82
#--------------------------




import builtins as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_1.LedgerEntry(var_0, var_1, var_2)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2023
    var_4 = 1
    var_5 = 'Cash'
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 'Test'
    var_9 = []
    var_10 = 'debit'
    var_11 = True
    var_12 = False
    var_13 = 1000.0



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/19 statements.
# Partially parsed test_read_initial_balances_call_empty_balances. Retrieved 4/19 statements.
# Partially parsed test_read_initial_balances_call_with_different_period. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = 31



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/42 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 100.0
    var_3 = 'USD'
    var_4 = 'Cash'
    var_5 = 'debit'
    var_6 = True
    var_7 = False
    var_8 = 1000.0



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test transaction'
    var_8 = []
    var_9 = 2023
    var_10 = 1
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 500.0



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/44 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2024
    var_6 = 1
    var_7 = 'Test Description'
    var_8 = []
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = 500.0



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/44 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = 1000.0



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/39 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Journal'
    var_4 = []
    var_5 = 2024
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 1000.0



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Test transaction'
    var_4 = []
    var_5 = 2024
    var_6 = 1
    var_7 = 15
    var_8 = 100.0
    var_9 = 'USD'
    var_10 = 'Cash'
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 500.0



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 23/81 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/28 statements.
# Partially parsed test_build_general_ledger_outside_period. Retrieved 14/47 statements.
# Partially parsed test_build_general_ledger_creates_new_ledger_for_uninitialized_account. Retrieved 15/44 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = '1000.00'
    var_9 = 15
    var_10 = 'Initial transaction'
    var_11 = 'source1'
    var_12 = '-100.00'
    var_13 = '100.00'
    var_14 = 2
    var_15 = 20
    var_16 = 'Second transaction'
    var_17 = 'source2'
    var_18 = '50.00'
    var_19 = '-50.00'
    var_20 = '900.00'
    var_21 = '950.00'
    var_22 = '0.00'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '5000.00'
    var_7 = []

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = '1000.00'
    var_9 = 2023
    var_10 = 'Before period'
    var_11 = 'source1'
    var_12 = '-100.00'
    var_13 = '100.00'

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = {}
    var_9 = 15
    var_10 = 'Transaction'
    var_11 = 'source1'
    var_12 = '100.00'
    var_13 = '-100.00'
    var_14 = '0.00'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/41 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 100.0
    var_3 = 'USD'
    var_4 = 'Test Account'
    var_5 = 'debit'
    var_6 = True
    var_7 = False



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 4/8 statements.


import pypara.accounting.generic as module_0

def test_case_0():
    var_0 = 'Test Account'
    var_1 = 'ASSET'
    var_2 = 1000
    var_3 = module_0.Balance(var_2)



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 9/20 statements.


import builtins as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_0.object()
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = module_0.object()
    var_8 = module_1.LedgerEntry(var_0, var_3, var_7)



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 15/45 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 2024
    var_4 = 1
    var_5 = 15
    var_6 = 100.0
    var_7 = 'USD'
    var_8 = 'Cash'
    var_9 = 'Test transaction'
    var_10 = []
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 500.0



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Ledger'
    var_3 = module_1.Ledger()
    var_4 = 'Cash'
    var_5 = 100.0
    var_6 = 'USD'
    var_7 = 'Test Journal'
    var_8 = []
    var_9 = 2024
    var_10 = 1
    var_11 = 'debit'
    var_12 = True
    var_13 = False



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_read_initial_balances_call. Retrieved 4/21 statements.
# Partially parsed test_read_initial_balances_call_different_period. Retrieved 4/23 statements.
# Partially parsed test_read_initial_balances_call_empty_balances. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/47 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = module_1.Ledger()
    var_3 = 'Test Account'
    var_4 = 100.0
    var_5 = 'USD'
    var_6 = 'Test Journal'
    var_7 = []
    var_8 = 2023
    var_9 = 1
    var_10 = 'debit'



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/27 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/43 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Description'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_build_general_ledger_posting_account_not_in_ledgers. Retrieved 14/44 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Accounts Payable'
    var_8 = 6
    var_9 = 15
    var_10 = 'Test entry'
    var_11 = 'source_object'
    var_12 = 100
    var_13 = 1000



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 6/18 statements.
# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 8/28 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 15/48 statements.
# Partially parsed test_build_general_ledger_filters_by_period. Retrieved 14/43 statements.
# Partially parsed test_build_general_ledger_multiple_postings_same_account. Retrieved 16/49 statements.


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
    var_4 = []
    var_5 = '1000'
    var_6 = 'Cash'
    var_7 = 1000

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Payable'
    var_8 = 6
    var_9 = 15
    var_10 = 'Test entry'
    var_11 = 'source'
    var_12 = 500
    var_13 = -500
    var_14 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 1
    var_3 = 30
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 15
    var_7 = 'In period'
    var_8 = 'source'
    var_9 = 100
    var_10 = 7
    var_11 = 'Out of period'
    var_12 = 200
    var_13 = {}

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = 6
    var_7 = 15
    var_8 = 'Entry 1'
    var_9 = 'source'
    var_10 = 100
    var_11 = 20
    var_12 = 'Entry 2'
    var_13 = 50
    var_14 = {}
    var_15 = 150



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/43 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Entry'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = []
    var_6 = 'Test transaction'
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/41 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Entry'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 500.0



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/41 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Description'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 500.0



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/43 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0
    var_2 = 'USD'
    var_3 = 'Test Description'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 'debit'
    var_8 = True
    var_9 = False
    var_10 = 500.0
    var_11 = 'Test Ledger'



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_ledger_constructor. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = '123456'
    var_2 = 1000



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_build_general_ledger. Retrieved 25/87 statements.
# Partially parsed test_build_general_ledger_with_new_account. Retrieved 16/56 statements.
# Partially parsed test_build_general_ledger_filters_by_period. Retrieved 17/53 statements.
# Partially parsed test_build_general_ledger_empty_journal. Retrieved 8/28 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 12
    var_7 = 31
    var_8 = 1000
    var_9 = 500
    var_10 = 15
    var_11 = 'Test entry 1'
    var_12 = 'source1'
    var_13 = 100
    var_14 = -100
    var_15 = 2
    var_16 = 20
    var_17 = 'Test entry 2'
    var_18 = 'source2'
    var_19 = -50
    var_20 = 50
    var_21 = 1100
    var_22 = 1050
    var_23 = 400
    var_24 = 450

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '3000'
    var_3 = 'Revenue'
    var_4 = 2024
    var_5 = 1
    var_6 = 12
    var_7 = 31
    var_8 = 5000
    var_9 = 3
    var_10 = 10
    var_11 = 'Revenue entry'
    var_12 = 'source'
    var_13 = -200
    var_14 = 200
    var_15 = 0

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 2024
    var_3 = 1
    var_4 = 6
    var_5 = 30
    var_6 = 1000
    var_7 = 3
    var_8 = 15
    var_9 = 'Inside period'
    var_10 = 'source1'
    var_11 = 100
    var_12 = 9
    var_13 = 'Outside period'
    var_14 = 'source2'
    var_15 = 200
    var_16 = 1100

def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = 2024
    var_3 = 1
    var_4 = 12
    var_5 = 31
    var_6 = 5000
    var_7 = []



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/49 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 'Test transaction'
    var_4 = []
    var_5 = 2024
    var_6 = 1
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = 'General Ledger'
    var_11 = module_1.Ledger()
    var_12 = 500.0



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_general_ledger_program_call. Retrieved 6/30 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 47/69 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Account'
    var_3 = ()
    var_4 = {}
    var_5 = type(var_2, var_3, var_4)
    var_6 = 'Amount'
    var_7 = ()
    var_8 = '__repr__'
    var_9 = 'Amount(100)'
    var_10 = lambda self: var_9
    var_11 = {var_8: var_10}
    var_12 = type(var_6, var_7, var_11)
    var_13 = 'Quantity'
    var_14 = ()
    var_15 = 'Quantity(100)'
    var_16 = lambda self: var_15
    var_17 = {var_8: var_16}
    var_18 = type(var_13, var_14, var_17)
    var_19 = 'Posting'
    var_20 = ()
    var_21 = 'date'
    var_22 = 'amount'
    var_23 = 'is_debit'
    var_24 = 'is_credit'
    var_25 = 'direction'
    var_26 = 'account'
    var_27 = 'journal'
    var_28 = 2023
    var_29 = 1
    var_30 = 15
    var_31 = True
    var_32 = False
    var_33 = 'debit'
    var_34 = 'Journal'
    var_35 = ()
    var_36 = 'description'
    var_37 = 'postings'
    var_38 = 'Test transaction'
    var_39 = 'PostingItem'
    var_40 = ()
    var_41 = ()
    var_42 = 'credit'
    var_43 = 'Ledger'
    var_44 = ()
    var_45 = {}
    var_46 = type(var_43, var_44, var_45)



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 14/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Journal'
    var_3 = []
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = 'Test Account'
    var_11 = True
    var_12 = False
    var_13 = 500.0



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 20/27 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_0.object()
    var_4 = 'Posting'
    var_5 = ()
    var_6 = 'date'
    var_7 = 'amount'
    var_8 = 'journal'
    var_9 = 'account'
    var_10 = 'direction'
    var_11 = 'is_debit'
    var_12 = 'is_credit'
    var_13 = 2024
    var_14 = 1
    var_15 = 15
    var_16 = 'debit'
    var_17 = True
    var_18 = False
    var_19 = module_0.object()



# Parsed testcases at query #116
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/45 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 2023
    var_6 = 1
    var_7 = 15
    var_8 = 'Test transaction'
    var_9 = []
    var_10 = 'debit'
    var_11 = 500.0



# Parsed testcases at query #117
#--------------------------

# Partially parsed test_build_general_ledger_predicate_filters_postings_within_period. Retrieved 24/86 statements.


def test_case_0():
    var_0 = '1000'
    var_1 = 'Cash'
    var_2 = '2000'
    var_3 = 'Payable'
    var_4 = 2024
    var_5 = 1
    var_6 = 31
    var_7 = 15
    var_8 = 'Inside period'
    var_9 = 'source1'
    var_10 = 100
    var_11 = -100
    var_12 = 2023
    var_13 = 12
    var_14 = 'Before period'
    var_15 = 'source2'
    var_16 = 50
    var_17 = -50
    var_18 = 2
    var_19 = 'After period'
    var_20 = 'source3'
    var_21 = 75
    var_22 = -75
    var_23 = 0



# Parsed testcases at query #118
#--------------------------

# Partially parsed test_read_initial_balances_protocol_call. Retrieved 4/20 statements.
# Partially parsed test_read_initial_balances_protocol_call_different_period. Retrieved 5/22 statements.
# Partially parsed test_read_initial_balances_protocol_call_zero_balance. Retrieved 5/20 statements.


import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2024
    var_2 = 1
    var_3 = 31

import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2024
    var_2 = 2
    var_3 = 1
    var_4 = 29

import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = module_0.ReadInitialBalances()
    var_1 = 2024
    var_2 = 3
    var_3 = 1
    var_4 = 31



# Parsed testcases at query #119
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 39/51 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'Journal'
    var_5 = ()
    var_6 = 'description'
    var_7 = 'postings'
    var_8 = 'Test Description'
    var_9 = []
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = type(var_4, var_5, var_10)
    var_12 = 'Posting'
    var_13 = ()
    var_14 = 'date'
    var_15 = 'journal'
    var_16 = 'amount'
    var_17 = 'direction'
    var_18 = 'is_debit'
    var_19 = 'is_credit'
    var_20 = 'account'
    var_21 = 2024
    var_22 = 1
    var_23 = 15
    var_24 = 'Amount'
    var_25 = ()
    var_26 = {}
    var_27 = type(var_24, var_25, var_26)
    var_28 = 'debit'
    var_29 = True
    var_30 = False
    var_31 = 'Ledger'
    var_32 = ()
    var_33 = {}
    var_34 = type(var_31, var_32, var_33)
    var_35 = 'Quantity'
    var_36 = ()
    var_37 = {}
    var_38 = type(var_35, var_36, var_37)



# Parsed testcases at query #120
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 12/48 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 'Test Journal'
    var_4 = []
    var_5 = 2023
    var_6 = 1
    var_7 = 100.0
    var_8 = 'USD'
    var_9 = 'debit'
    var_10 = 'Test Ledger'
    var_11 = module_1.Ledger()



# Parsed testcases at query #121
#--------------------------

# Partially parsed test_build_general_ledger_posting_account_not_in_ledgers. Retrieved 13/37 statements.


def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 12
    var_3 = 31
    var_4 = '1000'
    var_5 = 'Cash'
    var_6 = '2000'
    var_7 = 'Revenue'
    var_8 = 1000
    var_9 = 'Test entry'
    var_10 = 'source_obj'
    var_11 = -100
    var_12 = 100



# Parsed testcases at query #122
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 8/41 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 100.0
    var_3 = 'USD'
    var_4 = 'Cash'
    var_5 = 'debit'
    var_6 = True
    var_7 = False



# Parsed testcases at query #123
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 11/44 statements.


import typing as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Cash'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test transaction'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = module_1.Ledger()



# Parsed testcases at query #124
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 13/44 statements.


import typing as module_0

def test_case_0():
    var_0 = '_T'
    var_1 = module_0.TypeVar(var_0)
    var_2 = 'Test Account'
    var_3 = 100.0
    var_4 = 'USD'
    var_5 = 'Test Journal'
    var_6 = []
    var_7 = 2023
    var_8 = 1
    var_9 = 'debit'
    var_10 = True
    var_11 = False
    var_12 = 500.0



# Parsed testcases at query #125
#--------------------------

# Partially parsed test_ledger_entry_constructor. Retrieved 44/65 statements.


def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = {}
    var_3 = type(var_0, var_1, var_2)
    var_4 = 'debit'
    var_5 = 'Amount'
    var_6 = ()
    var_7 = 'value'
    var_8 = 100
    var_9 = {var_7: var_8}
    var_10 = type(var_5, var_6, var_9)
    var_11 = 'Posting'
    var_12 = ()
    var_13 = 'date'
    var_14 = 'amount'
    var_15 = 'direction'
    var_16 = 'is_debit'
    var_17 = 'is_credit'
    var_18 = 'account'
    var_19 = 'journal'
    var_20 = 2024
    var_21 = 1
    var_22 = 15
    var_23 = True
    var_24 = False
    var_25 = 'Journal'
    var_26 = ()
    var_27 = 'description'
    var_28 = 'postings'
    var_29 = 'Test transaction'
    var_30 = 'PostingItem'
    var_31 = ()
    var_32 = 'debit'
    var_33 = ()
    var_34 = 'credit'
    var_35 = 'Ledger'
    var_36 = ()
    var_37 = {}
    var_38 = type(var_35, var_36, var_37)
    var_39 = 'Quantity'
    var_40 = ()
    var_41 = 500
    var_42 = {var_7: var_41}
    var_43 = type(var_39, var_40, var_42)



