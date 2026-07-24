####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generallgederprogram_call_returns_correct_type. Retrieved 6/21 statements.


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
    var_9 = [var_0, var_1, var_1]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_correct_value. Retrieved 6/20 statements.


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
    var_9 = 100.0



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #5
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_correct_balances. Retrieved 10/25 statements.


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
    var_9 = 'USD'
    var_10 = 'EUR'
    var_11 = 100.0
    var_12 = 85.0
    var_13 = {var_9: var_11, var_10: var_12}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_build_general_ledger_initializes_with_provided_balances. Retrieved 23/37 statements.
# Partially parsed test_build_general_ledger_ignores_entries_outside_period. Retrieved 32/56 statements.
# Partially parsed test_build_general_ledger_creates_new_ledgers_for_untracked_accounts. Retrieved 23/36 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3

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
    var_10 = 'A'
    var_11 = 'B'
    var_12 = '100.00'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = '50.00'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = 15
    var_21 = [var_0, var_1, var_20]
    var_22 = {}
    var_23 = module_0.date(*var_21, **var_22)
    var_24 = 'Test Entry'
    var_25 = 'TestSource'
    var_26 = module_3.JournalEntry(var_23, var_24, var_25)
    var_27 = [var_0, var_1, var_20]
    var_28 = {}
    var_29 = module_0.date(*var_27, **var_28)
    var_30 = '20.00'
    var_31 = [var_30]
    var_32 = {}
    var_33 = module_2.Decimal(*var_31, **var_32)
    var_34 = [var_26]
    var_35 = '120.00'
    var_36 = [var_35]
    var_37 = {}
    var_38 = module_2.Decimal(*var_36, **var_37)

import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3

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
    var_10 = 'A'
    var_11 = '0.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_2.Decimal(*var_12, **var_13)
    var_15 = 15
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_0.date(*var_16, **var_17)
    var_19 = 'Inside'
    var_20 = 'Src'
    var_21 = module_3.JournalEntry(var_18, var_19, var_20)
    var_22 = [var_0, var_1, var_15]
    var_23 = {}
    var_24 = module_0.date(*var_22, **var_23)
    var_25 = '10.00'
    var_26 = [var_25]
    var_27 = {}
    var_28 = module_2.Decimal(*var_26, **var_27)
    var_29 = 2
    var_30 = [var_0, var_29, var_1]
    var_31 = {}
    var_32 = module_0.date(*var_30, **var_31)
    var_33 = 'Outside'
    var_34 = module_3.JournalEntry(var_32, var_33, var_20)
    var_35 = [var_0, var_29, var_1]
    var_36 = {}
    var_37 = module_0.date(*var_35, **var_36)
    var_38 = [var_25]
    var_39 = {}
    var_40 = module_2.Decimal(*var_38, **var_39)
    var_41 = 2022
    var_42 = 12
    var_43 = [var_41, var_42, var_5]
    var_44 = {}
    var_45 = module_0.date(*var_43, **var_44)
    var_46 = 'Early'
    var_47 = module_3.JournalEntry(var_45, var_46, var_20)
    var_48 = [var_41, var_42, var_5]
    var_49 = {}
    var_50 = module_0.date(*var_48, **var_49)
    var_51 = [var_25]
    var_52 = {}
    var_53 = module_2.Decimal(*var_51, **var_52)
    var_54 = [var_21, var_34, var_47]
    var_55 = [var_25]
    var_56 = {}
    var_57 = module_2.Decimal(*var_55, **var_56)

import datetime as module_0
import pypara.commons.zeitgeist as module_1
import pypara.accounting.journaling as module_2
import decimal as module_3
import pypara.accounting.ledger as module_4

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
    var_10 = 'New'
    var_11 = {}
    var_12 = 15
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = 'New Account Entry'
    var_17 = 'Src'
    var_18 = module_2.JournalEntry(var_15, var_16, var_17)
    var_19 = [var_0, var_1, var_12]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = '50.00'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_3.Decimal(*var_23, **var_24)
    var_26 = [var_0, var_1, var_12]
    var_27 = {}
    var_28 = module_0.date(*var_26, **var_27)
    var_29 = [var_22]
    var_30 = {}
    var_31 = module_3.Decimal(*var_29, **var_30)
    var_32 = [var_18]
    var_33 = module_4.build_general_ledger(var_9, var_32, var_11)
    var_34 = '0.00'
    var_35 = [var_34]
    var_36 = {}
    var_37 = module_3.Decimal(*var_35, **var_36)
    var_38 = [var_22]
    var_39 = {}
    var_40 = module_3.Decimal(*var_38, **var_39)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_build_general_ledger_returns_correct_type. Retrieved 23/34 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1.DateRange(var_4, var_9)
    var_11 = 'A'
    var_12 = 'B'
    var_13 = '100.00'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_2.Decimal(*var_14, **var_15)
    var_17 = 6
    var_18 = [var_0, var_17, var_1]
    var_19 = {}
    var_20 = module_0.date(*var_18, **var_19)
    var_21 = 'Test Entry'
    var_22 = 'TestSource'
    var_23 = module_3.JournalEntry(var_20, var_21, var_22)
    var_24 = [var_0, var_17, var_1]
    var_25 = {}
    var_26 = module_0.date(*var_24, **var_25)
    var_27 = '50.00'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_2.Decimal(*var_28, **var_29)
    var_31 = [var_0, var_17, var_1]
    var_32 = {}
    var_33 = module_0.date(*var_31, **var_32)
    var_34 = '-50.00'
    var_35 = [var_34]
    var_36 = {}
    var_37 = module_2.Decimal(*var_35, **var_36)
    var_38 = [var_23]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_build_general_ledger_success. Retrieved 11/53 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2

def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Revenue'
    var_2 = 'Expense'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = 31
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = module_1.DateRange(var_7, var_11)
    var_13 = 1000
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_2.Decimal(*var_14, **var_15)



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #14
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_correct_balances. Retrieved 6/20 statements.


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
    var_9 = 100.0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_correct_value. Retrieved 6/20 statements.


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
    var_9 = 100.0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_build_general_ledger_returns_correct_type. Retrieved 11/54 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2

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
    var_10 = '100.00'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_2.Decimal(*var_11, **var_12)
    var_14 = '50.00'
    var_15 = 15
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_0.date(*var_16, **var_17)



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #19
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #20
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #21
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_general_ledger_program_call_returns_expected_type. Retrieved 6/21 statements.


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
    var_9 = [var_0, var_1, var_1]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #24
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #25
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #26
#--------------------------




import pypara.accounting.ledger as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.LedgerEntry(var_0, var_1, var_2)
    var_4 = var_3.ledger
    var_5 = bool(var_3.ledger == var_0)
    assert var_5 is True
    var_6 = var_3.posting
    var_7 = bool(var_3.posting == var_1)
    assert var_7 is True
    var_8 = var_3.balance
    var_9 = bool(var_3.balance == var_2)
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_generalledgerprogram_call_returns_correct_type_and_period. Retrieved 9/24 statements.


import typing as module_0
import datetime as module_1
import pypara.commons.zeitgeist as module_2

def test_case_0():
    var_0 = '_T'
    var_1 = []
    var_2 = module_0.TypeVar(var_0, *var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_1.date(*var_5, **var_6)
    var_8 = 31
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_1.date(*var_9, **var_10)
    var_12 = module_2.DateRange(var_7, var_11)
    var_13 = [var_3, var_4, var_4]
    var_14 = {}
    var_15 = module_1.date(*var_13, **var_14)



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #32
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_outside_period. Retrieved 28/48 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3

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
    var_10 = 'InPeriod'
    var_11 = 'OutPeriod'
    var_12 = '100.00'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = 15
    var_17 = [var_0, var_1, var_16]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = 'Valid'
    var_21 = 'Src'
    var_22 = module_3.JournalEntry(var_19, var_20, var_21)
    var_23 = [var_0, var_1, var_16]
    var_24 = {}
    var_25 = module_0.date(*var_23, **var_24)
    var_26 = 2022
    var_27 = 12
    var_28 = [var_26, var_27, var_5]
    var_29 = {}
    var_30 = module_0.date(*var_28, **var_29)
    var_31 = 'Before'
    var_32 = module_3.JournalEntry(var_30, var_31, var_21)
    var_33 = [var_26, var_27, var_5]
    var_34 = {}
    var_35 = module_0.date(*var_33, **var_34)
    var_36 = 2
    var_37 = [var_0, var_36, var_1]
    var_38 = {}
    var_39 = module_0.date(*var_37, **var_38)
    var_40 = 'After'
    var_41 = module_3.JournalEntry(var_39, var_40, var_21)
    var_42 = [var_0, var_36, var_1]
    var_43 = {}
    var_44 = module_0.date(*var_42, **var_43)
    var_45 = [var_22, var_32, var_41]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_period. Retrieved 32/92 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2

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
    var_10 = 'In Period'
    var_11 = 'Out of Period'
    var_12 = 15
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = 'Valid'
    var_17 = None
    var_18 = [var_0, var_1, var_12]
    var_19 = {}
    var_20 = module_0.date(*var_18, **var_19)
    var_21 = [var_1]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = 100
    var_25 = [var_24]
    var_26 = {}
    var_27 = module_2.Decimal(*var_25, **var_26)
    var_28 = 2022
    var_29 = 12
    var_30 = [var_28, var_29, var_5]
    var_31 = {}
    var_32 = module_0.date(*var_30, **var_31)
    var_33 = 'Too Early'
    var_34 = [var_28, var_29, var_5]
    var_35 = {}
    var_36 = module_0.date(*var_34, **var_35)
    var_37 = [var_1]
    var_38 = {}
    var_39 = module_2.Decimal(*var_37, **var_38)
    var_40 = 50
    var_41 = [var_40]
    var_42 = {}
    var_43 = module_2.Decimal(*var_41, **var_42)
    var_44 = 2
    var_45 = [var_0, var_44, var_1]
    var_46 = {}
    var_47 = module_0.date(*var_45, **var_46)
    var_48 = 'Too Late'
    var_49 = [var_0, var_44, var_1]
    var_50 = {}
    var_51 = module_0.date(*var_49, **var_50)
    var_52 = [var_1]
    var_53 = {}
    var_54 = module_2.Decimal(*var_52, **var_53)
    var_55 = [var_40]
    var_56 = {}
    var_57 = module_2.Decimal(*var_55, **var_56)
    var_58 = {}
    var_59 = [var_24]
    var_60 = {}
    var_61 = module_2.Decimal(*var_59, **var_60)



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #38
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #40
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #41
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #42
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #43
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #44
#--------------------------

# Partially parsed test_build_general_ledger_predicate_filtering. Retrieved 30/50 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3

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
    var_10 = 'in_period'
    var_11 = 'out_period'
    var_12 = '100.00'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = 15
    var_17 = [var_0, var_1, var_16]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = 'Valid Entry'
    var_21 = 'SourceA'
    var_22 = module_3.JournalEntry(var_19, var_20, var_21)
    var_23 = [var_0, var_1, var_16]
    var_24 = {}
    var_25 = module_0.date(*var_23, **var_24)
    var_26 = 2022
    var_27 = 12
    var_28 = [var_26, var_27, var_5]
    var_29 = {}
    var_30 = module_0.date(*var_28, **var_29)
    var_31 = 'Old Entry'
    var_32 = 'SourceB'
    var_33 = module_3.JournalEntry(var_30, var_31, var_32)
    var_34 = [var_26, var_27, var_5]
    var_35 = {}
    var_36 = module_0.date(*var_34, **var_35)
    var_37 = 2
    var_38 = [var_0, var_37, var_1]
    var_39 = {}
    var_40 = module_0.date(*var_38, **var_39)
    var_41 = 'Future Entry'
    var_42 = 'SourceC'
    var_43 = module_3.JournalEntry(var_40, var_41, var_42)
    var_44 = [var_0, var_37, var_1]
    var_45 = {}
    var_46 = module_0.date(*var_44, **var_45)
    var_47 = [var_22, var_33, var_43]



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_ledger_entry_constructor_initialization. Retrieved 12/50 statements.


import typing as module_0
import datetime as module_1

def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = module_0.TypeVar(var_0, *var_1)
    var_3 = 100.0
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = {}
    var_8 = module_1.date(*var_6, **var_7)
    var_9 = 'Cash'
    var_10 = 'Test Journal'
    var_11 = 'debit'
    var_12 = True
    var_13 = False
    var_14 = 50.0



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #3
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_build_general_ledger_with_initial_balances. Retrieved 12/25 statements.
# Partially parsed test_build_general_ledger_processes_postings_within_period. Retrieved 10/22 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import pypara.accounting.ledger as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1.DateRange(var_4, var_9)
    var_11 = {}
    var_12 = []
    var_13 = module_2.build_general_ledger(var_10, var_12, var_11)
    var_14 = var_13.period
    var_15 = bool(var_13.period == var_10)
    assert var_15 is True
    var_16 = var_13.ledgers
    var_17 = bool(var_13.ledgers == {})
    assert var_17 is True

import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1.DateRange(var_4, var_9)
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = '100.00'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = []
    var_19 = [var_14]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)

import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1.DateRange(var_4, var_9)
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = '0.00'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #6
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generalladger_program_call_returns_correct_type. Retrieved 6/21 statements.


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
    var_9 = [var_0, var_1, var_1]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #10
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #11
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #12
#--------------------------

# Partially parsed test_build_general_ledger_does_not_fail_on_empty_inputs. Retrieved 10/50 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import pypara.accounting.ledger as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1.DateRange(var_4, var_9)
    var_11 = []
    var_12 = {}
    var_13 = module_2.build_general_ledger(var_10, var_11, var_12)
    var_14 = var_13.period
    var_15 = bool(var_13.period == var_10)
    assert var_15 is True
    var_16 = var_13.ledgers
    var_17 = bool(var_13.ledgers == {})
    assert var_17 is True



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_ledger_entry_constructor_initialization. Retrieved 12/42 statements.


import typing as module_0
import datetime as module_1

def test_case_0():
    var_0 = '_T'
    var_1 = []
    var_2 = module_0.TypeVar(var_0, *var_1)
    var_3 = 'Test Account'
    var_4 = 100.0
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_1.date(*var_7, **var_8)
    var_10 = 'Test Journal'
    var_11 = []
    var_12 = 'debit'
    var_13 = True
    var_14 = False



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_build_general_ledger_evaluates_predicate_to_false. Retrieved 25/44 statements.


import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

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
    var_9 = 'Account A'
    var_10 = 'Account B'
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = '100.00'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = 15
    var_19 = [var_0, var_1, var_18]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = None
    var_23 = '50.00'
    var_24 = [var_23]
    var_25 = {}
    var_26 = module_1.Decimal(*var_24, **var_25)
    var_27 = '20.00'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_1.Decimal(*var_28, **var_29)
    var_31 = 'Test Entry'
    var_32 = 'Source'
    var_33 = module_2.JournalEntry(var_21, var_31, var_32)
    var_34 = [var_33]
    var_35 = '150.00'
    var_36 = [var_35]
    var_37 = {}
    var_38 = module_1.Decimal(*var_36, **var_37)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_build_general_ledger_success. Retrieved 51/82 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1.DateRange(var_4, var_9)
    var_11 = 'Cash'
    var_12 = 'Revenue'
    var_13 = 'Expense'
    var_14 = '1000.00'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = 6
    var_19 = [var_0, var_18, var_1]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = 'Service Revenue'
    var_23 = 'Client A'
    var_24 = module_3.JournalEntry(var_21, var_22, var_23)
    var_25 = [var_0, var_18, var_1]
    var_26 = {}
    var_27 = module_0.date(*var_25, **var_26)
    var_28 = '500.00'
    var_29 = [var_28]
    var_30 = {}
    var_31 = module_2.Decimal(*var_29, **var_30)
    var_32 = int(var_0, var_18, var_1)
    var_33 = '-500.00'
    var_34 = [var_33]
    var_35 = {}
    var_36 = module_2.Decimal(*var_34, **var_35)
    var_37 = 7
    var_38 = [var_0, var_37, var_1]
    var_39 = {}
    var_40 = module_0.date(*var_38, **var_39)
    var_41 = 'Office Supplies'
    var_42 = 'Supplier B'
    var_43 = module_3.JournalEntry(var_40, var_41, var_42)
    var_44 = [var_0, var_37, var_1]
    var_45 = {}
    var_46 = module_0.date(*var_44, **var_45)
    var_47 = '-100.00'
    var_48 = [var_47]
    var_49 = {}
    var_50 = module_2.Decimal(*var_48, **var_49)
    var_51 = [var_0, var_37, var_1]
    var_52 = {}
    var_53 = module_0.date(*var_51, **var_52)
    var_54 = '100.00'
    var_55 = [var_54]
    var_56 = {}
    var_57 = module_2.Decimal(*var_55, **var_56)
    var_58 = 2022
    var_59 = [var_58, var_5, var_6]
    var_60 = {}
    var_61 = module_0.date(*var_59, **var_60)
    var_62 = 'Old Entry'
    var_63 = 'Legacy'
    var_64 = module_3.JournalEntry(var_61, var_62, var_63)
    var_65 = [var_58, var_5, var_6]
    var_66 = {}
    var_67 = module_0.date(*var_65, **var_66)
    var_68 = [var_54]
    var_69 = {}
    var_70 = module_2.Decimal(*var_68, **var_69)
    var_71 = [var_24, var_43, var_64]
    var_72 = [var_14]
    var_73 = {}
    var_74 = module_2.Decimal(*var_72, **var_73)
    var_75 = '1400.00'
    var_76 = [var_75]
    var_77 = {}
    var_78 = module_2.Decimal(*var_76, **var_77)
    var_79 = [var_33]
    var_80 = {}
    var_81 = module_2.Decimal(*var_79, **var_80)
    var_82 = [var_54]
    var_83 = {}
    var_84 = module_2.Decimal(*var_82, **var_83)
    var_85 = 2022
    var_86 = 12
    var_87 = 31
    var_88 = [var_85, var_86, var_87]
    var_89 = {}
    var_90 = module_0.date(*var_88, **var_89)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_build_general_ledger_returns_correct_type_and_period. Retrieved 17/45 statements.


import datetime as module_0
import decimal as module_1

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
    var_9 = 'Cash'
    var_10 = '100.00'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)
    var_14 = 15
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = 'Test'
    var_19 = [var_0, var_1, var_14]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = '50.00'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_1.Decimal(*var_23, **var_24)
    var_26 = [var_10]
    var_27 = {}
    var_28 = module_1.Decimal(*var_26, **var_27)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generalladger_program_call_returns_correct_type. Retrieved 6/21 statements.


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
    var_9 = [var_0, var_1, var_1]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_correct_balances. Retrieved 5/19 statements.


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



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #28
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #29
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #30
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #31
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #32
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_correct_type_and_value. Retrieved 5/19 statements.


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



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #34
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_outside_period. Retrieved 44/88 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3

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
    var_10 = 'InPeriod'
    var_11 = 'OutPeriod'
    var_12 = 'Existing'
    var_13 = [var_0, var_1, var_1]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = '100.00'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = 15
    var_21 = [var_0, var_1, var_20]
    var_22 = {}
    var_23 = module_0.date(*var_21, **var_22)
    var_24 = 'Valid'
    var_25 = 'Source'
    var_26 = module_3.JournalEntry(var_23, var_24, var_25)
    var_27 = [var_0, var_1, var_20]
    var_28 = {}
    var_29 = module_0.date(*var_27, **var_28)
    var_30 = '50.00'
    var_31 = [var_30]
    var_32 = {}
    var_33 = module_2.Decimal(*var_31, **var_32)
    var_34 = 2022
    var_35 = 12
    var_36 = [var_34, var_35, var_5]
    var_37 = {}
    var_38 = module_0.date(*var_36, **var_37)
    var_39 = 'Too Early'
    var_40 = module_3.JournalEntry(var_38, var_39, var_25)
    var_41 = [var_34, var_35, var_5]
    var_42 = {}
    var_43 = module_0.date(*var_41, **var_42)
    var_44 = [var_30]
    var_45 = {}
    var_46 = module_2.Decimal(*var_44, **var_45)
    var_47 = 2
    var_48 = 'Too Late'
    var_49 = [var_0, var_47, var_1]
    var_50 = {}
    var_51 = module_0.date(*var_49, **var_50)
    var_52 = [var_30]
    var_53 = {}
    var_54 = module_2.Decimal(*var_52, **var_53)
    var_55 = [var_0, var_1, var_1]
    var_56 = {}
    var_57 = module_0.date(*var_55, **var_56)
    var_58 = 'Boundary Start'
    var_59 = module_3.JournalEntry(var_57, var_58, var_25)
    var_60 = [var_0, var_1, var_1]
    var_61 = {}
    var_62 = module_0.date(*var_60, **var_61)
    var_63 = '10.00'
    var_64 = [var_63]
    var_65 = {}
    var_66 = module_2.Decimal(*var_64, **var_65)
    var_67 = [var_0, var_1, var_5]
    var_68 = {}
    var_69 = module_0.date(*var_67, **var_68)
    var_70 = 'Boundary End'
    var_71 = module_3.JournalEntry(var_69, var_70, var_25)
    var_72 = [var_0, var_1, var_5]
    var_73 = {}
    var_74 = module_0.date(*var_72, **var_73)
    var_75 = [var_63]
    var_76 = {}
    var_77 = module_2.Decimal(*var_75, **var_76)
    var_78 = '70.00'
    var_79 = [var_78]
    var_80 = {}
    var_81 = module_2.Decimal(*var_79, **var_80)



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #37
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #38
#--------------------------

# Partially parsed test_build_general_ledger_predicate_true. Retrieved 13/22 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import pypara.accounting.journaling as module_2

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
    var_10 = 'A'
    var_11 = 'B'
    var_12 = 15
    var_13 = [var_0, var_1, var_12]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = 'Inside'
    var_17 = 'src'
    var_18 = module_2.JournalEntry(var_15, var_16, var_17)



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




