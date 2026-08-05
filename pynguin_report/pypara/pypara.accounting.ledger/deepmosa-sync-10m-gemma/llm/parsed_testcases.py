####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #2
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_account'
    var_1 = 100.0



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_build_general_ledger_empty. Retrieved 20/26 statements.
# Partially parsed test_build_general_ledger_with_initial_and_postings. Retrieved 37/70 statements.


import datetime as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = 'DateRange'
    var_1 = ()
    var_2 = 'since'
    var_3 = 'until'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 12
    var_10 = 31
    var_11 = [var_4, var_9, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = {var_2: var_8, var_3: var_13}
    var_15 = type(var_0, var_1, var_14)
    var_16 = var_15()
    var_17 = []
    var_18 = {}
    var_19 = 'Test Account'
    var_20 = module_1.build_general_ledger(var_16, var_17, var_18)
    var_21 = [var_4, var_5, var_5]
    var_22 = {}
    var_23 = module_0.date(*var_21, **var_22)
    var_24 = var_20.period.since
    var_25 = bool(var_20.period.since == var_23)
    assert var_25 is True
    var_26 = var_20.ledgers
    var_27 = len(var_26)
    assert var_27 == 0

import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

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
    var_10 = 'DateRange'
    var_11 = ()
    var_12 = 'since'
    var_13 = 'until'
    var_14 = {var_12: var_4, var_13: var_9}
    var_15 = type(var_10, var_11, var_14)
    var_16 = var_15()
    var_17 = 'Cash'
    var_18 = 'Revenue'
    var_19 = '100.00'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)
    var_23 = 6
    var_24 = [var_0, var_23, var_1]
    var_25 = {}
    var_26 = module_0.date(*var_24, **var_25)
    var_27 = 'Sale'
    var_28 = 'None'
    var_29 = module_2.JournalEntry(var_26, var_27, var_28)
    var_30 = '50.00'
    var_31 = [var_30]
    var_32 = {}
    var_33 = module_1.Decimal(*var_31, **var_32)
    var_34 = [var_30]
    var_35 = {}
    var_36 = module_1.Decimal(*var_34, **var_35)
    var_37 = 2024
    var_38 = [var_37, var_1, var_1]
    var_39 = {}
    var_40 = module_0.date(*var_38, **var_39)
    var_41 = 'Old Sale'
    var_42 = module_2.JournalEntry(var_40, var_41, var_28)
    var_43 = '10.00'
    var_44 = [var_43]
    var_45 = {}
    var_46 = module_1.Decimal(*var_44, **var_45)
    var_47 = [var_29, var_42]
    var_48 = [var_19]
    var_49 = {}
    var_50 = module_1.Decimal(*var_48, **var_49)
    var_51 = [var_30]
    var_52 = {}
    var_53 = module_1.Decimal(*var_51, **var_52)
    var_54 = '0.00'
    var_55 = [var_54]
    var_56 = {}
    var_57 = module_1.Decimal(*var_55, **var_56)
    var_58 = [var_30]
    var_59 = {}
    var_60 = module_1.Decimal(*var_58, **var_59)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_outside_period. Retrieved 32/51 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2
import pypara.accounting.journaling as module_3
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
    var_10 = 'IN'
    var_11 = 'OUT'
    var_12 = '100.00'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = 15
    var_17 = [var_0, var_1, var_16]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = 'Inside'
    var_21 = 'Src1'
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
    var_32 = 'Src2'
    var_33 = module_3.JournalEntry(var_30, var_31, var_32)
    var_34 = [var_26, var_27, var_5]
    var_35 = {}
    var_36 = module_0.date(*var_34, **var_35)
    var_37 = 2
    var_38 = [var_0, var_37, var_1]
    var_39 = {}
    var_40 = module_0.date(*var_38, **var_39)
    var_41 = 'After'
    var_42 = 'Src3'
    var_43 = module_3.JournalEntry(var_40, var_41, var_42)
    var_44 = [var_0, var_37, var_1]
    var_45 = {}
    var_46 = module_0.date(*var_44, **var_45)
    var_47 = [var_22, var_33, var_43]
    var_48 = {}
    var_49 = module_4.build_general_ledger(var_9, var_47, var_48)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #13
#--------------------------

# Failed to parse test_ledger_entry_constructor_initializes_correctly.




# Parsed testcases at query #14
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_build_general_ledger_calculates_correctly. Retrieved 34/96 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import decimal as module_2

def test_case_0():
    var_0 = 'Cash'
    var_1 = 'Revenue'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = 31
    var_8 = [var_2, var_3, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = module_1.DateRange(var_6, var_10)
    var_12 = [var_2, var_3, var_3]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = '100.00'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = None
    var_20 = 15
    var_21 = [var_2, var_3, var_20]
    var_22 = {}
    var_23 = module_0.date(*var_21, **var_22)
    var_24 = '1'
    var_25 = [var_24]
    var_26 = {}
    var_27 = module_2.Decimal(*var_25, **var_26)
    var_28 = '50.00'
    var_29 = [var_28]
    var_30 = {}
    var_31 = module_2.Decimal(*var_29, **var_30)
    var_32 = [var_2, var_3, var_20]
    var_33 = {}
    var_34 = module_0.date(*var_32, **var_33)
    var_35 = [var_24]
    var_36 = {}
    var_37 = module_2.Decimal(*var_35, **var_36)
    var_38 = [var_28]
    var_39 = {}
    var_40 = module_2.Decimal(*var_38, **var_39)
    var_41 = [var_2, var_3, var_20]
    var_42 = {}
    var_43 = module_0.date(*var_41, **var_42)
    var_44 = 'Sale'
    var_45 = 2
    var_46 = [var_2, var_45, var_3]
    var_47 = {}
    var_48 = module_0.date(*var_46, **var_47)
    var_49 = '-1'
    var_50 = [var_49]
    var_51 = {}
    var_52 = module_2.Decimal(*var_50, **var_51)
    var_53 = '20.00'
    var_54 = [var_53]
    var_55 = {}
    var_56 = module_2.Decimal(*var_54, **var_55)
    var_57 = [var_2, var_45, var_3]
    var_58 = {}
    var_59 = module_0.date(*var_57, **var_58)
    var_60 = 'Expense'
    var_61 = '150.00'
    var_62 = [var_61]
    var_63 = {}
    var_64 = module_2.Decimal(*var_62, **var_63)
    var_65 = [var_28]
    var_66 = {}
    var_67 = module_2.Decimal(*var_65, **var_66)



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_ledger_entry_constructor.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_by_date_range. Retrieved 30/116 statements.


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
    var_10 = 'In Range'
    var_11 = 'Too Early'
    var_12 = 'Too Late'
    var_13 = None
    var_14 = 15
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = '1'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = '10'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_2.Decimal(*var_23, **var_24)
    var_26 = 2022
    var_27 = 12
    var_28 = [var_26, var_27, var_5]
    var_29 = {}
    var_30 = module_0.date(*var_28, **var_29)
    var_31 = [var_18]
    var_32 = {}
    var_33 = module_2.Decimal(*var_31, **var_32)
    var_34 = '5'
    var_35 = [var_34]
    var_36 = {}
    var_37 = module_2.Decimal(*var_35, **var_36)
    var_38 = 2
    var_39 = [var_0, var_38, var_1]
    var_40 = {}
    var_41 = module_0.date(*var_39, **var_40)
    var_42 = [var_18]
    var_43 = {}
    var_44 = module_2.Decimal(*var_42, **var_43)
    var_45 = [var_34]
    var_46 = {}
    var_47 = module_2.Decimal(*var_45, **var_46)
    var_48 = [var_0, var_1, var_14]
    var_49 = {}
    var_50 = module_0.date(*var_48, **var_49)
    var_51 = [var_26, var_27, var_5]
    var_52 = {}
    var_53 = module_0.date(*var_51, **var_52)
    var_54 = [var_0, var_38, var_1]
    var_55 = {}
    var_56 = module_0.date(*var_54, **var_55)
    var_57 = {}



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_outside_period. Retrieved 31/119 statements.


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
    var_10 = 'A'
    var_11 = 'B'
    var_12 = 'C'
    var_13 = None
    var_14 = 15
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = [var_1]
    var_19 = {}
    var_20 = module_2.Decimal(*var_18, **var_19)
    var_21 = 100
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_2.Decimal(*var_22, **var_23)
    var_25 = 2022
    var_26 = 12
    var_27 = [var_25, var_26, var_5]
    var_28 = {}
    var_29 = module_0.date(*var_27, **var_28)
    var_30 = [var_1]
    var_31 = {}
    var_32 = module_2.Decimal(*var_30, **var_31)
    var_33 = 50
    var_34 = [var_33]
    var_35 = {}
    var_36 = module_2.Decimal(*var_34, **var_35)
    var_37 = 2
    var_38 = [var_0, var_37, var_1]
    var_39 = {}
    var_40 = module_0.date(*var_38, **var_39)
    var_41 = [var_1]
    var_42 = {}
    var_43 = module_2.Decimal(*var_41, **var_42)
    var_44 = 200
    var_45 = [var_44]
    var_46 = {}
    var_47 = module_2.Decimal(*var_45, **var_46)
    var_48 = [var_0, var_1, var_14]
    var_49 = {}
    var_50 = module_0.date(*var_48, **var_49)
    var_51 = [var_25, var_26, var_5]
    var_52 = {}
    var_53 = module_0.date(*var_51, **var_52)
    var_54 = [var_0, var_37, var_1]
    var_55 = {}
    var_56 = module_0.date(*var_54, **var_55)
    var_57 = {}
    var_58 = [var_21]
    var_59 = {}
    var_60 = module_2.Decimal(*var_58, **var_59)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generallledgerprogram_call_returns_correct_type. Retrieved 7/22 statements.


import typing as module_0
import datetime as module_1

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



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_ledger_entry_constructor_initializes_correctly.




# Parsed testcases at query #24
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #25
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #26
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_build_general_ledger_filters_postings_outside_period. Retrieved 11/45 statements.


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
    var_9 = 'Balance'
    var_10 = ()
    var_11 = 'value'
    var_12 = 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_account'
    var_1 = 100.0



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 16/23 statements.
# Partially parsed test_build_general_ledger_with_initial_and_postings. Retrieved 49/86 statements.


import datetime as module_0
import pypara.accounting.ledger as module_1

def test_case_0():
    var_0 = 'DateRange'
    var_1 = ()
    var_2 = 'since'
    var_3 = 'until'
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 12
    var_10 = 31
    var_11 = [var_4, var_9, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = {var_2: var_8, var_3: var_13}
    var_15 = type(var_0, var_1, var_14)
    var_16 = var_15()
    var_17 = {}
    var_18 = []
    var_19 = module_1.build_general_ledger(var_16, var_18, var_17)
    var_20 = var_19.ledgers
    var_21 = bool(var_19.ledgers == {})
    assert var_21 is True

import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

def test_case_0():
    var_0 = 'Account'
    var_1 = ()
    var_2 = 'id'
    var_3 = 'A1'
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = var_5()
    var_7 = ()
    var_8 = 'A2'
    var_9 = {var_2: var_8}
    var_10 = type(var_0, var_7, var_9)
    var_11 = var_10()
    var_12 = 2023
    var_13 = 1
    var_14 = [var_12, var_13, var_13]
    var_15 = {}
    var_16 = module_0.date(*var_14, **var_15)
    var_17 = 12
    var_18 = 31
    var_19 = [var_12, var_17, var_18]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = 'DateRange'
    var_23 = ()
    var_24 = 'since'
    var_25 = 'until'
    var_26 = {var_24: var_16, var_25: var_21}
    var_27 = type(var_22, var_23, var_26)
    var_28 = var_27()
    var_29 = '100.00'
    var_30 = [var_29]
    var_31 = {}
    var_32 = module_1.Decimal(*var_30, **var_31)
    var_33 = 6
    var_34 = [var_12, var_33, var_13]
    var_35 = {}
    var_36 = module_0.date(*var_34, **var_35)
    var_37 = 'Test 1'
    var_38 = 'Src'
    var_39 = module_2.JournalEntry(var_36, var_37, var_38)
    var_40 = '50.00'
    var_41 = [var_40]
    var_42 = {}
    var_43 = module_1.Decimal(*var_41, **var_42)
    var_44 = [var_40]
    var_45 = {}
    var_46 = module_1.Decimal(*var_44, **var_45)
    var_47 = 2022
    var_48 = [var_47, var_17, var_18]
    var_49 = {}
    var_50 = module_0.date(*var_48, **var_49)
    var_51 = 'Test 2'
    var_52 = module_2.JournalEntry(var_50, var_51, var_38)
    var_53 = '10.00'
    var_54 = [var_53]
    var_55 = {}
    var_56 = module_1.Decimal(*var_54, **var_55)
    var_57 = 2024
    var_58 = [var_57, var_13, var_13]
    var_59 = {}
    var_60 = module_0.date(*var_58, **var_59)
    var_61 = 'Test 3'
    var_62 = module_2.JournalEntry(var_60, var_61, var_38)
    var_63 = [var_53]
    var_64 = {}
    var_65 = module_1.Decimal(*var_63, **var_64)
    var_66 = [var_39, var_52, var_62]
    var_67 = [var_40]
    var_68 = {}
    var_69 = module_1.Decimal(*var_67, **var_68)
    var_70 = [var_40]
    var_71 = {}
    var_72 = module_1.Decimal(*var_70, **var_71)



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #32
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #33
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generallledgerprogram_call_returns_correct_type. Retrieved 6/21 statements.


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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_build_general_ledger_populates_ledgers_correctly. Retrieved 49/91 statements.


import datetime as module_0
import decimal as module_1
import pypara.accounting.journaling as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2023
    var_7 = 12
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = 'Account A'
    var_13 = 'Account B'
    var_14 = 'Account C'
    var_15 = 2023
    var_16 = 1
    var_17 = [var_15, var_16, var_16]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = '100.00'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_1.Decimal(*var_21, **var_22)
    var_24 = [var_15, var_16, var_16]
    var_25 = {}
    var_26 = module_0.date(*var_24, **var_25)
    var_27 = '50.00'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_1.Decimal(*var_28, **var_29)
    var_31 = 6
    var_32 = [var_15, var_31, var_16]
    var_33 = {}
    var_34 = module_0.date(*var_32, **var_33)
    var_35 = 'Entry 1'
    var_36 = 'Source 1'
    var_37 = module_2.JournalEntry(var_34, var_35, var_36)
    var_38 = [var_15, var_31, var_16]
    var_39 = {}
    var_40 = module_0.date(*var_38, **var_39)
    var_41 = '20.00'
    var_42 = [var_41]
    var_43 = {}
    var_44 = module_1.Decimal(*var_42, **var_43)
    var_45 = [var_15, var_31, var_16]
    var_46 = {}
    var_47 = module_0.date(*var_45, **var_46)
    var_48 = '10.00'
    var_49 = [var_48]
    var_50 = {}
    var_51 = module_1.Decimal(*var_49, **var_50)
    var_52 = 2024
    var_53 = [var_52, var_16, var_16]
    var_54 = {}
    var_55 = module_0.date(*var_53, **var_54)
    var_56 = 'Entry 2'
    var_57 = 'Source 2'
    var_58 = module_2.JournalEntry(var_55, var_56, var_57)
    var_59 = [var_52, var_16, var_16]
    var_60 = {}
    var_61 = module_0.date(*var_59, **var_60)
    var_62 = [var_27]
    var_63 = {}
    var_64 = module_1.Decimal(*var_62, **var_63)
    var_65 = 8
    var_66 = [var_15, var_65, var_16]
    var_67 = {}
    var_68 = module_0.date(*var_66, **var_67)
    var_69 = 'Entry 3'
    var_70 = 'Source 3'
    var_71 = module_2.JournalEntry(var_68, var_69, var_70)
    var_72 = [var_15, var_65, var_16]
    var_73 = {}
    var_74 = module_0.date(*var_72, **var_73)
    var_75 = '30.00'
    var_76 = [var_75]
    var_77 = {}
    var_78 = module_1.Decimal(*var_76, **var_77)
    var_79 = [var_37, var_58, var_71]
    var_80 = [var_20]
    var_81 = {}
    var_82 = module_1.Decimal(*var_80, **var_81)
    var_83 = '80.00'
    var_84 = [var_83]
    var_85 = {}
    var_86 = module_1.Decimal(*var_84, **var_85)
    var_87 = [var_27]
    var_88 = {}
    var_89 = module_1.Decimal(*var_87, **var_88)
    var_90 = '60.00'
    var_91 = [var_90]
    var_92 = {}
    var_93 = module_1.Decimal(*var_91, **var_92)
    var_94 = '0'
    var_95 = [var_94]
    var_96 = {}
    var_97 = module_1.Decimal(*var_95, **var_96)
    var_98 = [var_75]
    var_99 = {}
    var_100 = module_1.Decimal(*var_98, **var_99)



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #37
#--------------------------

# Partially parsed test_build_general_ledger_returns_correct_type. Retrieved 12/65 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2023
    var_7 = 12
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = 'A'
    var_13 = 'B'
    var_14 = 2023
    var_15 = 1
    var_16 = [var_14, var_15, var_15]
    var_17 = {}
    var_18 = module_0.date(*var_16, **var_17)
    var_19 = '100'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)
    var_23 = 50
    var_24 = 6
    var_25 = [var_14, var_24, var_15]
    var_26 = {}
    var_27 = module_0.date(*var_25, **var_26)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_build_general_ledger_success. Retrieved 20/100 statements.
# Partially parsed test_build_general_ledger_ignores_out_of_period. Retrieved 18/47 statements.


import datetime as module_0
import pypara.commons.zeitgeist as module_1
import pypara.accounting.journaling as module_2

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
    var_13 = 1
    var_14 = -1
    var_15 = -1
    var_16 = 15
    var_17 = [var_3, var_4, var_16]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = 'Sale'
    var_21 = 'test'
    var_22 = module_2.JournalEntry(var_19, var_20, var_21)
    var_23 = 100
    var_24 = 50
    var_25 = [var_22]

import datetime as module_0
import pypara.commons.zeitgeist as module_1
import pypara.accounting.journaling as module_2
import pypara.accounting.ledger as module_3

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
    var_10 = 'Cash'
    var_11 = 2
    var_12 = [var_0, var_11, var_1]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = 'Late Sale'
    var_16 = 'test'
    var_17 = module_2.JournalEntry(var_14, var_15, var_16)
    var_18 = 100
    var_19 = {}
    var_20 = [var_17]
    var_21 = module_3.build_general_ledger(var_9, var_20, var_19)
    var_22 = var_21.ledgers
    var_23 = len(var_22)
    assert var_23 == 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generalledgerprogram_call_returns_correct_type. Retrieved 8/27 statements.


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



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_build_general_ledger_returns_correct_type. Retrieved 16/87 statements.


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
    var_10 = 'A1'
    var_11 = 100
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_2.Decimal(*var_12, **var_13)
    var_15 = 15
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_0.date(*var_16, **var_17)
    var_19 = None
    var_20 = [var_1]
    var_21 = {}
    var_22 = module_2.Decimal(*var_20, **var_21)
    var_23 = 50
    var_24 = [var_23]
    var_25 = {}
    var_26 = module_2.Decimal(*var_24, **var_25)
    var_27 = 'Test'



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

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_expected_value. Retrieved 6/20 statements.


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



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_build_general_ledger_success. Retrieved 38/75 statements.


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
    var_12 = 'C'
    var_13 = [var_0, var_1, var_1]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = '100.00'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = [var_0, var_1, var_1]
    var_21 = {}
    var_22 = module_0.date(*var_20, **var_21)
    var_23 = '50.00'
    var_24 = [var_23]
    var_25 = {}
    var_26 = module_2.Decimal(*var_24, **var_25)
    var_27 = 15
    var_28 = [var_0, var_1, var_27]
    var_29 = {}
    var_30 = module_0.date(*var_28, **var_29)
    var_31 = 'In-period transaction'
    var_32 = 'TestSource'
    var_33 = module_3.JournalEntry(var_30, var_31, var_32)
    var_34 = [var_0, var_1, var_27]
    var_35 = {}
    var_36 = module_0.date(*var_34, **var_35)
    var_37 = '20.00'
    var_38 = [var_37]
    var_39 = {}
    var_40 = module_2.Decimal(*var_38, **var_39)
    var_41 = [var_0, var_1, var_27]
    var_42 = {}
    var_43 = module_0.date(*var_41, **var_42)
    var_44 = [var_37]
    var_45 = {}
    var_46 = module_2.Decimal(*var_44, **var_45)
    var_47 = 2
    var_48 = [var_0, var_47, var_1]
    var_49 = {}
    var_50 = module_0.date(*var_48, **var_49)
    var_51 = 'Out-of-period transaction'
    var_52 = module_3.JournalEntry(var_50, var_51, var_32)
    var_53 = [var_0, var_47, var_1]
    var_54 = {}
    var_55 = module_0.date(*var_53, **var_54)
    var_56 = [var_23]
    var_57 = {}
    var_58 = module_2.Decimal(*var_56, **var_57)
    var_59 = [var_33, var_52]
    var_60 = '0'
    var_61 = [var_60]
    var_62 = {}
    var_63 = module_2.Decimal(*var_61, **var_62)
    var_64 = '120.00'
    var_65 = [var_64]
    var_66 = {}
    var_67 = module_2.Decimal(*var_65, **var_66)
    var_68 = '30.00'
    var_69 = [var_68]
    var_70 = {}
    var_71 = module_2.Decimal(*var_69, **var_70)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_build_general_ledger_returns_correct_type. Retrieved 15/64 statements.


import datetime as module_0
import decimal as module_1

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
    var_10 = [var_0, var_1, var_1]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '100.00'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = None
    var_18 = 6
    var_19 = [var_0, var_18, var_1]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = '50.00'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_1.Decimal(*var_23, **var_24)
    var_26 = [var_0, var_18, var_1]
    var_27 = {}
    var_28 = module_0.date(*var_26, **var_27)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal. Retrieved 13/32 statements.
# Partially parsed test_build_general_ledger_with_postings. Retrieved 9/15 statements.


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
    var_11 = 'Cash'
    var_12 = [var_0, var_1, var_1]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = '100.00'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = []
    var_20 = [var_15]
    var_21 = {}
    var_22 = module_2.Decimal(*var_20, **var_21)

import datetime as module_0
import pypara.commons.zeitgeist as module_1

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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_build_general_ledger_returns_correct_type. Retrieved 18/95 statements.


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
    var_11 = '100'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_2.Decimal(*var_12, **var_13)
    var_15 = None
    var_16 = 6
    var_17 = [var_0, var_16, var_1]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = '1'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = '50'
    var_25 = [var_24]
    var_26 = {}
    var_27 = module_2.Decimal(*var_25, **var_26)
    var_28 = [var_0, var_16, var_1]
    var_29 = {}
    var_30 = module_0.date(*var_28, **var_29)
    var_31 = 'Test'



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #15
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #16
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_build_general_ledger_success. Retrieved 18/35 statements.


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
    var_10 = 'Revenue'
    var_11 = 'Expense'
    var_12 = '100.00'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Decimal(*var_13, **var_14)
    var_16 = 15
    var_17 = [var_0, var_1, var_16]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = 'Sale'
    var_21 = [var_0, var_1, var_16]
    var_22 = {}
    var_23 = module_0.date(*var_21, **var_22)
    var_24 = '50.00'
    var_25 = [var_24]
    var_26 = {}
    var_27 = module_1.Decimal(*var_25, **var_26)



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_build_general_ledger_empty_journal_uses_initial_balances. Retrieved 15/32 statements.
# Partially parsed test_build_general_ledger_processes_postings_within_period. Retrieved 34/61 statements.
# Partially parsed test_build_general_ledger_creates_new_accounts_from_postings. Retrieved 21/35 statements.


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
    var_9 = 'AccountA'
    var_10 = 'AccountB'
    var_11 = '100.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = '0.00'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = []
    var_20 = [var_15]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)

import datetime as module_0
import pypara.accounting.journaling as module_1
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
    var_9 = 'AccountA'
    var_10 = 'AccountB'
    var_11 = 15
    var_12 = [var_0, var_1, var_11]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = 'In Period'
    var_16 = 'Src'
    var_17 = module_1.JournalEntry(var_14, var_15, var_16)
    var_18 = [var_0, var_1, var_11]
    var_19 = {}
    var_20 = module_0.date(*var_18, **var_19)
    var_21 = '50.00'
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_2.Decimal(*var_22, **var_23)
    var_25 = 2
    var_26 = [var_0, var_25, var_1]
    var_27 = {}
    var_28 = module_0.date(*var_26, **var_27)
    var_29 = 'Late'
    var_30 = module_1.JournalEntry(var_28, var_29, var_16)
    var_31 = [var_0, var_25, var_1]
    var_32 = {}
    var_33 = module_0.date(*var_31, **var_32)
    var_34 = [var_21]
    var_35 = {}
    var_36 = module_2.Decimal(*var_34, **var_35)
    var_37 = 2022
    var_38 = 12
    var_39 = [var_37, var_38, var_5]
    var_40 = {}
    var_41 = module_0.date(*var_39, **var_40)
    var_42 = 'Early'
    var_43 = module_1.JournalEntry(var_41, var_42, var_16)
    var_44 = [var_37, var_38, var_5]
    var_45 = {}
    var_46 = module_0.date(*var_44, **var_45)
    var_47 = [var_21]
    var_48 = {}
    var_49 = module_2.Decimal(*var_47, **var_48)
    var_50 = [var_17, var_30, var_43]
    var_51 = '0.00'
    var_52 = [var_51]
    var_53 = {}
    var_54 = module_2.Decimal(*var_52, **var_53)
    var_55 = [var_21]
    var_56 = {}
    var_57 = module_2.Decimal(*var_55, **var_56)

import datetime as module_0
import pypara.accounting.journaling as module_1
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
    var_9 = 'NewAccount'
    var_10 = 15
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = 'New'
    var_15 = 'Src'
    var_16 = module_1.JournalEntry(var_13, var_14, var_15)
    var_17 = [var_0, var_1, var_10]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = '100.00'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = [var_16]
    var_25 = {}
    var_26 = '0.00'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_2.Decimal(*var_27, **var_28)
    var_30 = [var_20]
    var_31 = {}
    var_32 = module_2.Decimal(*var_30, **var_31)



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #23
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'TestAccount'
    var_1 = 100.0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generalladgerprogram_call_returns_correct_type. Retrieved 7/21 statements.


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
    var_12 = [var_11]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_build_general_ledger_initializes_with_provided_balances. Retrieved 20/32 statements.
# Partially parsed test_build_general_ledger_creates_new_ledgers_for_untracked_accounts. Retrieved 20/30 statements.
# Partially parsed test_build_general_ledger_filters_by_date_range. Retrieved 27/46 statements.
# Partially parsed test_build_general_ledger_calculates_running_balances. Retrieved 26/46 statements.


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
    var_24 = 'Test'
    var_25 = 'Source'
    var_26 = module_3.JournalEntry(var_23, var_24, var_25)
    var_27 = [var_26]
    var_28 = [var_12]
    var_29 = {}
    var_30 = module_2.Decimal(*var_28, **var_29)
    var_31 = [var_16]
    var_32 = {}
    var_33 = module_2.Decimal(*var_31, **var_32)

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
    var_16 = 'Test'
    var_17 = 'Source'
    var_18 = module_2.JournalEntry(var_15, var_16, var_17)
    var_19 = [var_0, var_1, var_12]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = '10.00'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_3.Decimal(*var_23, **var_24)
    var_26 = [var_18]
    var_27 = module_4.build_general_ledger(var_9, var_26, var_11)
    var_28 = '0'
    var_29 = [var_28]
    var_30 = {}
    var_31 = module_3.Decimal(*var_29, **var_30)

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
    var_11 = '0'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_2.Decimal(*var_12, **var_13)
    var_15 = 15
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_0.date(*var_16, **var_17)
    var_19 = 'Inside'
    var_20 = 'S1'
    var_21 = module_3.JournalEntry(var_18, var_19, var_20)
    var_22 = 2
    var_23 = [var_0, var_22, var_1]
    var_24 = {}
    var_25 = module_0.date(*var_23, **var_24)
    var_26 = 'Outside'
    var_27 = 'S2'
    var_28 = module_3.JournalEntry(var_25, var_26, var_27)
    var_29 = [var_0, var_1, var_15]
    var_30 = {}
    var_31 = module_0.date(*var_29, **var_30)
    var_32 = '10.00'
    var_33 = [var_32]
    var_34 = {}
    var_35 = module_2.Decimal(*var_33, **var_34)
    var_36 = [var_0, var_22, var_1]
    var_37 = {}
    var_38 = module_0.date(*var_36, **var_37)
    var_39 = '20.00'
    var_40 = [var_39]
    var_41 = {}
    var_42 = module_2.Decimal(*var_40, **var_41)
    var_43 = [var_21, var_28]
    var_44 = [var_32]
    var_45 = {}
    var_46 = module_2.Decimal(*var_44, **var_45)

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
    var_11 = '100.00'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_2.Decimal(*var_12, **var_13)
    var_15 = 15
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_0.date(*var_16, **var_17)
    var_19 = 'Test'
    var_20 = 'Source'
    var_21 = module_3.JournalEntry(var_18, var_19, var_20)
    var_22 = [var_0, var_1, var_15]
    var_23 = {}
    var_24 = module_0.date(*var_22, **var_23)
    var_25 = '50.00'
    var_26 = [var_25]
    var_27 = {}
    var_28 = module_2.Decimal(*var_26, **var_27)
    var_29 = 16
    var_30 = [var_0, var_1, var_29]
    var_31 = {}
    var_32 = module_0.date(*var_30, **var_31)
    var_33 = '20.00'
    var_34 = [var_33]
    var_35 = {}
    var_36 = module_2.Decimal(*var_34, **var_35)
    var_37 = [var_21]
    var_38 = '150.00'
    var_39 = [var_38]
    var_40 = {}
    var_41 = module_2.Decimal(*var_39, **var_40)
    var_42 = '130.00'
    var_43 = [var_42]
    var_44 = {}
    var_45 = module_2.Decimal(*var_43, **var_44)



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_read_initial_balances_call_returns_correct_value. Retrieved 6/20 statements.
# Partially parsed test_read_initial_balances_call_with_different_input. Retrieved 7/21 statements.


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

import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 28
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 50.5



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_general_ledger_program_call_returns_correct_type. Retrieved 7/22 statements.


import typing as module_0
import datetime as module_1

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



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #32
#--------------------------

# Partially parsed test_ledger_constructor_initialization. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test Account'
    var_1 = 100.0



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #34
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #35
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #36
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #37
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #38
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #39
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




# Parsed testcases at query #40
#--------------------------

# Failed to parse test_ledger_entry_constructor_initialization.




