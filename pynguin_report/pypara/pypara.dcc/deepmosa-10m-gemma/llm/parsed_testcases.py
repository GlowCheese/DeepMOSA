####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.1'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.Decimal(*var_2, **var_3)
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = 'ACT360'
    var_8 = 'ACT/360'
    var_9 = {var_7, var_8}
    var_10 = set()
    var_11 = []
    var_12 = 'name'
    var_13 = 'altnames'
    var_14 = 'currencies'
    var_15 = 'calculate_fraction_method'
    var_16 = {var_12: var_6, var_13: var_9, var_14: var_10, var_15: var_5}
    var_17 = module_0.DCC(*var_11, **var_16)
    var_18 = var_0.register(var_17)
    var_19 = var_0.find(var_6)
    var_20 = bool(var_19 == var_17)
    assert var_20 is True
    var_21 = var_0.find(var_7)
    var_22 = bool(var_21 == var_17)
    assert var_22 is True
    var_23 = var_0.find(var_8)
    var_24 = bool(var_23 == var_17)
    assert var_24 is True
    var_25 = bool(var_17 in var_0.registry)
    assert var_25 is True

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.1'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.Decimal(*var_2, **var_3)
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = 'ACT360'
    var_8 = {var_7}
    var_9 = set()
    var_10 = []
    var_11 = 'name'
    var_12 = 'altnames'
    var_13 = 'currencies'
    var_14 = 'calculate_fraction_method'
    var_15 = {var_11: var_6, var_12: var_8, var_13: var_9, var_14: var_5}
    var_16 = module_0.DCC(*var_10, **var_15)
    var_17 = 'OTHER'
    var_18 = {var_17}
    var_19 = set()
    var_20 = []
    var_21 = 'name'
    var_22 = 'altnames'
    var_23 = 'currencies'
    var_24 = 'calculate_fraction_method'
    var_25 = {var_21: var_6, var_22: var_18, var_23: var_19, var_24: var_5}
    var_26 = module_0.DCC(*var_20, **var_25)
    var_27 = var_0.register(var_16)
    var_28 = var_0.register(var_26)
    var_29 = 'TypeError not raised'
    var_30 = AssertionError(var_29)

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.1'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.Decimal(*var_2, **var_3)
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = 'ACT360'
    var_8 = {var_7}
    var_9 = set()
    var_10 = []
    var_11 = 'name'
    var_12 = 'altnames'
    var_13 = 'currencies'
    var_14 = 'calculate_fraction_method'
    var_15 = {var_11: var_6, var_12: var_8, var_13: var_9, var_14: var_5}
    var_16 = module_0.DCC(*var_10, **var_15)
    var_17 = 'Other'
    var_18 = {var_7}
    var_19 = set()
    var_20 = []
    var_21 = 'name'
    var_22 = 'altnames'
    var_23 = 'currencies'
    var_24 = 'calculate_fraction_method'
    var_25 = {var_21: var_17, var_22: var_18, var_23: var_19, var_24: var_5}
    var_26 = module_0.DCC(*var_20, **var_25)
    var_27 = var_0.register(var_16)
    var_28 = var_0.register(var_26)
    var_29 = 'TypeError not raised'
    var_30 = AssertionError(var_29)



# Parsed testcases at query #2
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_us(var_5, var_10, var_13)
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '0.16666666666667'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_2.Decimal(*var_18, **var_19)
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '0.16944444444444'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.08333333333333'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.33333333333333'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = module_1.dcfc_30_360_us(var_4, var_7, var_10)
    var_12 = '0'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = bool(var_11 == var_15)
    assert var_16 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_interest_valid_dates. Retrieved 23/66 statements.
# Partially parsed test_interest_zero_fraction_due_to_invalid_date_order. Retrieved 22/64 statements.
# Partially parsed test_interest_with_end_date_as_none. Retrieved 19/60 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'DCC'
    var_1 = 'name'
    var_2 = 'altnames'
    var_3 = 'currencies'
    var_4 = 'calculate_fraction_method'
    var_5 = 'Actual/Actual'
    var_6 = 'A/A'
    var_7 = {var_6}
    var_8 = set()
    var_9 = 2023
    var_10 = 1
    var_11 = [var_9, var_10, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = 6
    var_15 = [var_9, var_14, var_10]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = 12
    var_19 = 31
    var_20 = [var_9, var_18, var_19]
    var_21 = {}
    var_22 = module_0.date(*var_20, **var_21)
    var_23 = '1000.00'
    var_24 = [var_23]
    var_25 = {}
    var_26 = module_1.Decimal(*var_24, **var_25)
    var_27 = '0.05'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_1.Decimal(*var_28, **var_29)
    var_31 = '25.00'
    var_32 = [var_31]
    var_33 = {}
    var_34 = module_1.Decimal(*var_32, **var_33)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'DCC'
    var_1 = 'name'
    var_2 = 'altnames'
    var_3 = 'currencies'
    var_4 = 'calculate_fraction_method'
    var_5 = 'Test'
    var_6 = set()
    var_7 = set()
    var_8 = 2023
    var_9 = 6
    var_10 = 1
    var_11 = [var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = [var_8, var_10, var_10]
    var_15 = {}
    var_16 = module_0.date(*var_14, **var_15)
    var_17 = 12
    var_18 = 31
    var_19 = [var_8, var_17, var_18]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = '1000.00'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_1.Decimal(*var_23, **var_24)
    var_26 = '0.05'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_1.Decimal(*var_27, **var_28)
    var_30 = '0'
    var_31 = [var_30]
    var_32 = {}
    var_33 = module_1.Decimal(*var_31, **var_32)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'DCC'
    var_1 = 'name'
    var_2 = 'altnames'
    var_3 = 'currencies'
    var_4 = 'calculate_fraction_method'
    var_5 = 'Test'
    var_6 = set()
    var_7 = set()
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = [var_8, var_9, var_9]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = '100.00'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Decimal(*var_17, **var_18)
    var_20 = '0.10'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_1.Decimal(*var_21, **var_22)
    var_24 = None
    var_25 = '10.00'
    var_26 = [var_25]
    var_27 = {}
    var_28 = module_1.Decimal(*var_26, **var_27)



# Parsed testcases at query #4
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 31
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 2
    var_11 = [var_0, var_10, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_us(var_5, var_9, var_13)
    var_15 = '0.0'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = bool(var_14 == var_18)
    assert var_19 is True
    var_20 = bool(True)
    assert var_20 is True



# Parsed testcases at query #5
#--------------------------




import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1._has_leap_day(var_4, var_9)
    assert var_10 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 3
    var_7 = 1
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = module_1._has_leap_day(var_5, var_10)
    assert var_11 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2020
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 29
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1._has_leap_day(var_5, var_9)
    assert var_10 is True

def test_case_0():
    pass



# Parsed testcases at query #6
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_act_act(var_5, var_10, var_13)
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '0.16942884946478'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_2.Decimal(*var_18, **var_19)
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_act_act(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '0.17216108990194'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_act_act(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.08243131970956'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_act_act(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.32625945055768'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = module_1.dcfc_act_act(var_4, var_7, var_10)
    var_12 = '0'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = bool(var_11 == var_15)
    assert var_16 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 10
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 5
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 15
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_act_act(var_5, var_9, var_13)
    var_15 = '0'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = bool(var_14 == var_18)
    assert var_19 is True



# Parsed testcases at query #7
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_us(var_5, var_10, var_13)
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '0.16666666666667'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_2.Decimal(*var_18, **var_19)
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '0.16944444444444'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.08333333333333'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.33333333333333'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = module_1.dcfc_30_360_us(var_4, var_7, var_10)
    var_12 = '0'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = bool(var_11 == var_15)
    assert var_16 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_interest_standard_calculation. Retrieved 18/45 statements.
# Partially parsed test_interest_with_invalid_date_range_returns_zero. Retrieved 16/41 statements.
# Partially parsed test_interest_with_end_date_as_none_uses_asof. Retrieved 15/40 statements.


import decimal as module_0
import datetime as module_1

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'Actual/360'
    var_5 = 'A/360'
    var_6 = {var_5}
    var_7 = set()
    var_8 = '1000.00'
    var_9 = '0.05'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = 2023
    var_14 = 1
    var_15 = [var_13, var_14, var_14]
    var_16 = {}
    var_17 = module_1.date(*var_15, **var_16)
    var_18 = 6
    var_19 = [var_13, var_18, var_14]
    var_20 = {}
    var_21 = module_1.date(*var_19, **var_20)
    var_22 = 12
    var_23 = 31
    var_24 = [var_13, var_22, var_23]
    var_25 = {}
    var_26 = module_1.date(*var_24, **var_25)
    var_27 = '25.00'

import decimal as module_0
import datetime as module_1

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'Test'
    var_5 = set()
    var_6 = set()
    var_7 = '1000.00'
    var_8 = '0.05'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)
    var_12 = 2023
    var_13 = 1
    var_14 = [var_12, var_13, var_13]
    var_15 = {}
    var_16 = module_1.date(*var_14, **var_15)
    var_17 = 12
    var_18 = 31
    var_19 = [var_12, var_17, var_18]
    var_20 = {}
    var_21 = module_1.date(*var_19, **var_20)
    var_22 = 6
    var_23 = [var_12, var_22, var_13]
    var_24 = {}
    var_25 = module_1.date(*var_23, **var_24)

import decimal as module_0
import datetime as module_1

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'Test'
    var_5 = set()
    var_6 = set()
    var_7 = '100.00'
    var_8 = '0.1'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)
    var_12 = 2023
    var_13 = 1
    var_14 = [var_12, var_13, var_13]
    var_15 = {}
    var_16 = module_1.date(*var_14, **var_15)
    var_17 = 2
    var_18 = [var_12, var_17, var_13]
    var_19 = {}
    var_20 = module_1.date(*var_18, **var_19)
    var_21 = '1.0'
    var_22 = None



# Parsed testcases at query #9
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 5
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 10
    var_6 = var_3.day
    assert var_6 == 5

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = module_0._construct_date(var_0, var_1, var_1)
    var_3 = 'ValueError not raised for year 0'
    var_4 = AssertionError(var_3)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 1
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = 'ValueError not raised for month 0'
    var_5 = AssertionError(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 0
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = 'ValueError not raised for day 0'
    var_5 = AssertionError(var_4)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 32
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 10
    var_6 = var_3.day
    assert var_6 == 31

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 2
    var_6 = var_3.day
    assert var_6 == 28

def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------




import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = '0.16986301369863'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = module_2.dcfc_act_365_a(var_5, var_10, var_13)
    var_19 = 14
    var_20 = round(var_18, var_19)
    var_21 = bool(var_20 == var_17)
    assert var_21 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = '0.17213114754098'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = module_2.dcfc_act_365_a(var_5, var_11, var_14)
    var_20 = 14
    var_21 = round(var_19, var_20)
    var_22 = bool(var_21 == var_18)
    assert var_22 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = '1.08196721311475'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = module_2.dcfc_act_365_a(var_5, var_11, var_14)
    var_20 = 14
    var_21 = round(var_19, var_20)
    var_22 = bool(var_21 == var_18)
    assert var_22 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = '1.32513661202186'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = module_2.dcfc_act_365_a(var_5, var_11, var_14)
    var_20 = 14
    var_21 = round(var_19, var_20)
    var_22 = bool(var_21 == var_18)
    assert var_22 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = '0'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = module_2.dcfc_act_365_a(var_4, var_7, var_10)
    var_16 = bool(var_15 == var_14)
    assert var_16 is True



# Parsed testcases at query #11
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_german(var_5, var_10, var_13)
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '0.16666666666667'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_2.Decimal(*var_18, **var_19)
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_german(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '0.16944444444444'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_german(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.08333333333333'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_german(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.33055555555556'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2
    var_7 = 28
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_0, var_6, var_7]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_german(var_5, var_10, var_13)
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '0.07777777777778'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_2.Decimal(*var_18, **var_19)
    var_21 = bool(var_16 == var_20)
    assert var_21 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_calculate_daily_fraction_logic. Retrieved 9/67 statements.


def test_case_0():
    var_0 = '0'
    var_1 = 0
    var_2 = 5
    var_3 = 10
    var_4 = 'Test'
    var_5 = set()
    var_6 = set()
    var_7 = '0.1'
    var_8 = 1



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dcfc_30_360_isda_calculation_standard_dates. Retrieved 11/15 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 14
    var_12 = '0.16666666666667'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Decimal(*var_13, **var_14)

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = module_1.dcfc_30_360_isda(var_5, var_11, var_11)
    var_13 = 14
    var_14 = round(var_12, var_13)
    var_15 = '0.16944444444444'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = bool(var_14 == var_18)
    assert var_19 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = module_1.dcfc_30_360_isda(var_5, var_11, var_11)
    var_13 = 14
    var_14 = round(var_12, var_13)
    var_15 = '1.08333333333333'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = bool(var_14 == var_18)
    assert var_19 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = module_1.dcfc_30_360_isda(var_5, var_11, var_11)
    var_13 = 14
    var_14 = round(var_12, var_13)
    var_15 = '1.33333333333333'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = bool(var_14 == var_18)
    assert var_19 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = module_1.dcfc_30_360_isda(var_4, var_4, var_4)
    var_6 = '0'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_2.Decimal(*var_7, **var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2
    var_6 = [var_0, var_5, var_1]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = module_1.dcfc_30_360_isda(var_4, var_8, var_8)
    var_10 = '30'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_2.Decimal(*var_11, **var_12)
    var_14 = '360'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = var_13 / var_17
    var_19 = bool(var_9 == var_18)
    assert var_19 is True



# Parsed testcases at query #2
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = 'ACTUAL/ACTUAL'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = 0.1
    var_7 = lambda s, a, e, f: var_6
    var_8 = []
    var_9 = 'name'
    var_10 = 'altnames'
    var_11 = 'currencies'
    var_12 = 'calculate_fraction_method'
    var_13 = {var_9: var_1, var_10: var_4, var_11: var_5, var_12: var_7}
    var_14 = module_0.DCC(*var_8, **var_13)
    var_15 = var_0.register(var_14)
    var_16 = var_0.find(var_1)
    var_17 = bool(var_16 == var_14)
    assert var_17 is True
    var_18 = var_0.find(var_2)
    var_19 = bool(var_18 == var_14)
    assert var_19 is True
    var_20 = var_0.find(var_3)
    var_21 = bool(var_20 == var_14)
    assert var_21 is True
    var_22 = bool(var_14 in var_0.registry)
    assert var_22 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = set()
    var_3 = set()
    var_4 = 0.1
    var_5 = lambda s, a, e, f: var_4
    var_6 = []
    var_7 = 'name'
    var_8 = 'altnames'
    var_9 = 'currencies'
    var_10 = 'calculate_fraction_method'
    var_11 = {var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_5}
    var_12 = module_0.DCC(*var_6, **var_11)
    var_13 = set()
    var_14 = set()
    var_15 = 0.2
    var_16 = lambda s, a, e, f: var_15
    var_17 = []
    var_18 = 'name'
    var_19 = 'altnames'
    var_20 = 'currencies'
    var_21 = 'calculate_fraction_method'
    var_22 = {var_18: var_1, var_19: var_13, var_20: var_14, var_21: var_16}
    var_23 = module_0.DCC(*var_17, **var_22)
    var_24 = var_0.register(var_12)
    var_25 = var_0.register(var_23)
    var_26 = 'TypeError not raised'
    var_27 = AssertionError(var_26)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = 'ACT/ACT'
    var_3 = {var_2}
    var_4 = set()
    var_5 = 0.1
    var_6 = lambda s, a, e, f: var_5
    var_7 = []
    var_8 = 'name'
    var_9 = 'altnames'
    var_10 = 'currencies'
    var_11 = 'calculate_fraction_method'
    var_12 = {var_8: var_1, var_9: var_3, var_10: var_4, var_11: var_6}
    var_13 = module_0.DCC(*var_7, **var_12)
    var_14 = 'New/New'
    var_15 = {var_2}
    var_16 = set()
    var_17 = 0.2
    var_18 = lambda s, a, e, f: var_17
    var_19 = []
    var_20 = 'name'
    var_21 = 'altnames'
    var_22 = 'currencies'
    var_23 = 'calculate_fraction_method'
    var_24 = {var_20: var_14, var_21: var_15, var_22: var_16, var_23: var_18}
    var_25 = module_0.DCC(*var_19, **var_24)
    var_26 = var_0.register(var_13)
    var_27 = var_0.register(var_25)
    var_28 = 'TypeError not raised'
    var_29 = AssertionError(var_28)

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/Act'
    var_2 = set()
    var_3 = set()
    var_4 = 0.1
    var_5 = lambda s, a, e, f: var_4
    var_6 = []
    var_7 = 'name'
    var_8 = 'altnames'
    var_9 = 'currencies'
    var_10 = 'calculate_fraction_method'
    var_11 = {var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_5}
    var_12 = module_0.DCC(*var_6, **var_11)
    var_13 = '30/360'
    var_14 = {var_1}
    var_15 = set()
    var_16 = 0.2
    var_17 = lambda s, a, e, f: var_16
    var_18 = []
    var_19 = 'name'
    var_20 = 'altnames'
    var_21 = 'currencies'
    var_22 = 'calculate_fraction_method'
    var_23 = {var_19: var_13, var_20: var_14, var_21: var_15, var_22: var_17}
    var_24 = module_0.DCC(*var_18, **var_23)
    var_25 = var_0.register(var_12)
    var_26 = var_0.register(var_24)
    var_27 = 'TypeError not raised'
    var_28 = AssertionError(var_27)



# Parsed testcases at query #3
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_act_act(var_5, var_10, var_13)
    var_15 = '0.16942884946478'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = bool(var_14 == var_18)
    assert var_19 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_act_act(var_5, var_11, var_14)
    var_16 = '0.17216108990194'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = bool(var_15 == var_19)
    assert var_20 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_act_act(var_5, var_11, var_14)
    var_16 = '1.08243131970956'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = bool(var_15 == var_19)
    assert var_20 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_act_act(var_5, var_11, var_14)
    var_16 = '1.32625945055768'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = bool(var_15 == var_19)
    assert var_20 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = [var_0, var_1, var_1]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = module_1.dcfc_act_act(var_4, var_7, var_10)
    var_12 = '0'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = bool(var_11 == var_15)
    assert var_16 is True



# Parsed testcases at query #4
#--------------------------




import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = None
    var_6 = module_1._next_payment_date(var_4, var_1, var_5)
    var_7 = 2015
    var_8 = [var_7, var_1, var_1]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 15
    var_6 = module_1._next_payment_date(var_4, var_1, var_5)
    var_7 = 2015
    var_8 = [var_7, var_1, var_5]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 4
    var_6 = None
    var_7 = module_1._next_payment_date(var_4, var_5, var_6)
    var_8 = [var_0, var_5, var_1]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = bool(var_7 == var_10)
    assert var_11 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = '2'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Decimal(*var_6, **var_7)
    var_9 = None
    var_10 = module_2._next_payment_date(var_4, var_8, var_9)
    var_11 = 7
    var_12 = [var_0, var_11, var_1]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = bool(var_10 == var_14)
    assert var_15 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._next_payment_date(var_5, var_1, var_2)
    var_7 = 2025
    var_8 = [var_7, var_1, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 30
    var_6 = module_1._next_payment_date(var_4, var_1, var_5)
    var_7 = 2024
    var_8 = [var_7, var_1, var_5]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 31
    var_6 = module_1._next_payment_date(var_4, var_1, var_5)
    var_7 = 2025
    var_8 = [var_7, var_1, var_5]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_last_payment_date_semi_annual_start_of_year. Retrieved 9/12 statements.


import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2015
    var_6 = 12
    var_7 = 31
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = module_1._last_payment_date(var_4, var_10, var_1)
    var_12 = [var_5, var_1, var_1]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = bool(var_11 == var_14)
    assert var_15 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2015
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 12
    var_6 = 31
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = module_1._last_payment_date(var_4, var_9, var_1)
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2015
    var_6 = 12
    var_7 = 31
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 2
    var_12 = module_1._last_payment_date(var_4, var_10, var_11)
    var_13 = 7
    var_14 = [var_5, var_13, var_1]
    var_15 = {}
    var_16 = module_0.date(*var_14, **var_15)
    var_17 = bool(var_12 == var_16)
    assert var_17 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2015
    var_6 = 8
    var_7 = 31
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 2
    var_12 = module_1._last_payment_date(var_4, var_10, var_11)
    var_13 = 7
    var_14 = [var_5, var_13, var_1]
    var_15 = {}
    var_16 = module_0.date(*var_14, **var_15)
    var_17 = bool(var_12 == var_16)
    assert var_17 is True

import datetime as module_0

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2015
    var_6 = 4
    var_7 = 30
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 2
    var_12 = [var_5, var_1, var_1]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 6
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2015
    var_7 = 4
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = module_1._last_payment_date(var_5, var_11, var_2)
    var_13 = [var_0, var_1, var_2]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2008
    var_1 = 7
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2015
    var_6 = 10
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 4
    var_12 = module_1._last_payment_date(var_4, var_10, var_11)
    var_13 = [var_5, var_1, var_1]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 12
    var_2 = 9
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2015
    var_7 = 4
    var_8 = [var_6, var_1, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 1
    var_12 = module_1._last_payment_date(var_5, var_10, var_11)
    var_13 = [var_0, var_1, var_2]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2016
    var_7 = 1
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = 2
    var_13 = module_1._last_payment_date(var_5, var_11, var_12)
    var_14 = 2015
    var_15 = [var_14, var_1, var_2]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = bool(var_13 == var_17)
    assert var_18 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2012
    var_1 = 12
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2015
    var_7 = 31
    var_8 = [var_6, var_1, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 2
    var_12 = module_1._last_payment_date(var_5, var_10, var_11)
    var_13 = [var_6, var_1, var_2]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2014
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2015
    var_6 = 12
    var_7 = 31
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 28
    var_12 = module_1._last_payment_date(var_4, var_10, var_1, var_11)
    var_13 = [var_5, var_1, var_11]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True



# Parsed testcases at query #6
#--------------------------




import pypara.dcc as module_0
import datetime as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = [var_0, var_1, var_2]
    var_5 = {}
    var_6 = module_1.date(*var_4, **var_5)
    var_7 = bool(var_3 == var_6)
    assert var_7 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 0
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 0
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0
import datetime as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = 28
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_1.date(*var_5, **var_6)
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import pypara.dcc as module_0
import datetime as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 31
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = 30
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_1.date(*var_5, **var_6)
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import pypara.dcc as module_0
import datetime as module_1

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = [var_0, var_1, var_2]
    var_5 = {}
    var_6 = module_1.date(*var_4, **var_5)
    var_7 = bool(var_3 == var_6)
    assert var_7 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 1
    var_3 = module_0._construct_date(var_0, var_1, var_2)



# Parsed testcases at query #7
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 10
    var_2 = 27
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = var_3.month
    assert var_5 == 10
    var_6 = var_3.day
    assert var_6 == 27



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_interest_valid_dates. Retrieved 20/58 statements.
# Partially parsed test_interest_invalid_dates_returns_zero. Retrieved 18/54 statements.
# Partially parsed test_interest_end_date_is_asof_if_none. Retrieved 18/54 statements.


import decimal as module_0
import datetime as module_1

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'Test'
    var_5 = 'T'
    var_6 = {var_5}
    var_7 = set()
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = {}
    var_12 = module_1.date(*var_10, **var_11)
    var_13 = 6
    var_14 = [var_8, var_13, var_9]
    var_15 = {}
    var_16 = module_1.date(*var_14, **var_15)
    var_17 = 12
    var_18 = 31
    var_19 = [var_8, var_17, var_18]
    var_20 = {}
    var_21 = module_1.date(*var_19, **var_20)
    var_22 = '1000'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_0.Decimal(*var_23, **var_24)
    var_26 = '0.05'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_0.Decimal(*var_27, **var_28)
    var_30 = '25.0'
    var_31 = [var_30]
    var_32 = {}
    var_33 = module_0.Decimal(*var_31, **var_32)

import decimal as module_0
import datetime as module_1

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'Test'
    var_5 = 'T'
    var_6 = {var_5}
    var_7 = set()
    var_8 = 2023
    var_9 = 6
    var_10 = 1
    var_11 = [var_8, var_9, var_10]
    var_12 = {}
    var_13 = module_1.date(*var_11, **var_12)
    var_14 = [var_8, var_10, var_10]
    var_15 = {}
    var_16 = module_1.date(*var_14, **var_15)
    var_17 = 12
    var_18 = 31
    var_19 = [var_8, var_17, var_18]
    var_20 = {}
    var_21 = module_1.date(*var_19, **var_20)
    var_22 = '1000'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_0.Decimal(*var_23, **var_24)
    var_26 = '0.05'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_0.Decimal(*var_27, **var_28)

import decimal as module_0
import datetime as module_1

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'Test'
    var_5 = 'T'
    var_6 = {var_5}
    var_7 = set()
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = {}
    var_12 = module_1.date(*var_10, **var_11)
    var_13 = 6
    var_14 = [var_8, var_13, var_9]
    var_15 = {}
    var_16 = module_1.date(*var_14, **var_15)
    var_17 = '1000'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_0.Decimal(*var_18, **var_19)
    var_21 = '0.10'
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_0.Decimal(*var_22, **var_23)
    var_25 = None
    var_26 = '100.0'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_0.Decimal(*var_27, **var_28)



# Parsed testcases at query #9
#--------------------------




import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = 0
    var_9 = module_1._last_payment_date(var_4, var_7, var_1, var_8)
    var_10 = [var_0, var_1, var_1]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = bool(var_9 == var_12)
    assert var_13 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_calculate_fraction_valid_dates. Retrieved 16/20 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 6
    var_6 = [var_0, var_5, var_1]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 12
    var_10 = 31
    var_11 = [var_0, var_9, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = 'TestDC'
    var_15 = 'Test'
    var_16 = {var_15}
    var_17 = set()
    var_18 = '2'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Decimal(*var_19, **var_20)
    var_22 = '0.5'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_1.Decimal(*var_23, **var_24)



# Parsed testcases at query #11
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0._buffer_main
    var_2 = bool(var_0._buffer_main == {})
    assert var_2 is True
    var_3 = var_0._buffer_altn
    var_4 = bool(var_0._buffer_altn == {})
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------




import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = 1
    var_9 = 1
    var_10 = [var_1, var_1, var_1]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = [var_1, var_1, var_1]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = module_1._last_payment_date(var_15, var_12, var_1, var_1)
    var_17 = [var_1, var_1, var_1]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = bool(var_16 == var_19)
    assert var_20 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dcfc_30_360_us_boundary_day_31_to_31. Retrieved 9/12 statements.


import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = [var_6, var_7, var_2]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_us(var_5, var_10, var_13)
    var_15 = 14
    var_16 = round(var_14, var_15)
    var_17 = '0.16666666666667'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_2.Decimal(*var_18, **var_19)
    var_21 = bool(var_16 == var_20)
    assert var_21 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 12
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 2
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '0.16944444444444'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2007
    var_1 = 10
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2008
    var_7 = 11
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.08333333333333'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 2
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2009
    var_7 = 5
    var_8 = 31
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_30_360_us(var_5, var_11, var_14)
    var_16 = 14
    var_17 = round(var_15, var_16)
    var_18 = '1.33333333333333'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_2.Decimal(*var_19, **var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 3
    var_7 = [var_0, var_6, var_2]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = [var_0, var_6, var_2]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '0.1666666666666666666666666667'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2
    var_7 = 30
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_0, var_1, var_7]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = 3
    var_15 = [var_0, var_14, var_2]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = [var_0, var_14, var_2]
    var_19 = {}
    var_20 = module_0.date(*var_18, **var_19)
    var_21 = module_1.dcfc_30_360_us(var_13, var_17, var_20)
    var_22 = '1'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_2.Decimal(*var_23, **var_24)
    var_26 = '6'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_2.Decimal(*var_27, **var_28)
    var_30 = var_25 / var_29
    var_31 = bool(var_21 == var_30)
    assert var_31 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dcfc_30_360_us_d1_is_31_logic. Retrieved 9/13 statements.


import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2
    var_7 = 28
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_0, var_6, var_7]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_us(var_5, var_10, var_13)



# Parsed testcases at query #15
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 31
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = [var_0, var_1, var_6]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = module_1.dcfc_30_360_us(var_5, var_9, var_12)
    var_14 = '0'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = bool(var_13 == var_17)
    assert var_18 is True



