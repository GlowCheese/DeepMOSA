####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0.registry
    var_2 = bool(var_0.registry == [])
    assert var_2 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0.table
    var_2 = bool(var_0.table == {})
    assert var_2 is True



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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_last_day_of_month_true_non_leap_year_february. Retrieved 5/9 statements.


import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._is_last_day_of_month(var_5)
    assert var_6 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._is_last_day_of_month(var_5)
    assert var_6 is False

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._is_last_day_of_month(var_5)
    assert var_6 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 28
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._is_last_day_of_month(var_5)
    assert var_6 is False

import datetime as module_0

def test_case_0():
    var_0 = 'lydate'
    var_1 = 2023
    var_2 = 2
    var_3 = 28
    var_4 = [var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 27
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._is_last_day_of_month(var_5)
    assert var_6 is False

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._is_last_day_of_month(var_5)
    assert var_6 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 4
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = module_1._is_last_day_of_month(var_5)
    assert var_6 is False



# Parsed testcases at query #4
#--------------------------




import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.5'
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
    var_23 = 'act/360'
    var_24 = var_0.find(var_23)
    var_25 = bool(var_24 == var_17)
    assert var_25 is True
    var_26 = var_0.registry
    var_27 = len(var_26)
    assert var_27 == 1

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.Decimal(*var_2, **var_3)
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = set()
    var_8 = set()
    var_9 = []
    var_10 = 'name'
    var_11 = 'altnames'
    var_12 = 'currencies'
    var_13 = 'calculate_fraction_method'
    var_14 = {var_10: var_6, var_11: var_7, var_12: var_8, var_13: var_5}
    var_15 = module_0.DCC(*var_9, **var_14)
    var_16 = set()
    var_17 = set()
    var_18 = []
    var_19 = 'name'
    var_20 = 'altnames'
    var_21 = 'currencies'
    var_22 = 'calculate_fraction_method'
    var_23 = {var_19: var_6, var_20: var_16, var_21: var_17, var_22: var_5}
    var_24 = module_0.DCC(*var_18, **var_23)
    var_25 = var_0.register(var_15)
    var_26 = var_0.register(var_24)

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.Decimal(*var_2, **var_3)
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = 'ALT'
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

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = '0.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.Decimal(*var_2, **var_3)
    var_5 = lambda s, a, e, f: var_4
    var_6 = 'Act/360'
    var_7 = set()
    var_8 = set()
    var_9 = []
    var_10 = 'name'
    var_11 = 'altnames'
    var_12 = 'currencies'
    var_13 = 'calculate_fraction_method'
    var_14 = {var_10: var_6, var_11: var_7, var_12: var_8, var_13: var_5}
    var_15 = module_0.DCC(*var_9, **var_14)
    var_16 = 'Other'
    var_17 = {var_6}
    var_18 = set()
    var_19 = []
    var_20 = 'name'
    var_21 = 'altnames'
    var_22 = 'currencies'
    var_23 = 'calculate_fraction_method'
    var_24 = {var_20: var_16, var_21: var_17, var_22: var_18, var_23: var_5}
    var_25 = module_0.DCC(*var_19, **var_24)
    var_26 = var_0.register(var_15)
    var_27 = var_0.register(var_25)



# Parsed testcases at query #5
#--------------------------




import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2017
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 10
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = [var_0, var_1, var_5]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = '9'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Decimal(*var_13, **var_14)
    var_16 = '365'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Decimal(*var_17, **var_18)
    var_20 = var_15 / var_19
    var_21 = module_2.dcfc_act_365_a(var_4, var_8, var_11)
    var_22 = bool(var_21 == var_20)
    assert var_22 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2008
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 3
    var_6 = [var_0, var_5, var_1]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = [var_0, var_5, var_1]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = '60'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Decimal(*var_13, **var_14)
    var_16 = '366'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Decimal(*var_17, **var_18)
    var_20 = var_15 / var_19
    var_21 = module_2.dcfc_act_365_a(var_4, var_8, var_11)
    var_22 = bool(var_21 == var_20)
    assert var_22 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2017
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2018
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_6, var_7, var_7]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = '1'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = '365'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Decimal(*var_19, **var_20)
    var_22 = var_17 / var_21
    var_23 = module_2.dcfc_act_365_a(var_5, var_10, var_13)
    var_24 = bool(var_23 == var_22)
    assert var_24 is True

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
    var_14 = '62'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = '365'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Decimal(*var_19, **var_20)
    var_22 = var_17 / var_21
    var_23 = module_2.dcfc_act_365_a(var_5, var_10, var_13)
    var_24 = 14
    var_25 = round(var_23, var_24)
    var_26 = '0.16986301369863'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_1.Decimal(*var_27, **var_28)
    var_30 = bool(var_25 == var_29)
    assert var_30 is True

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
    var_15 = '63'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = '366'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)
    var_23 = var_18 / var_22
    var_24 = module_2.dcfc_act_365_a(var_5, var_11, var_14)
    var_25 = 14
    var_26 = round(var_24, var_25)
    var_27 = '0.17213114754098'
    var_28 = [var_27]
    var_29 = {}
    var_30 = module_1.Decimal(*var_28, **var_29)
    var_31 = bool(var_26 == var_30)
    assert var_31 is True



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
    var_7 = [var_0, var_6, var_2]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = [var_0, var_1, var_2]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = 3
    var_14 = [var_0, var_13, var_2]
    var_15 = {}
    var_16 = module_0.date(*var_14, **var_15)
    var_17 = [var_0, var_13, var_2]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = module_1.dcfc_30_360_us(var_12, var_16, var_19)
    var_21 = '1'
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_2.Decimal(*var_22, **var_23)
    var_25 = '6'
    var_26 = [var_25]
    var_27 = {}
    var_28 = module_2.Decimal(*var_26, **var_27)
    var_29 = var_24 / var_28
    var_30 = bool(var_20 == var_29)
    assert var_30 is True



# Parsed testcases at query #7
#--------------------------




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
    var_6 = 29
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = [var_0, var_1, var_6]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '28'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = '366'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_1.Decimal(*var_18, **var_19)
    var_21 = var_16 / var_20
    var_22 = module_2.dcfc_act_365_l(var_5, var_9, var_12)
    var_23 = bool(var_22 == var_21)
    assert var_23 is True

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
    var_6 = 31
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = [var_0, var_1, var_6]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '3'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = '365'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_1.Decimal(*var_18, **var_19)
    var_21 = var_16 / var_20
    var_22 = module_2.dcfc_act_365_l(var_5, var_9, var_12)
    var_23 = bool(var_22 == var_21)
    assert var_23 is True

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = '0'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Decimal(*var_6, **var_7)
    var_9 = module_2.dcfc_act_365_l(var_4, var_4, var_4)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

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
    var_11 = module_1.dcfc_act_365_l(var_5, var_10, var_10)
    var_12 = 14
    var_13 = round(var_11, var_12)
    var_14 = '0.16939890710383'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = bool(var_13 == var_17)
    assert var_18 is True

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
    var_12 = module_1.dcfc_act_365_l(var_5, var_11, var_11)
    var_13 = 14
    var_14 = round(var_12, var_13)
    var_15 = '0.17213114754098'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = bool(var_14 == var_18)
    assert var_19 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dcfc_30_360_german_leap_year_case. Retrieved 12/14 statements.


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
    var_8 = 29
    var_9 = [var_6, var_7, var_8]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = 14
    var_16 = '0.16944444444444'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Decimal(*var_17, **var_18)

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
    var_15 = '28'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = '360'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_2.Decimal(*var_20, **var_21)
    var_23 = var_18 / var_22
    var_24 = bool(var_14 == var_23)
    assert var_24 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2024
    var_1 = 2
    var_2 = 29
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 3
    var_7 = 31
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_0, var_6, var_7]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_german(var_5, var_10, var_13)
    var_15 = '30'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = '360'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_2.Decimal(*var_20, **var_21)
    var_23 = var_18 / var_22
    var_24 = bool(var_14 == var_23)
    assert var_24 is True



# Parsed testcases at query #9
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 8
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = module_1.dcfc_30_360_us(var_5, var_8, var_11)
    var_13 = '0'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_2.Decimal(*var_14, **var_15)
    var_17 = bool(var_12 == var_16)
    assert var_17 is True



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dcfc_30_360_us_triggers_d1_is_31_logic. Retrieved 8/13 statements.


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
    var_7 = [var_0, var_6, var_1]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = [var_0, var_6, var_1]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = module_1.dcfc_30_360_us(var_5, var_9, var_12)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dcfc_30_360_us_d1_is_31_logic. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 31
    var_1 = 1
    var_2 = 2023
    var_3 = 15
    var_4 = 2
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 7
    var_7 = 31
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_0, var_6, var_7]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = module_1.dcfc_30_360_us(var_5, var_10, var_13)
    var_15 = '1'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = '12'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_2.Decimal(*var_20, **var_21)
    var_23 = var_18 / var_22
    var_24 = bool(var_14 == var_23)
    assert var_24 is True



# Parsed testcases at query #14
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
    var_14 = module_1.dcfc_30_e_360(var_5, var_10, var_13)
    var_15 = '0.1666666666666666666666666667'
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
    var_15 = module_1.dcfc_30_e_360(var_5, var_11, var_14)
    var_16 = '61'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = '360'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = var_19 / var_23
    var_25 = bool(var_15 == var_24)
    assert var_25 is True

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
    var_15 = module_1.dcfc_30_e_360(var_5, var_11, var_14)
    var_16 = '390'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = '360'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = var_19 / var_23
    var_25 = bool(var_15 == var_24)
    assert var_25 is True

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
    var_15 = module_1.dcfc_30_e_360(var_5, var_11, var_14)
    var_16 = '479'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_2.Decimal(*var_17, **var_18)
    var_20 = '360'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_2.Decimal(*var_21, **var_22)
    var_24 = var_19 / var_23
    var_25 = bool(var_15 == var_24)
    assert var_25 is True

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
    var_11 = module_1.dcfc_30_e_360(var_4, var_7, var_10)
    var_12 = '0'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_2.Decimal(*var_13, **var_14)
    var_16 = bool(var_11 == var_15)
    assert var_16 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_8 = module_1._get_date_range(var_4, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = module_1._get_date_range(var_4, var_8)
    var_10 = list(var_9)
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = [var_13]
    var_15 = bool(var_10 == var_14)
    assert var_15 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 4
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = module_1._get_date_range(var_4, var_8)
    var_10 = list(var_9)
    var_11 = [var_0, var_1, var_1]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = 2
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = 3
    var_19 = [var_0, var_1, var_18]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = [var_13, var_17, var_21]
    var_23 = bool(var_10 == var_22)
    assert var_23 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 5
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = [var_0, var_1, var_1]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = module_1._get_date_range(var_5, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True



# Parsed testcases at query #2
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

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0.registry
    var_2 = bool(var_0.registry == [])
    assert var_2 is True

import pypara.dcc as module_0

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = var_0.table
    var_2 = bool(var_0.table == {})
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dcfc_act_act_standard_non_leap. Retrieved 18/19 statements.
# Partially parsed test_dcfc_act_act_leap_year_transition. Retrieved 15/17 statements.


import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2018
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2019
    var_6 = [var_5, var_1, var_1]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 2007
    var_10 = 12
    var_11 = 28
    var_12 = [var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = 2008
    var_16 = 2
    var_17 = [var_15, var_16, var_11]
    var_18 = {}
    var_19 = module_0.date(*var_17, **var_18)
    var_20 = module_1.dcfc_act_act(var_14, var_19, var_19)
    var_21 = 14
    var_22 = round(var_20, var_21)
    var_23 = '0.16942884946478'
    var_24 = [var_23]
    var_25 = {}
    var_26 = module_2.Decimal(*var_24, **var_25)
    var_27 = bool(var_22 == var_26)
    assert var_27 is True

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
    var_9 = [var_0, var_1, var_2]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = [var_6, var_7, var_8]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = module_1.dcfc_act_act(var_11, var_14, var_14)
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
    var_12 = module_1.dcfc_act_act(var_5, var_11, var_11)
    var_13 = 14
    var_14 = round(var_12, var_13)
    var_15 = '1.08243131970956'
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
    var_12 = module_1.dcfc_act_act(var_5, var_11, var_11)
    var_13 = 14
    var_14 = round(var_12, var_13)
    var_15 = '1.32625945055768'
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
    var_5 = [var_0, var_1, var_1]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = module_1.dcfc_act_act(var_4, var_7, var_7)
    var_9 = '0'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_2.Decimal(*var_10, **var_11)
    var_13 = bool(var_8 == var_12)
    assert var_13 is True

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
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = module_1.dcfc_act_act(var_4, var_8, var_8)
    var_10 = '1'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_2.Decimal(*var_11, **var_12)
    var_14 = '365'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = var_13 / var_17
    var_19 = bool(var_9 == var_18)
    assert var_19 is True



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
    var_16 = [var_5, var_1, var_1]
    var_17 = {}
    var_18 = module_0.date(*var_16, **var_17)
    var_19 = [var_5, var_6, var_7]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = module_1._last_payment_date(var_18, var_21, var_1)
    var_23 = [var_5, var_1, var_1]
    var_24 = {}
    var_25 = module_0.date(*var_23, **var_24)
    var_26 = bool(var_22 == var_25)
    assert var_26 is True
    var_27 = [var_0, var_1, var_1]
    var_28 = {}
    var_29 = module_0.date(*var_27, **var_28)
    var_30 = [var_5, var_6, var_7]
    var_31 = {}
    var_32 = module_0.date(*var_30, **var_31)
    var_33 = 2
    var_34 = module_1._last_payment_date(var_29, var_32, var_33)
    var_35 = 7
    var_36 = [var_5, var_35, var_1]
    var_37 = {}
    var_38 = module_0.date(*var_36, **var_37)
    var_39 = bool(var_34 == var_38)
    assert var_39 is True
    var_40 = [var_0, var_1, var_1]
    var_41 = {}
    var_42 = module_0.date(*var_40, **var_41)
    var_43 = 8
    var_44 = [var_5, var_43, var_7]
    var_45 = {}
    var_46 = module_0.date(*var_44, **var_45)
    var_47 = module_1._last_payment_date(var_42, var_46, var_33)
    var_48 = [var_5, var_35, var_1]
    var_49 = {}
    var_50 = module_0.date(*var_48, **var_49)
    var_51 = bool(var_47 == var_50)
    assert var_51 is True
    var_52 = [var_0, var_1, var_1]
    var_53 = {}
    var_54 = module_0.date(*var_52, **var_53)
    var_55 = 4
    var_56 = 30
    var_57 = [var_5, var_55, var_56]
    var_58 = {}
    var_59 = module_0.date(*var_57, **var_58)
    var_60 = module_1._last_payment_date(var_54, var_59, var_33)
    var_61 = [var_5, var_1, var_1]
    var_62 = {}
    var_63 = module_0.date(*var_61, **var_62)
    var_64 = bool(var_60 == var_63)
    assert var_64 is True
    var_65 = 6
    var_66 = [var_0, var_65, var_1]
    var_67 = {}
    var_68 = module_0.date(*var_66, **var_67)
    var_69 = [var_5, var_55, var_56]
    var_70 = {}
    var_71 = module_0.date(*var_69, **var_70)
    var_72 = module_1._last_payment_date(var_68, var_71, var_1)
    var_73 = [var_0, var_65, var_1]
    var_74 = {}
    var_75 = module_0.date(*var_73, **var_74)
    var_76 = bool(var_72 == var_75)
    assert var_76 is True
    var_77 = 2008
    var_78 = [var_77, var_35, var_35]
    var_79 = {}
    var_80 = module_0.date(*var_78, **var_79)
    var_81 = 10
    var_82 = [var_5, var_81, var_65]
    var_83 = {}
    var_84 = module_0.date(*var_82, **var_83)
    var_85 = module_1._last_payment_date(var_80, var_84, var_55)
    var_86 = [var_5, var_35, var_35]
    var_87 = {}
    var_88 = module_0.date(*var_86, **var_87)
    var_89 = bool(var_85 == var_88)
    assert var_89 is True
    var_90 = 9
    var_91 = [var_0, var_6, var_90]
    var_92 = {}
    var_93 = module_0.date(*var_91, **var_92)
    var_94 = [var_5, var_6, var_55]
    var_95 = {}
    var_96 = module_0.date(*var_94, **var_95)
    var_97 = module_1._last_payment_date(var_93, var_96, var_1)
    var_98 = [var_0, var_6, var_90]
    var_99 = {}
    var_100 = module_0.date(*var_98, **var_99)
    var_101 = bool(var_97 == var_100)
    assert var_101 is True
    var_102 = 2012
    var_103 = 15
    var_104 = [var_102, var_6, var_103]
    var_105 = {}
    var_106 = module_0.date(*var_104, **var_105)
    var_107 = 2016
    var_108 = [var_107, var_1, var_65]
    var_109 = {}
    var_110 = module_0.date(*var_108, **var_109)
    var_111 = module_1._last_payment_date(var_106, var_110, var_33)
    var_112 = [var_5, var_6, var_103]
    var_113 = {}
    var_114 = module_0.date(*var_112, **var_113)
    var_115 = bool(var_111 == var_114)
    assert var_115 is True
    var_116 = [var_102, var_6, var_103]
    var_117 = {}
    var_118 = module_0.date(*var_116, **var_117)
    var_119 = [var_5, var_6, var_7]
    var_120 = {}
    var_121 = module_0.date(*var_119, **var_120)
    var_122 = module_1._last_payment_date(var_118, var_121, var_33)
    var_123 = [var_5, var_6, var_103]
    var_124 = {}
    var_125 = module_0.date(*var_123, **var_124)
    var_126 = bool(var_122 == var_125)
    assert var_126 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2024
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 3
    var_6 = 15
    var_7 = [var_0, var_5, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 31
    var_11 = module_1._last_payment_date(var_4, var_9, var_1, var_10)
    var_12 = 2
    var_13 = 29
    var_14 = [var_0, var_12, var_13]
    var_15 = {}
    var_16 = module_0.date(*var_14, **var_15)
    var_17 = bool(var_11 == var_16)
    assert var_17 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.date(*var_1, **var_2)
    var_4 = 2
    var_5 = [var_4, var_0, var_0]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = module_1._last_payment_date(var_3, var_7, var_0)
    var_9 = [var_0, var_0, var_0]
    var_10 = {}
    var_11 = module_0.date(*var_9, **var_10)
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 6
    var_7 = [var_0, var_6, var_2]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 12
    var_11 = module_1._last_payment_date(var_5, var_9, var_10)
    var_12 = [var_0, var_1, var_2]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = bool(var_11 == var_14)
    assert var_15 is True

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
    var_8 = 0
    var_9 = module_1._last_payment_date(var_4, var_7, var_8)



# Parsed testcases at query #5
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
    var_1 = -1
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = -1
    var_3 = module_0._construct_date(var_0, var_1, var_2)

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



# Parsed testcases at query #6
#--------------------------




import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'ACT/360'
    var_3 = 'Actual/360'
    var_4 = {var_2, var_3}
    var_5 = set()
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Decimal(*var_7, **var_8)
    var_10 = lambda s, a, e, f: var_9
    var_11 = []
    var_12 = 'name'
    var_13 = 'altnames'
    var_14 = 'currencies'
    var_15 = 'calculate_fraction_method'
    var_16 = {var_12: var_1, var_13: var_4, var_14: var_5, var_15: var_10}
    var_17 = module_0.DCC(*var_11, **var_16)
    var_18 = var_0.register(var_17)
    var_19 = var_0.find(var_1)
    var_20 = bool(var_19 == var_17)
    assert var_20 is True
    var_21 = var_0.find(var_2)
    var_22 = bool(var_21 == var_17)
    assert var_22 is True
    var_23 = var_0.find(var_3)
    var_24 = bool(var_23 == var_17)
    assert var_24 is True
    var_25 = var_0.registry
    var_26 = len(var_25)
    assert var_26 == 1

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = set()
    var_3 = set()
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Decimal(*var_5, **var_6)
    var_8 = lambda s, a, e, f: var_7
    var_9 = []
    var_10 = 'name'
    var_11 = 'altnames'
    var_12 = 'currencies'
    var_13 = 'calculate_fraction_method'
    var_14 = {var_10: var_1, var_11: var_2, var_12: var_3, var_13: var_8}
    var_15 = module_0.DCC(*var_9, **var_14)
    var_16 = set()
    var_17 = set()
    var_18 = '0.1'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Decimal(*var_19, **var_20)
    var_22 = lambda s, a, e, f: var_21
    var_23 = []
    var_24 = 'name'
    var_25 = 'altnames'
    var_26 = 'currencies'
    var_27 = 'calculate_fraction_method'
    var_28 = {var_24: var_1, var_25: var_16, var_26: var_17, var_27: var_22}
    var_29 = module_0.DCC(*var_23, **var_28)
    var_30 = var_0.register(var_15)
    var_31 = var_0.register(var_29)

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'ACT/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Decimal(*var_6, **var_7)
    var_9 = lambda s, a, e, f: var_8
    var_10 = []
    var_11 = 'name'
    var_12 = 'altnames'
    var_13 = 'currencies'
    var_14 = 'calculate_fraction_method'
    var_15 = {var_11: var_1, var_12: var_3, var_13: var_4, var_14: var_9}
    var_16 = module_0.DCC(*var_10, **var_15)
    var_17 = 'Other'
    var_18 = {var_2}
    var_19 = set()
    var_20 = '0.1'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_1.Decimal(*var_21, **var_22)
    var_24 = lambda s, a, e, f: var_23
    var_25 = []
    var_26 = 'name'
    var_27 = 'altnames'
    var_28 = 'currencies'
    var_29 = 'calculate_fraction_method'
    var_30 = {var_26: var_17, var_27: var_18, var_28: var_19, var_29: var_24}
    var_31 = module_0.DCC(*var_25, **var_30)
    var_32 = var_0.register(var_16)
    var_33 = var_0.register(var_31)

import pypara.dcc as module_0
import decimal as module_1

def test_case_0():
    var_0 = module_0.DCCRegistryMachinery()
    var_1 = 'Act/360'
    var_2 = 'ACT/360'
    var_3 = {var_2}
    var_4 = set()
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Decimal(*var_6, **var_7)
    var_9 = lambda s, a, e, f: var_8
    var_10 = []
    var_11 = 'name'
    var_12 = 'altnames'
    var_13 = 'currencies'
    var_14 = 'calculate_fraction_method'
    var_15 = {var_11: var_1, var_12: var_3, var_13: var_4, var_14: var_9}
    var_16 = module_0.DCC(*var_10, **var_15)
    var_17 = var_0.register(var_16)
    var_18 = 'New'
    var_19 = {var_2}
    var_20 = set()
    var_21 = '0.1'
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_1.Decimal(*var_22, **var_23)
    var_25 = lambda s, a, e, f: var_24
    var_26 = []
    var_27 = 'name'
    var_28 = 'altnames'
    var_29 = 'currencies'
    var_30 = 'calculate_fraction_method'
    var_31 = {var_27: var_18, var_28: var_19, var_29: var_20, var_30: var_25}
    var_32 = module_0.DCC(*var_26, **var_31)
    var_33 = var_0.register(var_32)



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

# Partially parsed test_dcfc_act_act_icma_calculation_logic. Retrieved 14/18 statements.
# Partially parsed test_dcfc_act_act_icma_metadata. Retrieved 1/2 statements.


import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 9
    var_7 = 10
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = 2020
    var_12 = [var_0, var_1, var_2]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = [var_0, var_6, var_7]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = [var_11, var_1, var_2]
    var_19 = {}
    var_20 = module_0.date(*var_18, **var_19)
    var_21 = '1'
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_1.Decimal(*var_22, **var_23)
    var_25 = module_2.dcfc_act_act_icma(var_14, var_17, var_20, var_24)

import datetime as module_0
import decimal as module_1
import pypara.dcc as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 11
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 31
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = '2'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = module_2.dcfc_act_act_icma(var_4, var_8, var_12, var_16)
    var_18 = '10'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Decimal(*var_19, **var_20)
    var_22 = '30'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_1.Decimal(*var_23, **var_24)
    var_26 = var_21 / var_25
    var_27 = [var_13]
    var_28 = {}
    var_29 = module_1.Decimal(*var_27, **var_28)
    var_30 = var_26 / var_29
    var_31 = bool(var_17 == var_30)
    assert var_31 is True

import datetime as module_0
import pypara.dcc as module_1
import decimal as module_2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 11
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 31
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = None
    var_14 = module_1.dcfc_act_act_icma(var_4, var_8, var_12, var_13)
    var_15 = '10'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = '30'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_2.Decimal(*var_20, **var_21)
    var_23 = var_18 / var_22
    var_24 = '1'
    var_25 = [var_24]
    var_26 = {}
    var_27 = module_2.Decimal(*var_25, **var_26)
    var_28 = var_23 / var_27
    var_29 = bool(var_14 == var_28)
    assert var_29 is True

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
    var_11 = '1'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = module_2.dcfc_act_act_icma(var_4, var_7, var_10, var_14)

def test_case_0():
    var_0 = '__dcc'



# Parsed testcases at query #9
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
    var_10 = [var_0, var_1, var_1]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = [var_0, var_1, var_1]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = 0
    var_17 = module_1._last_payment_date(var_12, var_15, var_1, var_16)
    var_18 = [var_0, var_1, var_1]
    var_19 = {}
    var_20 = module_0.date(*var_18, **var_19)
    var_21 = bool(var_17 == var_20)
    assert var_21 is True



# Parsed testcases at query #10
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
    var_17 = '0.16666444444444'
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
    var_15 = '28'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_2.Decimal(*var_16, **var_17)
    var_19 = '360'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_2.Decimal(*var_20, **var_21)
    var_23 = var_18 / var_22
    var_24 = bool(var_14 == var_23)
    assert var_24 is True



# Parsed testcases at query #11
#--------------------------




import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0, var_0, var_0]
    var_2 = {}
    var_3 = module_0.date(*var_1, **var_2)
    var_4 = [var_0, var_0, var_0]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = 1
    var_8 = 1
    var_9 = module_1._last_payment_date(var_3, var_6, var_7, var_8)
    var_10 = bool(var_9 == var_3)
    assert var_10 is True



# Parsed testcases at query #12
#--------------------------




import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)
    var_4 = var_3.year
    assert var_4 == 2023
    var_5 = module_0._construct_date(var_0, var_1, var_2)
    var_6 = var_5.month
    assert var_6 == 5
    var_7 = module_0._construct_date(var_0, var_1, var_2)
    var_8 = var_7.day
    assert var_8 == 15

import pypara.dcc as module_0

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 15
    var_3 = module_0._construct_date(var_0, var_1, var_2)

import pypara.dcc as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 13
    var_2 = 1
    var_3 = module_0._construct_date(var_0, var_1, var_2)

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



# Parsed testcases at query #13
#--------------------------




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
    var_6 = 3
    var_7 = [var_0, var_6, var_2]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = [var_0, var_6, var_2]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = module_1.dcfc_30_360_us(var_5, var_9, var_12)
    var_14 = '0.5'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_2.Decimal(*var_15, **var_16)
    var_18 = bool(var_13 == var_17)
    assert var_18 is True
    var_19 = [var_0, var_1, var_2]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = [var_0, var_6, var_2]
    var_23 = {}
    var_24 = module_0.date(*var_22, **var_23)
    var_25 = [var_0, var_6, var_2]
    var_26 = {}
    var_27 = module_0.date(*var_25, **var_26)
    var_28 = module_1.dcfc_30_360_us(var_21, var_24, var_27)
    var_29 = '60'
    var_30 = [var_29]
    var_31 = {}
    var_32 = module_2.Decimal(*var_30, **var_31)
    var_33 = '360'
    var_34 = [var_33]
    var_35 = {}
    var_36 = module_2.Decimal(*var_34, **var_35)
    var_37 = var_32 / var_36
    var_38 = bool(var_28 == var_37)
    assert var_38 is True



# Parsed testcases at query #14
#--------------------------




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
    var_11 = 1
    var_12 = module_1._last_payment_date(var_4, var_10, var_11)
    var_13 = [var_5, var_1, var_1]
    var_14 = {}
    var_15 = module_0.date(*var_13, **var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True

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
    var_10 = 1
    var_11 = module_1._last_payment_date(var_4, var_9, var_10)
    var_12 = [var_0, var_1, var_1]
    var_13 = {}
    var_14 = module_0.date(*var_12, **var_13)
    var_15 = bool(var_11 == var_14)
    assert var_15 is True

def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_interest_valid_dates. Retrieved 18/28 statements.
# Partially parsed test_interest_with_end_date_none. Retrieved 16/26 statements.
# Partially parsed test_interest_invalid_date_order_returns_zero. Retrieved 18/28 statements.


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
    var_14 = '0.05'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = '1000.00'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Decimal(*var_19, **var_20)
    var_22 = 'USD'
    var_23 = 'TestDCC'
    var_24 = 'Test'
    var_25 = {var_24}
    var_26 = '25.00'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_1.Decimal(*var_27, **var_28)

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
    var_9 = '0.10'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = '100.00'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = 'EUR'
    var_18 = 'TestDCC'
    var_19 = 'Test'
    var_20 = {var_19}
    var_21 = None
    var_22 = '10.00'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_1.Decimal(*var_23, **var_24)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 1
    var_7 = [var_0, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 6
    var_11 = [var_0, var_10, var_6]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = '0.05'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = '1000.00'
    var_19 = [var_18]
    var_20 = {}
    var_21 = module_1.Decimal(*var_19, **var_20)
    var_22 = 'USD'
    var_23 = 'TestDCC'
    var_24 = 'Test'
    var_25 = {var_24}
    var_26 = '0.00'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_1.Decimal(*var_27, **var_28)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dcfc_30_360_us_predicate_true. Retrieved 17/21 statements.


import datetime as module_0
import pypara.dcc as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 30
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.date(*var_3, **var_4)
    var_6 = 2
    var_7 = 31
    var_8 = [var_0, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = [var_0, var_1, var_2]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = 8
    var_15 = [var_0, var_14, var_7]
    var_16 = {}
    var_17 = module_0.date(*var_15, **var_16)
    var_18 = module_1.dcfc_30_360_us(var_13, var_17, var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = {}
    var_21 = module_0.date(*var_19, **var_20)
    var_22 = [var_0, var_14, var_7]
    var_23 = {}
    var_24 = module_0.date(*var_22, **var_23)
    var_25 = [var_0, var_14, var_7]
    var_26 = {}
    var_27 = module_0.date(*var_25, **var_26)
    var_28 = [var_0, var_1, var_2]
    var_29 = {}
    var_30 = module_0.date(*var_28, **var_29)
    var_31 = [var_0, var_14, var_7]
    var_32 = {}
    var_33 = module_0.date(*var_31, **var_32)
    var_34 = module_1.dcfc_30_360_us(var_30, var_33, var_33)



