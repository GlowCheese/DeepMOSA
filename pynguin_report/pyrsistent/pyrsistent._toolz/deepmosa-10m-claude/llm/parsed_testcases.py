####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_in_nested_dict_access. Retrieved 25/38 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = [var_0]
    var_18 = 'total'
    var_19 = [var_1, var_18]
    var_20 = 'apple'
    var_21 = [var_1, var_4, var_20]
    var_22 = 10
    var_23 = [var_1, var_4, var_22]
    var_24 = [var_1, var_18]

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'missing'
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'default_val'
    var_6 = module_0.get_in(var_1, var_4, var_5)
    assert var_6 == 'default_val'
    var_7 = 'b'
    var_8 = [var_2, var_7]
    var_9 = {}
    var_10 = {var_2: var_9}
    var_11 = 42
    var_12 = module_0.get_in(var_8, var_10, var_11)
    assert var_12 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 0
    var_7 = [var_0, var_6]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 1
    var_9 = [var_0, var_2]
    var_10 = module_0.get_in(var_9, var_5)
    assert var_10 == 3
    var_11 = 5
    var_12 = [var_0, var_11]
    var_13 = module_0.get_in(var_12, var_5)
    assert var_13 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.get_in(var_2, var_7, no_default=var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = [var_0, var_1, var_2, var_3]
    var_10 = module_0.get_in(var_9, var_8)
    assert var_10 == 'value'
    var_11 = 'e'
    var_12 = [var_0, var_1, var_2, var_11]
    var_13 = module_0.get_in(var_12, var_8)
    assert var_13 is None



# Parsed testcases at query #2
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'value'
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = module_0.get_in(var_2, var_5)
    assert var_6 == 'value'
    var_7 = 0
    var_8 = 1
    var_9 = [var_7, var_8]
    var_10 = 2
    var_11 = [var_8, var_10]
    var_12 = 3
    var_13 = 4
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = module_0.get_in(var_9, var_15)
    assert var_16 == 3
    var_17 = []
    var_18 = {var_0: var_1}
    var_19 = module_0.get_in(var_17, var_18)
    var_20 = bool(var_19 == {'a': 'b'})
    assert var_20 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_in_with_nested_dict. Retrieved 17/25 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_missing_list_index_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_default_value. Retrieved 18/26 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'y'
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.get_in(var_2, var_0, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = [var_7, var_0]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'name'
    var_2 = 'apple'
    var_3 = {var_1: var_2}
    var_4 = 'orange'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 1
    var_9 = [var_0, var_8, var_1]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 'orange'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_in_nested_dict_access. Retrieved 17/25 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_invalid_index_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_default_value. Retrieved 18/26 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = [var_0, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 3

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'name'
    var_2 = 'price'
    var_3 = 'Apple'
    var_4 = 1.5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = 0
    var_9 = [var_0, var_8, var_1]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 'Apple'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_in_with_nested_dict. Retrieved 17/25 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_default. Retrieved 17/25 statements.
# Partially parsed test_get_in_missing_nested_key_returns_default. Retrieved 17/25 statements.
# Partially parsed test_get_in_out_of_bounds_index. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_custom_default. Retrieved 18/26 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'apple'
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.get_in(var_0, var_3, var_4)
    var_6 = bool(var_5 == {'a': 1})
    assert var_6 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_1, var_4]
    var_6 = [var_0, var_5]
    var_7 = 0
    var_8 = [var_0, var_0, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 3

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True
    var_7 = module_0.get_in(var_1, var_5, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_36_evaluates_to_false. Retrieved 10/16 statements.


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'purchase'
    var_1 = 'items'
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Apple'
    var_5 = 'Orange'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.get_in(var_3, var_8)
    assert var_9 == 'Apple'



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'purchase'
    var_1 = 'items'
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Apple'
    var_5 = 'Orange'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.get_in(var_3, var_8)
    assert var_9 == 'Apple'
    var_10 = bool(not False)
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_get_in_with_nested_dict. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_simple_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_invalid_index_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_custom_default. Retrieved 18/26 statements.
# Partially parsed test_get_in_no_default_raises_key_error. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

def test_case_0():
    var_0 = {}
    var_1 = 'y'
    var_2 = [var_1]
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_2, var_3]
    var_5 = [var_1, var_4]
    var_6 = [var_0, var_5]
    var_7 = 0
    var_8 = [var_0, var_0, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 3

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 10
    var_2 = 20
    var_3 = 'value'
    var_4 = 100
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = 2
    var_9 = [var_0, var_8, var_3]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 100



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_in_nested_dict_access. Retrieved 17/20 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/19 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/20 statements.
# Partially parsed test_get_in_missing_nested_key_returns_none. Retrieved 17/20 statements.
# Partially parsed test_get_in_out_of_bounds_index_returns_none. Retrieved 17/20 statements.
# Partially parsed test_get_in_missing_key_with_default. Retrieved 18/21 statements.
# Partially parsed test_get_in_missing_key_no_default_raises_keyerror. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'apple'
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

def test_case_0():
    var_0 = {}
    var_1 = 'y'
    var_2 = [var_1]
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_0, var_1]
    var_7 = module_0.get_in(var_6, var_5)
    assert var_7 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'custom_default'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'custom_default'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 0
    var_10 = 1
    var_11 = [var_0, var_9, var_10]
    var_12 = module_0.get_in(var_11, var_8)
    assert var_12 == 'b'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'items'
    var_7 = 10
    var_8 = [var_6, var_7]
    var_9 = True
    var_10 = module_0.get_in(var_8, var_5, no_default=var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'value'
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 'value'
    var_5 = 'nonexistent'
    var_6 = [var_5]
    var_7 = {var_0: var_2}
    var_8 = 'default_value'
    var_9 = module_0.get_in(var_6, var_7, var_8)
    assert var_9 == 'default_value'
    var_10 = 'x'
    var_11 = 'y'
    var_12 = [var_10, var_11]
    var_13 = {}
    var_14 = {var_10: var_13}
    var_15 = None
    var_16 = module_0.get_in(var_12, var_14, var_15)
    assert var_16 is None



# Parsed testcases at query #3
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'purchase'
    var_1 = 'items'
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Apple'
    var_5 = 'Orange'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.get_in(var_3, var_8)
    assert var_9 == 'Apple'



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'purchase'
    var_1 = 'items'
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Apple'
    var_5 = 'Orange'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.get_in(var_3, var_8)
    assert var_9 == 'Apple'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_in_with_nested_dict_and_list. Retrieved 17/25 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_invalid_list_index_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_default_value. Retrieved 18/26 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0.get_in(var_0, var_3, var_4)
    var_6 = bool(var_5 == {'a': 1})
    assert var_6 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]
    var_10 = [var_1, var_0]
    var_11 = module_0.get_in(var_10, var_9)
    assert var_11 == 6

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'name'
    var_2 = 'item1'
    var_3 = {var_1: var_2}
    var_4 = 'item2'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 1
    var_9 = [var_0, var_8, var_1]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 'item2'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_in_nested_dict_access. Retrieved 17/25 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_invalid_index_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_default_value. Retrieved 18/26 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = [var_0]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 10
    var_16 = [var_1, var_4, var_15]

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 'total'
    var_16 = [var_1, var_15]
    var_17 = 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = module_0.get_in(var_1, var_2, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = [var_0, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 3

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'name'
    var_2 = 'apple'
    var_3 = {var_1: var_2}
    var_4 = 'orange'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 1
    var_9 = [var_0, var_8, var_1]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 'orange'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'value'
    var_3 = {var_0: var_2}
    var_4 = 'default_value'
    var_5 = False
    var_6 = module_0.get_in(var_1, var_3, var_4, var_5)
    assert var_6 == 'value'
    var_7 = 'missing_key'
    var_8 = [var_7]
    var_9 = {var_0: var_2}
    var_10 = module_0.get_in(var_8, var_9, var_4, var_5)
    assert var_10 == 'default_value'
    var_11 = 10
    var_12 = [var_11]
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_0, var_13, var_14]
    var_16 = module_0.get_in(var_12, var_15, var_4, var_5)
    assert var_16 == 'default_value'
    var_17 = 'key'
    var_18 = [var_17]
    var_19 = 123
    var_20 = module_0.get_in(var_18, var_19, var_4, var_5)
    assert var_20 == 'default_value'



# Parsed testcases at query #8
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'credit card'
    var_3 = 'Alice'
    var_4 = 'items'
    var_5 = 'costs'
    var_6 = 'Apple'
    var_7 = 'Orange'
    var_8 = [var_6, var_7]
    var_9 = 0.5
    var_10 = 1.25
    var_11 = [var_9, var_10]
    var_12 = {var_4: var_8, var_5: var_11}
    var_13 = '5555-1234-1234-1234'
    var_14 = {var_0: var_3, var_1: var_12, var_2: var_13}
    var_15 = 0
    var_16 = [var_1, var_4, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 'Alice'
    var_20 = 1
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 == 'Orange'
    var_23 = [var_1, var_5, var_15]
    var_24 = module_0.get_in(var_23, var_14)
    var_25 = bool(var_24 == 0.5)
    assert var_25 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'purchase'
    var_2 = 'Alice'
    var_3 = 'items'
    var_4 = 'Apple'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 'total'
    var_9 = [var_1, var_8]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 is None
    var_11 = 'apple'
    var_12 = [var_1, var_3, var_11]
    var_13 = module_0.get_in(var_12, var_7)
    assert var_13 is None
    var_14 = 10
    var_15 = [var_1, var_3, var_14]
    var_16 = module_0.get_in(var_15, var_7)
    assert var_16 is None

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'purchase'
    var_1 = 'items'
    var_2 = 'Apple'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'total'
    var_7 = [var_0, var_6]
    var_8 = 0
    var_9 = module_0.get_in(var_7, var_5, var_8)
    assert var_9 == 0
    var_10 = 'nonexistent'
    var_11 = [var_10]
    var_12 = 'default_value'
    var_13 = module_0.get_in(var_11, var_5, var_12)
    assert var_13 == 'default_value'

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'y'
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.get_in(var_2, var_0, no_default=var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'Apple'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = 'items'
    var_5 = 10
    var_6 = [var_4, var_5]
    var_7 = True
    var_8 = module_0.get_in(var_6, var_3, no_default=var_7)
    var_9 = bool(False)
    assert var_9 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.get_in(var_3, var_2)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = [var_7, var_7]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 'a'
    var_10 = 1
    var_11 = [var_10, var_10]
    var_12 = module_0.get_in(var_11, var_6)
    assert var_12 == 'd'
    var_13 = 2
    var_14 = [var_7, var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None



