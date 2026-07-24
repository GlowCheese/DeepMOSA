####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_20 = 'total'
    var_21 = [var_1, var_20]
    var_22 = module_0.get_in(var_21, var_14)
    assert var_22 is None
    var_23 = 'apple'
    var_24 = [var_1, var_4, var_23]
    var_25 = module_0.get_in(var_24, var_14)
    assert var_25 is None
    var_26 = 10
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_14)
    assert var_28 is None
    var_29 = [var_1, var_20]
    var_30 = module_0.get_in(var_29, var_14, var_15)
    assert var_30 == 0

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_0, var_1]
    var_6 = module_0.get_in(var_5, var_4)
    assert var_6 == 1
    var_7 = 'c'
    var_8 = [var_0, var_7]
    var_9 = 'default_val'
    var_10 = module_0.get_in(var_8, var_4, var_9)
    assert var_10 == 'default_val'
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = 42
    var_14 = module_0.get_in(var_12, var_4, var_13)
    assert var_14 == 42

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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = [var_4]
    var_6 = True
    var_7 = module_0.get_in(var_5, var_3, no_default=var_6)
    var_8 = bool(False)
    assert var_8 is True

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
    var_0 = 'items'
    var_1 = 'name'
    var_2 = 'Apple'
    var_3 = {var_1: var_2}
    var_4 = 'Orange'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 0
    var_9 = [var_0, var_8, var_1]
    var_10 = module_0.get_in(var_9, var_7)
    assert var_10 == 'Apple'
    var_11 = 1
    var_12 = [var_0, var_11, var_1]
    var_13 = module_0.get_in(var_12, var_7)
    assert var_13 == 'Orange'
    var_14 = 2
    var_15 = [var_0, var_14, var_1]
    var_16 = module_0.get_in(var_15, var_7)
    assert var_16 is None



# Parsed testcases at query #2
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
    var_10 = 'name'
    var_11 = [var_10]
    var_12 = 'Alice'
    var_13 = {var_10: var_12}
    var_14 = module_0.get_in(var_11, var_13)
    assert var_14 == 'Alice'
    var_15 = 'nonexistent'
    var_16 = [var_15]
    var_17 = {}
    var_18 = 'default_value'
    var_19 = module_0.get_in(var_16, var_17, var_18)
    assert var_19 == 'default_value'



# Parsed testcases at query #4
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'value'
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 'value'
    var_5 = 'outer'
    var_6 = 'inner'
    var_7 = 'found'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = [var_5, var_6]
    var_11 = module_0.get_in(var_10, var_9)
    assert var_11 == 'found'
    var_12 = 'items'
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_12: var_16}
    var_18 = 0
    var_19 = [var_12, var_18]
    var_20 = module_0.get_in(var_19, var_17)
    assert var_20 == 1



# Parsed testcases at query #5
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'purchase'
    var_1 = 'items'
    var_2 = 0
    var_3 = [var_0, var_1, var_2]
    var_4 = 'name'
    var_5 = 'credit card'
    var_6 = 'Alice'
    var_7 = 'costs'
    var_8 = 'Apple'
    var_9 = 'Orange'
    var_10 = [var_8, var_9]
    var_11 = 0.5
    var_12 = 1.25
    var_13 = [var_11, var_12]
    var_14 = {var_1: var_10, var_7: var_13}
    var_15 = '5555-1234-1234-1234'
    var_16 = {var_4: var_6, var_0: var_14, var_5: var_15}
    var_17 = module_0.get_in(var_3, var_16)
    assert var_17 == 'Apple'



# Parsed testcases at query #6
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'value'
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 'value'
    var_5 = 'outer'
    var_6 = 'inner'
    var_7 = 'found'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = [var_5, var_6]
    var_11 = module_0.get_in(var_10, var_9)
    assert var_11 == 'found'
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = [var_12]
    var_17 = module_0.get_in(var_16, var_15)
    assert var_17 == 2



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'nonexistent_key'
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'default_value'
    var_6 = False
    var_7 = module_0.get_in(var_1, var_4, var_5, var_6)
    assert var_7 == 'default_value'
    var_8 = 10
    var_9 = [var_8]
    var_10 = 2
    var_11 = 3
    var_12 = [var_3, var_10, var_11]
    var_13 = module_0.get_in(var_9, var_12, var_5, var_6)
    assert var_13 == 'default_value'
    var_14 = 'key'
    var_15 = [var_14]
    var_16 = None
    var_17 = module_0.get_in(var_15, var_16, var_5, var_6)
    assert var_17 == 'default_value'



# Parsed testcases at query #8
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_in_nested_dict_access. Retrieved 17/20 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/19 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/20 statements.
# Partially parsed test_get_in_invalid_list_index_returns_none. Retrieved 17/20 statements.
# Partially parsed test_get_in_out_of_bounds_index_returns_none. Retrieved 17/20 statements.
# Partially parsed test_get_in_with_default_value. Retrieved 18/21 statements.


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
    var_0 = 'data'
    var_1 = 'id'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = [var_0, var_2, var_1]
    var_9 = module_0.get_in(var_8, var_7)
    assert var_9 == 2

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_6]
    var_8 = True
    var_9 = module_0.get_in(var_2, var_7, no_default=var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 123
    var_4 = True
    var_5 = module_0.get_in(var_2, var_3, no_default=var_4)
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_3]
    var_5 = 'custom'
    var_6 = module_0.get_in(var_4, var_2, var_5)
    assert var_6 == 'custom'



# Parsed testcases at query #2
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'fallback'
    var_6 = False
    var_7 = module_0.get_in(var_1, var_4, var_5, var_6)
    assert var_7 == 'fallback'
    var_8 = False
    assert var_8 is False



# Parsed testcases at query #3
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = 'value'
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 'value'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = 'found'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {var_5: var_10}
    var_12 = [var_5, var_6, var_7]
    var_13 = module_0.get_in(var_12, var_11)
    assert var_13 == 'found'
    var_14 = 'items'
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_14: var_18}
    var_20 = [var_14, var_15]
    var_21 = module_0.get_in(var_20, var_19)
    assert var_21 == 2



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_in_nested_dict_access. Retrieved 17/25 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_default. Retrieved 17/25 statements.
# Partially parsed test_get_in_invalid_index_returns_default. Retrieved 17/25 statements.
# Partially parsed test_get_in_out_of_bounds_index_returns_default. Retrieved 17/25 statements.
# Partially parsed test_get_in_with_custom_default. Retrieved 18/26 statements.
# Partially parsed test_get_in_empty_keys. Retrieved 4/12 statements.


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
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = {var_0: var_1}
    var_3 = []

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_0, var_1, var_5]
    var_7 = [var_1, var_0]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 4

import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 2
    var_7 = [var_0, var_6]
    var_8 = module_0.get_in(var_7, var_5)
    assert var_8 == 30



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_36_evaluates_to_false. Retrieved 11/18 statements.


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
    var_10 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_in_nested_dict_access. Retrieved 17/25 statements.
# Partially parsed test_get_in_single_key. Retrieved 16/24 statements.
# Partially parsed test_get_in_missing_key_returns_none. Retrieved 17/25 statements.
# Partially parsed test_get_in_invalid_list_index. Retrieved 17/25 statements.
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
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = [var_7, var_0]
    var_9 = module_0.get_in(var_8, var_6)
    assert var_9 == 1



# Parsed testcases at query #7
#--------------------------




import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'value'
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 'value'
    var_5 = 'x'
    var_6 = 'y'
    var_7 = [var_5, var_6]
    var_8 = 42
    var_9 = {var_6: var_8}
    var_10 = {var_5: var_9}
    var_11 = module_0.get_in(var_7, var_10)
    assert var_11 == 42
    var_12 = 0
    var_13 = 1
    var_14 = [var_12, var_13]
    var_15 = 2
    var_16 = 3
    var_17 = [var_13, var_15, var_16]
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = [var_18, var_19, var_20]
    var_22 = [var_17, var_21]
    var_23 = module_0.get_in(var_14, var_22)
    assert var_23 == 2



# Parsed testcases at query #8
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



