####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/14 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_single_element. Retrieved 7/10 statements.
# Partially parsed test_pmap_constructor_multiple_elements. Retrieved 17/20 statements.
# Partially parsed test_pmap_constructor_weakref_support. Retrieved 7/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8, var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = [var_3, var_4, var_4]
    var_6 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'd'
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'b'
    var_8 = 2
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = None
    var_12 = 'c'
    var_13 = 3
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = [var_6, var_10, var_11, var_15]

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 12/17 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_large. Retrieved 6/15 statements.
# Partially parsed test_pmap_constructor_slots. Retrieved 10/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_0, var_8]
    var_10 = '__weakref__'
    var_11 = '_cached_hash'

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = [var_1]
    var_5 = 10

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 1
    var_7 = '_size'
    var_8 = '_buckets'
    var_9 = '__weakref__'
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pmap_eq_pmap_vs_non_mapping. Retrieved 5/7 statements.
# Partially parsed test_pmap_eq_same_buckets. Retrieved 4/6 statements.
# Partially parsed test_pmap_eq_pmap_vs_custom_mapping. Retrieved 6/18 statements.
# Partially parsed test_pmap_eq_pmap_vs_custom_mapping_different_content. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = module_0.m(**var_7)
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = hash(var_5)
    var_12 = hash(var_10)
    var_13 = bool(not var_5 == var_10)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = hash(var_5)
    var_11 = hash(var_9)
    var_12 = bool(var_5 == var_9)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 22/30 statements.


def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = '__weakref__'
    var_8 = [var_0, var_0, var_0]
    var_9 = 0
    var_10 = 'a'
    var_11 = 1
    var_12 = (var_10, var_11)
    var_13 = [var_12]
    var_14 = 'b'
    var_15 = 2
    var_16 = (var_14, var_15)
    var_17 = 'c'
    var_18 = 3
    var_19 = (var_17, var_18)
    var_20 = [var_16, var_19]
    var_21 = [var_13, var_20, var_0]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 7/9 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 2/5 statements.
# Partially parsed test_pmap_items_eq_different_keys. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = {var_4: var_1}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_nonexistent_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_argument. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_key. Retrieved 4/6 statements.
# Partially parsed test_pmap_items_contains_with_none_value. Retrieved 4/6 statements.
# Partially parsed test_pmap_items_contains_with_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_list_unpacking_failure. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'not_a_tuple'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = (var_6,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = None
    var_5 = 'value'
    var_6 = (var_4, var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'a'
    var_5 = None
    var_6 = (var_4, var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]



# Parsed testcases at query #7
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = var_6['a']
    assert var_7 == 1
    var_8 = var_6['b']
    assert var_8 == 2
    var_9 = len(var_6)
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 16
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4['a']
    assert var_5 == 1
    var_6 = len(var_4)
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8['x']
    assert var_9 == 10
    var_10 = var_8['y']
    assert var_10 == 20
    var_11 = len(var_8)
    assert var_11 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4['key']
    assert var_5 == 'value'
    var_6 = len(var_4)
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = None
    var_10 = module_0._turbo_mapping(var_8, var_9)
    var_11 = var_10['a']
    assert var_11 == 1
    var_12 = var_10['b']
    assert var_12 == 2
    var_13 = var_10['c']
    assert var_13 == 3
    var_14 = var_10['d']
    assert var_14 == 4
    var_15 = len(var_10)
    assert var_15 == 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 32
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = var_6['a']
    assert var_7 == 1
    var_8 = var_6['b']
    assert var_8 == 2
    var_9 = len(var_6)
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 42
    var_5 = None
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._turbo_mapping(var_6, var_5)
    var_8 = var_7['key1']
    assert var_8 == 'value1'
    var_9 = var_7['key2']
    assert var_9 == 42
    var_10 = var_7['key3']
    assert var_10 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4['a']
    assert var_5 == 1
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_contains_predicate_line_4_evaluates_to_false. Retrieved 13/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 999
    var_7 = (var_0, var_6)
    var_8 = 'nonexistent'
    var_9 = (var_8, var_2)
    var_10 = 'x'
    var_11 = 99
    var_12 = (var_10, var_11)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 8/10 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_new_keys. Retrieved 7/9 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/7 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 5/8 statements.
# Partially parsed test_update_with_dict. Retrieved 9/11 statements.
# Partially parsed test_update_with_multiple_maps_rightmost_wins. Retrieved 10/12 statements.
# Partially parsed test_update_with_custom_merge_function. Retrieved 10/12 statements.
# Partially parsed test_update_with_no_maps. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r
    var_7 = 'a'
    var_8 = {var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = 'a'
    var_11 = 3
    var_12 = {var_10: var_11}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = 'a'
    var_10 = 3
    var_11 = {var_9: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r
    var_7 = 3
    var_8 = 4
    var_9 = 'c'
    var_10 = 'd'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r
    var_7 = {}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'c'
    var_9 = {var_7: var_1, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r
    var_7 = 'a'
    var_8 = 'c'
    var_9 = 10
    var_10 = 3
    var_11 = {var_7: var_9, var_8: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r
    var_5 = 2
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = 3
    var_10 = 'a'
    var_11 = {var_10: var_9}
    var_12 = module_0.m(**var_11)
    var_13 = 'a'
    var_14 = 4
    var_15 = {var_13: var_14}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = lambda l, r: l + r
    var_6 = 2
    var_7 = [var_6]
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 3
    var_12 = [var_11]
    var_13 = 'a'
    var_14 = {var_13: var_12}
    var_15 = module_0.m(**var_14)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r
    var_11 = 'c'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/14 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_large. Retrieved 18/22 statements.
# Partially parsed test_pmap_constructor_preserves_buckets_reference. Retrieved 7/10 statements.
# Partially parsed test_pmap_constructor_with_weakref_slot. Retrieved 9/14 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8, var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = (var_7, var_8)
    var_10 = [var_6, var_9]
    var_11 = None
    var_12 = 'key4'
    var_13 = 'value4'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = [var_3, var_10, var_11, var_15, var_11]
    var_17 = 4

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = '_size'
    var_8 = '_buckets'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/6 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 6/8 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/9 statements.
# Partially parsed test_update_with_new_keys. Retrieved 6/8 statements.
# Partially parsed test_update_with_empty_map. Retrieved 4/6 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 4/6 statements.
# Partially parsed test_update_with_dict. Retrieved 7/9 statements.
# Partially parsed test_update_with_custom_function. Retrieved 7/8 statements.
# Partially parsed test_update_with_rightmost_wins. Retrieved 8/9 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)
    var_9 = 3
    var_10 = 'a'
    var_11 = {var_10: var_9}
    var_12 = module_0.m(**var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = 'a'
    var_10 = 3
    var_11 = {var_9: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 4
    var_8 = 'c'
    var_9 = 'd'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.m(**var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = {}
    var_7 = module_0.m(**var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_6: var_1, var_7: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 5
    var_8 = 10
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r
    var_5 = 2
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = 'a'
    var_10 = 3
    var_11 = {var_9: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 3
    var_9 = 'c'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 22/35 statements.


def test_case_0():
    var_0 = 0
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_4]
    var_6 = 1
    var_7 = 'a'
    var_8 = (var_7, var_6)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)
    var_12 = (var_8, var_11)
    var_13 = [var_12]
    var_14 = 'c'
    var_15 = 3
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_13, var_17]
    var_19 = '_size'
    var_20 = '_buckets'
    var_21 = '_cached_hash'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_key_value_pair. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_key_value_pair. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_nonexistent_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_argument. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_values. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_pmap. Retrieved 2/4 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = (var_6,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = None
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 19/28 statements.


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 'a'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = None
    var_6 = 2
    var_7 = 'b'
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = [var_4, var_5, var_9]
    var_11 = '__weakref__'
    var_12 = '_cached_hash'
    var_13 = 42
    var_14 = 5
    var_15 = 'value'
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_5, var_17]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pmap_items_eq_same_object. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_objects_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 7/10 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 4/7 statements.
# Partially parsed test_pmap_items_eq_with_string. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_missing_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_value. Retrieved 4/6 statements.
# Partially parsed test_pmap_items_contains_with_complex_value. Retrieved 7/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'not_a_tuple'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = (var_6,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'a'
    var_5 = None
    var_6 = (var_4, var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = 'a'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = (var_7, var_11)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 5/7 statements.
# Partially parsed test_pmap_eq_with_generic_mapping. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(not var_5 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'world'
    var_1 = 'bar'
    var_2 = 'hello'
    var_3 = 'foo'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'hello'
    var_7 = 'foo'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = -3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.m(**var_6)
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = -3
    var_12 = {var_8: var_0, var_9: var_1, var_10: var_11}
    var_13 = bool(var_7 == var_12)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(not var_5 != var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_6: var_0, var_7: var_1, var_8: var_9}
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 12/17 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_large_map. Retrieved 18/23 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8, var_0]
    var_10 = '__weakref__'
    var_11 = '_cached_hash'

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'key2'
    var_5 = 'val2'
    var_6 = (var_4, var_5)
    var_7 = 'key3'
    var_8 = 'val3'
    var_9 = (var_7, var_8)
    var_10 = [var_6, var_9]
    var_11 = None
    var_12 = 'key4'
    var_13 = 'val4'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = [var_3, var_10, var_11, var_15]
    var_17 = 4



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_contains_returns_false_on_unpacking_exception. Retrieved 11/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'not_a_tuple'
    var_7 = 42
    var_8 = 3
    var_9 = [var_2, var_3, var_8]
    var_10 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap_instance. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = bool(var_2 == {})
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = bool(var_4 == {'a': 1})
    assert var_6 is True
    var_7 = var_4['a']
    assert var_7 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_10 is True
    var_11 = var_8['a']
    assert var_11 == 1
    var_12 = var_8['b']
    assert var_12 == 2
    var_13 = var_8['c']
    assert var_13 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = bool(var_8 == {'x': 10, 'y': 20})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8['key1']
    assert var_9 == 'value1'
    var_10 = var_8['key2']
    assert var_10 == 'value2'
    var_11 = var_8['key3']
    assert var_11 == 'value3'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 256
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'one'
    var_1 = 'two'
    var_2 = 'three'
    var_3 = 'four'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 0
    var_10 = module_0._turbo_mapping(var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = var_10['one']
    assert var_12 == 1
    var_13 = var_10['two']
    assert var_13 == 2
    var_14 = var_10['three']
    assert var_14 == 3
    var_15 = var_10['four']
    assert var_15 == 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'two'
    var_2 = 3
    var_3 = (var_2, var_2)
    var_4 = 'one'
    var_5 = 2
    var_6 = 'three'
    var_7 = {var_0: var_4, var_1: var_5, var_3: var_6}
    var_8 = 0
    var_9 = module_0._turbo_mapping(var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_9[1]
    assert var_11 == 'one'
    var_12 = var_9['two']
    assert var_12 == 2
    var_13 = var_9[3, 3]
    assert var_13 == 'three'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l
    var_7 = 3
    var_8 = 'c'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_keys. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_with_dict_items. Retrieved 7/10 statements.
# Partially parsed test_pmap_items_eq_with_none. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_eq_predicate_line_3_returns_false. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'not a PMapItems'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_contains_predicate_evaluates_to_false. Retrieved 12/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 999
    var_7 = (var_0, var_6)
    var_8 = 'c'
    var_9 = (var_8, var_2)
    var_10 = 'z'
    var_11 = (var_10, var_6)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_eq_pmap_vs_dict_like_mapping. Retrieved 6/18 statements.
# Partially parsed test_eq_pmap_vs_custom_mapping_different_content. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = bool(not var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]
    var_8 = var_5 == var_7

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = hash(var_5)
    var_11 = hash(var_9)
    var_12 = bool(var_5 == var_9)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = hash(var_5)
    var_12 = hash(var_10)
    var_13 = bool(not var_5 == var_10)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 'z'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.m(**var_6)
    var_8 = bool(var_7 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = var_5 == var_9
    var_11 = bool((var_5 == var_9) == (var_9 == var_5))
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.m(**var_12)
    var_14 = bool(var_5 == var_9)
    assert var_14 is True
    var_15 = bool(var_9 == var_13)
    assert var_15 is True
    var_16 = bool(var_5 == var_13)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'not a map'
    var_5 = var_3 == var_4



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/14 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_large. Retrieved 18/22 statements.
# Partially parsed test_pmap_constructor_returns_instance. Retrieved 9/15 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8, var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = 'key3'
    var_9 = 'value3'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = 'key4'
    var_13 = 'value4'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = [var_3, var_4, var_11, var_4, var_15]
    var_17 = 4

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 1
    var_7 = '_size'
    var_8 = '_buckets'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_contains_returns_false_on_exception. Retrieved 11/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'single_value'
    var_7 = 42
    var_8 = None
    var_9 = 3
    var_10 = [var_2, var_3, var_9]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 7/9 statements.
# Partially parsed test_pmap_eq_with_mapping_protocol. Retrieved 7/19 statements.
# Partially parsed test_pmap_eq_with_mapping_protocol_different. Retrieved 9/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_0: var_2, var_1: var_3, var_6: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'not a mapping'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = module_0.pmap()
    var_2 = bool(var_0 == var_1)
    assert var_2 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = {}
    var_2 = bool(var_0 == var_1)
    assert var_2 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = bool(not var_3 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_0: var_2, var_6: var_7}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_value. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_value. Retrieved 4/6 statements.
# Partially parsed test_pmap_items_contains_with_empty_map. Retrieved 2/4 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 99
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 1
    var_7 = (var_6,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'a'
    var_5 = None
    var_6 = (var_4, var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_turbo_mapping_exception_handler. Retrieved 3/13 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pmap_eq_non_mapping. Retrieved 5/7 statements.
# Partially parsed test_pmap_eq_pmap_vs_regular_mapping. Retrieved 8/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(not var_5 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_0, var_10: var_1}
    var_12 = module_0.m(**var_11)
    var_13 = bool(var_5 == var_12)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = (var_6, var_0)
    var_8 = 'b'
    var_9 = (var_8, var_1)
    var_10 = [var_7, var_9]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(not var_5 != var_9)
    assert var_10 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_turbo_mapping_exception_handling. Retrieved 3/11 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_turbo_mapping_exception_handler. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_keys. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 4/7 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_type_dict_items. Retrieved 7/10 statements.
# Partially parsed test_pmap_items_eq_different_type_none. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_type_string. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_key_value_pair. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_key_value_pair. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_nonexistent_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_argument. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_values. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_matching_pair. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = (var_6,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = None
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'foo'
    var_2 = 'y'
    var_3 = 'bar'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'x'
    var_7 = 'y'
    var_8 = (var_6, var_7)
    var_9 = 'foo'
    var_10 = 'bar'
    var_11 = (var_9, var_10)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_eq_pmap_vs_non_mapping. Retrieved 4/6 statements.
# Partially parsed test_eq_pmap_vs_mapping_protocol. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'not a mapping'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(not var_5 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = hash(var_5)
    var_11 = hash(var_9)
    var_12 = bool(var_5 == var_9)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 4
    var_8 = 'c'
    var_9 = 'd'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = hash(var_5)
    var_13 = hash(var_11)
    var_14 = bool(not var_5 == var_11)
    assert var_14 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_contains_predicate_line_4_evaluates_to_false. Retrieved 13/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 999
    var_7 = (var_0, var_6)
    var_8 = 'nonexistent'
    var_9 = (var_8, var_2)
    var_10 = 'x'
    var_11 = 99
    var_12 = (var_10, var_11)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 12/16 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_large_map. Retrieved 24/29 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8, var_0]
    var_10 = 2
    var_11 = '__weakref__'

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'val2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'key3'
    var_8 = 'val3'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = None
    var_12 = 'key4'
    var_13 = 'val4'
    var_14 = (var_12, var_13)
    var_15 = 'key5'
    var_16 = 'val5'
    var_17 = (var_15, var_16)
    var_18 = 'key6'
    var_19 = 'val6'
    var_20 = (var_18, var_19)
    var_21 = [var_14, var_17, var_20]
    var_22 = [var_6, var_10, var_11, var_21, var_11]
    var_23 = 6



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_large_map. Retrieved 24/28 statements.
# Partially parsed test_pmap_constructor_attributes. Retrieved 10/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8, var_0]
    var_10 = 2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'key3'
    var_8 = 'value3'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = None
    var_12 = 'key4'
    var_13 = 'value4'
    var_14 = (var_12, var_13)
    var_15 = 'key5'
    var_16 = 'value5'
    var_17 = (var_15, var_16)
    var_18 = 'key6'
    var_19 = 'value6'
    var_20 = (var_18, var_19)
    var_21 = [var_14, var_17, var_20]
    var_22 = [var_6, var_10, var_11, var_21]
    var_23 = 6

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = '_size'
    var_8 = '_buckets'
    var_9 = '_cached_hash'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_contains_predicate_line_4_evaluates_to_false. Retrieved 11/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 999
    var_7 = (var_0, var_6)
    var_8 = 'z'
    var_9 = (var_8, var_2)
    var_10 = (var_8, var_6)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_turbo_mapping_exception_predicate_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_pmap_items_eq_same_object. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_objects_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_keys. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_not_pmap_items_type. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_with_dict_items. Retrieved 7/10 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 2/5 statements.
# Partially parsed test_pmap_items_eq_one_empty_one_not. Retrieved 5/8 statements.
# Partially parsed test_pmap_items_eq_with_none_values. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_with_string. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_contains_returns_false_on_unpacking_exception. Retrieved 13/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'single_value'
    var_7 = (var_6,)
    var_8 = 'string'
    var_9 = 42
    var_10 = 3
    var_11 = [var_2, var_3, var_10]
    var_12 = None



# Parsed testcases at query #47
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'c'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_eq_predicate_line_3_evaluates_to_true. Retrieved 8/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = 'not a PMapItems'
    var_8 = [var_6]
    var_9 = var_6 == var_7
    assert var_9 is False
    var_10 = var_6 == {'a': 1, 'b': 2}
    assert var_10 is False
    var_11 = var_6 == 42
    assert var_11 is False
    var_12 = var_6 == None
    assert var_12 is False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap_instance. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = bool(var_2 == {})
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1
    var_7 = bool(var_4 == {'a': 1})
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3
    var_13 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 32
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['b']
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['x']
    assert var_10 == 10
    var_11 = var_8['y']
    assert var_11 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = None
    var_7 = module_0._turbo_mapping(var_5, var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7['a']
    assert var_9 == 1
    var_10 = var_7['b']
    assert var_10 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 'e'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = None
    var_12 = module_0._turbo_mapping(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = var_12['a']
    assert var_14 == 1
    var_15 = var_12['e']
    assert var_15 == 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 1024
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 42
    var_5 = None
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._turbo_mapping(var_6, var_5)
    var_8 = var_7['key1']
    assert var_8 == 'value1'
    var_9 = var_7['key2']
    assert var_9 == 42
    var_10 = var_7['key3']
    assert var_10 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = None
    var_7 = module_0._turbo_mapping(var_5, var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7['a']
    assert var_9 == 2



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_contains_predicate_line_4_evaluates_to_false. Retrieved 11/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 999
    var_7 = (var_0, var_6)
    var_8 = 'nonexistent'
    var_9 = (var_8, var_2)
    var_10 = (var_8, var_6)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_value. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_list. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_wrong_length_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_none_values. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 999
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = [var_6, var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = None
    var_8 = (var_6, var_7)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pmap_eq_with_dict_predicate. Retrieved 8/22 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/14 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_single_element. Retrieved 6/10 statements.
# Partially parsed test_pmap_constructor_multiple_collisions. Retrieved 12/16 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_8, var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = 1

def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = 'key3'
    var_8 = 3
    var_9 = (var_7, var_8)
    var_10 = [var_3, var_6, var_9]
    var_11 = [var_0, var_10, var_0]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_large. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_with_collisions. Retrieved 13/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 2
    var_5 = 'b'
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = None
    var_9 = [var_3, var_7, var_8]

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = [[(i, f'val_{i}')] for i in var_1]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = 3
    var_9 = 'c'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = [var_6, var_7, var_11]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_missing_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_arg. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_values. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_list_as_arg. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = (var_6,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = None
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = [var_6, var_7]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_contains_predicate_evaluates_to_false. Retrieved 12/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 999
    var_7 = (var_0, var_6)
    var_8 = 'c'
    var_9 = (var_8, var_2)
    var_10 = 'z'
    var_11 = (var_10, var_6)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pmap_eq_with_non_dict_mapping. Retrieved 7/22 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_turbo_mapping_preserves_values. Retrieved 9/12 statements.
# Partially parsed test_turbo_mapping_returns_pmap_instance. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = var_2._size
    assert var_4 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 32
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['b']
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['x']
    assert var_10 == 10
    var_11 = var_8['y']
    assert var_11 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'one'
    var_4 = 'two'
    var_5 = 'three'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8[1]
    assert var_10 == 'one'
    var_11 = var_8[2]
    assert var_11 == 'two'
    var_12 = var_8[3]
    assert var_12 == 'three'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_2)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5['a']
    assert var_7 is None
    var_8 = var_5['b']
    assert var_8 == 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {f'key_{i}': i for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 5
    var_4 = {var_0: var_3, var_1: var_3, var_2: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = var_6['a']
    assert var_8 == 5
    var_9 = var_6['b']
    assert var_9 == 5
    var_10 = var_6['c']
    assert var_10 == 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 128
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1



# Parsed testcases at query #61
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = 'not a PMapItems object'
    var_8 = var_6 == var_7
    assert var_8 is False
    var_9 = 42
    var_10 = var_6 == var_9
    assert var_10 is False
    var_11 = {var_0: var_2, var_1: var_3}
    var_12 = var_6 == var_11
    assert var_12 is False
    var_13 = (var_0, var_2)
    var_14 = (var_1, var_3)
    var_15 = [var_13, var_14]
    var_16 = var_6 == var_15
    assert var_16 is False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 23/37 statements.


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_1, var_5, var_1]
    var_7 = 1
    var_8 = (var_2, var_3)
    var_9 = [var_8]
    var_10 = 'key2'
    var_11 = 'value2'
    var_12 = (var_10, var_11)
    var_13 = 'key3'
    var_14 = 'value3'
    var_15 = (var_13, var_14)
    var_16 = [var_12, var_15]
    var_17 = [var_9, var_16, var_1]
    var_18 = 3
    var_19 = '_size'
    var_20 = '_buckets'
    var_21 = '__weakref__'
    var_22 = '_cached_hash'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 6/9 statements.
# Partially parsed test_update_with_keep_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_new_keys. Retrieved 5/8 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 5/8 statements.
# Partially parsed test_update_with_custom_function. Retrieved 9/11 statements.
# Partially parsed test_update_with_dict. Retrieved 8/11 statements.
# Partially parsed test_update_with_multiple_dicts. Retrieved 8/10 statements.
# Partially parsed test_update_with_overwrite_function. Retrieved 7/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'c'
    var_9 = {var_7: var_1, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = {var_11: var_0}
    var_13 = module_0.m(**var_12)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = 'a'
    var_10 = 3
    var_11 = {var_9: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 3
    var_6 = 'b'
    var_7 = 'c'
    var_8 = {var_6: var_4, var_7: var_5}
    var_9 = module_0.m(**var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 5
    var_7 = 'a'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_2}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = 'a'
    var_11 = {var_10: var_9}
    var_12 = module_0.m(**var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 3
    var_9 = 4
    var_10 = {var_6: var_8, var_7: var_9}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r
    var_5 = 'a'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 3
    var_9 = {var_5: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l * r
    var_7 = 2
    var_8 = 5
    var_9 = 'x'
    var_10 = 'z'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_instances_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items_type. Retrieved 7/11 statements.
# Partially parsed test_pmap_items_eq_with_string. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_with_none. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_eq_pmap_vs_non_mapping. Retrieved 5/7 statements.
# Partially parsed test_eq_pmap_vs_custom_mapping. Retrieved 6/18 statements.
# Partially parsed test_eq_pmap_vs_custom_mapping_different. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(not var_5 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(not var_5 == [1, 2, 3])
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = hash(var_5)
    var_11 = hash(var_9)
    var_12 = bool(var_5 == var_9)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = hash(var_5)
    var_12 = hash(var_10)
    var_13 = bool(not var_5 == var_10)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l + r
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/13 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_multiple_buckets. Retrieved 15/20 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = 2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'c'
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = [var_6]
    var_8 = 'b'
    var_9 = 2
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = None
    var_13 = [var_7, var_11, var_12]
    var_14 = 3



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pmap_items_contains_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_missing_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_non_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_single_value. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_triple_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_empty_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_none_values. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_multiple_valid_items. Retrieved 8/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = ()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = None
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 'x'
    var_9 = 10
    var_10 = (var_8, var_9)
    var_11 = 'y'
    var_12 = 20
    var_13 = (var_11, var_12)
    var_14 = 'z'
    var_15 = 30
    var_16 = (var_14, var_15)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 7/10 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap_instance. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = bool(var_2 == {})
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1
    var_7 = bool(var_4 == {'a': 1})
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3
    var_13 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['b']
    assert var_9 == 2
    var_10 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['x']
    assert var_10 == 10
    var_11 = var_8['y']
    assert var_11 == 20
    var_12 = bool(var_8 == {'x': 10, 'y': 20})
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 'p'
    var_3 = 'q'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = None
    var_7 = module_0._turbo_mapping(var_5, var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7['p']
    assert var_9 == 100
    var_10 = var_7['q']
    assert var_10 == 200
    var_11 = bool(var_7 == {'p': 100, 'q': 200})
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 8
    var_10 = module_0._turbo_mapping(var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = var_10['a']
    assert var_12 == 1
    var_13 = var_10['b']
    assert var_13 == 2
    var_14 = var_10['c']
    assert var_14 == 3
    var_15 = var_10['d']
    assert var_15 == 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = None
    var_5 = 0
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._turbo_mapping(var_6, var_4)
    var_8 = var_7['key1']
    assert var_8 == 'value1'
    var_9 = var_7['key2']
    assert var_9 is None
    var_10 = var_7['key3']
    assert var_10 == 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pmap_items_eq. Retrieved 16/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_0: var_2, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = {}
    var_13 = module_0.pmap(var_12)
    var_14 = {}
    var_15 = module_0.pmap(var_14)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pmap_eq_with_generic_mapping. Retrieved 3/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(not var_5 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != 'not a mapping')
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.m(**var_5)
    var_7 = bool(var_3 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = bool(not var_3 == var_7)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pmap_eq_with_non_dict_mapping. Retrieved 7/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_turbo_mapping_exception_handler. Retrieved 3/11 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_contains_returns_false_on_unpacking_exception. Retrieved 11/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'not_a_tuple'
    var_7 = [var_2]
    var_8 = None
    var_9 = 42
    var_10 = 'single_string'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 5/7 statements.
# Partially parsed test_pmap_eq_with_generic_mapping. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(not var_5 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.m(**var_6)
    var_8 = 'c'
    var_9 = 'b'
    var_10 = 'a'
    var_11 = {var_8: var_2, var_9: var_1, var_10: var_0}
    var_12 = module_0.m(**var_11)
    var_13 = bool(var_7 == var_12)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_6: var_0, var_7: var_1, var_8: var_9}
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_turbo_mapping_with_mapping_object. Retrieved 8/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = var_6['a']
    assert var_7 == 1
    var_8 = var_6['b']
    assert var_8 == 2
    var_9 = len(var_6)
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = var_6['a']
    assert var_7 == 1
    var_8 = var_6['b']
    assert var_8 == 2
    var_9 = len(var_6)
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8['a']
    assert var_9 == 1
    var_10 = var_8['b']
    assert var_10 == 2
    var_11 = len(var_8)
    assert var_11 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4['key']
    assert var_5 == 'value'
    var_6 = len(var_4)
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 32
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4['a']
    assert var_5 == 1
    var_6 = len(var_4)
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = 'a'
    var_8 = bool('a' in var_6)
    assert var_8 is True
    var_9 = 'b'
    var_10 = bool('b' in var_6)
    assert var_10 is True
    var_11 = 'c'
    var_12 = bool('c' not in var_6)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'z'
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8['x']
    assert var_9 == 10
    var_10 = var_8['y']
    assert var_10 == 20
    var_11 = var_8['z']
    assert var_11 == 30
    var_12 = len(var_8)
    assert var_12 == 3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_contains_returns_false_on_exception. Retrieved 10/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 42
    var_7 = 'single'
    var_8 = 3
    var_9 = [var_2, var_3, var_8]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_update_with_predicate_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_contains_returns_false_on_unpacking_exception. Retrieved 11/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'single_value'
    var_7 = 42
    var_8 = None
    var_9 = 3
    var_10 = [var_2, var_3, var_9]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r
    var_11 = 'c'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_keys. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 7/11 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap_instance. Retrieved 5/6 statements.
# Partially parsed test_turbo_mapping_from_mapping_object. Retrieved 11/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = bool(var_2 == {})
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = bool(var_4 == {'a': 1})
    assert var_6 is True
    var_7 = var_4['a']
    assert var_7 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_10 is True
    var_11 = var_8['a']
    assert var_11 == 1
    var_12 = var_8['b']
    assert var_12 == 2
    var_13 = var_8['c']
    assert var_13 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = bool(var_6 == {'x': 10, 'y': 20})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = bool(var_8 == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {f'key{i}': i for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 'e'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = 8
    var_12 = module_0._turbo_mapping(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = var_12['a']
    assert var_14 == 1
    var_15 = var_12['b']
    assert var_15 == 2
    var_16 = var_12['c']
    assert var_16 == 3
    var_17 = var_12['d']
    assert var_17 == 4
    var_18 = var_12['e']
    assert var_18 == 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'float'
    var_3 = 'value'
    var_4 = 42
    var_5 = 3.14
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8['key']
    assert var_9 == 'value'
    var_10 = var_8['number']
    assert var_10 == 42
    var_11 = var_8['float']
    var_12 = bool(var_8['float'] == 3.14)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'three'
    var_3 = 'one'
    var_4 = 'two'
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8[1]
    assert var_10 == 'one'
    var_11 = var_8[2]
    assert var_11 == 'two'
    var_12 = var_8['three']
    assert var_12 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = module_0._turbo_mapping(var_3, var_2)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_4['a']
    assert var_6 is None
    var_7 = var_4['b']
    assert var_7 is None

def test_case_0():
    var_0 = 'first'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'second'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 'third'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_contains_returns_false_on_exception. Retrieved 11/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'single_string'
    var_7 = 42
    var_8 = 3
    var_9 = [var_2, var_3, var_8]
    var_10 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_eq_predicate_line_3_returns_false. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 7/9 statements.
# Partially parsed test_pmap_eq_with_custom_mapping. Retrieved 7/19 statements.
# Partially parsed test_pmap_eq_with_custom_mapping_different_content. Retrieved 8/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = {var_0: var_2, var_6: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(not var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = bool(not var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'not a mapping'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_turbo_mapping_large_initial. Retrieved 8/11 statements.
# Partially parsed test_turbo_mapping_returns_pmap. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = var_2._size
    assert var_3 == 0
    var_4 = var_2._buckets
    var_5 = len(var_4)
    assert var_5 == 8

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = var_6._size
    assert var_7 == 2
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['b']
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 16
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 1
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 16
    var_8 = var_4['a']
    assert var_8 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8._size
    assert var_9 == 2
    var_10 = var_8['x']
    assert var_10 == 10
    var_11 = var_8['y']
    assert var_11 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 1
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 8
    var_8 = var_4['a']
    assert var_8 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = {str(i): i for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 10
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 20
    var_8 = var_4[var_0]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 4
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = var_8._size
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._buckets
    var_6 = 'evolver'
    var_7 = hasattr(var_5, var_6)
    var_8 = bool(var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'single'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 1
    var_6 = var_4['single']
    assert var_6 == 42
    var_7 = var_4._buckets
    var_8 = len(var_7)
    assert var_8 == 8



# Parsed testcases at query #28
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_6: var_0, var_7: var_1, var_8: var_9}
    var_11 = var_5 == var_10
    assert var_11 is False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap_instance. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = var_2._size
    assert var_4 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2
    var_12 = var_8['c']
    assert var_12 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 32
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['x']
    assert var_8 == 10
    var_9 = var_6['y']
    assert var_9 == 20
    var_10 = var_6._buckets
    var_11 = len(var_10)
    assert var_11 == 32

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'val2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['key1']
    assert var_10 == 'val1'
    var_11 = var_8['key2']
    assert var_11 == 'val2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = None
    var_10 = module_0._turbo_mapping(var_8, var_9)
    var_11 = var_10._buckets
    var_12 = len(var_11)
    assert var_12 == 8

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5['a']
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = 'list'
    var_3 = 'none'
    var_4 = 42
    var_5 = 'hello'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = None
    var_11 = {var_0: var_4, var_1: var_5, var_2: var_9, var_3: var_10}
    var_12 = module_0._turbo_mapping(var_11, var_10)
    var_13 = var_12['int']
    assert var_13 == 42
    var_14 = var_12['str']
    assert var_14 == 'hello'
    var_15 = var_12['list']
    var_16 = bool(var_12['list'] == [1, 2, 3])
    assert var_16 is True
    var_17 = var_12['none']
    assert var_17 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = {f'key_{i}': i for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 'e'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = None
    var_12 = module_0._turbo_mapping(var_10, var_11)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = var_12['a']
    assert var_14 == 1
    var_15 = var_12['b']
    assert var_15 == 2
    var_16 = var_12['c']
    assert var_16 == 3
    var_17 = var_12['d']
    assert var_17 == 4
    var_18 = var_12['e']
    assert var_18 == 5



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_missing_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_values. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_list. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_empty_map. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_matching_value. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = (var_6,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'extra'
    var_9 = (var_6, var_7, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = None
    var_8 = (var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = [var_6, var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = (var_4, var_5)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_turbo_mapping_exception_handler. Retrieved 3/12 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_contains_returns_false_on_unpacking_exception. Retrieved 11/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'single_value'
    var_7 = 3
    var_8 = [var_2, var_3, var_7]
    var_9 = 42
    var_10 = None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_eq_pmap_vs_non_mapping. Retrieved 5/7 statements.
# Partially parsed test_eq_pmap_vs_string. Retrieved 4/6 statements.
# Partially parsed test_eq_pmap_after_modifications. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = bool(not var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = [var_0, var_1, var_6]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'string'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 'z'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.m(**var_6)
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 'z'
    var_11 = {var_8: var_0, var_9: var_1, var_10: var_2}
    var_12 = bool(var_7 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.m(**var_8)
    var_10 = 'd'
    var_11 = 'c'
    var_12 = 'b'
    var_13 = 'a'
    var_14 = {var_10: var_3, var_11: var_2, var_12: var_1, var_13: var_0}
    var_15 = module_0.m(**var_14)
    var_16 = bool(var_9 == var_15)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'alice'
    var_1 = 'nyc'
    var_2 = 'name'
    var_3 = 'city'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'name'
    var_7 = 'city'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'c'
    var_7 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(var_5 != var_10)
    assert var_11 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_turbo_mapping_exception_handler. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: l + r



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pmap_eq_non_mapping. Retrieved 4/6 statements.
# Partially parsed test_pmap_eq_pmap_vs_custom_mapping. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = bool(not var_5 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(not var_5 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'not a mapping'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = {}
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = hash(var_5)
    var_12 = hash(var_10)
    var_13 = bool(not var_5 == var_10)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = hash(var_5)
    var_11 = hash(var_9)
    var_12 = bool(var_5 == var_9)
    assert var_12 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_contains_returns_false_on_exception. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'invalid'



