####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_type_dict_items. Retrieved 7/10 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 2/5 statements.
# Partially parsed test_pmap_items_eq_one_empty_one_not. Retrieved 5/8 statements.


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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 17/27 statements.


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = 'c'
    var_10 = 3
    var_11 = (var_9, var_10)
    var_12 = [var_8, var_11]
    var_13 = [var_1, var_5, var_1, var_12]
    var_14 = '_size'
    var_15 = '_buckets'
    var_16 = '_cached_hash'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 6/9 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_new_keys. Retrieved 6/9 statements.
# Partially parsed test_update_with_empty_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_dict. Retrieved 7/10 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 4/7 statements.
# Partially parsed test_update_with_custom_merge_function. Retrieved 7/9 statements.
# Partially parsed test_update_with_multiple_dicts. Retrieved 13/15 statements.
# Partially parsed test_update_with_no_maps. Retrieved 3/6 statements.


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
    var_6 = 10
    var_7 = lambda l, r: l + r * var_6
    var_8 = 3
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_1, var_10: var_8}
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
    var_7 = 'a'
    var_8 = 'c'
    var_9 = 5
    var_10 = 3
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'd'
    var_13 = 17
    var_14 = 35
    var_15 = {var_7: var_13, var_12: var_14}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)



# Parsed testcases at query #4
#--------------------------

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = var_6._size
    assert var_7 == 2
    var_8 = var_6._buckets
    var_9 = len(var_8)
    assert var_9 == 16
    var_10 = var_6['a']
    assert var_10 == 1
    var_11 = var_6['b']
    assert var_11 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = None
    var_7 = module_0._turbo_mapping(var_5, var_6)
    var_8 = var_7._size
    assert var_8 == 2
    var_9 = var_7['x']
    assert var_9 == 10
    var_10 = var_7['y']
    assert var_10 == 20

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
    var_9 = var_8._size
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
    var_9 = 8
    var_10 = module_0._turbo_mapping(var_8, var_9)
    var_11 = var_10._size
    assert var_11 == 4
    var_12 = var_10._buckets
    var_13 = len(var_12)
    assert var_13 == 8
    var_14 = var_10['a']
    assert var_14 == 1
    var_15 = var_10['b']
    assert var_15 == 2
    var_16 = var_10['c']
    assert var_16 == 3
    var_17 = var_10['d']
    assert var_17 == 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_3)
    var_6 = var_5._size
    assert var_6 == 2
    var_7 = var_5['a']
    assert var_7 == 1
    var_8 = var_5['b']
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

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
    var_9 = var_8._size
    assert var_9 == 3
    var_10 = var_8._buckets
    var_11 = len(var_10)
    assert var_11 == 6

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 1
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 100
    var_8 = var_4['a']
    assert var_8 == 1



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_turbo_mapping_exception_handling. Retrieved 12/47 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 10
    var_6 = 20
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = None
    var_9 = module_0._turbo_mapping(var_7, var_8)
    var_10 = bool(var_9 is not None)
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pmap_items_eq_same_object. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_objects_same_map. Retrieved 6/9 statements.
# Partially parsed test_pmap_items_eq_different_maps_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_maps_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items. Retrieved 7/10 statements.
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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_turbo_mapping_exception_handling. Retrieved 1/38 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 4/6 statements.


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
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_0.m(**var_8)
    var_10 = [var_0, var_1]
    var_11 = [var_3, var_4]
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_10, var_13: var_11}
    var_15 = module_0.m(**var_14)
    var_16 = bool(var_9 == var_15)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_0.m(**var_8)
    var_10 = {var_0: var_1}
    var_11 = {var_3: var_4}
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_10, var_13: var_11}
    var_15 = module_0.m(**var_14)
    var_16 = bool(var_9 == var_15)
    assert var_16 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pmap_eq_with_mapping_protocol. Retrieved 6/18 statements.
# Partially parsed test_pmap_eq_with_different_mapping_protocol. Retrieved 7/19 statements.


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
    var_6 = 'not a mapping'
    var_7 = var_5 == var_6

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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/14 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_slots. Retrieved 11/18 statements.


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
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = '_size'
    var_8 = '_buckets'
    var_9 = '_PMap__weakref__'
    var_10 = '_cached_hash'



# Parsed testcases at query #14
#--------------------------




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
    var_8 = var_5 == var_7
    assert var_8 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_missing_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_arg. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_multiple_matching_items. Retrieved 8/10 statements.


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
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 'a'
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = 'b'
    var_12 = 2
    var_13 = (var_11, var_12)
    var_14 = 'c'
    var_15 = 3
    var_16 = (var_14, var_15)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_turbo_mapping_exception_path. Retrieved 3/12 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------




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
    var_8 = var_5 == var_7
    assert var_8 is False



# Parsed testcases at query #18
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
    var_7 = 123
    var_8 = 3
    var_9 = (var_2, var_3, var_8)
    var_10 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_large_map. Retrieved 18/23 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_8]
    var_10 = '__weakref__'

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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 22/31 statements.


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
    var_14 = None
    var_15 = 'c'
    var_16 = 3
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = [var_13, var_14, var_18]
    var_20 = '__weakref__'
    var_21 = '_cached_hash'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_contains_returns_false_on_exception. Retrieved 10/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'invalid'
    var_7 = 42
    var_8 = None
    var_9 = (var_2,)



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_contains_returns_false_on_unpacking_exception. Retrieved 7/10 statements.
# Partially parsed test_contains_returns_false_on_non_iterable. Retrieved 7/10 statements.
# Partially parsed test_contains_returns_false_on_wrong_unpacking_length. Retrieved 8/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'single_string'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = (var_2, var_3, var_6)



# Parsed testcases at query #24
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
    var_8 = var_6.__eq__(var_7)
    assert var_8 is False



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_update_with_keep_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 7/9 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 4/7 statements.
# Partially parsed test_update_with_dict_input. Retrieved 9/11 statements.
# Partially parsed test_update_with_complex_merge_function. Retrieved 12/14 statements.
# Partially parsed test_update_with_overwrites_all_keys. Retrieved 12/14 statements.
# Partially parsed test_update_with_returns_pmap. Retrieved 5/8 statements.


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
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_1, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 17
    var_15 = 35
    var_16 = {var_12: var_14, var_13: var_15}

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
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
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
    var_6 = lambda l, r: r
    var_7 = 'a'
    var_8 = 'c'
    var_9 = 5
    var_10 = 10
    var_11 = {var_7: var_9, var_8: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.m(**var_6)
    var_8 = lambda l, r: l + r
    var_9 = [var_2]
    var_10 = 'a'
    var_11 = {var_10: var_9}
    var_12 = module_0.m(**var_11)
    var_13 = 'a'
    var_14 = 3
    var_15 = [var_14]
    var_16 = {var_13: var_15}

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
    var_8 = lambda l, r: r
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 'c'
    var_12 = 10
    var_13 = 20
    var_14 = 30
    var_15 = {var_9: var_12, var_10: var_13, var_11: var_14}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 4/6 statements.
# Partially parsed test_pmap_eq_with_different_mapping_types. Retrieved 8/11 statements.
# Partially parsed test_pmap_eq_with_list. Retrieved 4/6 statements.


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
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
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
    var_6 = [var_0, var_1]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_update_with_single_map_with_merge_function. Retrieved 4/7 statements.
# Partially parsed test_update_with_keeps_leftmost_element. Retrieved 8/10 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 12/14 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_new_keys_only. Retrieved 6/8 statements.
# Partially parsed test_update_with_original_pmap_unchanged. Retrieved 5/8 statements.
# Partially parsed test_update_with_custom_merge_function. Retrieved 7/9 statements.
# Partially parsed test_update_with_replaces_when_key_not_in_evolver. Retrieved 5/7 statements.
# Partially parsed test_update_with_multiple_overlapping_keys. Retrieved 11/13 statements.
# Partially parsed test_update_with_with_dict_input. Retrieved 9/11 statements.


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
    var_7 = 10
    var_8 = 3
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)
    var_13 = 'a'
    var_14 = 'd'
    var_15 = 20
    var_16 = 4
    var_17 = {var_13: var_15, var_14: var_16}

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
    var_4 = lambda l, r: r
    var_5 = 2
    var_6 = 3
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_0.m(**var_9)

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
    var_0 = 10
    var_1 = 20
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 5
    var_8 = 30
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
    var_4 = lambda l, r: l * r
    var_5 = 5
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 10
    var_8 = 20
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)
    var_13 = 100
    var_14 = 200
    var_15 = 300
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = {var_16: var_13, var_17: var_14, var_18: var_15}
    var_20 = module_0.m(**var_19)

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
    var_9 = 99
    var_10 = 3
    var_11 = {var_7: var_9, var_8: var_10}



# Parsed testcases at query #30
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = 'd'
    var_8 = 3
    var_9 = 4
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = module_0.PMapItems(var_5)
    var_13 = module_0.PMapItems(var_11)
    var_14 = 'not a PMapItems instance'
    var_15 = var_12.__eq__(var_14)
    assert var_15 is False
    var_16 = 123
    var_17 = var_12.__eq__(var_16)
    assert var_17 is False
    var_18 = (var_0, var_2)
    var_19 = (var_1, var_3)
    var_20 = [var_18, var_19]
    var_21 = var_12.__eq__(var_20)
    assert var_21 is False
    var_22 = None
    var_23 = var_12.__eq__(var_22)
    assert var_23 is False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['b']
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
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['key']
    assert var_6 == 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 8
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
    var_5 = 1000
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['b']
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_2)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5['a']
    assert var_7 is None
    var_8 = var_5['b']
    assert var_8 == 2

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
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['b']
    assert var_9 == 2



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/14 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_single_element. Retrieved 7/11 statements.
# Partially parsed test_pmap_constructor_with_collisions. Retrieved 9/13 statements.
# Partially parsed test_pmap_constructor_large_map. Retrieved 14/21 statements.


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
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 1

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_0, var_7, var_0]

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 32
    var_3 = var_1 * var_2
    var_4 = 'k1'
    var_5 = 'v1'
    var_6 = (var_4, var_5)
    var_7 = 'k2'
    var_8 = 'v2'
    var_9 = (var_7, var_8)
    var_10 = 'k3'
    var_11 = 'v3'
    var_12 = (var_10, var_11)
    var_13 = 3



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_single_element. Retrieved 7/11 statements.
# Partially parsed test_pmap_constructor_collision. Retrieved 10/14 statements.


def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_0, var_8]
    var_10 = 2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = [var_3, var_4, var_4]
    var_6 = 1

def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_0, var_7, var_0]
    var_9 = 2



# Parsed testcases at query #34
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
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = var_6.__eq__(var_7)
    assert var_8 is False



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 5/7 statements.
# Partially parsed test_pmap_eq_with_mapping_protocol. Retrieved 6/18 statements.
# Partially parsed test_pmap_eq_with_different_mapping_protocol. Retrieved 7/19 statements.


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



# Parsed testcases at query #37
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
    var_10 = var_6._buckets
    var_11 = len(var_10)
    assert var_11 == 32

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
    var_8 = var_6['x']
    assert var_8 == 10
    var_9 = var_6['y']
    assert var_9 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = None
    var_11 = module_0._turbo_mapping(var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_11['a']
    assert var_13 == 1
    var_14 = var_11['b']
    assert var_14 == 2
    var_15 = var_11['c']
    assert var_15 == 3

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 256
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1
    var_7 = var_4._buckets
    var_8 = len(var_7)
    assert var_8 == 256

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
    var_13 = bool(var_12 >= 8)
    assert var_13 is True
    var_14 = len(var_10)
    assert var_14 == 4

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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = 'list'
    var_3 = 'float'
    var_4 = 42
    var_5 = 'hello'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 3.14
    var_11 = {var_0: var_4, var_1: var_5, var_2: var_9, var_3: var_10}
    var_12 = None
    var_13 = module_0._turbo_mapping(var_11, var_12)
    var_14 = var_13['int']
    assert var_14 == 42
    var_15 = var_13['str']
    assert var_15 == 'hello'
    var_16 = var_13['list']
    var_17 = bool(var_13['list'] == [1, 2, 3])
    assert var_17 is True
    var_18 = var_13['float']
    var_19 = bool(var_13['float'] == 3.14)
    assert var_19 is True



# Parsed testcases at query #38
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
    var_6 = 42
    var_7 = 'single'
    var_8 = 3
    var_9 = [var_2, var_3, var_8]
    var_10 = None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pmap_eq_with_dict_when_isinstance_other_dict_is_false. Retrieved 6/20 statements.


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



# Parsed testcases at query #40
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
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = var_6.__eq__(var_7)
    assert var_8 is False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap_instance. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
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
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0

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
    var_2 = 'val1'
    var_3 = 'val2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 8
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['key1']
    assert var_8 == 'val1'
    var_9 = var_6['key2']
    assert var_9 == 'val2'

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
    var_0 = 'single'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['single']
    assert var_6 == 'value'

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
    var_0 = 5
    var_1 = 10
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = None
    var_7 = module_0._turbo_mapping(var_5, var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7['x']
    assert var_9 == 5
    var_10 = var_7['y']
    assert var_10 == 10



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_contains_with_invalid_unpacking. Retrieved 11/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'invalid'
    var_7 = None
    var_8 = 42
    var_9 = 3
    var_10 = [var_2, var_3, var_9]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 6/9 statements.
# Partially parsed test_update_with_keep_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_keep_right. Retrieved 9/11 statements.
# Partially parsed test_update_with_empty_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_new_keys. Retrieved 5/8 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 5/8 statements.
# Partially parsed test_update_with_dict. Retrieved 7/10 statements.
# Partially parsed test_update_with_custom_function. Retrieved 12/14 statements.
# Partially parsed test_update_with_overwrites_in_order. Retrieved 10/12 statements.


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
    var_7 = 5
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 'b'
    var_12 = 10
    var_13 = {var_11: var_12}

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
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.m(**var_6)
    var_8 = lambda l, r: l + r
    var_9 = [var_2]
    var_10 = 'a'
    var_11 = {var_10: var_9}
    var_12 = module_0.m(**var_11)
    var_13 = 'b'
    var_14 = 3
    var_15 = [var_14]
    var_16 = {var_13: var_15}

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



# Parsed testcases at query #44
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
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = var_5 == var_9
    assert var_10 is False



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_eq_predicate_line_3_evaluates_true. Retrieved 13/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = module_0.PMapItems(var_10)
    var_12 = 'not a PMapItems object'
    var_13 = [var_6]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_multiple_entries. Retrieved 17/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = 2
    var_6 = 'b'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_8]
    var_10 = '__weakref__'

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 4
    var_4 = 'd'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = 2
    var_9 = 'b'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = 3
    var_13 = 'c'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = [var_6, var_7, var_11, var_15]



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_update_with_single_map_with_merge_function. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps_with_merge_function. Retrieved 6/9 statements.
# Partially parsed test_update_with_keeps_leftmost_element. Retrieved 8/10 statements.
# Partially parsed test_update_with_empty_map_argument. Retrieved 5/7 statements.
# Partially parsed test_update_with_new_keys_only. Retrieved 7/9 statements.
# Partially parsed test_update_with_rightmost_wins. Retrieved 8/10 statements.
# Partially parsed test_update_with_original_map_unchanged. Retrieved 7/9 statements.
# Partially parsed test_update_with_custom_merge_function. Retrieved 7/9 statements.
# Partially parsed test_update_with_multiple_arguments_left_to_right. Retrieved 9/11 statements.
# Partially parsed test_update_with_dict_argument. Retrieved 9/11 statements.
# Partially parsed test_update_with_mixed_pmap_and_dict. Retrieved 10/12 statements.
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
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r
    var_7 = 5
    var_8 = 3
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)
    var_13 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 10
    var_8 = 5
    var_9 = 'a'
    var_10 = 'c'
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
    var_7 = 'a'
    var_8 = {var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = 3
    var_11 = 'a'
    var_12 = {var_11: var_10}
    var_13 = module_0.m(**var_12)
    var_14 = 4
    var_15 = 'a'
    var_16 = {var_15: var_14}
    var_17 = module_0.m(**var_16)

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
    var_9 = 5
    var_10 = 3
    var_11 = {var_7: var_9, var_8: var_10}

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
    var_11 = 'c'
    var_12 = 3
    var_13 = 4
    var_14 = {var_10: var_12, var_11: var_13}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap. Retrieved 5/7 statements.


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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6._size
    assert var_8 == 2
    var_9 = var_6['a']
    assert var_9 == 1
    var_10 = var_6['b']
    assert var_10 == 2

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
    var_8 = var_6._size
    assert var_8 == 2
    var_9 = var_6['a']
    assert var_9 == 1
    var_10 = var_6['b']
    assert var_10 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = 'z'
    var_7 = 30
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = None
    var_11 = module_0._turbo_mapping(var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_11._size
    assert var_13 == 3
    var_14 = var_11['x']
    assert var_14 == 10
    var_15 = var_11['y']
    assert var_15 == 20
    var_16 = var_11['z']
    assert var_16 == 30

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 100
    var_7 = var_5._size
    assert var_7 == 100
    var_8 = var_5[50]
    assert var_8 == 100

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
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
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = '_buckets'
    var_8 = hasattr(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = var_6._buckets
    var_11 = bool(var_6._buckets is not None)
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_turbo_mapping_exception_handling. Retrieved 1/8 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pmap_values_eq_same_object. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_different_object_same_values. Retrieved 6/9 statements.
# Partially parsed test_pmap_values_eq_different_type. Retrieved 7/9 statements.
# Partially parsed test_pmap_values_eq_with_none. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_empty_pmap. Retrieved 4/8 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_2, var_3]

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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_turbo_mapping_exception_handler. Retrieved 1/12 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_single_element. Retrieved 7/11 statements.
# Partially parsed test_pmap_constructor_multiple_collisions. Retrieved 10/14 statements.


def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_4, var_0, var_8]
    var_10 = 2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = [var_3, var_4, var_4]
    var_6 = 1

def test_case_0():
    var_0 = None
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_0, var_7, var_0]
    var_9 = 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_type_list. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_empty_maps. Retrieved 4/7 statements.
# Partially parsed test_pmap_items_eq_none. Retrieved 4/6 statements.


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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 29/37 statements.


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
    var_10 = [var_0, var_0, var_0, var_0]
    var_11 = 0
    var_12 = 'key1'
    var_13 = 'value1'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = 'key2'
    var_17 = 'value2'
    var_18 = (var_16, var_17)
    var_19 = 'key3'
    var_20 = 'value3'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = 'key4'
    var_24 = 'value4'
    var_25 = (var_23, var_24)
    var_26 = [var_25]
    var_27 = [var_0, var_15, var_22, var_0, var_26]
    var_28 = 4



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pmap_values_eq_self_returns_true. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_missing_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_arg. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_multiple_valid_items. Retrieved 8/10 statements.


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
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmap_values_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_different_instance. Retrieved 6/9 statements.
# Partially parsed test_pmap_values_eq_with_list. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_with_none. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_with_dict_values. Retrieved 7/10 statements.
# Partially parsed test_pmap_values_eq_empty_map. Retrieved 2/4 statements.
# Partially parsed test_pmap_values_eq_self_returns_true. Retrieved 6/9 statements.
# Partially parsed test_pmap_values_eq_other_object_returns_false. Retrieved 7/10 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'other'



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_nonexistent_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_wrong_value. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_value. Retrieved 6/8 statements.
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
    var_6 = 'c'
    var_7 = 3
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pmap_values_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_different_instance. Retrieved 6/9 statements.
# Partially parsed test_pmap_values_eq_with_list. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_with_none. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_with_dict_values. Retrieved 7/10 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_items_eq_different_type. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_key_value_pair. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_key_value_pair. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_nonexistent_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_argument. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_tuple_length. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_map. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_matching_pair. Retrieved 6/8 statements.
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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'x'
    var_7 = 10
    var_8 = (var_6, var_7)
    var_9 = 'y'
    var_10 = 20
    var_11 = (var_9, var_10)

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
    var_9 = 'a'
    var_10 = 1
    var_11 = (var_9, var_10)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_eq_predicate_line_3_true. Retrieved 9/14 statements.


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
    var_8 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_with_empty_buckets. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_with_multiple_items_in_bucket. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = None
    var_9 = [var_3, var_7, var_8]
    var_10 = '__weakref__'

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = [var_6]
    var_8 = None
    var_9 = [var_7, var_8, var_8]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_keys. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 7/11 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items_type. Retrieved 6/8 statements.
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_update_with_keep_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 7/9 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 4/7 statements.
# Partially parsed test_update_with_custom_function. Retrieved 7/9 statements.
# Partially parsed test_update_with_dict_input. Retrieved 9/11 statements.
# Partially parsed test_update_with_replaces_values. Retrieved 7/9 statements.
# Partially parsed test_update_with_preserves_type. Retrieved 5/9 statements.


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
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_1, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 17
    var_15 = 35
    var_16 = {var_12: var_14, var_13: var_15}

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
    var_6 = 'a'
    var_7 = {var_6: var_1}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 3
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = {var_11: var_1}
    var_13 = module_0.m(**var_12)

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
    var_10 = 20
    var_11 = {var_7: var_9, var_8: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 2
    var_7 = lambda l, r: r * var_6
    var_8 = 5
    var_9 = 'x'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = [var_3]



# Parsed testcases at query #25
#--------------------------




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
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = var_5 == var_14
    assert var_15 is False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 13/18 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_preserves_buckets. Retrieved 22/25 statements.


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

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 1
    var_1 = 'one'
    var_2 = (var_0, var_1)
    var_3 = 4
    var_4 = 'four'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 2
    var_8 = 'two'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = 3
    var_12 = 'three'
    var_13 = (var_11, var_12)
    var_14 = 5
    var_15 = 'five'
    var_16 = (var_14, var_15)
    var_17 = 6
    var_18 = 'six'
    var_19 = (var_17, var_18)
    var_20 = [var_13, var_16, var_19]
    var_21 = [var_6, var_10, var_20]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_turbo_mapping_exception_handler_executes. Retrieved 3/11 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_contains_returns_false_on_unpacking_exception. Retrieved 6/8 statements.


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
    var_8 = 'string'
    var_9 = 42
    var_10 = None
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pmap_eq_pmap_vs_non_mapping. Retrieved 4/6 statements.
# Partially parsed test_pmap_eq_pmap_vs_other_mapping. Retrieved 8/11 statements.


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
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_4}
    var_8 = module_0.m(**var_7)
    var_9 = bool(not var_3 == var_8)
    assert var_9 is True

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
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'nested'
    var_5 = 'dict'
    var_6 = {var_4: var_5}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_3, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = [var_0, var_1, var_2]
    var_12 = {var_4: var_5}
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_11, var_14: var_12}
    var_16 = module_0.m(**var_15)
    var_17 = bool(var_10 == var_16)
    assert var_17 is True

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
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = hash(var_5)
    var_12 = hash(var_10)
    var_13 = bool(not var_5 == var_10)
    assert var_13 is True



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_values_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_different_instance_same_content. Retrieved 6/9 statements.
# Partially parsed test_pmap_values_eq_different_pmap. Retrieved 8/11 statements.
# Partially parsed test_pmap_values_eq_with_other_type. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_with_none. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_with_string. Retrieved 6/8 statements.
# Partially parsed test_pmap_values_eq_empty_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_values_eq_empty_pmap_different_instances. Retrieved 2/5 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/7 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 5/8 statements.
# Partially parsed test_update_with_new_keys_only. Retrieved 7/9 statements.
# Partially parsed test_update_with_custom_merge_function. Retrieved 6/8 statements.
# Partially parsed test_update_with_overwrites_with_rightmost. Retrieved 7/9 statements.
# Partially parsed test_update_with_dict_input. Retrieved 7/9 statements.
# Partially parsed test_update_with_returns_pmap. Retrieved 6/9 statements.


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
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_1, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 17
    var_15 = 35
    var_16 = {var_12: var_14, var_13: var_15}

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
    var_0 = 10
    var_1 = 5
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_1, var_9: var_7}
    var_11 = module_0.m(**var_10)

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r
    var_7 = 'c'
    var_8 = 3
    var_9 = {var_7: var_8}

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
    var_8 = 'c'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/7 statements.
# Partially parsed test_pmap_constructor_large. Retrieved 22/26 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_8]
    var_10 = 2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = 'c'
    var_9 = 3
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = 'd'
    var_13 = 4
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = 'e'
    var_17 = 5
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_3, var_4, var_11, var_4, var_15, var_4, var_4, var_19]
    var_21 = 5



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_turbo_mapping_returns_pmap. Retrieved 5/7 statements.


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
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['key']
    assert var_6 == 'value'

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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = 'val3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_8['key1']
    assert var_10 == 'val1'
    var_11 = var_8['key2']
    assert var_11 == 'val2'
    var_12 = var_8['key3']
    assert var_12 == 'val3'

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
    var_13 = bool(var_12 >= 2 * 4)
    assert var_13 is True

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'exists'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = 'exists'
    var_6 = bool('exists' in var_4)
    assert var_6 is True
    var_7 = 'not_exists'
    var_8 = bool('not_exists' not in var_4)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {f'key{i}': i for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100
    var_6 = var_4['key50']
    assert var_6 == 50
    var_7 = var_4['key99']
    assert var_7 == 99

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'num'
    var_1 = 'str'
    var_2 = 'list'
    var_3 = 42
    var_4 = 'hello'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = {var_0: var_3, var_1: var_4, var_2: var_8}
    var_10 = None
    var_11 = module_0._turbo_mapping(var_9, var_10)
    var_12 = var_11['num']
    assert var_12 == 42
    var_13 = var_11['str']
    assert var_13 == 'hello'
    var_14 = var_11['list']
    var_15 = bool(var_11['list'] == [1, 2, 3])
    assert var_15 is True



# Parsed testcases at query #35
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
    var_9 = var_6 == 'not a PMapItems'
    assert var_9 is False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pmap_eq_predicate_line_15_evaluates_to_false. Retrieved 10/12 statements.


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
    var_9 = var_5 == var_8
    assert var_9 is False



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_missing_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_triple_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_empty_map. Retrieved 2/4 statements.
# Partially parsed test_pmap_items_contains_with_none_value. Retrieved 4/6 statements.
# Partially parsed test_pmap_items_contains_with_numeric_keys. Retrieved 6/8 statements.


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
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)

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
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 1
    var_7 = 'one'
    var_8 = (var_6, var_7)
    var_9 = 2
    var_10 = 'two'
    var_11 = (var_9, var_10)
    var_12 = 1
    var_13 = 'two'
    var_14 = (var_12, var_13)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 20/29 statements.


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_1, var_5, var_1]
    var_7 = 1
    var_8 = 'a'
    var_9 = (var_8, var_7)
    var_10 = [var_9]
    var_11 = 'b'
    var_12 = 2
    var_13 = (var_11, var_12)
    var_14 = 'c'
    var_15 = 3
    var_16 = (var_14, var_15)
    var_17 = [var_13, var_16]
    var_18 = [var_10, var_1, var_17, var_1]
    var_19 = '__weakref__'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pmap_items_eq_same_instance. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_different_instances_same_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_items_eq_different_content. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_keys. Retrieved 9/12 statements.
# Partially parsed test_pmap_items_eq_different_type. Retrieved 7/11 statements.
# Partially parsed test_pmap_items_eq_with_non_pmap_items. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_eq_with_string. Retrieved 6/8 statements.
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



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = len(var_2)
    assert var_4 == 0



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_pmap_items_contains_with_valid_item. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_key. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_invalid_value. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_non_tuple_arg. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_single_element. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_three_element_tuple. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_none_values. Retrieved 6/8 statements.
# Partially parsed test_pmap_items_contains_with_list_arg. Retrieved 6/8 statements.
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



# Parsed testcases at query #43
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
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}
    var_10 = var_5 == var_9
    assert var_10 is False



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_turbo_mapping_exception_path. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #45
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



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/15 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 3/8 statements.
# Partially parsed test_pmap_constructor_multiple_items. Retrieved 15/18 statements.
# Partially parsed test_pmap_constructor_weakref_slot. Retrieved 4/8 statements.
# Partially parsed test_pmap_constructor_cached_hash_slot. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = None
    var_9 = [var_3, var_7, var_8]
    var_10 = 2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'val2'
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = [var_6]
    var_8 = None
    var_9 = 'key3'
    var_10 = 'val3'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = [var_7, var_8, var_12]
    var_14 = 3

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 1
    var_3 = '__weakref__'

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 0
    var_3 = '_cached_hash'



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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
    var_9 = None
    var_10 = 42



# Parsed testcases at query #51
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



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_turbo_mapping_exception_predicate_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



