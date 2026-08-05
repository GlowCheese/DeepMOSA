####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
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
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(var_5 != var_11)
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
    var_11 = bool(var_5 != var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1, 2, 3])
    assert var_4 is True
    var_5 = bool(var_3 != 'not a map')
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




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
    var_1 = 'key'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['a']
    assert var_8 == 1
    var_9 = var_6['key']
    assert var_9 == 'value'

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
    var_7 = iter(var_6)
    var_8 = None
    var_9 = module_0._turbo_mapping(var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_9['x']
    assert var_11 == 10
    var_12 = var_9['y']
    assert var_12 == 20

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
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 99
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['z']
    assert var_6 == 99



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pmap_items_repr. Retrieved 1/18 statements.


def test_case_0():
    var_0 = "pmap_items([('a', 1), ('b', 2)])"



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pmap_items_contains_valid_pair. Retrieved 6/36 statements.
# Partially parsed test_pmap_items_contains_invalid_pair_value. Retrieved 3/24 statements.
# Failed to parse test_pmap_items_contains_non_iterable_arg.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pmap_update_with_merge_rightmost. Retrieved 12/14 statements.
# Partially parsed test_pmap_update_with_merge_leftmost. Retrieved 12/14 statements.
# Partially parsed test_pmap_update_with_addition. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_no_overlap. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_empty_maps. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 5
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 10
    var_15 = 4
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 5
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 10
    var_15 = 4
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = lambda l, r: l

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
    var_9 = lambda l, r: l + r

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
    var_8 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r
    var_5 = {}
    var_6 = module_0.m(**var_5)
    var_7 = {}



# Parsed testcases at query #6
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'key2'
    var_5 = 'val2'
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = 2
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._buckets[0][0]
    var_14 = bool(var_12._buckets[0][0] == ('key1', 'val1'))
    assert var_14 is True
    var_15 = var_12._buckets[1][0]
    var_16 = bool(var_12._buckets[1][0] == ('key2', 'val2'))
    assert var_16 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_repr_returns_correct_string_format. Retrieved 1/13 statements.


def test_case_0():
    var_0 = "pmap_items([('a', 1), ('b', 2)])"



# Parsed testcases at query #8
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = None
    var_4 = 'y'
    var_5 = 20
    var_6 = (var_4, var_5)
    var_7 = 'z'
    var_8 = 30
    var_9 = (var_7, var_8)
    var_10 = [var_6, var_9]
    var_11 = [var_2, var_3, var_10]
    var_12 = 3
    var_13 = [var_12, var_11]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = var_15._size
    var_17 = bool(var_15._size == var_12)
    assert var_17 is True
    var_18 = var_15._buckets
    var_19 = bool(var_15._buckets == var_11)
    assert var_19 is True



# Parsed testcases at query #9
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = bool(not 123 in var_3)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = None
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_2, var_3, var_7]
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_0.PMap(*var_9, **var_10)
    var_12 = var_11._size
    assert var_12 == 2
    var_13 = var_11._buckets
    var_14 = bool(var_11._buckets == var_8)
    assert var_14 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    assert var_6 == 0
    var_7 = var_5._buckets
    var_8 = bool(var_5._buckets == [None, None])
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pmap_eq_not_mapping_is_not_implemented. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'a'
    var_5 = {var_4: var_0}



# Parsed testcases at query #12
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = 'c'
    var_8 = 3
    var_9 = (var_7, var_8)
    var_10 = [var_6, var_9]
    var_11 = None
    var_12 = [var_3, var_10, var_11]
    var_13 = 3
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._size
    assert var_17 == 3
    var_18 = var_16._buckets[var_1]
    var_19 = len(var_18)
    assert var_19 == 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pmap_eq_different_cached_hashes. Retrieved 3/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.m(**var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmap_update_with_addition. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_leftmost. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 10/12 statements.
# Partially parsed test_pmap_update_with_no_changes. Retrieved 4/6 statements.
# Partially parsed test_pmap_update_with_new_keys. Retrieved 5/7 statements.


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
    var_9 = lambda l, r: l + r

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
    var_8 = 'a'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 10
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'd'
    var_13 = 4
    var_14 = {var_12: var_13}
    var_15 = lambda l, r: r

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
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = lambda l, r: r



# Parsed testcases at query #15
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = [var_5, var_7]
    var_9 = 1
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'k1'
    var_2 = 'v1'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = 1
    var_7 = 'k2'
    var_8 = 'v2'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = (var_6, var_10)
    var_12 = [var_5, var_11]
    var_13 = 2
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._size
    var_18 = bool(var_16._size == var_13)
    assert var_18 is True
    var_19 = var_16._buckets
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = hash(var_1)
    var_22 = 2
    var_23 = var_21 % var_22
    var_24 = var_23 == var_0
    var_25 = var_1 if var_24 else var_7
    var_26 = bool(var_16[var_25])
    assert var_26 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_contains_invalid_argument_type_returns_false.




# Parsed testcases at query #17
#--------------------------




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
    var_9 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 16
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = bool(var_4 == {'x': 10})
    assert var_5 is True
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 16

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
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True

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
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_2)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True
    var_7 = var_5._buckets
    var_8 = len(var_7)
    assert var_8 == 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pmap_eq_different_cached_hashes. Retrieved 3/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.m(**var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = module_0.PMapItems(var_4)
    var_7 = bool(var_5 == var_5)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.PMapItems(var_4)
    var_7 = module_0.PMapItems(var_5)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.PMapItems(var_4)
    var_8 = module_0.PMapItems(var_6)
    var_9 = bool(var_7 != var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = {var_0: var_1}
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = {}
    var_3 = module_0.PMapItems(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pmap_eq_different_cached_hashes. Retrieved 6/9 statements.


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
    var_12 = bool(var_5 != var_9)
    assert var_12 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 5/8 statements.


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



# Parsed testcases at query #22
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'a'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k1'
    var_1 = 'v1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'k2'
    var_5 = 'v2'
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = None
    var_9 = [var_3, var_7, var_8]
    var_10 = 2
    var_11 = [var_10, var_9]
    var_12 = {}
    var_13 = module_0.PMap(*var_11, **var_12)
    var_14 = var_13._size
    assert var_14 == 2
    var_15 = var_13._buckets
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = var_13['k1']
    assert var_17 == 'v1'
    var_18 = var_13['k2']
    assert var_18 == 'v2'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_contains_invalid_arg_type.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_contains_predicate_evaluates_to_false_on_invalid_argument_type. Retrieved 2/6 statements.


def test_case_0():
    var_0 = True
    var_1 = 123



# Parsed testcases at query #25
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 'c'
    var_9 = 3
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = [var_0, var_7, var_11]
    var_13 = 3
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._size
    var_18 = bool(var_16._size == var_13)
    assert var_18 is True
    var_19 = var_16._buckets
    var_20 = bool(var_16._buckets == var_12)
    assert var_20 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_contains_exception_returns_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #27
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
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'c': 1, 'b': 2})
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
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    var_12 = bool(var_5 != var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1])
    assert var_4 is True
    var_5 = bool(var_3 != 'a: 1')
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 6/9 statements.


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
    var_8 = 3
    var_9 = 'b'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 5



# Parsed testcases at query #30
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_on_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #33
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

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
    var_8 = [var_6, var_7]
    var_9 = 2
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12['a']
    assert var_15 == 1
    var_16 = var_12['b']
    assert var_16 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'x'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9.x
    assert var_10 == 10

import pyrsistent._pmap as module_0

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
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = len(var_12)
    assert var_13 == 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pmap_items_contains_valid_pair. Retrieved 5/16 statements.
# Partially parsed test_pmap_items_contains_invalid_value. Retrieved 3/14 statements.
# Partially parsed test_pmap_items_contains_missing_key. Retrieved 3/14 statements.
# Partially parsed test_pmap_items_contains_non_iterable_arg. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = (var_5, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 1
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = None



# Parsed testcases at query #35
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = ('b', 1) in var_3
    assert var_4 is False



# Parsed testcases at query #36
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = (var_0, var_2)
    var_7 = 'a'
    var_8 = 1
    var_9 = (var_7, var_8)
    var_10 = bool(('a', 1) in var_5)
    assert var_10 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pmap_update_with_merging_values. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_preference. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_pmap_update_with_no_overlapping_keys. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_empty_map. Retrieved 5/7 statements.


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
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 'a'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = lambda l, r: l

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
    var_12 = 'd'
    var_13 = 17
    var_14 = 35
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = lambda l, r: r

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
    var_8 = lambda l, r: r

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



# Parsed testcases at query #38
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = (var_2,)
    var_4 = None
    var_5 = 'key2'
    var_6 = 'val2'
    var_7 = (var_5, var_6)
    var_8 = 'key3'
    var_9 = 'val3'
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = [var_3, var_4, var_11]
    var_13 = 3
    var_14 = [var_13, var_12]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = var_16._size
    assert var_17 == 3
    var_18 = var_16._buckets
    var_19 = len(var_18)
    assert var_19 == 3
    var_20 = var_16._buckets[0][0]
    var_21 = bool(var_16._buckets[0][0] == ('key1', 'val1'))
    assert var_21 is True
    var_22 = var_16._buckets[2][1]
    var_23 = bool(var_16._buckets[2][1] == ('key3', 'val3'))
    assert var_23 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pmap_eq_not_implement_not_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 5



# Parsed testcases at query #40
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = 2
    var_9 = 'b'
    var_10 = (var_9, var_8)
    var_11 = [var_10]
    var_12 = (var_8, var_11)
    var_13 = [var_5, var_7, var_12]
    var_14 = 2
    var_15 = [var_14, var_13]
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = var_17._size
    var_19 = bool(var_17._size == var_14)
    assert var_19 is True
    var_20 = var_17._buckets
    var_21 = bool(var_17._buckets == var_13)
    assert var_21 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'key'
    var_2 = 'val'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = var_10._size
    assert var_11 == 1
    var_12 = var_10._buckets[0]
    var_13 = bool(var_10._buckets[0] == [('key', 'val')])
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_contains_invalid_arg_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #42
#--------------------------




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
    var_9 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 10
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 2

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
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = bool(var_8 == {'x': 10, 'y': 20})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #43
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_7 is True
    var_8 = len(var_6)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 8
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = bool(var_6 == {'x': 10, 'y': 20})
    assert var_7 is True
    var_8 = len(var_6)
    assert var_8 == 2

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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 1
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = bool(var_8 == {'a': 1, 'b': 2, 'c': 3})
    assert var_9 is True
    var_10 = len(var_8)
    assert var_10 == 3



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_update_with_does_not_use_existing_value_when_key_absent. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'b'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 3
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)



# Parsed testcases at query #45
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #46
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

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
    var_11 = [var_6, var_10]
    var_12 = 3
    var_13 = [var_12, var_11]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = var_15._size
    assert var_16 == 3
    var_17 = 0
    var_18 = var_15._buckets[var_17]
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_15._buckets[1][0]
    var_21 = bool(var_15._buckets[1][0] == ('key3', 'val3'))
    assert var_21 is True



# Parsed testcases at query #47
#--------------------------

# Failed to parse test_contains_evaluates_false_on_ununpackable_arg.




# Parsed testcases at query #48
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 2
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = var_10._size
    assert var_11 == 2
    var_12 = var_10._buckets
    var_13 = bool(var_10._buckets == [('a', 1), ('b', 2)])
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'val'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets[1]
    var_12 = bool(var_9._buckets[1] == [('key', 'val')])
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = var_5._size
    assert var_7 == 0



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 5



# Parsed testcases at query #50
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #51
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

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
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True



# Parsed testcases at query #52
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 1) in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 3) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('c', 1) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 1
    var_7 = bool(1 not in var_5)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = (var_6,)
    var_8 = bool(('a',) not in var_5)
    assert var_8 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pmap_update_with_merging_values. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_preference. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 9/11 statements.
# Partially parsed test_pmap_update_with_no_changes. Retrieved 4/6 statements.
# Partially parsed test_pmap_update_with_overwriting_existing_keys. Retrieved 6/8 statements.


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
    var_4 = 2
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = 'a'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = lambda l, r: l

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
    var_10 = 'd'
    var_11 = 4
    var_12 = {var_10: var_11}
    var_13 = lambda l, r: r

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
    var_6 = 10
    var_7 = 'a'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = lambda l, r: r



# Parsed testcases at query #54
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = [var_6, var_5]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = [var_0, var_9, var_0]
    var_11 = [var_2, var_5]
    var_12 = {}
    var_13 = module_0.PMap(*var_11, **var_12)
    var_14 = [var_2, var_10]
    var_15 = {}
    var_16 = module_0.PMap(*var_14, **var_15)
    var_17 = bool(var_13 != var_16)
    assert var_17 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_4._size
    assert var_6 == 0



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_update_with_does_not_always_trigger_ternary_true_branch. Retrieved 5/7 statements.


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
    var_8 = lambda l, r: l + r



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_contains_predicate_evaluates_to_false_on_invalid_tuple_unpacking. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false. Retrieved 7/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #58
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = 2
    var_9 = 'b'
    var_10 = (var_9, var_8)
    var_11 = [var_10]
    var_12 = (var_8, var_11)
    var_13 = [var_5, var_7, var_12]
    var_14 = 2
    var_15 = [var_14, var_13]
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = var_17._size
    var_19 = bool(var_17._size == var_14)
    assert var_19 is True
    var_20 = var_17._buckets
    var_21 = bool(var_17._buckets == var_13)
    assert var_21 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'val'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'val2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_6)
    var_8 = [var_7]
    var_9 = [var_3, var_8]
    var_10 = {}
    var_11 = module_0.PMap(*var_9, **var_10)
    var_12 = var_11[1]
    assert var_12 == 'val'
    var_13 = var_11[2]
    assert var_13 == 'val2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = var_10.key
    assert var_11 == 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = [var_2, var_6]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = 'a'
    var_11 = bool('a' in var_9)
    assert var_11 is True
    var_12 = 'b'
    var_13 = bool('b' not in var_9)
    assert var_13 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_pmapvalues_str.




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
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
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
    var_7 = {var_6: var_0}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [('a', 1)])
    assert var_4 is True
    var_5 = bool(var_3 != 5)
    assert var_5 is True

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
    var_8 = bool(var_3 != var_7)
    assert var_8 is True
    var_9 = bool(var_7 != var_3)
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pmap_values_str_empty. Retrieved 1/6 statements.
# Partially parsed test_pmap_values_str_with_elements. Retrieved 5/10 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = True
    var_3 = (var_2,)
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #4
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.PMapItems(var_4)
    var_7 = module_0.PMapItems(var_5)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.PMapItems(var_4)
    var_8 = module_0.PMapItems(var_6)
    var_9 = bool(var_7 != var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = {var_0: var_1}
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = {}
    var_3 = module_0.PMapItems(var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'a'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 3
    var_8 = 'c'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = [var_2, var_6, var_10]
    var_12 = 3
    var_13 = [var_12, var_11]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = var_15._size
    assert var_16 == 3
    var_17 = var_15._buckets
    var_18 = bool(var_15._buckets == var_11)
    assert var_18 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pmap_eq_not_mapping_returns_not_implemented. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 5



# Parsed testcases at query #7
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = bool(var_6 == {'a': 1, 'b': 2})
    assert var_7 is True
    var_8 = len(var_6)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 8
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = bool(var_6 == {'x': 10, 'y': 20})
    assert var_7 is True
    var_8 = len(var_6)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 4
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
    var_5 = var_4['a']
    assert var_5 == 1



# Parsed testcases at query #8
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'a'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = None
    var_8 = [var_2, var_6, var_7]
    var_9 = 2
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pmap_constructor_stores_reference_to_buckets. Retrieved 9/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = 2
    var_9 = 'b'
    var_10 = (var_9, var_8)
    var_11 = [var_10]
    var_12 = (var_8, var_11)
    var_13 = [var_5, var_7, var_12]
    var_14 = 2
    var_15 = [var_14, var_13]
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = var_17._size
    assert var_18 == 2
    var_19 = var_17._buckets
    var_20 = bool(var_17._buckets == var_13)
    assert var_20 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = var_10._buckets[0]
    assert var_11 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pmapvalues_eq_identity. Retrieved 5/18 statements.
# Partially parsed test_pmapvalues_eq_not_identity. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_turbo_mapping_predicate_is_false. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pmap_eq_different_cached_hashes. Retrieved 3/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = module_0.m(**var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_via_exception. Retrieved 6/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)



# Parsed testcases at query #14
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
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
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
    var_7 = {var_6: var_0}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1, 2, 3])
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1])
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 1) in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'a'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 3) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 'c'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = bool(('c', 1) not in var_5)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = 'a'
    var_5 = bool('a' not in var_3)
    assert var_5 is True
    var_6 = 1
    var_7 = bool(1 not in var_3)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = 'a'
    var_5 = 1
    var_6 = 'extra'
    var_7 = (var_4, var_5, var_6)
    var_8 = bool(('a', 1, 'extra') not in var_3)
    assert var_8 is True
    var_9 = 1
    var_10 = (var_9,)
    var_11 = bool((1,) not in var_3)
    assert var_11 is True



# Parsed testcases at query #16
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = [var_5, var_7]
    var_9 = 1
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'key'
    var_2 = 'val'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = var_10._size
    assert var_11 == 1
    var_12 = var_10._buckets
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = var_10._buckets[0][0]
    assert var_14 == 'key'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = 2
    var_9 = 'b'
    var_10 = (var_9, var_8)
    var_11 = [var_10]
    var_12 = (var_8, var_11)
    var_13 = [var_5, var_7, var_12]
    var_14 = 2
    var_15 = [var_14, var_13]
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = var_17._size
    assert var_18 == 2
    var_19 = var_17._buckets
    var_20 = bool(var_17._buckets == var_13)
    assert var_20 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    assert var_6 == 0
    var_7 = var_5._buckets
    var_8 = bool(var_5._buckets == [None, None])
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = var_10._size
    assert var_11 == 1
    var_12 = var_10['key']
    assert var_12 == 'value'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pmap_items_contains_valid_pair. Retrieved 5/19 statements.
# Partially parsed test_pmap_items_contains_invalid_value. Retrieved 3/17 statements.
# Partially parsed test_pmap_items_contains_non_tuple. Retrieved 3/17 statements.
# Partially parsed test_pmap_items_contains_malformed_tuple. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = 'b'
    var_9 = 2
    var_10 = (var_8, var_9)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 1
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = None

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = (var_3,)
    var_5 = 1
    var_6 = 'a'
    var_7 = (var_5, var_6)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_update_with_addition. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_leftmost. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 12/14 statements.
# Partially parsed test_pmap_update_with_no_overlap. Retrieved 5/7 statements.


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
    var_9 = lambda l, r: l + r

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
    var_8 = 'a'
    var_9 = 3
    var_10 = {var_8: var_9}
    var_11 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 10
    var_7 = 3
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'c'
    var_13 = 'd'
    var_14 = 5
    var_15 = 4
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = lambda l, r: r

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
    var_8 = lambda l, r: r



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pmap_update_with_merging_logic. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 10/13 statements.
# Partially parsed test_pmap_update_with_leftmost_logic. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_new_keys. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_empty_maps. Retrieved 4/6 statements.


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
    var_12 = 'd'
    var_13 = 17
    var_14 = 35
    var_15 = {var_11: var_13, var_12: var_14}

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
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)
    var_8 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_on_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 5/8 statements.


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



# Parsed testcases at query #23
#--------------------------




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
    var_8 = hash(var_3)
    var_9 = hash(var_7)
    var_10 = bool(var_3 != var_7)
    assert var_10 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_update_with_not_existing_key. Retrieved 4/7 statements.


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pmap_constructor_initializes_size_and_buckets. Retrieved 11/14 statements.
# Partially parsed test_pmap_constructor_with_empty_buckets. Retrieved 3/8 statements.
# Partially parsed test_pmap_constructor_preserves_object_identity. Retrieved 6/9 statements.


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
    var_10 = 2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = 1



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_contains_invalid_arg_type_returns_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = None
    var_1 = 123



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_eq_returns_true_when_identity_is_same.




# Parsed testcases at query #28
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = 2
    var_9 = 'b'
    var_10 = (var_9, var_8)
    var_11 = [var_10]
    var_12 = (var_8, var_11)
    var_13 = [var_5, var_7, var_12]
    var_14 = 2
    var_15 = [var_14, var_13]
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = var_17._size
    var_19 = bool(var_17._size == var_14)
    assert var_19 is True
    var_20 = var_17._buckets
    var_21 = bool(var_17._buckets == var_13)
    assert var_21 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    var_7 = bool(var_5._size == var_2)
    assert var_7 is True
    var_8 = var_5._buckets
    var_9 = bool(var_5._buckets == var_1)
    assert var_9 is True



# Parsed testcases at query #29
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
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 != {'a': 1, 'b': 3})
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
    var_7 = {var_6: var_0}
    var_8 = module_0.m(**var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True
    var_10 = bool(var_5 != {'a': 1, 'b': 2, 'c': 3})
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = bool(var_3 != [1, 2, 3])
    assert var_4 is True
    var_5 = bool(var_3 != 'not a map')
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = bool(not None in var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = bool(not 123 in var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = bool(not 'not_a_tuple' in var_3)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = bool(not ('a', 1, 'extra') in var_3)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------




import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = 'itervalues'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = iter(var_6)
    var_8 = lambda self: var_7
    var_9 = {var_2: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.PMapValues(var_13)
    var_15 = bool(var_14 == var_14)
    assert var_15 is True



# Parsed testcases at query #32
#--------------------------




import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = 'itervalues'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = iter(var_6)
    var_8 = lambda self: var_7
    var_9 = {var_2: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.PMapValues(var_13)
    var_15 = bool(var_14 == var_14)
    assert var_15 is True

import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = 'itervalues'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = iter(var_6)
    var_8 = lambda self: var_7
    var_9 = {var_2: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = ()
    var_15 = [var_3, var_4, var_5]
    var_16 = iter(var_15)
    var_17 = lambda self: var_16
    var_18 = {var_2: var_17}
    var_19 = [var_0, var_14, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = module_1.PMapValues(var_13)
    var_24 = module_1.PMapValues(var_22)
    var_25 = bool(var_23 != var_24)
    assert var_25 is True

import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = 'itervalues'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = iter(var_6)
    var_8 = lambda self: var_7
    var_9 = {var_2: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.PMapValues(var_13)
    var_15 = bool(var_14 != [1, 2, 3])
    assert var_15 is True

import builtins as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = 'MockMap'
    var_1 = ()
    var_2 = 'itervalues'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = iter(var_6)
    var_8 = lambda self: var_7
    var_9 = {var_2: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = module_1.PMapValues(var_13)
    var_15 = bool(var_14 != None)
    assert var_15 is True



# Parsed testcases at query #33
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = [var_5, var_7]
    var_9 = 1
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = (var_2, var_9)
    var_11 = [var_5, var_10]
    var_12 = 2
    var_13 = [var_12, var_11]
    var_14 = {}
    var_15 = module_0.PMap(*var_13, **var_14)
    var_16 = var_15._size
    assert var_16 == 2
    var_17 = var_15['a']
    assert var_17 == 1
    var_18 = var_15['b']
    assert var_18 == 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_via_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 123



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pmap_update_with_addition. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_pmap_update_with_no_overlap. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_dict_input. Retrieved 8/10 statements.


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
    var_4 = lambda l, r: r
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: r
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 5
    var_8 = 10
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #37
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'a'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    assert var_10 == 1
    var_11 = var_9._buckets
    var_12 = bool(var_9._buckets == var_5)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    assert var_5 == 0
    var_6 = var_4._buckets
    var_7 = bool(var_4._buckets == [])
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = (var_0, var_1)
    var_3 = (var_2,)
    var_4 = None
    var_5 = 'key2'
    var_6 = 'val2'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_8]
    var_10 = 2
    var_11 = [var_10, var_9]
    var_12 = {}
    var_13 = module_0.PMap(*var_11, **var_12)
    var_14 = var_13._buckets
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13._buckets[0][0]
    var_17 = bool(var_13._buckets[0][0] == ('key', 'val'))
    assert var_17 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_contains_raises_exception_returns_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_pmap_values_eq_identity.
# Failed to parse test_pmap_values_eq_not_identity.
# Failed to parse test_pmap_values_eq_with_other_types.




# Parsed testcases at query #40
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = 2
    var_9 = 'b'
    var_10 = (var_9, var_8)
    var_11 = [var_10]
    var_12 = (var_8, var_11)
    var_13 = [var_5, var_7, var_12]
    var_14 = 2
    var_15 = [var_14, var_13]
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = var_17._size
    var_19 = bool(var_17._size == var_14)
    assert var_19 is True
    var_20 = var_17._buckets
    var_21 = bool(var_17._buckets == var_13)
    assert var_21 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0
    var_3 = [var_2, var_1]
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = var_5._size
    var_7 = bool(var_5._size == var_2)
    assert var_7 is True
    var_8 = var_5._buckets
    var_9 = bool(var_5._buckets == var_1)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = hash(var_0)
    var_2 = 1
    var_3 = var_1 % var_2
    var_4 = 'value'
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = (var_3, var_6)
    var_8 = [var_7]
    var_9 = 1
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12['key']
    assert var_13 == 'value'
    var_14 = len(var_12)
    assert var_14 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7, var_6]
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = 'a'
    var_12 = bool('a' in var_10)
    assert var_12 is True
    var_13 = 'b'
    var_14 = bool('b' not in var_10)
    assert var_14 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 4/7 statements.


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
    var_8 = 'b'



# Parsed testcases at query #42
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = module_0._turbo_mapping(var_1, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapValues(var_4)
    var_6 = bool(var_5 == var_5)
    assert var_6 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_contains_fails_on_non_iterable_arg. Retrieved 1/5 statements.
# Partially parsed test_contains_fails_on_single_element_tuple. Retrieved 2/7 statements.
# Partially parsed test_contains_fails_on_non_tuple_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 1
    var_1 = (var_0,)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 5



# Parsed testcases at query #46
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1
    var_7 = [var_6, var_5]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_6)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = bool(var_9._buckets == var_5)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

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
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_update_with_predicate_is_false. Retrieved 6/9 statements.


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
    var_8 = 5
    var_9 = 'b'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 5



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pmap_items_contains_valid_pair. Retrieved 7/19 statements.
# Partially parsed test_pmap_items_contains_invalid_value. Retrieved 3/14 statements.
# Partially parsed test_pmap_items_contains_missing_key. Retrieved 3/14 statements.
# Partially parsed test_pmap_items_contains_non_iterable_arg. Retrieved 3/14 statements.
# Partially parsed test_pmap_items_contains_wrong_tuple_size. Retrieved 3/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = 'a'
    var_8 = 1
    var_9 = (var_7, var_8)
    var_10 = bool(('a', 1) in var_5)
    assert var_10 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 1
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'not_a_tuple'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = 'extra'
    var_6 = (var_3, var_4, var_5)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_on_exception. Retrieved 1/8 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #51
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = (var_0, var_4)
    var_6 = None
    var_7 = (var_2, var_6)
    var_8 = [var_5, var_7]
    var_9 = 1
    var_10 = [var_9, var_8]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)
    var_13 = var_12._size
    var_14 = bool(var_12._size == var_9)
    assert var_14 is True
    var_15 = var_12._buckets
    var_16 = bool(var_12._buckets == var_8)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    var_8 = bool(var_4._buckets == var_0)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = var_4._size
    var_6 = bool(var_4._size == var_1)
    assert var_6 is True
    var_7 = var_4._buckets
    assert var_7 is None



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_turbo_mapping_predicate_is_false. Retrieved 4/30 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_update_with_predicate_false_when_key_not_in_evolver. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'b'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 3
    var_5 = 'a'
    var_6 = {var_5: var_4}
    var_7 = module_0.m(**var_6)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_pmap_items_contains_valid_pair. Retrieved 5/14 statements.
# Partially parsed test_pmap_items_contains_invalid_value. Retrieved 3/12 statements.
# Partially parsed test_pmap_items_contains_missing_key. Retrieved 3/12 statements.
# Partially parsed test_pmap_items_contains_invalid_format. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = (var_5, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 1
    var_5 = (var_3, var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = 'a'
    var_6 = (var_5,)
    var_7 = (var_6,)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_pmap_eq_not_implement_not_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'not a mapping'



