####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 7/8 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 6/7 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/9 statements.
# Partially parsed test_update_with_empty_map. Retrieved 4/5 statements.
# Partially parsed test_update_with_new_keys. Retrieved 7/8 statements.
# Partially parsed test_update_with_complex_merge. Retrieved 7/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 'a'
    var_8 = {var_7: var_1}
    var_9 = module_0.m(**var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = 'a'
    var_8 = {var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = 3
    var_11 = 'a'
    var_12 = {var_11: var_10}
    var_13 = module_0.m(**var_12)

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
    var_8 = 'c'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)

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
    var_6 = lambda l, r: l + r

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
    var_9 = 3
    var_10 = 'c'
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
    var_6 = lambda l, r: l * r
    var_7 = 3
    var_8 = 4
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_contains_non_existing_key. Retrieved 5/7 statements.
# Partially parsed test_contains_non_existing_value. Retrieved 5/7 statements.
# Partially parsed test_contains_invalid_arg. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 2
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = None



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_eq_identical_instance.
# Failed to parse test_eq_different_type.
# Failed to parse test_eq_different_map.
# Failed to parse test_eq_same_map.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 1
    var_3 = 'a'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_0, var_6)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pmap_constructor_creates_instance_with_correct_size_and_buckets. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_pmapitems_eq_same_instance.
# Failed to parse test_pmapitems_eq_different_type.
# Failed to parse test_pmapitems_eq_different_pmap.
# Partially parsed test_pmapitems_eq_different_pmap_with_items. Retrieved 4/8 statements.
# Partially parsed test_pmapitems_eq_different_pmap_with_different_items. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = {var_0: var_1}
    var_5 = [var_4]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = [var_6]



# Parsed testcases at query #7
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 8
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

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
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(var_6 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = {var_0: var_1, var_3: var_4}
    var_11 = module_0.pmap(var_10)
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {f'key_{i}': i for i in var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100
    var_6 = module_0.pmap(var_2)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = [CollidingHash(i) for i in var_1]
    var_3 = {k: k.val for k in var_2}
    var_4 = 0
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 5



# Parsed testcases at query #8
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapView(var_5)
    var_7 = var_6._map
    var_8 = bool(var_6._map == var_5)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapView(var_4)
    var_6 = module_0.pmap(var_4)
    var_7 = var_5._map
    var_8 = bool(var_5._map == var_6)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'not a mapping'
    var_1 = module_0.PMapView(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pmap_equality_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_equality_with_different_dict. Retrieved 6/7 statements.
# Partially parsed test_pmap_equality_with_different_length. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_with_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_equality_with_different_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_equality_with_same_buckets. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_1, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = len(var_3)
    var_6 = var_1 * var_5
    var_7 = 8
    var_8 = var_6 or var_7
    var_9 = bool(not var_8)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_eq_with_different_cached_hash. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = len(var_1)
    var_3 = var_0 * var_2
    var_4 = 8
    var_5 = var_3 or var_4
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #16
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
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = var_5._cached_hash
    var_11 = bool(var_5._cached_hash == var_9._cached_hash)
    assert var_11 is True
    var_12 = bool(var_5 == var_9)
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (1 in var_1))
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pmapitems_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_non_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_invalid_arg. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 123
    var_7 = 'a'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = (var_8, var_9, var_10)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test__turbo_mapping_with_collision. Retrieved 6/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(var_2 == var_4)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

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
    var_9 = {var_0: var_1, var_3: var_4}
    var_10 = module_0.pmap(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

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
    var_9 = {var_0: var_1, var_3: var_4}
    var_10 = module_0.pmap(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 0
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 8



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pmap_constructor_creates_instance_with_given_size_and_buckets. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 5
    var_1 = None
    var_2 = [var_1]
    var_3 = 10
    var_4 = var_2 * var_3
    var_5 = [var_0, var_4]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pmap_equality_with_itself. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_with_identical_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_with_different_pmap. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_equality_with_different_dict. Retrieved 6/7 statements.
# Partially parsed test_pmap_equality_with_different_size. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_with_cached_hash. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_with_different_cached_hash. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pmap_constructor_creates_instance_with_correct_size_and_buckets. Retrieved 11/12 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = [var_1, var_5, var_9]
    var_11 = [var_0, var_10]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_eq_different_pmaps_same_content. Retrieved 2/4 statements.
# Partially parsed test_eq_pmap_and_dict_same_content. Retrieved 5/6 statements.
# Partially parsed test_eq_pmap_and_dict_different_content. Retrieved 6/7 statements.
# Partially parsed test_eq_pmap_and_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_eq_pmaps_different_sizes. Retrieved 3/5 statements.
# Partially parsed test_eq_pmaps_same_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_eq_pmaps_different_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_eq_pmaps_same_buckets. Retrieved 2/5 statements.
# Partially parsed test_eq_pmaps_different_buckets_same_content. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_contains_returns_false_for_invalid_arg. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_cached_hash_inequality. Retrieved 4/6 statements.


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
    var_10 = bool(var_5 != var_9)
    assert var_10 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_contains_returns_false_for_non_tuple_arg. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_cached_hash_inequality. Retrieved 15/19 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_4, var_6]
    var_8 = (var_1, var_7)
    var_9 = [var_8]
    var_10 = [var_0, var_9]
    var_11 = (var_2, var_3)
    var_12 = (var_5, var_0)
    var_13 = [var_11, var_12]
    var_14 = (var_1, var_13)
    var_15 = [var_14]
    var_16 = [var_0, var_15]



# Parsed testcases at query #36
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = bool(not 42 in var_1)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_eq_predicate_line_15. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_8]
    var_10 = {var_1: var_2, var_5: var_0}



# Parsed testcases at query #38
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5 == {'a': 1, 'b': 3}
    assert var_6 is False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_contains_with_non_tuple_arg. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_update_with_basic. Retrieved 6/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 9/10 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 6/7 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_keep_left. Retrieved 9/10 statements.
# Partially parsed test_update_with_keep_right. Retrieved 9/10 statements.
# Partially parsed test_update_with_complex_merge. Retrieved 11/13 statements.
# Partially parsed test_update_with_string_concatenation. Retrieved 6/7 statements.
# Partially parsed test_update_with_list_concatenation. Retrieved 12/13 statements.


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
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)

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
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = 5
    var_13 = {var_11: var_12}

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
    var_8 = 'c'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
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
    var_6 = lambda l, r: l
    var_7 = 3
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = 5
    var_13 = {var_11: var_12}

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
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = 5
    var_13 = {var_11: var_12}

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
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 5
    var_15 = 6
    var_16 = {var_12: var_14, var_13: var_15}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
    var_7 = ' there'
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)

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
    var_10 = lambda l, r: l + r
    var_11 = 5
    var_12 = 6
    var_13 = [var_11, var_12]
    var_14 = 'a'
    var_15 = {var_14: var_13}
    var_16 = module_0.m(**var_15)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test___contains___with_existing_key_value_pair. Retrieved 5/7 statements.
# Partially parsed test___contains___with_non_existing_key_value_pair. Retrieved 5/7 statements.
# Partially parsed test___contains___with_non_tuple_arg. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = []



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_update_with_basic_merge. Retrieved 5/9 statements.
# Partially parsed test_update_with_rightmost_value. Retrieved 7/11 statements.
# Partially parsed test_update_with_leftmost_value. Retrieved 7/11 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 9/13 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 7/11 statements.
# Partially parsed test_update_with_empty_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_empty_original. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = 3
    var_4 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: r
    var_3 = 3
    var_4 = 'a'
    var_5 = 5
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l
    var_3 = 3
    var_4 = 'a'
    var_5 = 5
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l * r
    var_3 = 3
    var_4 = 'a'
    var_5 = 'd'
    var_6 = 4
    var_7 = {var_4: var_3, var_5: var_6}
    var_8 = 6

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = 3
    var_4 = 'd'
    var_5 = 4
    var_6 = {var_4: var_5}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = lambda l, r: l + r
    var_2 = 1
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pmapitems_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_non_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_invalid_arg. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = None



# Parsed testcases at query #45
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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 6/7 statements.


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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 8/9 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = (var_4, var_0)
    var_6 = [var_3, var_5]
    var_7 = [var_6]
    var_8 = [var_0, var_7]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pmapitems_contains_returns_false_for_invalid_arg. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #50
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
    var_10 = bool(var_5 != var_9)
    assert var_10 is True



# Parsed testcases at query #51
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
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.m(**var_8)
    var_10 = var_5._cached_hash
    assert var_10 is None
    var_11 = var_9._cached_hash
    assert var_11 is None
    var_12 = bool(var_5 == var_9)
    assert var_12 is True
    var_13 = var_5._cached_hash
    var_14 = bool(var_5._cached_hash == var_9._cached_hash)
    assert var_14 is True
    var_15 = 3
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_0, var_17: var_15}
    var_19 = module_0.m(**var_18)
    var_20 = var_19._cached_hash
    assert var_20 is None
    var_21 = bool(var_5 != var_19)
    assert var_21 is True
    var_22 = var_5._cached_hash
    var_23 = bool(var_5._cached_hash != var_19._cached_hash)
    assert var_23 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test__turbo_mapping_with_collision. Retrieved 4/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0

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
    var_5 = 10
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
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: i for i in var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pmap_equality_with_non_dict_mapping. Retrieved 3/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)



# Parsed testcases at query #54
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 6/7 statements.


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



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_cached_hash_comparison. Retrieved 4/6 statements.


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
    var_10 = bool(var_5 != var_9)
    assert var_10 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_contains_returns_false_for_non_tuple_arg. Retrieved 2/4 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = []



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_eq_with_non_dict_mapping. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'NonDictMapping'
    var_7 = {}



# Parsed testcases at query #61
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 8
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = var_2._size
    assert var_3 == 0
    var_4 = var_2._buckets
    var_5 = len(var_4)
    assert var_5 == 8

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
    assert var_11 == 8
    var_12 = bool(var_8 == var_6)
    assert var_12 is True

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
    var_12 = var_11._size
    assert var_12 == 3
    var_13 = var_11._buckets
    var_14 = len(var_13)
    assert var_14 == 8
    var_15 = bool(var_11 == {'a': 1, 'b': 2, 'c': 3})
    assert var_15 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: str(i) for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._size
    assert var_5 == 100
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 200
    var_8 = bool(var_4 == var_2)
    assert var_8 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_cached_hash_comparison. Retrieved 15/19 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_4, var_6]
    var_8 = (var_1, var_7)
    var_9 = [var_8]
    var_10 = [var_0, var_9]
    var_11 = (var_2, var_3)
    var_12 = (var_5, var_0)
    var_13 = [var_11, var_12]
    var_14 = (var_1, var_13)
    var_15 = [var_14]
    var_16 = [var_0, var_15]



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #64
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (None in var_1))
    assert var_3 is True
    var_4 = False
    var_5 = bool(False == ('not a tuple' in var_1))
    assert var_5 is True
    var_6 = False
    var_7 = bool(False == (123 in var_1))
    assert var_7 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_pmap_constructor_creates_instance_with_correct_size_and_buckets. Retrieved 11/12 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = 2
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = [var_1, var_5, var_9]
    var_11 = [var_0, var_10]



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #67
#--------------------------




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



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pmap_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_eq_different_sizes. Retrieved 2/4 statements.
# Partially parsed test_pmap_eq_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_eq_with_other_mapping. Retrieved 7/10 statements.
# Partially parsed test_pmap_eq_with_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_with_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_eq_with_different_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_eq_with_same_buckets. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_pmapvalues_eq_self.
# Failed to parse test_pmapvalues_eq_other_instance.
# Failed to parse test_pmapvalues_eq_non_pmapvalues.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test__turbo_mapping_with_large_initial. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0

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
    var_5 = 4
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
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8['a']
    assert var_10 == 1
    var_11 = var_8['b']
    assert var_11 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: str(i) for i in var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 5/9 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 7/12 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 5/9 statements.
# Partially parsed test_update_with_empty_map. Retrieved 4/7 statements.
# Partially parsed test_update_with_keep_left. Retrieved 5/9 statements.
# Partially parsed test_update_with_keep_right. Retrieved 5/9 statements.
# Partially parsed test_update_with_complex_merge. Retrieved 5/9 statements.
# Partially parsed test_update_with_dict_argument. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = 3
    var_4 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = 3
    var_4 = 5
    var_5 = 6
    var_6 = 9

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = 3
    var_4 = 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = module_0.pmap()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l
    var_3 = 3
    var_4 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: r
    var_3 = 3
    var_4 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l * r
    var_3 = 3
    var_4 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = 'a'
    var_4 = 'c'
    var_5 = 3
    var_6 = 4
    var_7 = {var_3: var_5, var_4: var_6}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 1
    var_3 = 'a'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_0, var_6)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pmapitems_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_non_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_invalid_arg. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 123
    var_7 = 'a'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = (var_8, var_9, var_10)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_update_with_with_single_map. Retrieved 6/7 statements.
# Partially parsed test_update_with_with_multiple_maps. Retrieved 9/10 statements.
# Partially parsed test_update_with_with_leftmost_value. Retrieved 8/9 statements.
# Partially parsed test_update_with_with_new_key. Retrieved 5/6 statements.
# Partially parsed test_update_with_with_empty_map. Retrieved 3/4 statements.


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
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_1, var_9: var_7}
    var_11 = module_0.m(**var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l + r
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
    var_4 = lambda l, r: l + r
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
    var_4 = lambda l, r: l + r



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = 2
    var_1 = []
    var_2 = len(var_1)
    var_3 = var_0 * var_2
    var_4 = 8
    var_5 = var_3 or var_4
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #12
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (42 in var_1))
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__turbo_mapping_with_large_input. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0

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
    var_5 = 8
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_3)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5['a']
    assert var_7 == 1
    var_8 = var_5['b']
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: str(i) for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmapitems_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_non_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_invalid_arg. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = 'a'
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = 'b'
    var_12 = 2
    var_13 = (var_11, var_12)
    var_14 = (var_10, var_13)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmapitems_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_non_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_invalid_arg. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = 'a'
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = 'b'
    var_12 = 2
    var_13 = (var_11, var_12)
    var_14 = (var_10, var_13)



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = 2
    var_1 = []
    var_2 = len(var_1)
    var_3 = var_0 * var_2
    var_4 = 8
    var_5 = var_3 or var_4
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_eq_predicate_line_15. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_6]
    var_8 = [var_4, var_7]
    var_9 = [var_0, var_8]
    var_10 = {var_1: var_2, var_5: var_0}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_constructor_creates_instance_with_correct_size_and_buckets. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pmap_equality_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_equality_with_different_lengths. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_with_cached_hash. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_with_different_cached_hash. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_same_buckets. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_eq_with_non_dict_mapping. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #23
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (1 in var_1))
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_1, var_5, var_8]
    var_10 = [var_0, var_9]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pmap_equality_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_equality_with_dict_different_content. Retrieved 6/7 statements.
# Partially parsed test_pmap_equality_with_other_mapping. Retrieved 7/10 statements.
# Partially parsed test_pmap_equality_with_other_mapping_different_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_equality_different_sizes. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_with_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_equality_with_different_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_equality_same_buckets. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__turbo_mapping_with_empty_dict. Retrieved 7/10 statements.
# Partially parsed test__turbo_mapping_with_non_empty_dict. Retrieved 15/20 statements.
# Partially parsed test__turbo_mapping_with_pre_size. Retrieved 15/20 statements.
# Partially parsed test__turbo_mapping_with_list_of_tuples. Retrieved 17/22 statements.
# Partially parsed test__turbo_mapping_with_collision. Retrieved 13/17 statements.
# Partially parsed test__turbo_mapping_with_large_initial. Retrieved 7/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = 0
    var_4 = 8
    var_5 = [var_1]
    var_6 = var_4 * var_5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = 8
    var_8 = [var_5]
    var_9 = var_7 * var_8
    var_10 = 0
    var_11 = (var_0, var_2)
    var_12 = [var_11]
    var_13 = (var_1, var_3)
    var_14 = [var_13]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = None
    var_8 = [var_7]
    var_9 = var_5 * var_8
    var_10 = 0
    var_11 = (var_0, var_2)
    var_12 = [var_11]
    var_13 = (var_1, var_3)
    var_14 = [var_13]

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
    var_9 = 8
    var_10 = [var_7]
    var_11 = var_9 * var_10
    var_12 = 0
    var_13 = (var_0, var_1)
    var_14 = [var_13]
    var_15 = (var_3, var_4)
    var_16 = [var_15]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_3)
    var_6 = None
    var_7 = [var_6]
    var_8 = var_3 * var_7
    var_9 = 0
    var_10 = (var_0, var_2)
    var_11 = (var_1, var_3)
    var_12 = [var_10, var_11]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: i for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100
    var_6 = var_4[var_0]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_eq_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_eq_different_sizes. Retrieved 2/4 statements.
# Partially parsed test_eq_with_dict. Retrieved 5/6 statements.
# Partially parsed test_eq_with_other_mapping. Retrieved 7/10 statements.
# Partially parsed test_eq_cached_hash_mismatch. Retrieved 3/7 statements.
# Partially parsed test_eq_same_buckets. Retrieved 2/6 statements.
# Partially parsed test_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'not a mapping'



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = len(var_6)
    var_9 = var_4 * var_8
    var_10 = 8
    var_11 = var_9 or var_10
    var_12 = bool(not var_11)
    assert var_12 is True



# Parsed testcases at query #30
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (None in var_1))
    assert var_3 is True
    var_4 = False
    var_5 = bool(False == ('not a tuple' in var_1))
    assert var_5 is True
    var_6 = False
    var_7 = bool(False == (123 in var_1))
    assert var_7 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_turbo_mapping_with_empty_dict. Retrieved 7/10 statements.
# Partially parsed test_turbo_mapping_with_collision. Retrieved 4/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = 0
    var_4 = 8
    var_5 = [var_1]
    var_6 = var_4 * var_5

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_eq_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_eq_different_content. Retrieved 3/5 statements.
# Partially parsed test_eq_with_dict. Retrieved 5/6 statements.
# Partially parsed test_eq_with_dict_different_content. Retrieved 6/7 statements.
# Partially parsed test_eq_with_other_mapping. Retrieved 7/10 statements.
# Partially parsed test_eq_with_other_mapping_different_content. Retrieved 8/11 statements.
# Partially parsed test_eq_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_eq_different_sizes. Retrieved 3/5 statements.
# Partially parsed test_eq_with_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_eq_with_different_cached_hash. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pmapitems_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_non_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmapitems_contains_invalid_arg. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = 'b'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]
    var_6 = 'c'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 2
    var_11 = (var_9, var_10)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = 1
    var_6 = None



# Parsed testcases at query #36
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pmap_equality_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_equality_with_dict_different_content. Retrieved 6/7 statements.
# Partially parsed test_pmap_equality_with_other_mapping. Retrieved 7/10 statements.
# Partially parsed test_pmap_equality_different_sizes. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_cached_hash_mismatch. Retrieved 2/6 statements.
# Partially parsed test_pmap_equality_same_buckets. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 3
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_update_with_merges_values_from_multiple_maps. Retrieved 11/12 statements.
# Partially parsed test_update_with_keeps_leftmost_value. Retrieved 8/9 statements.
# Partially parsed test_update_with_uses_rightmost_value_when_key_not_in_left. Retrieved 5/6 statements.
# Partially parsed test_update_with_empty_map. Retrieved 3/4 statements.
# Partially parsed test_update_with_no_overlapping_keys. Retrieved 9/10 statements.
# Partially parsed test_update_with_all_overlapping_keys. Retrieved 7/8 statements.


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
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l + r
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
    var_4 = lambda l, r: l + r

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
    var_8 = 'c'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 'd'
    var_12 = 4
    var_13 = {var_11: var_12}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l * r
    var_7 = 3
    var_8 = 4
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)



# Parsed testcases at query #39
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (None in var_1))
    assert var_3 is True
    var_4 = False
    var_5 = bool(False == ('not_a_tuple' in var_1))
    assert var_5 is True
    var_6 = False
    var_7 = bool(False == (123 in var_1))
    assert var_7 is True



