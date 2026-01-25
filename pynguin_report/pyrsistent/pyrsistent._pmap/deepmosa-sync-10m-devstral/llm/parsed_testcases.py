####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_eq_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_eq_different_content. Retrieved 3/5 statements.
# Partially parsed test_eq_with_dict. Retrieved 5/6 statements.
# Partially parsed test_eq_with_different_length. Retrieved 3/5 statements.
# Partially parsed test_eq_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_eq_with_cached_hash. Retrieved 2/4 statements.
# Partially parsed test_eq_with_different_cached_hash. Retrieved 3/5 statements.
# Partially parsed test_eq_with_same_buckets. Retrieved 2/4 statements.
# Partially parsed test_eq_with_different_buckets. Retrieved 3/5 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pmapview_setattr_immutable. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_getattr_existing_key. Retrieved 2/3 statements.
# Partially parsed test_getattr_nonexistent_key. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)
    var_5 = bool(var_4 == {})
    assert var_5 is True

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
    var_8 = dict(var_6)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True

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
    var_8 = dict(var_6)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
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
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = dict(var_8)
    var_11 = bool(var_10 == {'a': 1, 'b': 2})
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 3
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]
    var_9 = None
    var_10 = module_0._turbo_mapping(var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = dict(var_10)
    var_13 = bool(var_12 == {'a': 3, 'b': 2})
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: str(i) for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100
    var_6 = dict(var_4)
    var_7 = bool(var_6 == var_2)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_equality_with_different_cached_hash. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 7/8 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 6/7 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_keep_left. Retrieved 8/9 statements.
# Partially parsed test_update_with_keep_right. Retrieved 8/9 statements.
# Partially parsed test_update_with_complex_merge. Retrieved 7/8 statements.
# Partially parsed test_update_with_string_concatenation. Retrieved 7/8 statements.
# Partially parsed test_update_with_list_concatenation. Retrieved 15/16 statements.
# Partially parsed test_update_with_new_key. Retrieved 5/6 statements.


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
    var_6 = lambda l, r: l * r
    var_7 = 3
    var_8 = 4
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)

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
    var_8 = '!'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)

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
    var_14 = 7
    var_15 = 8
    var_16 = [var_14, var_15]
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_13, var_18: var_16}
    var_20 = module_0.m(**var_19)

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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_cached_hash_comparison. Retrieved 15/19 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = (var_1, var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = (var_6, var_0)
    var_8 = [var_5, var_7]
    var_9 = [var_2, var_8]
    var_10 = [var_0, var_9]
    var_11 = (var_1, var_1)
    var_12 = (var_3, var_4)
    var_13 = (var_6, var_0)
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = [var_0, var_15]



# Parsed testcases at query #8
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'not_iterable'
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test___contains___with_existing_item_returns_true. Retrieved 5/7 statements.
# Partially parsed test___contains___with_non_existing_item_returns_false. Retrieved 5/7 statements.
# Partially parsed test___contains___with_non_tuple_arg_returns_false. Retrieved 5/7 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_eq_predicate_line_15. Retrieved 10/11 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_4, var_6]
    var_8 = [var_1, var_7]
    var_9 = [var_0, var_8]
    var_10 = {var_2: var_3, var_5: var_0}



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_eq_same_instance.
# Failed to parse test_eq_different_instances_same_map.
# Partially parsed test_eq_different_maps. Retrieved 4/9 statements.
# Partially parsed test_eq_non_pmapitems_instance. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = {var_0: var_1}
    var_5 = [var_4]

def test_case_0():
    var_0 = []
    var_1 = 'not a PMapItems'



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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
    var_8 = False
    var_9 = bool(False == ([1, 2] in var_1))
    assert var_9 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__turbo_mapping_with_collision. Retrieved 4/20 statements.
# Partially parsed test__turbo_mapping_with_large_initial. Retrieved 7/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)
    var_5 = bool(var_4 == {})
    assert var_5 is True

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
    var_10 = dict(var_8)
    var_11 = bool(var_10 == var_6)
    assert var_11 is True

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
    var_8 = dict(var_6)
    var_9 = bool(var_8 == var_4)
    assert var_9 is True

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
    var_13 = dict(var_11)
    var_14 = bool(var_13 == {'a': 1, 'b': 2, 'c': 3})
    assert var_14 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: str(i) for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100
    var_6 = range(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_eq_with_non_dict_mapping. Retrieved 14/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'MockMapping'
    var_3 = ()
    var_4 = '__len__'
    var_5 = 'items'
    var_6 = lambda self: var_1
    var_7 = 'a'
    var_8 = (var_7, var_0)
    var_9 = 'b'
    var_10 = (var_9, var_1)
    var_11 = [var_8, var_10]
    var_12 = lambda self: var_11
    var_13 = {var_4: var_6, var_5: var_12}
    var_14 = [var_2, var_3, var_13]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pmapitems_contains_returns_false_on_invalid_arg. Retrieved 2/4 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = []



# Parsed testcases at query #18
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
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_6}
    var_11 = module_0.m(**var_10)
    assert var_11 is False
    var_12 = bool(var_5 == var_11)
    assert var_12 is True



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_pmapitems_eq_same_instance.
# Failed to parse test_pmapitems_eq_different_type.
# Failed to parse test_pmapitems_eq_different_pmap.
# Failed to parse test_pmapitems_eq_different_pmap_with_items.
# Failed to parse test_pmapitems_eq_different_pmap_with_different_items.




# Parsed testcases at query #20
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)
    var_4 = {var_0: var_1}
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_update_with_merge_values. Retrieved 7/8 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/9 statements.
# Partially parsed test_update_with_new_keys. Retrieved 5/6 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 9/10 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 7/8 statements.


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
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_9: var_7, var_10: var_8}
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
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l * r
    var_7 = 3
    var_8 = 'a'
    var_9 = {var_8: var_7}
    var_10 = module_0.m(**var_9)
    var_11 = 'b'
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
    var_6 = lambda l, r: l + r
    var_7 = 3
    var_8 = 4
    var_9 = 'c'
    var_10 = 'd'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)



# Parsed testcases at query #22
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = bool(not var_1 == {})
    assert var_2 is True



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pmap_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_eq_different_instances_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_eq_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_eq_with_different_dict. Retrieved 6/7 statements.
# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_with_different_size. Retrieved 3/5 statements.
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



# Parsed testcases at query #25
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

# Partially parsed test_eq_with_non_dict_mapping. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = 'b'
    var_5 = (var_4, var_1)
    var_6 = [var_3, var_5]



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_contains_returns_false_for_invalid_arg. Retrieved 2/4 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 6/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 12/13 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 7/8 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_keep_left. Retrieved 7/8 statements.
# Partially parsed test_update_with_keep_right. Retrieved 7/8 statements.
# Partially parsed test_update_with_complex_merge. Retrieved 5/7 statements.


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
    var_8 = 4
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)
    var_13 = 'a'
    var_14 = 'd'
    var_15 = 5
    var_16 = 6
    var_17 = {var_13: var_15, var_14: var_16}

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
    var_8 = 4
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
    var_7 = 3
    var_8 = 4
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 4
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_1, var_8: var_6}
    var_10 = module_0.m(**var_9)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_update_with_basic_merge. Retrieved 6/7 statements.
# Partially parsed test_update_with_leftmost_priority. Retrieved 8/9 statements.
# Partially parsed test_update_with_rightmost_priority. Retrieved 8/9 statements.
# Partially parsed test_update_with_new_keys. Retrieved 8/9 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 5/6 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 9/10 statements.
# Partially parsed test_update_with_multiple_overlaps. Retrieved 13/14 statements.
# Partially parsed test_update_with_string_concatenation. Retrieved 7/8 statements.
# Partially parsed test_update_with_list_concatenation. Retrieved 15/16 statements.
# Partially parsed test_update_with_single_map. Retrieved 5/6 statements.


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
    var_4 = lambda l, r: l + r
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = 'c'
    var_10 = 3
    var_11 = {var_9: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l + r
    var_5 = {}
    var_6 = module_0.m(**var_5)
    var_7 = {}

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
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.m(**var_6)
    var_8 = lambda l, r: l + r
    var_9 = 10
    var_10 = 20
    var_11 = 'a'
    var_12 = 'b'
    var_13 = {var_11: var_9, var_12: var_10}
    var_14 = module_0.m(**var_13)
    var_15 = 'c'
    var_16 = 'd'
    var_17 = 30
    var_18 = 40
    var_19 = {var_15: var_17, var_16: var_18}

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
    var_8 = '!'
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)

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
    var_14 = 7
    var_15 = 8
    var_16 = [var_14, var_15]
    var_17 = 'a'
    var_18 = 'c'
    var_19 = {var_17: var_13, var_18: var_16}
    var_20 = module_0.m(**var_19)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l * r
    var_5 = 2
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test__turbo_mapping_with_collision. Retrieved 4/17 statements.
# Partially parsed test__turbo_mapping_with_large_initial_size. Retrieved 6/8 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: str(i) for i in var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 100



# Parsed testcases at query #33
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
    var_7 = bool(False == (1 in var_1))
    assert var_7 is True
    var_8 = False
    var_9 = bool(False == ([1, 2] in var_1))
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_eq_identical_pmaps. Retrieved 15/17 statements.
# Partially parsed test_eq_same_pmap. Retrieved 10/11 statements.
# Partially parsed test_eq_different_sizes. Retrieved 19/21 statements.
# Partially parsed test_eq_different_content. Retrieved 16/18 statements.
# Partially parsed test_eq_with_dict. Retrieved 11/12 statements.
# Partially parsed test_eq_with_different_dict. Retrieved 12/13 statements.
# Partially parsed test_eq_with_non_mapping. Retrieved 10/11 statements.
# Partially parsed test_eq_with_cached_hash. Retrieved 15/19 statements.
# Partially parsed test_eq_with_different_cached_hash. Retrieved 15/19 statements.
# Partially parsed test_eq_with_same_buckets. Retrieved 10/12 statements.


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
    var_11 = (var_2, var_3)
    var_12 = [var_11]
    var_13 = (var_6, var_0)
    var_14 = [var_13]
    var_15 = [var_1, var_12, var_14]
    var_16 = [var_0, var_15]

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
    var_11 = 3
    var_12 = (var_2, var_3)
    var_13 = [var_12]
    var_14 = (var_6, var_0)
    var_15 = [var_14]
    var_16 = 'c'
    var_17 = (var_16, var_11)
    var_18 = [var_17]
    var_19 = [var_1, var_13, var_15, var_18]
    var_20 = [var_11, var_19]

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
    var_11 = (var_2, var_3)
    var_12 = [var_11]
    var_13 = 3
    var_14 = (var_6, var_13)
    var_15 = [var_14]
    var_16 = [var_1, var_12, var_15]
    var_17 = [var_0, var_16]

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
    var_11 = {var_2: var_3, var_6: var_0}

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
    var_11 = 3
    var_12 = {var_2: var_3, var_6: var_11}

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
    var_11 = (var_2, var_3)
    var_12 = [var_11]
    var_13 = (var_6, var_0)
    var_14 = [var_13]
    var_15 = [var_1, var_12, var_14]
    var_16 = [var_0, var_15]

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
    var_11 = (var_2, var_3)
    var_12 = [var_11]
    var_13 = (var_6, var_0)
    var_14 = [var_13]
    var_15 = [var_1, var_12, var_14]
    var_16 = [var_0, var_15]

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
    var_9 = [var_0, var_4, var_8]
    var_10 = [var_6, var_9]
    var_11 = [var_6, var_9]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pmap_items_contains_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmap_items_contains_non_existing_item. Retrieved 5/7 statements.
# Partially parsed test_pmap_items_contains_invalid_arg. Retrieved 5/7 statements.


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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_eq_predicate_line_15. Retrieved 10/12 statements.


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



# Parsed testcases at query #37
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = bool(var_1 == var_1)
    assert var_2 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = bool(not var_1 == {})
    assert var_2 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.PMapItems(var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)
    var_6 = 3
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.PMapItems(var_7)
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__turbo_mapping_with_collision. Retrieved 4/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)
    var_5 = bool(var_4 == {})
    assert var_5 is True

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
    var_10 = dict(var_8)
    var_11 = bool(var_10 == var_6)
    assert var_11 is True

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
    var_8 = dict(var_6)
    var_9 = bool(var_8 == var_4)
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
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = dict(var_8)
    var_11 = bool(var_10 == {'a': 1, 'b': 2})
    assert var_11 is True

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
    var_10 = 0
    var_11 = module_0._turbo_mapping(var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = dict(var_11)
    var_14 = bool(var_13 == {'a': 1, 'b': 2, 'c': 3})
    assert var_14 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_update_with_basic_merge. Retrieved 5/9 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 6/11 statements.
# Partially parsed test_update_with_new_key. Retrieved 3/7 statements.
# Partially parsed test_update_with_keep_left. Retrieved 4/9 statements.
# Partially parsed test_update_with_empty_map. Retrieved 3/6 statements.
# Partially parsed test_update_with_dict. Retrieved 7/10 statements.
# Partially parsed test_update_with_no_overlap. Retrieved 3/7 statements.


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
    var_5 = 9

def test_case_0():
    var_0 = 1
    var_1 = lambda l, r: l + r
    var_2 = 2

def test_case_0():
    var_0 = 1
    var_1 = lambda l, r: l
    var_2 = 2
    var_3 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda l, r: l + r
    var_2 = module_0.pmap()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = lambda l, r: l + r
    var_3 = 'a'
    var_4 = 3
    var_5 = {var_3: var_4}
    var_6 = 4

def test_case_0():
    var_0 = 1
    var_1 = lambda l, r: l + r
    var_2 = 2



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 11/12 statements.


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




def test_case_0():
    var_0 = 2
    var_1 = []
    var_2 = len(var_1)
    var_3 = var_0 * var_2
    var_4 = 8
    var_5 = var_3 or var_4
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = bool(not 1 in var_1)
    assert var_2 is True



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pmap_constructor. Retrieved 9/10 statements.


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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pmap_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_eq_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_eq_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_eq_with_dict_different_content. Retrieved 6/7 statements.
# Partially parsed test_pmap_eq_with_other_mapping. Retrieved 7/10 statements.
# Partially parsed test_pmap_eq_with_other_mapping_different_content. Retrieved 8/11 statements.
# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_different_lengths. Retrieved 3/5 statements.
# Partially parsed test_pmap_eq_with_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_eq_with_different_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_eq_same_buckets. Retrieved 2/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pmap_equality_with_itself. Retrieved 2/3 statements.
# Partially parsed test_pmap_equality_with_different_pmap. Retrieved 2/4 statements.
# Partially parsed test_pmap_equality_with_different_pmap_different_values. Retrieved 3/5 statements.
# Partially parsed test_pmap_equality_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_equality_with_dict_different_values. Retrieved 6/7 statements.
# Partially parsed test_pmap_equality_with_dict_different_keys. Retrieved 5/6 statements.
# Partially parsed test_pmap_equality_with_dict_different_length. Retrieved 4/5 statements.
# Partially parsed test_pmap_equality_with_other_mapping. Retrieved 7/10 statements.
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
    var_2 = 'a'
    var_3 = 'c'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2: var_0}

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

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_eq_with_non_dict_mapping. Retrieved 6/16 statements.


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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test__turbo_mapping_with_empty_dict. Retrieved 7/10 statements.


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



# Parsed testcases at query #22
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (1 in var_1))
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = False
    var_3 = bool(False == (None in var_1))
    assert var_3 is True



# Parsed testcases at query #24
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
    var_10 = bool(not var_5._buckets == var_9._buckets)
    assert var_10 is True



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 9/10 statements.


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
    var_12 = 'a'
    var_13 = {var_12: var_0}
    var_14 = module_0.m(**var_13)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_update_with_when_key_not_in_evolver. Retrieved 5/6 statements.


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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pmap_eq_same_instance. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_different_instances_same_content. Retrieved 2/4 statements.
# Partially parsed test_pmap_eq_different_sizes. Retrieved 2/4 statements.
# Partially parsed test_pmap_eq_with_dict. Retrieved 5/6 statements.
# Partially parsed test_pmap_eq_with_pmap_different_content. Retrieved 3/5 statements.
# Partially parsed test_pmap_eq_with_pmap_same_buckets. Retrieved 2/6 statements.
# Partially parsed test_pmap_eq_with_pmap_different_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_pmap_eq_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_pmap_eq_with_mapping_different_length. Retrieved 2/9 statements.


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




def test_case_0():
    var_0 = 2
    var_1 = []
    var_2 = len(var_1)
    var_3 = var_0 * var_2
    var_4 = 8
    var_5 = var_3 or var_4
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapItems(var_0)
    var_2 = bool(not 1 in var_1)
    assert var_2 is True



# Parsed testcases at query #31
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



# Parsed testcases at query #32
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



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_contains_returns_false_for_non_tuple_arg.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_eq_with_non_mapping_other. Retrieved 4/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = 'not a mapping'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_equality_with_same_pmap_instance. Retrieved 2/3 statements.
# Partially parsed test_equality_with_different_pmap_same_content. Retrieved 2/4 statements.
# Partially parsed test_equality_with_different_pmap_different_content. Retrieved 3/5 statements.
# Partially parsed test_equality_with_dict_same_content. Retrieved 5/6 statements.
# Partially parsed test_equality_with_dict_different_content. Retrieved 6/7 statements.
# Partially parsed test_equality_with_different_size. Retrieved 3/5 statements.
# Partially parsed test_equality_with_non_mapping. Retrieved 2/3 statements.
# Partially parsed test_equality_with_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_equality_with_different_cached_hash. Retrieved 2/6 statements.
# Partially parsed test_equality_with_same_buckets. Retrieved 2/5 statements.


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



