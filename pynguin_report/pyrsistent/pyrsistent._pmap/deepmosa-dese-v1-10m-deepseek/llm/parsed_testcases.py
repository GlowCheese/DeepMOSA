####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_creates_pmap_with_correct_slots. Retrieved 1/5 statements.
# Partially parsed test_constructor_via_new_with_size_and_buckets. Retrieved 11/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.pmap(var_7)
    var_9 = len(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = list(var_1)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = len(var_2)
    assert var_3 == 2

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
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_0.pmap(var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k1'
    var_1 = 'v1'
    var_2 = (var_0, var_1)
    var_3 = 'k2'
    var_4 = 'v2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = hash(var_3)
    var_5 = hash(var_3)
    var_6 = '_cached_hash'
    var_7 = hasattr(var_3, var_6)

def test_case_0():
    var_0 = '_cached_hash'

def test_case_0():
    var_0 = 3
    var_1 = 'x'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'y'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = 'z'
    var_8 = 3
    var_9 = (var_7, var_8)
    var_10 = (var_3, var_6, var_9)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair_present. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_value_pair_absent. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test___contains___with_wrong_length_tuple. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_tuple. Retrieved 7/10 statements.
# Partially parsed test___contains___with_single_element_tuple. Retrieved 7/10 statements.
# Partially parsed test___contains___with_none_argument. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 3
    var_7 = (var_0, var_2, var_6)

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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_creates_pmap_with_correct_size_and_buckets. Retrieved 8/9 statements.
# Partially parsed test_constructor_creates_pmap_with_zero_size_and_empty_buckets. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_pmap_with_large_size_and_buckets. Retrieved 2/5 statements.
# Partially parsed test_constructor_creates_pmap_with_none_buckets. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_pmap_with_single_bucket. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)

def test_case_0():
    var_0 = 0
    var_1 = ()

def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)

def test_case_0():
    var_0 = 0
    var_1 = None

def test_case_0():
    var_0 = 1
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = (var_3,)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_turbo_mapping_with_initial_having_collisions. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_with_initial_as_mapping_subclass. Retrieved 6/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 10
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

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
    var_8 = dict(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 20
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = dict(var_6)

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

def test_case_0():
    var_0 = 'key1'
    var_1 = 5
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = 0

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
    var_9 = module_0._turbo_mapping(var_8, var_5)
    var_10 = len(var_9)
    assert var_10 == 4
    var_11 = dict(var_9)

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'baz'
    var_2 = 'bar'
    var_3 = 'qux'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = module_0._turbo_mapping(var_4, var_5)
    var_8 = hash(var_6)
    var_9 = hash(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}
    var_4 = 0
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 100



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_update_with_merges_values_using_update_fn. Retrieved 5/7 statements.
# Partially parsed test_update_with_keeps_leftmost_value_when_update_fn_returns_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_inserts_new_key_from_single_map. Retrieved 5/7 statements.
# Partially parsed test_update_with_inserts_new_keys_from_multiple_maps. Retrieved 8/10 statements.
# Partially parsed test_update_with_overwrites_with_rightmost_when_update_fn_returns_right. Retrieved 8/10 statements.
# Partially parsed test_update_with_on_empty_map. Retrieved 5/7 statements.
# Partially parsed test_update_with_no_maps_returns_same_map. Retrieved 4/6 statements.
# Partially parsed test_update_with_using_operator_add. Retrieved 5/8 statements.
# Partially parsed test_update_with_handles_none_values. Retrieved 5/7 statements.
# Partially parsed test_update_with_returns_new_map_when_changes_made. Retrieved 5/7 statements.
# Partially parsed test_update_with_returns_same_map_when_no_changes. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.m()
    var_1 = lambda l, r: l + r
    var_2 = 1
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.m()
    var_2 = lambda l, r: r if r is not var_0 else l
    var_3 = 1
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = module_0.m()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_creates_pmap_with_colliding_keys. Retrieved 4/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = {var_1: var_2, var_4: var_5}
    var_9 = module_0.pmap(var_8)
    var_10 = var_9._buckets
    var_11 = dict(var_10)
    var_12 = dict(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._buckets
    var_3 = len(var_2)
    assert var_3 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = frozenset(var_5)
    var_7 = 'tuple_key'
    var_8 = 'frozenset_key'
    var_9 = {var_2: var_7, var_6: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = [var_3, var_4]
    var_12 = frozenset(var_11)
    var_13 = var_10[var_12]
    assert var_13 == 'frozenset_key'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._buckets
    var_3 = len(var_2)
    assert var_3 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = -1
    var_1 = -2
    var_2 = 'minus_one'
    var_3 = 'minus_two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_update_with_does_not_call_update_fn_when_key_not_in_evolver. Retrieved 5/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()
    var_4 = module_0.m()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_eq_equal_pmap_different_buckets. Retrieved 5/10 statements.
# Partially parsed test_eq_other_mapping_protocol. Retrieved 6/18 statements.
# Partially parsed test_eq_other_mapping_protocol_not_equal. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = var_2 == var_2
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'c'
    var_4 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'c'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2 == var_6
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = (var_3, var_0)
    var_5 = 'b'
    var_6 = (var_5, var_1)
    var_7 = [var_4, var_6]
    var_8 = var_2 == var_7
    assert var_8 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = hash(var_2)
    var_6 = hash(var_4)
    var_7 = var_2 == var_4
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = hash(var_2)
    var_5 = hash(var_3)
    var_6 = var_2 == var_3
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}



# Parsed testcases at query #9
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_update_with_does_not_call_update_fn_when_key_not_in_evolver. Retrieved 5/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    assert var_0 == 0
    var_1 = 1
    var_2 = module_0.m()
    var_3 = 2
    var_4 = module_0.m()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_contains_predicate_true. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)



# Parsed testcases at query #12
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'c'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2 == var_6
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = {var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = var_3 == var_7
    assert var_8 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_contains_with_non_tuple_arg. Retrieved 8/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_contains_returns_false_on_invalid_arg. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = 3
    var_6 = (var_1, var_4, var_5)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_constructor_creates_pmap_with_correct_size_and_buckets. Retrieved 8/9 statements.
# Partially parsed test_constructor_creates_pmap_with_zero_size_and_empty_buckets. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_pmap_with_large_size_and_buckets. Retrieved 2/5 statements.
# Partially parsed test_constructor_creates_pmap_with_none_buckets. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_pmap_with_single_element_buckets. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)

def test_case_0():
    var_0 = 0
    var_1 = ()

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)

def test_case_0():
    var_0 = 0
    var_1 = None

def test_case_0():
    var_0 = 1
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = (var_3,)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_turbo_mapping_predicate_at_line_7_false. Retrieved 6/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_eq_with_dict_and_different_cached_hash. Retrieved 8/10 statements.
# Partially parsed test_eq_with_dict_and_same_cached_hash. Retrieved 8/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
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
    var_7 = var_5 == var_6
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
    var_7 = var_5 == var_6
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
    var_7 = var_5 == var_6
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
    var_8 = var_5 == var_7
    assert var_8 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test___contains___with_existing_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test___contains___with_existing_key_but_different_value. Retrieved 7/10 statements.
# Partially parsed test___contains___with_non_existing_key. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test___contains___with_wrong_length_tuple. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_pmap. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 3
    var_7 = (var_0, var_2, var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_creates_hashable_pmap. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._buckets
    var_7 = len(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)

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
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

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
    var_8 = len(var_7)
    assert var_8 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_0.pmap(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = hash(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = dict(var_1)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'single'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = len(var_3)
    assert var_4 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_eq_pmap_vs_other_mapping. Retrieved 6/18 statements.
# Partially parsed test_eq_cached_hash_mismatch. Retrieved 5/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = var_2 == var_2
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2 == var_6
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = var_2 == var_4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2._buckets
    var_5 = var_3._buckets
    var_6 = var_4 == var_5
    assert var_6 is True
    var_7 = var_2 == var_3
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = module_0._turbo_mapping(var_3, var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_constructor_creates_pmap_with_given_size_and_buckets. Retrieved 8/9 statements.
# Partially parsed test_constructor_returns_pmap_instance. Retrieved 2/4 statements.
# Partially parsed test_constructor_sets_size_and_buckets_correctly. Retrieved 11/12 statements.
# Partially parsed test_constructor_with_empty_pmap. Retrieved 2/3 statements.
# Partially parsed test_constructor_sets_correct_size_for_non_empty_pmap. Retrieved 9/10 statements.
# Partially parsed test_constructor_preserves_bucket_structure. Retrieved 9/10 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)

def test_case_0():
    var_0 = 0
    var_1 = ()

def test_case_0():
    var_0 = 3
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = 'c'
    var_8 = 3
    var_9 = (var_7, var_8)
    var_10 = (var_3, var_6, var_9)

def test_case_0():
    var_0 = 0
    var_1 = ()

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = len(var_6)
    var_8 = len(var_6)

def test_case_0():
    var_0 = 'k1'
    var_1 = 'v1'
    var_2 = (var_0, var_1)
    var_3 = None
    var_4 = 'k2'
    var_5 = 'v2'
    var_6 = (var_4, var_5)
    var_7 = (var_2, var_3, var_6)
    var_8 = 2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_creates_pmap_that_is_hashable. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = {var_1: var_2, var_4: var_5}
    var_9 = module_0.pmap(var_8)
    var_10 = var_9._buckets
    var_11 = dict(var_10)
    var_12 = dict(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = var_0._buckets
    var_2 = len(var_1)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._buckets
    var_3 = len(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = hash(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

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
    var_8 = len(var_7)
    assert var_8 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = list(var_5)
    var_7 = set(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = var_3[var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3.b

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'one'
    var_1 = 'two'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'tuple_key'
    var_4 = {var_2: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = 123
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 'int'
    var_6 = 'tuple'
    var_7 = {var_0: var_2, var_1: var_5, var_4: var_6}
    var_8 = module_0.pmap(var_7)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_contains_with_invalid_arg. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = (var_0, var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_contains_with_valid_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test_contains_with_key_in_map_but_wrong_value. Retrieved 7/10 statements.
# Partially parsed test_contains_with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test_contains_with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test_contains_with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test_contains_with_empty_tuple. Retrieved 7/10 statements.
# Partially parsed test_contains_with_none_argument. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)

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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None



# Parsed testcases at query #26
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_update_with_single_map_merge. Retrieved 5/7 statements.
# Partially parsed test_update_with_multiple_maps_merge. Retrieved 10/12 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_new_key. Retrieved 6/8 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/7 statements.
# Partially parsed test_update_with_no_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_identity_merge. Retrieved 6/8 statements.
# Partially parsed test_update_with_complex_merge. Retrieved 11/13 statements.
# Partially parsed test_update_with_preserves_original. Retrieved 5/7 statements.
# Partially parsed test_update_with_merge_on_nonexistent_key. Retrieved 6/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'c'
    var_7 = 3
    var_8 = 4
    var_9 = {var_5: var_7, var_6: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = 3
    var_5 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = {}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r
    var_4 = 3
    var_5 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: max(l, r)
    var_4 = 5
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'c'
    var_8 = 3
    var_9 = 4
    var_10 = {var_6: var_8, var_7: var_9}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = 3
    var_5 = module_0.m()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_contains_with_valid_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test_contains_with_key_in_map_but_different_value. Retrieved 7/10 statements.
# Partially parsed test_contains_with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test_contains_with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test_contains_with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test_contains_with_empty_tuple. Retrieved 7/10 statements.
# Partially parsed test_contains_with_none_argument. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 3
    var_7 = (var_0, var_2, var_6)

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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_constructor_with_none_buckets_raises_error. Retrieved 2/5 statements.
# Partially parsed test_constructor_preserves_hash_collision_handling. Retrieved 2/11 statements.
# Partially parsed test_constructor_creates_immutable_instance. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.pmap(var_7)
    var_9 = len(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = dict(var_1)

def test_case_0():
    var_0 = 0
    var_1 = None

def test_case_0():
    var_0 = 'val1'
    var_1 = 'val2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0.pmap(var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'str'
    var_2 = 2
    var_3 = (var_0, var_2)
    var_4 = 'int'
    var_5 = 'string'
    var_6 = 'tuple'
    var_7 = {var_0: var_4, var_1: var_5, var_3: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 'string'
    var_6 = 2
    var_7 = [var_4, var_6]
    var_8 = 'nested'
    var_9 = 'dict'
    var_10 = {var_8: var_9}
    var_11 = {var_0: var_4, var_1: var_5, var_2: var_7, var_3: var_10}
    var_12 = module_0.pmap(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = hash(var_3)
    var_7 = hash(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)
    var_2 = {str(i): i for i in var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = len(var_3)
    assert var_4 == 1000



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_constructor_pmap_is_immutable. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.pmap(var_7)
    var_9 = len(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = list(var_1)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = len(var_5)
    assert var_6 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = module_0.m()
    var_3 = len(var_2)
    assert var_3 == 2

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
    var_8 = hash(var_5)
    var_9 = hash(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k1'
    var_1 = 'k2'
    var_2 = 'v1'
    var_3 = 'v2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'missing'
    var_3 = var_1[var_2]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.m()
    var_1 = var_0.missing



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 10/11 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/9 statements.
# Partially parsed test_update_with_keep_rightmost. Retrieved 8/9 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/6 statements.
# Partially parsed test_update_with_no_maps. Retrieved 4/5 statements.
# Partially parsed test_update_with_new_key. Retrieved 5/6 statements.
# Partially parsed test_update_with_identity_function. Retrieved 7/8 statements.
# Partially parsed test_update_with_constant_function. Retrieved 8/9 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 5/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 3
    var_8 = 4
    var_9 = {var_5: var_7, var_6: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.m()
    var_1 = lambda l, r: l + r
    var_2 = 1
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l + r
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l
    var_4 = 5
    var_5 = 3
    var_6 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 42
    var_4 = lambda l, r: var_3
    var_5 = 5
    var_6 = 3
    var_7 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()



# Parsed testcases at query #32
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_constructor_preserves_hash_collision_handling. Retrieved 2/10 statements.
# Partially parsed test_constructor_handles_large_number_of_efficiently. Retrieved 7/10 statements.
# Partially parsed test_constructor_creates_pmap_that_is_immutable. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.pmap(var_7)
    var_9 = len(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = dict(var_1)

def test_case_0():
    var_0 = 'val1'
    var_1 = 'val2'

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
    var_8 = hash(var_5)
    var_9 = hash(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}
    var_4 = module_0.pmap(var_3)
    var_5 = len(var_4)
    assert var_5 == 1000
    var_6 = range(var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = len(var_3)
    assert var_4 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 'v'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = 'b'
    var_6 = 3
    var_7 = (var_5, var_6)
    var_8 = [var_2, var_4, var_7]
    var_9 = module_0.pmap(var_8)
    var_10 = len(var_9)
    assert var_10 == 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_contains_returns_false_on_invalid_arg. Retrieved 8/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)



# Parsed testcases at query #35
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair_present. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_value_pair_absent. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 7/10 statements.
# Partially parsed test___contains___with_tuple_wrong_length. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_map. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_constructor_sets_correct_attributes. Retrieved 2/6 statements.
# Failed to parse test_constructor_returns_pmap_instance.
# Partially parsed test_constructor_allows_weakref_support. Retrieved 4/8 statements.
# Partially parsed test_constructor_initializes_without_cached_hash. Retrieved 2/7 statements.
# Partially parsed test_constructor_handles_non_empty_buckets. Retrieved 6/11 statements.
# Failed to parse test_constructor_supports_generic_types.
# Partially parsed test_constructor_maintains_slots. Retrieved 3/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._buckets
    var_7 = len(var_6)

def test_case_0():
    var_0 = None
    var_1 = (var_0, var_0, var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

def test_case_0():
    var_0 = None
    var_1 = '_cached_hash'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._buckets
    var_3 = len(var_2)

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = [var_4, var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = hash(var_3)

def test_case_0():
    var_0 = '__dict__'
    var_1 = '_size'
    var_2 = '_buckets'



# Parsed testcases at query #38
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = '_cached_hash'
    var_7 = hasattr(var_2, var_6)
    var_8 = hasattr(var_5, var_6)
    var_9 = var_2._cached_hash
    var_10 = var_5._cached_hash
    var_11 = var_9 != var_10
    var_12 = var_7 and var_8 and var_11



# Parsed testcases at query #39
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair_present. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_value_pair_absent. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test___contains___with_tuple_of_wrong_length. Retrieved 7/10 statements.
# Partially parsed test___contains___with_empty_tuple. Retrieved 7/10 statements.
# Partially parsed test___contains___with_empty_map. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = (var_0, var_2, var_3)

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
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_update_with_merge_function. Retrieved 8/10 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 9/10 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 15/16 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 4/5 statements.
# Partially parsed test_update_with_new_key. Retrieved 8/9 statements.
# Partially parsed test_update_with_overwrites_existing. Retrieved 10/11 statements.
# Partially parsed test_update_with_identity_function. Retrieved 11/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 3
    var_7 = {var_4: var_6, var_5: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_0}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = 3
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'd'
    var_8 = 10
    var_9 = 4
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'b'
    var_12 = 'c'
    var_13 = 13
    var_14 = {var_6: var_13, var_11: var_1, var_12: var_4, var_7: var_9}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r * var_1
    var_4 = 3
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 6
    var_9 = {var_6: var_8, var_7: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r
    var_4 = 5
    var_5 = 7
    var_6 = module_0.m()
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_4, var_8: var_1, var_9: var_5}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_turbo_mapping_with_collision_handling. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_preserves_hash_collision_buckets. Retrieved 5/17 statements.
# Partially parsed test_turbo_mapping_initial_length_hint_fallback. Retrieved 1/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 100
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 200
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2

def test_case_0():
    var_0 = 'a'
    var_1 = 5
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

def test_case_0():
    var_0 = 0



# Parsed testcases at query #42
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair_present. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_value_pair_absent. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test___contains___with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_map. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_eq_with_mapping_protocol. Retrieved 6/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = var_2 == var_2
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2 == var_6
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = [var_0, var_1]
    var_4 = var_2 == var_3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = hash(var_2)
    var_6 = hash(var_4)
    var_7 = var_2 == var_4
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.pmap(var_4)
    var_7 = var_5 == var_6
    assert var_7 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_creates_pmap_that_is_immutable. Retrieved 4/7 statements.
# Partially parsed test_constructor_creates_pmap_with_correct_iteritems. Retrieved 6/10 statements.
# Partially parsed test_constructor_creates_pmap_with_get_method. Retrieved 6/10 statements.
# Partially parsed test_constructor_creates_pmap_with_keys_and_values_and_items. Retrieved 6/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._buckets
    var_7 = len(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._buckets
    var_3 = len(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = len(var_5)
    assert var_6 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = len(var_2)
    assert var_3 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k1'
    var_1 = 'v1'
    var_2 = (var_0, var_1)
    var_3 = 'k2'
    var_4 = 'v2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.pmap(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = hash(var_3)
    var_7 = hash(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = hash(var_3)
    var_5 = '_cached_hash'
    var_6 = hasattr(var_3, var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = var_3 < var_7

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = repr(var_3)
    assert var_4 == "pmap({'a': 1})"
    var_5 = str(var_3)
    assert var_5 == "pmap({'a': 1})"

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = reversed(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 'default'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair_present. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_value_pair_absent. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 7/10 statements.
# Partially parsed test___contains___with_wrong_length_tuple. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_map. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair_present. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_value_pair_absent. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 7/10 statements.
# Partially parsed test___contains___with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_map. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_update_with_merges_values_using_update_fn. Retrieved 5/7 statements.
# Partially parsed test_update_with_keeps_leftmost_value_when_update_fn_returns_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_inserts_new_key_from_single_map. Retrieved 6/8 statements.
# Partially parsed test_update_with_inserts_new_keys_from_multiple_maps. Retrieved 9/11 statements.
# Partially parsed test_update_with_overwrites_with_rightmost_when_update_fn_returns_right. Retrieved 8/10 statements.
# Partially parsed test_update_with_on_empty_map. Retrieved 7/9 statements.
# Partially parsed test_update_with_returns_same_instance_if_no_changes. Retrieved 5/7 statements.
# Partially parsed test_update_with_using_custom_update_fn. Retrieved 6/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'c'
    var_7 = 3
    var_8 = {var_6: var_7}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 'a'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = {var_3: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.m()
    var_1 = lambda l, r: l + r
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l
    var_4 = {}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'z'
    var_5 = {var_3: var_4}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_update_with_merges_values_using_update_fn. Retrieved 4/7 statements.
# Partially parsed test_update_with_keeps_leftmost_value_when_update_fn_returns_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_inserts_new_keys_from_multiple_maps. Retrieved 8/10 statements.
# Partially parsed test_update_with_overwrites_existing_keys_using_update_fn. Retrieved 10/12 statements.
# Partially parsed test_update_with_empty_maps_returns_original. Retrieved 4/6 statements.
# Partially parsed test_update_with_single_map_merges_values. Retrieved 7/9 statements.
# Partially parsed test_update_with_non_existing_key_uses_new_value. Retrieved 5/7 statements.
# Partially parsed test_update_with_returns_new_pmap_when_changes_made. Retrieved 5/7 statements.
# Partially parsed test_update_with_returns_same_pmap_when_no_changes. Retrieved 5/7 statements.
# Partially parsed test_update_with_complex_update_fn. Retrieved 8/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = 10
    var_5 = 20
    var_6 = module_0.m()
    var_7 = 'a'
    var_8 = 100
    var_9 = {var_7: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r * var_1
    var_4 = 3
    var_5 = 4
    var_6 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l + r
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = module_0.m()
    var_3 = lambda l, r: max(l, r)
    var_4 = 3
    var_5 = 15
    var_6 = 20
    var_7 = module_0.m()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_eq_same_buckets. Retrieved 5/7 statements.
# Partially parsed test_eq_different_cached_hash. Retrieved 5/7 statements.
# Partially parsed test_eq_other_mapping. Retrieved 6/18 statements.
# Partially parsed test_eq_other_mapping_different. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = var_2 == var_2
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2 == var_6
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = [var_0, var_1]
    var_4 = var_2 == var_3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair_present. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_value_pair_absent. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test___contains___with_wrong_length_tuple. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_tuple. Retrieved 7/10 statements.
# Partially parsed test___contains___with_single_element_tuple. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'c'
    var_7 = (var_6, var_2)

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
    var_6 = 3
    var_7 = (var_0, var_2, var_6)

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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0,)



# Parsed testcases at query #8
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = module_0.pmap(var_7)
    var_9 = len(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = list(var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = len(var_5)
    assert var_6 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = module_0.pmap()
    var_3 = len(var_2)
    assert var_3 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = module_0.pmap(var_2)
    var_5 = len(var_4)
    assert var_5 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'inner'
    var_5 = {var_4: var_3}
    var_6 = module_0.pmap(var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = hash(var_3)
    var_7 = hash(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0.pmap(var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k1'
    var_1 = 'v1'
    var_2 = (var_0, var_1)
    var_3 = 'k2'
    var_4 = 'v2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.pmap(var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_turbo_mapping_with_collision_keys. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_initial_length_exception_falls_back. Retrieved 1/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 10
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

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
    var_8 = dict(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 20
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = dict(var_6)

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

def test_case_0():
    var_0 = 'key1'
    var_1 = 5
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = 0

def test_case_0():
    var_0 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = module_0._turbo_mapping(var_4, var_5)
    var_8 = hash(var_6)
    var_9 = hash(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

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



# Parsed testcases at query #10
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
    assert var_7 is True
    var_8 = {var_0: var_2, var_1: var_3}
    var_9 = module_0.pmap(var_8)
    var_10 = 3
    var_11 = {var_0: var_2, var_1: var_10}
    var_12 = var_9 == var_11
    assert var_12 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_update_with_merge_function. Retrieved 8/10 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 9/10 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 15/16 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 4/5 statements.
# Partially parsed test_update_with_new_key. Retrieved 8/9 statements.
# Partially parsed test_update_with_overwrites_existing. Retrieved 10/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 3
    var_7 = {var_4: var_6, var_5: var_1}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_0}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = 3
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'd'
    var_8 = 10
    var_9 = 4
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'b'
    var_12 = 'c'
    var_13 = 13
    var_14 = {var_6: var_13, var_11: var_1, var_12: var_4, var_7: var_9}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r * var_1
    var_4 = 3
    var_5 = module_0.m()
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 6
    var_9 = {var_6: var_8, var_7: var_1}



# Parsed testcases at query #12
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
    assert var_7 is True
    var_8 = 'c'
    var_9 = 3
    var_10 = {var_0: var_2, var_1: var_3, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = {var_0: var_2, var_1: var_3}
    var_13 = var_11 == var_12
    assert var_13 is False
    var_14 = {var_0: var_2, var_1: var_3}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_0: var_2, var_1: var_9}
    var_17 = var_15 == var_16
    assert var_17 is False



# Parsed testcases at query #13
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_update_with_single_map. Retrieved 5/7 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 10/12 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_keep_rightmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_new_key. Retrieved 6/8 statements.
# Partially parsed test_update_with_empty_map. Retrieved 5/7 statements.
# Partially parsed test_update_with_no_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_identity. Retrieved 6/8 statements.
# Partially parsed test_update_with_constant. Retrieved 7/9 statements.
# Partially parsed test_update_with_original_unchanged. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'c'
    var_7 = 3
    var_8 = 4
    var_9 = {var_5: var_7, var_6: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = 3
    var_5 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = {}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l
    var_4 = 3
    var_5 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 42
    var_4 = lambda l, r: var_3
    var_5 = 3
    var_6 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l + r
    var_4 = module_0.m()



# Parsed testcases at query #15
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #16
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_contains_with_non_iterable_arg. Retrieved 5/8 statements.
# Partially parsed test_contains_with_wrong_length_iterable. Retrieved 7/10 statements.
# Partially parsed test_contains_with_non_tuple_arg. Retrieved 5/8 statements.
# Partially parsed test_contains_with_string_arg. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = (var_1,)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = 3
    var_6 = (var_1, var_4, var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'ab'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_eq_same_buckets. Retrieved 5/7 statements.
# Partially parsed test_eq_different_cached_hash. Retrieved 5/7 statements.
# Partially parsed test_eq_other_mapping. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = var_2 == var_2
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = var_2 == var_4
    assert var_5 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 3
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2 == var_6
    assert var_7 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = var_2 == var_4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = var_2 == var_3
    assert var_4 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_turbo_mapping_with_collision_keys. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_handles_exception_in_len. Retrieved 1/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 100
    var_2 = (var_0, var_1)
    var_3 = 'q'
    var_4 = 200
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2

def test_case_0():
    var_0 = 'key1'
    var_1 = 5
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = 4

def test_case_0():
    var_0 = 0



# Parsed testcases at query #20
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_update_with_merges_values_using_update_fn. Retrieved 4/7 statements.
# Partially parsed test_update_with_keeps_leftmost_value_when_update_fn_returns_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_inserts_new_keys_from_multiple_maps. Retrieved 8/10 statements.
# Partially parsed test_update_with_applies_update_fn_for_overlapping_keys. Retrieved 10/12 statements.
# Partially parsed test_update_with_returns_same_instance_if_no_changes. Retrieved 5/7 statements.
# Partially parsed test_update_with_handles_empty_maps. Retrieved 5/7 statements.
# Partially parsed test_update_with_uses_update_fn_for_each_key_collision. Retrieved 9/13 statements.
# Partially parsed test_update_with_preserves_non_updated_keys. Retrieved 7/9 statements.
# Partially parsed test_update_with_works_with_different_map_types. Retrieved 8/10 statements.
# Partially parsed test_update_with_handles_none_values. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = 2
    var_4 = module_0.m()
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = module_0.m()
    var_3 = lambda x, y: x * y
    var_4 = 2
    var_5 = 3
    var_6 = module_0.m()
    var_7 = 'b'
    var_8 = 4
    var_9 = {var_7: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.m()
    var_1 = lambda l, r: r
    var_2 = {}
    var_3 = 1
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.m()
    var_3 = 'c'
    var_4 = 'd'
    var_5 = module_0.m()
    var_6 = 'x'
    var_7 = 'e'
    var_8 = {var_6: var_7}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = lambda l, r: r
    var_5 = 20
    var_6 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l + r
    var_3 = 'a'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 100
    var_7 = dict(b=var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.m()
    var_2 = lambda l, r: r if r is not var_0 else l
    var_3 = 5
    var_4 = module_0.m()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_contains_with_non_tuple_arg. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'extra'
    var_5 = (var_0, var_1, var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_turbo_mapping_with_collision_keys. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_with_initial_len_exception. Retrieved 1/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 10
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

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
    var_8 = dict(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 20
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = dict(var_6)

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

def test_case_0():
    var_0 = 'key1'
    var_1 = 5
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = 0

def test_case_0():
    var_0 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = module_0._turbo_mapping(var_4, var_5)
    var_8 = hash(var_6)
    var_9 = hash(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #24
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = var_2 == var_5
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_contains_with_invalid_arg. Retrieved 8/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_with_does_not_call_update_fn_when_key_not_in_evolver. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    assert var_0 == 0
    var_1 = 1
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 3
    var_5 = module_0.m()



# Parsed testcases at query #27
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = dict(var_3)
    var_5 = var_2 == var_4
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l + r
    var_3 = 2
    var_4 = module_0.m()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_turbo_mapping_with_collision. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_preserves_hash. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = dict(var_2)

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2

def test_case_0():
    var_0 = 'key1'
    var_1 = 5
    var_2 = 'key2'
    var_3 = 100
    var_4 = 200
    var_5 = 0

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = hash(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}
    var_4 = 0
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 100

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

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
    var_9 = len(var_8)
    assert var_9 == 3



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_contains_with_non_tuple_arg. Retrieved 8/11 statements.
# Partially parsed test_contains_with_single_value_arg. Retrieved 6/9 statements.
# Partially parsed test_contains_with_string_arg. Retrieved 7/10 statements.
# Partially parsed test_contains_with_none_arg. Retrieved 7/10 statements.
# Partially parsed test_contains_with_list_arg. Retrieved 7/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'key'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_0, var_2]



# Parsed testcases at query #31
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_contains_with_invalid_arg. Retrieved 8/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'extra'
    var_7 = (var_0, var_2, var_6)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_update_with_key_not_in_evolver. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: l + r
    var_3 = 2
    var_4 = module_0.m()



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
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
    assert var_7 is True



