####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_eq_pmap_vs_other_mapping. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    var_7 = (var_6, var_0)
    var_8 = 'b'
    var_9 = (var_8, var_1)
    var_10 = [var_7, var_9]
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_9 = var_5 == var_8
    assert var_9 is False

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
    var_12 = var_5 == var_9
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
    var_13 = var_5 == var_10
    assert var_13 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_creates_pmap_with_given_size_and_buckets. Retrieved 8/9 statements.
# Partially parsed test_constructor_returns_pmap_instance. Retrieved 2/4 statements.
# Partially parsed test_constructor_sets_size_to_zero_for_empty_pmap. Retrieved 2/3 statements.
# Partially parsed test_constructor_sets_buckets_to_empty_tuple_for_empty_pmap. Retrieved 2/3 statements.
# Partially parsed test_constructor_sets_size_correctly_for_non_empty_pmap. Retrieved 7/8 statements.
# Partially parsed test_constructor_sets_buckets_correctly_for_non_empty_pmap. Retrieved 7/8 statements.
# Partially parsed test_constructor_creates_pmap_with_mixed_type_keys_and_values. Retrieved 10/11 statements.
# Partially parsed test_constructor_creates_pmap_with_none_key_and_value. Retrieved 7/8 statements.
# Partially parsed test_constructor_creates_pmap_with_duplicate_keys_in_buckets. Retrieved 7/8 statements.
# Partially parsed test_constructor_creates_pmap_with_empty_buckets_but_non_zero_size. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_pmap_with_single_bucket. Retrieved 5/6 statements.
# Partially parsed test_constructor_creates_pmap_with_large_number_of_buckets. Retrieved 3/6 statements.
# Partially parsed test_constructor_sets_cached_hash_to_none_by_default. Retrieved 3/5 statements.
# Partially parsed test_constructor_does_not_initialize_weakref. Retrieved 3/5 statements.
# Partially parsed test_constructor_creates_pmap_with_custom_object_keys. Retrieved 4/13 statements.
# Partially parsed test_constructor_creates_pmap_with_pmap_as_value. Retrieved 6/10 statements.
# Partially parsed test_constructor_creates_pmap_with_empty_string_key_and_value. Retrieved 4/5 statements.
# Partially parsed test_constructor_creates_pmap_with_boolean_key_and_value. Retrieved 6/7 statements.
# Partially parsed test_constructor_creates_pmap_with_tuple_key. Retrieved 6/7 statements.
# Partially parsed test_constructor_creates_pmap_with_dict_as_value. Retrieved 8/9 statements.
# Partially parsed test_constructor_creates_pmap_with_list_as_value. Retrieved 7/8 statements.
# Partially parsed test_constructor_creates_pmap_with_set_as_value. Retrieved 7/8 statements.
# Partially parsed test_constructor_handles_negative_size. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_pmap_with_function_as_value. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = [var_0, var_7]

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 'one'
    var_2 = (var_0, var_1)
    var_3 = 'two'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 3.0
    var_7 = [var_0, var_4, var_6]
    var_8 = (var_6, var_7)
    var_9 = (var_2, var_5, var_8)
    var_10 = [var_6, var_9]

def test_case_0():
    var_0 = None
    var_1 = 'null'
    var_2 = (var_0, var_1)
    var_3 = 'key'
    var_4 = (var_3, var_0)
    var_5 = (var_2, var_4)
    var_6 = 2
    var_7 = [var_6, var_5]

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'value2'
    var_4 = (var_0, var_3)
    var_5 = (var_2, var_4)
    var_6 = 2
    var_7 = [var_6, var_5]

def test_case_0():
    var_0 = 5
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'single'
    var_1 = 'item'
    var_2 = (var_0, var_1)
    var_3 = (var_2,)
    var_4 = 1
    var_5 = [var_4, var_3]

def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)
    var_2 = 2

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]
    var_3 = '_cached_hash'

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]
    var_3 = '__weakref__'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'val1'
    var_3 = 'val2'

def test_case_0():
    var_0 = 1
    var_1 = 'inner'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = (var_3,)
    var_5 = [var_0, var_4]
    var_6 = 'outer'

def test_case_0():
    var_0 = ''
    var_1 = (var_0, var_0)
    var_2 = (var_1,)
    var_3 = 1
    var_4 = [var_3, var_2]

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = (var_0, var_1)
    var_3 = (var_1, var_0)
    var_4 = (var_2, var_3)
    var_5 = 2
    var_6 = [var_5, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'tuple_key'
    var_4 = (var_2, var_3)
    var_5 = (var_4,)
    var_6 = [var_0, var_5]

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = (var_0, var_5)
    var_7 = (var_6,)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = (var_0, var_4)
    var_6 = (var_5,)
    var_7 = [var_1, var_6]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = {var_1, var_2, var_3}
    var_5 = (var_0, var_4)
    var_6 = (var_5,)
    var_7 = [var_1, var_6]

def test_case_0():
    var_0 = -1
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'func'
    var_1 = 1



# Parsed testcases at query #3
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
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = var_5 == var_12
    assert var_13 is True
    var_14 = var_9 == var_12
    assert var_14 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_returns_pmap_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_sets_size_and_buckets. Retrieved 3/7 statements.
# Partially parsed test_constructor_handles_cached_hash_attribute. Retrieved 4/9 statements.
# Partially parsed test_constructor_preserves_weakref_slot. Retrieved 4/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._size
    assert var_6 == 2
    var_7 = var_5._buckets
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = var_1._buckets
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 0
    var_3 = '_cached_hash'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_eq_returns_true_for_same_instance. Retrieved 6/9 statements.
# Partially parsed test_eq_returns_false_for_different_instance_with_same_values. Retrieved 8/12 statements.
# Partially parsed test_eq_returns_false_for_list_with_same_values. Retrieved 7/10 statements.
# Partially parsed test_eq_returns_false_for_tuple_with_same_values. Retrieved 7/10 statements.
# Partially parsed test_eq_returns_false_for_empty_values_view. Retrieved 3/6 statements.
# Partially parsed test_eq_returns_false_for_none. Retrieved 5/8 statements.
# Partially parsed test_eq_returns_false_for_different_values_view. Retrieved 12/16 statements.


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
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_2, var_3]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_2, var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = []

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = 'c'
    var_9 = 'd'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.pmap(var_10)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test___contains___with_valid_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test___contains___with_valid_key_but_wrong_value. Retrieved 7/10 statements.
# Partially parsed test___contains___with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test___contains___with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test___contains___with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test___contains___with_empty_map. Retrieved 5/8 statements.
# Partially parsed test___contains___with_nested_structure_in_value. Retrieved 8/11 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_1, var_2]
    var_7 = (var_0, var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_preserves_immutability_after_creation. Retrieved 7/10 statements.
# Partially parsed test_constructor_with_empty_buckets_results_in_empty_pmap. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_non_empty_buckets_results_in_correct_pmap. Retrieved 14/18 statements.


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
    var_10 = bool(var_9 == var_0)
    assert var_10 is True
    var_11 = var_8['key1']
    assert var_11 == 'value1'
    var_12 = var_8['key2']
    assert var_12 == 'value2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = dict(var_0)
    var_3 = bool(var_2 == {})
    assert var_3 is True

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
    var_7 = var_5['a']
    assert var_7 == 1
    var_8 = var_5['b']
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5['x']
    assert var_7 == 10
    var_8 = var_5['y']
    assert var_8 == 20

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
    var_9 = var_7['k1']
    assert var_9 == 'v1'
    var_10 = var_7['k2']
    assert var_10 == 'v2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'inner_key'
    var_1 = 'inner_value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'outer_key'
    var_5 = {var_4: var_3}
    var_6 = module_0.pmap(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6['outer_key']['inner_key']
    assert var_8 == 'inner_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = len(var_3)
    assert var_6 == 1
    var_7 = 'b'
    var_8 = bool('b' not in var_3)
    assert var_8 is True
    var_9 = 'b'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'first'
    var_2 = (var_0, var_1)
    var_3 = 'last'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0.pmap(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6['key']
    assert var_8 == 'last'

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 3
    var_1 = None
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = 'c'
    var_10 = 3
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = [var_1, var_1, var_8, var_1, var_12]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_update_with_merges_values_using_update_fn. Retrieved 5/7 statements.
# Partially parsed test_update_with_keeps_leftmost_value_when_update_fn_returns_left. Retrieved 8/10 statements.
# Partially parsed test_update_with_keeps_rightmost_value_when_update_fn_returns_right. Retrieved 8/10 statements.
# Partially parsed test_update_with_inserts_new_key_when_key_not_present. Retrieved 5/7 statements.
# Partially parsed test_update_with_handles_multiple_maps. Retrieved 10/12 statements.
# Partially parsed test_update_with_returns_same_instance_when_no_changes. Retrieved 4/6 statements.
# Partially parsed test_update_with_works_with_empty_pmap. Retrieved 5/7 statements.
# Partially parsed test_update_with_uses_initial_value_for_first_merge. Retrieved 8/10 statements.
# Partially parsed test_update_with_handles_non_integer_values. Retrieved 6/8 statements.
# Partially parsed test_update_with_merges_from_dict_and_pmap. Retrieved 8/10 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l + r
    var_5 = 2
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_5, var_7: var_0}
    var_9 = module_0.m(**var_8)
    var_10 = 'a'
    var_11 = 'c'
    var_12 = 3
    var_13 = 4
    var_14 = {var_10: var_12, var_11: var_13}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l
    var_5 = {}
    var_6 = module_0.m(**var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = lambda l, r: l + r
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l * r
    var_5 = 2
    var_6 = 'a'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)
    var_9 = 'a'
    var_10 = 3
    var_11 = {var_9: var_10}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = ' '
    var_5 = lambda l, r: l + var_4 + r
    var_6 = 'world'
    var_7 = 'a'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l + r
    var_5 = 'a'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 3
    var_9 = 'a'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_eq_same_instance.
# Failed to parse test_eq_different_type.
# Failed to parse test_eq_same_map.
# Failed to parse test_eq_different_map.




# Parsed testcases at query #10
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapValues(var_0)
    var_2 = var_1 == var_1
    assert var_2 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapValues(var_0)
    var_2 = module_0.PMapValues(var_0)
    var_3 = var_1 == var_2
    assert var_3 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapValues(var_0)
    var_2 = 'not a view'
    var_3 = var_1 == var_2
    assert var_3 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMapValues(var_0)
    var_2 = None
    var_3 = var_1 == var_2
    assert var_3 is False



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_with_keyword_arguments. Retrieved 2/5 statements.
# Partially parsed test_constructor_with_mixed_input_types. Retrieved 8/11 statements.
# Partially parsed test_constructor_creates_hashable_instance. Retrieved 5/7 statements.
# Partially parsed test_constructor_with_complex_nested_structure. Retrieved 8/14 statements.
# Partially parsed test_constructor_pmap_is_immutable. Retrieved 4/7 statements.


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
    var_10 = bool(var_9 == var_0)
    assert var_10 is True
    var_11 = var_8['key1']
    assert var_11 == 'value1'
    var_12 = var_8['key2']
    assert var_12 == 'value2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = dict(var_0)
    var_3 = bool(var_2 == {})
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2

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
    var_7 = var_5['x']
    assert var_7 == 10
    var_8 = var_5['y']
    assert var_8 == 20

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
    var_9 = var_7['k1']
    assert var_9 == 'v1'
    var_10 = var_7['k2']
    assert var_10 == 'v2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_0.pmap(var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True
    var_6 = bool(var_3 is not var_4)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3[None]
    assert var_4 == 'null'
    var_5 = len(var_3)
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5[1]
    assert var_6 == 'one'
    var_7 = var_5[2]
    assert var_7 == 'two'
    var_8 = len(var_5)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1.5
    var_1 = 2.7
    var_2 = 'one point five'
    var_3 = 'two point seven'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5[1.5]
    assert var_6 == 'one point five'
    var_7 = var_5[2.7]
    assert var_7 == 'two point seven'
    var_8 = len(var_5)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'tuple key'
    var_4 = {var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5[1, 2]
    assert var_6 == 'tuple key'
    var_7 = len(var_5)
    assert var_7 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = []
    var_3 = module_0.pmap(var_2)
    var_4 = module_0.pmap()
    var_5 = len(var_1)
    assert var_5 == 0
    var_6 = len(var_3)
    assert var_6 == 0
    var_7 = len(var_4)
    assert var_7 == 0
    var_8 = bool(var_1 == var_3)
    assert var_8 is True
    var_9 = bool(var_3 == var_4)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0.pmap(var_5)
    var_7 = var_6['a']
    assert var_7 == 2
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 2
    var_5 = 'c'
    var_6 = 3
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = hash(var_3)

def test_case_0():
    var_0 = 'vec'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'inner'
    var_7 = 'data'
    var_8 = {var_6: var_7}
    var_9 = [var_2, var_3, var_4]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3['a']
    assert var_4 == 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 6/9 statements.
# Partially parsed test_eq_different_pmap_items_same_map. Retrieved 6/10 statements.
# Partially parsed test_eq_different_pmap_items_different_map. Retrieved 8/12 statements.
# Partially parsed test_eq_different_pmap_items_different_content. Retrieved 10/14 statements.
# Partially parsed test_eq_with_non_pmap_items_instance. Retrieved 9/12 statements.
# Partially parsed test_eq_with_none. Retrieved 7/10 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_0: var_2, var_6: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)
    var_7 = (var_1, var_3)
    var_8 = [var_6, var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmap_constructor_handles_same_hash_collisions. Retrieved 4/16 statements.
# Partially parsed test_pmap_constructor_with_large_number_of_items. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._size
    assert var_6 == 2
    var_7 = var_5._buckets
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

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
    var_8 = var_7._size
    assert var_8 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = var_1._buckets
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5['key1']
    assert var_6 == 'value1'
    var_7 = var_5['key2']
    assert var_7 == 'value2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3[None]
    assert var_4 == 'null'
    var_5 = var_3._size
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3['key']
    assert var_4 is None
    var_5 = var_3._size
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_3 is not var_5)
    assert var_6 is True
    var_7 = bool(var_3 == var_5)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {str(i): i for i in var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    assert var_4 == 100
    var_5 = var_3[var_0]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = var_4['a']
    assert var_5 == 2
    var_6 = var_4._size
    assert var_6 == 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_eq_pmap_vs_other_mapping_equal. Retrieved 8/11 statements.
# Partially parsed test_eq_pmap_vs_other_mapping_different. Retrieved 9/12 statements.
# Partially parsed test_eq_cached_hash_mismatch. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    var_8 = var_5 == var_7
    assert var_8 is False

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
    var_6 = 'a'
    var_7 = (var_6, var_0)
    var_8 = 'b'
    var_9 = 3
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]

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
    var_11 = var_5 == var_10

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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_10 = var_5._buckets
    var_11 = var_9._buckets
    var_12 = var_10 == var_11
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
    var_11 = var_5._buckets
    var_12 = var_10._buckets
    var_13 = var_11 == var_12
    assert var_13 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_contains_with_invalid_arg_returns_false. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = (var_0, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__turbo_mapping_handles_collisions. Retrieved 6/19 statements.
# Partially parsed test__turbo_mapping_with_large_initial_and_no_pre_size. Retrieved 8/10 statements.
# Partially parsed test__turbo_mapping_with_large_initial_and_small_pre_size. Retrieved 8/10 statements.


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
    var_0 = {}
    var_1 = 10
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
    var_5 = 0
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
    var_5 = 20
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
    var_6 = [var_2, var_5]
    var_7 = 15
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = dict(var_8)
    var_11 = bool(var_10 == {'a': 1, 'b': 2})
    assert var_11 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 5
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = 0

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
    var_7 = range(var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}
    var_4 = 50
    var_5 = module_0._turbo_mapping(var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 100
    var_7 = range(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test___eq___with_other_mapping_equal. Retrieved 8/11 statements.
# Partially parsed test___eq___with_other_mapping_not_equal. Retrieved 9/12 statements.
# Partially parsed test___eq___cached_hash_mismatch. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_9 = var_5 == var_8
    assert var_9 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    var_6 = 'a'
    var_7 = (var_6, var_0)
    var_8 = 'b'
    var_9 = 3
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = [var_0, var_1]
    var_7 = var_5 == var_6

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
    var_11 = var_5 == var_10
    assert var_11 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 6/9 statements.
# Partially parsed test_eq_different_pmapitems_same_map. Retrieved 6/10 statements.
# Partially parsed test_eq_different_pmapitems_different_map_same_content. Retrieved 8/12 statements.
# Partially parsed test_eq_different_pmapitems_different_map_different_content. Retrieved 10/14 statements.
# Partially parsed test_eq_with_non_pmapitems_instance. Retrieved 9/12 statements.
# Partially parsed test_eq_with_none. Retrieved 7/10 statements.
# Partially parsed test_eq_empty_maps. Retrieved 4/8 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_0: var_2, var_6: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)
    var_7 = (var_1, var_3)
    var_8 = [var_6, var_7]

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
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_creates_pmap_that_is_hashable. Retrieved 7/9 statements.
# Partially parsed test_constructor_creates_pmap_that_is_immutable. Retrieved 4/7 statements.


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
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_0)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = dict(var_12)
    var_14 = dict(var_7)
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = var_1._buckets
    var_4 = len(var_3)
    assert var_4 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    assert var_4 == 1
    var_5 = var_3['key']
    assert var_5 == 'value'

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
    var_8 = var_7._size
    assert var_8 == 3
    var_9 = var_7['a']
    assert var_9 == 1
    var_10 = var_7['b']
    assert var_10 == 2
    var_11 = var_7['c']
    assert var_11 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    assert var_4 == 1
    var_5 = var_3['key']
    assert var_5 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = len(var_1)
    assert var_3 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = hash(var_5)

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._buckets
    var_7 = bool(var_5._buckets is not None)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'two'
    var_2 = 3
    var_3 = 4
    var_4 = (var_2, var_3)
    var_5 = 'one'
    var_6 = 2
    var_7 = 'tuple'
    var_8 = {var_0: var_5, var_1: var_6, var_4: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = var_9._size
    assert var_10 == 3
    var_11 = var_9[1]
    assert var_11 == 'one'
    var_12 = var_9['two']
    assert var_12 == 2
    var_13 = var_9[3, 4]
    assert var_13 == 'tuple'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = var_1._buckets
    var_4 = len(var_3)
    assert var_4 == 0

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
    var_8 = bool(var_7 == {'a', 'b'})
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = var_4._size
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 2



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_turbo_mapping_with_collision. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_preserves_hashability. Retrieved 10/12 statements.
# Partially parsed test_turbo_mapping_with_non_integer_len_fallback. Retrieved 5/15 statements.


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
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['x']
    assert var_8 == 10
    var_9 = var_6['y']
    assert var_9 == 20

def test_case_0():
    var_0 = 'a'
    var_1 = 5
    var_2 = 'b'
    var_3 = 100
    var_4 = 200
    var_5 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'val2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 0
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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = hash(var_6)
    var_8 = 'c'
    var_9 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1

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
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = bool(var_4 == {'a': 1, 'b': 2})
    assert var_7 is True
    var_8 = dict(var_6)
    var_9 = bool(var_8 == {'a': 1, 'b': 2})
    assert var_9 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_pmap_is_immutable. Retrieved 4/7 statements.
# Partially parsed test_constructor_pmap_preserves_insertion_order. Retrieved 11/14 statements.
# Partially parsed test_constructor_pmap_implements_mapping_protocol. Retrieved 4/7 statements.
# Partially parsed test_constructor_pmap_iteritems_yields_key_value_pairs. Retrieved 6/10 statements.
# Partially parsed test_constructor_pmap_iterkeys_yields_keys. Retrieved 6/10 statements.
# Partially parsed test_constructor_pmap_itervalues_yields_values. Retrieved 6/10 statements.
# Partially parsed test_constructor_pmap_get_method_with_default. Retrieved 6/10 statements.


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
    var_10 = bool(var_9 == var_0)
    assert var_10 is True
    var_11 = var_8['a']
    assert var_11 == 1
    var_12 = var_8['b']
    assert var_12 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = list(var_1)
    var_4 = bool(var_3 == [])
    assert var_4 is True

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
    var_7 = var_5['x']
    assert var_7 == 10
    var_8 = var_5['y']
    assert var_8 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 'alpha'
    var_3 = 'beta'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5['alpha']
    assert var_7 == 100
    var_8 = var_5['beta']
    assert var_8 == 200

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.pmap(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7['key1']
    assert var_9 == 'value1'
    var_10 = var_7['key2']
    assert var_10 == 'value2'

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
    var_10 = bool(var_8 == var_9)
    assert var_10 is True
    var_11 = bool(var_5 == var_7)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'bar'
    var_1 = 'foo'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = var_3.foo
    assert var_4 == 'bar'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'missing'
    var_3 = var_1[var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = var_1.missing
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 'v'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = 'm'
    var_7 = 3
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = module_0.pmap(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0.pmap(var_5)
    var_7 = var_6['k']
    assert var_7 == 2
    var_8 = len(var_6)
    assert var_8 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = dict(var_1)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True
    var_7 = bool(not var_5 != {'a': 1, 'b': 2})
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True
    var_7 = bool(not var_3 != var_5)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_0: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(var_3 != var_7)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

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
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = var_3 < var_7
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool(True)
    assert var_10 is True

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
    var_0 = 'present'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'present'
    var_5 = bool('present' in var_3)
    assert var_5 is True
    var_6 = 'absent'
    var_7 = bool('absent' not in var_3)
    assert var_7 is True

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = reversed(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_constructor_handles_nested_pmaps. Retrieved 8/11 statements.
# Partially parsed test_constructor_preserves_hash_collisions_handling. Retrieved 4/16 statements.
# Partially parsed test_constructor_with_empty_buckets. Retrieved 2/7 statements.
# Partially parsed test_constructor_does_not_allow_direct_instantiation_without_factory. Retrieved 2/6 statements.
# Partially parsed test_constructor_pmap_is_hashable_when_empty. Retrieved 2/4 statements.
# Partially parsed test_constructor_pmap_is_hashable_with_elements. Retrieved 7/9 statements.


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
    var_10 = bool(var_9 == var_0)
    assert var_10 is True
    var_11 = var_8['key1']
    assert var_11 == 'value1'
    var_12 = var_8['key2']
    assert var_12 == 'value2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = dict(var_0)
    var_3 = bool(var_2 == {})
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'inner'
    var_5 = {var_4: var_3}
    var_6 = module_0.pmap(var_5)
    var_7 = var_6['inner']['a']
    assert var_7 == 1
    var_8 = var_6[var_4]
    var_9 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'val1'
    var_3 = 'val2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5['a']
    assert var_6 == 1
    var_7 = var_5['b']
    assert var_7 == 2
    var_8 = module_0.pmap(var_5)
    var_9 = var_8['a']
    assert var_9 == 1
    var_10 = var_8['b']
    assert var_10 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._size
    assert var_6 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = 200
    var_2 = 'alpha'
    var_3 = 'beta'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5['alpha']
    assert var_6 == 100
    var_7 = var_5['beta']
    assert var_7 == 200
    var_8 = len(var_5)
    assert var_8 == 2

def test_case_0():
    var_0 = ()
    var_1 = 0
    var_2 = [var_1, var_0]

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = hash(var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'k1'
    var_1 = 'k2'
    var_2 = 'v1'
    var_3 = 'v2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = hash(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_1: var_3, var_0: var_2}
    var_7 = module_0.pmap(var_6)
    var_8 = hash(var_5)
    var_9 = hash(var_7)
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 2
    var_5 = {var_0: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = hash(var_3)
    var_8 = hash(var_6)
    var_9 = bool(var_7 != var_8)
    assert var_9 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_update_with_does_not_call_update_fn_when_key_not_in_evolver. Retrieved 5/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    assert var_0 == 0
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_2: var_1}
    var_4 = module_0.m(**var_3)
    var_5 = 2
    var_6 = 'b'
    var_7 = {var_6: var_5}
    var_8 = module_0.m(**var_7)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_creates_pmap_with_given_size_and_buckets. Retrieved 8/9 statements.
# Partially parsed test_constructor_returns_pmap_instance. Retrieved 2/4 statements.
# Partially parsed test_constructor_sets_size_to_zero_for_empty_pmap. Retrieved 2/3 statements.
# Partially parsed test_constructor_sets_buckets_to_empty_tuple_for_empty_pmap. Retrieved 2/3 statements.
# Partially parsed test_constructor_sets_size_correctly_for_non_empty_pmap. Retrieved 7/8 statements.
# Partially parsed test_constructor_assigns_buckets_directly. Retrieved 8/9 statements.
# Partially parsed test_constructor_creates_pmap_with_single_key_value_pair. Retrieved 5/6 statements.
# Partially parsed test_constructor_handles_large_size_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_sets_cached_hash_to_none_by_default. Retrieved 3/5 statements.
# Partially parsed test_constructor_does_not_modify_input_buckets. Retrieved 9/10 statements.


def test_case_0():
    var_0 = 2
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = [var_0, var_7]

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 2
    var_8 = [var_7, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 'single_key'
    var_2 = 'single_value'
    var_3 = (var_1, var_2)
    var_4 = (var_3,)
    var_5 = [var_0, var_4]

def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)
    var_2 = 'key'

def test_case_0():
    var_0 = 0
    var_1 = ()
    var_2 = [var_0, var_1]
    var_3 = '_cached_hash'

def test_case_0():
    var_0 = 'k1'
    var_1 = 'v1'
    var_2 = (var_0, var_1)
    var_3 = 'k2'
    var_4 = 'v2'
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 2
    var_8 = [var_7, var_6]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_eq_pmap_vs_other_mapping_equal. Retrieved 8/11 statements.
# Partially parsed test_eq_pmap_vs_pmap_with_same_buckets. Retrieved 5/11 statements.
# Partially parsed test_eq_pmap_vs_pmap_with_different_cached_hash. Retrieved 5/7 statements.
# Partially parsed test_eq_pmap_vs_pmap_with_different_buckets_same_items. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    var_8 = var_5 == var_7
    assert var_8 is False

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
    var_6 = 'a'
    var_7 = (var_6, var_0)
    var_8 = 'b'
    var_9 = (var_8, var_1)
    var_10 = [var_7, var_9]
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    assert var_10 is True



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_turbo_mapping_with_collision_keys. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_initial_len_exception_falls_back. Retrieved 1/10 statements.


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
    var_0 = {}
    var_1 = 10
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
    var_5 = 0
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
    var_5 = 20
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
    var_7 = 0
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = dict(var_8)
    var_11 = bool(var_10 == {'a': 1, 'b': 2})
    assert var_11 is True

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
    var_8 = bool(var_6 == var_7)
    assert var_8 is True
    var_9 = hash(var_6)
    var_10 = hash(var_7)
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = dict(var_4)
    var_7 = bool(var_6 == {'a': 1})
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_eq_with_cached_hash_mismatch_returns_false. Retrieved 9/12 statements.


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
    var_8 = var_5 == var_7
    assert var_8 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_update_with_merge_function. Retrieved 4/7 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_update_with_empty_map. Retrieved 7/9 statements.
# Partially parsed test_update_with_no_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_new_key. Retrieved 6/8 statements.
# Partially parsed test_update_with_overwrites_existing. Retrieved 7/9 statements.
# Partially parsed test_update_with_identity_function. Retrieved 7/9 statements.


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
    var_6 = lambda l, r: l + r
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_1, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 10
    var_15 = 4
    var_16 = {var_12: var_14, var_13: var_15}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = lambda l, r: r
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}

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
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: r * var_1
    var_7 = 'a'
    var_8 = 5
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
    var_8 = 4
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_contains_with_existing_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test_contains_with_existing_key_but_different_value. Retrieved 7/10 statements.
# Partially parsed test_contains_with_non_existing_key. Retrieved 8/11 statements.
# Partially parsed test_contains_with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test_contains_with_wrong_length_tuple. Retrieved 8/11 statements.
# Partially parsed test_contains_with_empty_pmap. Retrieved 5/8 statements.


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_contains_predicate_true. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = (var_0, var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 6/9 statements.
# Partially parsed test_eq_different_pmap_items_same_map. Retrieved 6/10 statements.
# Partially parsed test_eq_different_pmap_items_different_map. Retrieved 8/12 statements.
# Partially parsed test_eq_different_pmap_items_different_content. Retrieved 10/14 statements.
# Partially parsed test_eq_with_non_pmap_items_instance. Retrieved 9/12 statements.
# Partially parsed test_eq_with_none. Retrieved 7/10 statements.
# Partially parsed test_eq_empty_pmap_items. Retrieved 4/8 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = 'c'
    var_8 = {var_0: var_2, var_6: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)
    var_7 = (var_1, var_3)
    var_8 = [var_6, var_7]

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
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = {}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #9
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5.a
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5.c
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = 'b'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = 'a'
    var_5 = {var_4: var_3}
    var_6 = module_0.m(**var_5)
    var_7 = var_6.a.b
    assert var_7 == 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'valid_identifier'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = var_3.valid_identifier
    assert var_4 == 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key.with.dot'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = 'key.with.dot'
    var_4 = {var_3: var_1}
    var_5 = module_0.m(**var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_creates_hashable_pmap. Retrieved 5/7 statements.


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
    var_10 = bool(var_9 == var_0)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._size
    assert var_6 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._buckets
    var_5 = bool(var_3._buckets is not None)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_1._size
    assert var_3 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3['key']
    assert var_4 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5['x']
    assert var_6 == 10
    var_7 = var_5['y']
    assert var_7 == 20

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0.pmap(var_5)
    var_7 = var_6['a']
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = hash(var_3)

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
    var_9 = var_8[1]
    assert var_9 == 'int'
    var_10 = var_8['str']
    assert var_10 == 'string'
    var_11 = var_8[1, 2]
    assert var_11 == 'tuple'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 2})
    assert var_6 is True



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_update_with_merges_values_using_update_fn. Retrieved 7/9 statements.
# Partially parsed test_update_with_keeps_leftmost_value_when_update_fn_returns_left. Retrieved 9/11 statements.
# Partially parsed test_update_with_inserts_new_key_from_multiple_maps. Retrieved 8/10 statements.
# Partially parsed test_update_with_handles_empty_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_merges_colliding_keys_from_multiple_maps. Retrieved 9/11 statements.
# Partially parsed test_update_with_returns_same_instance_when_no_changes. Retrieved 5/7 statements.
# Partially parsed test_update_with_works_with_dict_and_pmap. Retrieved 9/11 statements.
# Partially parsed test_update_with_uses_default_value_for_new_keys. Retrieved 7/9 statements.


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
    var_12 = 'b'
    var_13 = {var_11: var_10, var_12: var_1}
    var_14 = module_0.m(**var_13)

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
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = {var_13: var_0, var_14: var_5, var_15: var_9}
    var_17 = module_0.m(**var_16)

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
    var_2 = 'b'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    var_5 = lambda l, r: l * r
    var_6 = 2
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_6, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = 3
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_11, var_13: var_11}
    var_15 = module_0.m(**var_14)
    var_16 = 6
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_16, var_18: var_16}
    var_20 = module_0.m(**var_19)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = lambda l, r: l
    var_7 = {}
    var_8 = module_0.m(**var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = lambda l, r: l + r
    var_5 = 'a'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 3
    var_9 = 'b'
    var_10 = {var_9: var_8}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_8, var_13: var_8}
    var_15 = module_0.m(**var_14)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = module_0.m(**var_2)
    var_4 = None
    var_5 = lambda l, r: l + r if l is not var_4 else r
    var_6 = 2
    var_7 = 'b'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_6}
    var_13 = module_0.m(**var_12)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_turbo_mapping_with_collision_keys. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_with_initial_length_hint_failure. Retrieved 1/10 statements.


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
    var_0 = {}
    var_1 = 10
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
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = dict(var_6)
    var_9 = bool(var_8 == var_4)
    assert var_9 is True

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
    var_8 = bool(var_6 == var_7)
    assert var_8 is True
    var_9 = hash(var_6)
    var_10 = hash(var_7)
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = dict(var_4)
    var_7 = bool(var_6 == var_2)
    assert var_7 is True

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
    var_11 = module_0._turbo_mapping(var_10, var_6)
    var_12 = len(var_11)
    assert var_12 == 5
    var_13 = dict(var_11)
    var_14 = bool(var_13 == var_10)
    assert var_14 is True



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
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = var_5 == var_6
    assert var_7 is True



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = var_6 == var_6
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = 'not a PMapItems'
    var_8 = var_6 == var_7
    assert var_8 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = module_0.PMapItems(var_5)
    var_7 = module_0.PMapItems(var_5)
    var_8 = var_6 == var_7
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 3
    var_7 = 4
    var_8 = 'c'
    var_9 = 'd'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = module_0.PMapItems(var_5)
    var_13 = module_0.PMapItems(var_11)
    var_14 = var_12 == var_13
    assert var_14 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_eq_cached_hash_mismatch. Retrieved 6/12 statements.
# Partially parsed test_eq_cached_hash_match. Retrieved 5/11 statements.
# Partially parsed test_eq_with_other_mapping. Retrieved 6/18 statements.
# Partially parsed test_eq_with_other_mapping_different. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_12 = var_5 == var_11
    assert var_12 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    var_9 = module_0.m(**var_8)
    var_10 = var_5._buckets
    var_11 = var_9._buckets
    var_12 = var_10 == var_11
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
    var_9 = module_0.m(**var_8)
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_8 = 3
    var_9 = {var_6: var_0, var_7: var_8}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_contains_with_valid_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test_contains_with_valid_key_but_wrong_value. Retrieved 7/10 statements.
# Partially parsed test_contains_with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test_contains_with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test_contains_with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test_contains_with_empty_map. Retrieved 5/8 statements.


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



# Parsed testcases at query #19
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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_eq_with_different_cached_hash. Retrieved 9/12 statements.


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
    var_8 = var_5 == var_7
    assert var_8 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_eq_pmap_vs_other_mapping_equal. Retrieved 8/11 statements.
# Partially parsed test_eq_pmap_vs_dict_with_same_hash. Retrieved 5/7 statements.
# Partially parsed test_eq_pmap_vs_dict_with_different_hash. Retrieved 5/7 statements.
# Partially parsed test_eq_pmap_with_identical_buckets. Retrieved 9/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = var_7 == var_10
    assert var_11 is False

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
    var_9 = module_0.m(**var_8)
    var_10 = var_5 == var_9
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    assert var_10 is True

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
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_8: var_2, var_9: var_0, var_10: var_1}
    var_12 = var_7 == var_11
    assert var_12 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_turbo_mapping_with_collision_handling. Retrieved 6/19 statements.
# Partially parsed test_turbo_mapping_preserves_hashability. Retrieved 10/12 statements.


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
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_6['x']
    assert var_8 == 10
    var_9 = var_6['y']
    assert var_9 == 20

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
    var_10 = var_8['key1']
    assert var_10 == 100
    var_11 = var_8['key2']
    assert var_11 == 200

def test_case_0():
    var_0 = 'a'
    var_1 = 5
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = 4

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = hash(var_6)
    var_8 = 'c'
    var_9 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = 'a'
    var_8 = bool('a' in var_6)
    assert var_8 is True
    var_9 = 'c'
    var_10 = bool('c' not in var_6)
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
    var_7 = list(var_6)
    var_8 = set(var_7)
    var_9 = bool(var_8 == {'a', 'b'})
    assert var_9 is True

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
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_eq_same_buckets. Retrieved 5/7 statements.
# Partially parsed test_eq_different_cached_hash. Retrieved 5/7 statements.
# Partially parsed test_eq_with_other_mapping. Retrieved 6/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_12 = var_5 == var_11
    assert var_12 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = [var_0, var_1]
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
    var_9 = module_0.m(**var_8)
    var_10 = var_5 == var_9
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
    var_10 = var_5 == var_9
    assert var_10 is False

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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_update_with_merges_values_using_update_fn. Retrieved 7/9 statements.
# Partially parsed test_update_with_keeps_leftmost_value_when_update_fn_returns_left. Retrieved 9/11 statements.
# Partially parsed test_update_with_inserts_new_key_from_maps. Retrieved 9/11 statements.
# Partially parsed test_update_with_handles_multiple_maps_and_merge_fn. Retrieved 12/14 statements.
# Partially parsed test_update_with_returns_same_instance_if_no_changes. Retrieved 5/7 statements.
# Partially parsed test_update_with_on_empty_pmap. Retrieved 8/10 statements.
# Partially parsed test_update_with_uses_update_fn_only_for_existing_keys. Retrieved 9/15 statements.


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
    var_12 = 'b'
    var_13 = {var_11: var_10, var_12: var_1}
    var_14 = module_0.m(**var_13)

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
    var_9 = 'c'
    var_10 = 3
    var_11 = {var_9: var_10}
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = {var_12: var_0, var_13: var_5, var_14: var_10}
    var_16 = module_0.m(**var_15)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.m(**var_3)
    var_5 = lambda l, r: l * r
    var_6 = 2
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_6, var_8: var_6}
    var_10 = module_0.m(**var_9)
    var_11 = 'a'
    var_12 = 'c'
    var_13 = 3
    var_14 = 5
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 6
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = {var_17: var_16, var_18: var_6, var_19: var_14}
    var_21 = module_0.m(**var_20)

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
    var_0 = {}
    var_1 = module_0.m(**var_0)
    var_2 = lambda l, r: l + r
    var_3 = 'x'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = 20
    var_7 = 'y'
    var_8 = {var_7: var_6}
    var_9 = module_0.m(**var_8)
    var_10 = 'x'
    var_11 = 'y'
    var_12 = {var_10: var_4, var_11: var_6}
    var_13 = module_0.m(**var_12)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    assert var_0 == 1
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.m(**var_5)
    var_7 = 10
    var_8 = 30
    var_9 = 'a'
    var_10 = 'c'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.m(**var_11)
    var_13 = 11
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = {var_14: var_13, var_15: var_2, var_16: var_8}
    var_18 = module_0.m(**var_17)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_update_with_merge_function. Retrieved 4/7 statements.
# Partially parsed test_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_update_with_empty_maps. Retrieved 4/6 statements.
# Partially parsed test_update_with_new_key. Retrieved 5/7 statements.
# Partially parsed test_update_with_overwrites_existing. Retrieved 6/8 statements.
# Partially parsed test_update_with_identity_function. Retrieved 7/9 statements.
# Partially parsed test_update_with_constant_function. Retrieved 8/10 statements.


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
    var_6 = lambda l, r: l + r
    var_7 = 3
    var_8 = 'a'
    var_9 = 'c'
    var_10 = {var_8: var_1, var_9: var_7}
    var_11 = module_0.m(**var_10)
    var_12 = 'a'
    var_13 = 'd'
    var_14 = 10
    var_15 = 4
    var_16 = {var_12: var_14, var_13: var_15}

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
    var_6 = lambda l, r: r * var_1
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
    var_6 = lambda l, r: r
    var_7 = 5
    var_8 = 7
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
    var_6 = 99
    var_7 = lambda l, r: var_6
    var_8 = 5
    var_9 = 7
    var_10 = 'a'
    var_11 = 'c'
    var_12 = {var_10: var_8, var_11: var_9}
    var_13 = module_0.m(**var_12)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_constructor_preserves_hash_collisions_handling. Retrieved 2/10 statements.
# Partially parsed test_constructor_creates_pmap_with_zero_size_and_empty_buckets. Retrieved 5/8 statements.
# Partially parsed test_constructor_creates_pmap_that_is_hashable. Retrieved 7/9 statements.
# Partially parsed test_constructor_creates_pmap_with_identical_buckets_as_input. Retrieved 8/10 statements.
# Partially parsed test_constructor_creates_pmap_with_correct_internal_structure. Retrieved 12/15 statements.
# Partially parsed test_constructor_creates_pmap_that_is_immutable. Retrieved 4/7 statements.


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
    var_10 = var_9._size
    var_11 = bool(var_9._size == var_0)
    assert var_11 is True
    var_12 = var_9._buckets
    var_13 = dict(var_12)
    var_14 = dict(var_7)
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = var_1._buckets
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    assert var_4 == 1
    var_5 = var_3['key']
    assert var_5 == 'value'

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
    var_8 = var_7._size
    assert var_8 == 3
    var_9 = var_7['a']
    assert var_9 == 1
    var_10 = var_7['b']
    assert var_10 == 2
    var_11 = var_7['c']
    assert var_11 == 3

def test_case_0():
    var_0 = 'first'
    var_1 = 'second'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = list(var_1)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    assert var_4 == 1
    var_5 = var_3['key']
    assert var_5 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = var_1._buckets
    var_4 = None
    var_5 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = hash(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._size
    var_7 = var_5._buckets
    var_8 = [var_6, var_7]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = '_size'
    var_7 = hasattr(var_5, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = '_buckets'
    var_10 = hasattr(var_5, var_9)
    var_11 = bool(var_10)
    assert var_11 is True
    var_12 = var_5._size
    assert var_12 == 2
    var_13 = var_5._buckets
    var_14 = var_5._buckets
    var_15 = [var_14]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

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
    var_9 = var_8._size
    assert var_9 == 3
    var_10 = var_8[1]
    assert var_10 == 'int'
    var_11 = var_8['str']
    assert var_11 == 'string'
    var_12 = var_8[1, 2]
    assert var_12 == 'tuple'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = var_4._size
    assert var_5 == 1
    var_6 = var_4['a']
    assert var_6 == 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_contains_with_valid_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test_contains_with_valid_key_value_pair_second_item. Retrieved 7/10 statements.
# Partially parsed test_contains_with_valid_key_but_different_value. Retrieved 7/10 statements.
# Partially parsed test_contains_with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test_contains_with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test_contains_with_none_argument. Retrieved 7/10 statements.
# Partially parsed test_contains_with_non_existent_key. Retrieved 8/11 statements.
# Partially parsed test_contains_with_existing_key_and_matching_value. Retrieved 7/10 statements.
# Partially parsed test_contains_with_empty_pmap. Retrieved 5/8 statements.


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
    var_6 = (var_1, var_3)

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
    var_6 = None

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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = (var_0, var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = 'any'
    var_3 = 1
    var_4 = (var_2, var_3)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_eq_with_different_cached_hash. Retrieved 9/12 statements.


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
    var_8 = var_5 == var_7
    assert var_8 is False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_constructor_returns_pmap_instance. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_large_dict. Retrieved 5/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5._size
    assert var_6 == 2
    var_7 = var_5._buckets
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = var_1._size
    assert var_2 == 0
    var_3 = var_1._buckets
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5['x']
    assert var_6 == 10
    var_7 = var_5['y']
    assert var_7 == 20

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
    var_8 = var_7._size
    assert var_8 == 3
    var_9 = var_7['a']
    assert var_9 == 1
    var_10 = var_7['b']
    assert var_10 == 2
    var_11 = var_7['c']
    assert var_11 == 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3['key']
    assert var_4 is None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3['key']
    assert var_4 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3['key']
    assert var_4 == 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'empty'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3['']
    assert var_4 == 'empty'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5[1]
    assert var_6 == 'one'
    var_7 = var_5[2]
    assert var_7 == 'two'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'tuple'
    var_4 = {var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5[1, 2]
    assert var_6 == 'tuple'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.pmap(var_4)
    var_6 = bool(var_3 is not var_5)
    assert var_6 is True
    var_7 = bool(var_3 == var_5)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = {var_0: var_1, var_0: var_2}
    var_4 = module_0.pmap(var_3)
    var_5 = var_4['a']
    assert var_5 == 2
    var_6 = var_4._size
    assert var_6 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'single'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    assert var_4 == 1
    var_5 = var_3['single']
    assert var_5 == 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {str(i): i for i in var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    assert var_4 == 100
    var_5 = var_3[var_0]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_eq_other_mapping_type. Retrieved 6/18 statements.
# Partially parsed test_eq_other_mapping_type_not_equal. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5 == var_5
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
    var_10 = var_5 == var_9
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
    var_11 = var_5 == var_10
    assert var_11 is False

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
    var_9 = var_5 == var_8
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
    var_10 = var_5 == var_9
    assert var_10 is False

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
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.m(**var_10)
    var_12 = var_7 == var_11
    assert var_12 is False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = [var_0, var_1]
    var_7 = var_5 == var_6

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
    var_13 = var_5 == var_10
    assert var_13 is False

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
    var_8 = var_5 == var_7
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_contains_with_valid_key_value_pair. Retrieved 7/10 statements.
# Partially parsed test_contains_with_key_not_in_map. Retrieved 8/11 statements.
# Partially parsed test_contains_with_wrong_value_for_key. Retrieved 8/11 statements.
# Partially parsed test_contains_with_non_tuple_argument. Retrieved 6/9 statements.
# Partially parsed test_contains_with_tuple_of_wrong_length. Retrieved 8/11 statements.
# Partially parsed test_contains_with_empty_tuple. Retrieved 7/10 statements.
# Partially parsed test_contains_with_non_iterable_argument. Retrieved 7/10 statements.
# Partially parsed test_contains_with_none_argument. Retrieved 7/10 statements.
# Partially parsed test_contains_with_tuple_key_not_hashable. Retrieved 9/12 statements.
# Partially parsed test_contains_with_exact_match_for_multiple_items. Retrieved 9/12 statements.


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
    var_6 = 3
    var_7 = (var_0, var_6)

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
    var_6 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_2, var_3]
    var_7 = 3
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
    var_8 = (var_1, var_4)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_constructor_preserves_hash_collisions. Retrieved 2/10 statements.
# Partially parsed test_constructor_does_not_share_internal_state. Retrieved 7/9 statements.
# Partially parsed test_constructor_preserves_order_of_insertion_iteration. Retrieved 8/12 statements.
# Partially parsed test_constructor_with_custom_hashable_objects. Retrieved 4/16 statements.


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
    var_10 = bool(var_9 == var_0)
    assert var_10 is True
    var_11 = var_8['a']
    assert var_11 == 1
    var_12 = var_8['b']
    assert var_12 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = dict(var_0)
    var_3 = bool(var_2 == {})
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'inner'
    var_5 = {var_4: var_3}
    var_6 = module_0.pmap(var_5)
    var_7 = var_6['inner']['x']
    assert var_7 == 10

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.m(**var_4)
    var_6 = var_5['a']
    assert var_6 == 1
    var_7 = var_5['b']
    assert var_7 == 2
    var_8 = len(var_5)
    assert var_8 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'val1'
    var_3 = 'val2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5['key1']
    assert var_6 == 'val1'
    var_7 = var_5['key2']
    assert var_7 == 'val2'

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
    var_8 = var_7['k1']
    assert var_8 == 'v1'
    var_9 = var_7['k2']
    assert var_9 == 'v2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = module_0.pmap(var_2)
    var_5 = len(var_3)
    assert var_5 == 1
    var_6 = len(var_4)
    assert var_6 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0.pmap(var_5)
    var_7 = var_6['a']
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = None
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3[None]
    assert var_4 == 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'false_value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3[False]
    assert var_4 == 'false_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'true_value'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3[True]
    assert var_4 == 'true_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'answer'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3[42]
    assert var_4 == 'answer'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = 'pi'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3[3.14]
    assert var_4 == 'pi'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'tuple_value'
    var_4 = {var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = var_5[1, 2]
    assert var_6 == 'tuple_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = list(var_1)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1000
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}
    var_4 = module_0.pmap(var_3)
    var_5 = len(var_4)
    assert var_5 == 1000
    var_6 = var_4[500]
    assert var_6 == 1000

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 'a'
    var_2 = 'm'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_eq_with_different_cached_hash. Retrieved 9/12 statements.


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
    var_8 = var_5 == var_7
    assert var_8 is False



