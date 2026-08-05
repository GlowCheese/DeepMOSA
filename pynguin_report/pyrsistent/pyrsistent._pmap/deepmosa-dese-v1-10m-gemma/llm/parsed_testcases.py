####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pmap_items_contains_valid_pair. Retrieved 5/16 statements.
# Partially parsed test_pmap_items_contains_invalid_value. Retrieved 3/13 statements.
# Partially parsed test_pmap_items_contains_nonexistent_key. Retrieved 3/13 statements.
# Partially parsed test_pmap_items_contains_invalid_format. Retrieved 3/13 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_multiple_items. Retrieved 13/14 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

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
    var_11 = [var_3, var_10]
    var_12 = 3



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pmap_constructor_initializes_attributes. Retrieved 8/11 statements.
# Partially parsed test_pmap_constructor_empty_state. Retrieved 1/6 statements.
# Partially parsed test_pmap_constructor_with_none_buckets. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 2

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = None
    var_1 = 3
    var_2 = 'c'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1



# Parsed testcases at query #4
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapValues(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0.PMapValues(var_4)
    var_7 = module_0.PMapValues(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapValues(var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 17/18 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_none_bucket_elements. Retrieved 11/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = hash(var_0)
    var_2 = 10
    var_3 = var_1 % var_2
    var_4 = 1
    var_5 = (var_0, var_4)
    var_6 = [var_5]
    var_7 = (var_3, var_6)
    var_8 = 'b'
    var_9 = hash(var_8)
    var_10 = var_9 % var_2
    var_11 = 2
    var_12 = (var_8, var_11)
    var_13 = [var_12]
    var_14 = (var_10, var_13)
    var_15 = [var_7, var_14]
    var_16 = 2

def test_case_0():
    var_0 = []
    var_1 = 0

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = hash(var_1)
    var_3 = 2
    var_4 = var_2 % var_3
    var_5 = 1
    var_6 = (var_1, var_5)
    var_7 = (var_4, var_6)
    var_8 = [var_7]
    var_9 = [var_0, var_8]
    var_10 = 1



# Parsed testcases at query #6
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()

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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pmap_update_with_addition. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_precedence. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 11/13 statements.
# Partially parsed test_pmap_update_with_no_changes. Retrieved 4/6 statements.
# Partially parsed test_pmap_update_with_new_keys. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = module_0.m()
    var_3 = 5
    var_4 = 30
    var_5 = module_0.m()
    var_6 = 'b'
    var_7 = 'd'
    var_8 = 40
    var_9 = {var_6: var_3, var_7: var_8}
    var_10 = lambda l, r: l + r

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
    var_2 = 2
    var_3 = module_0.m()
    var_4 = lambda l, r: r



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_contains_predicate_false_on_invalid_tuple_unpacking. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #9
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
    var_7 = len(var_6)
    assert var_7 == 2

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
    var_8 = None
    var_9 = module_0._turbo_mapping(var_6, var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._buckets
    var_6 = len(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 4
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
    var_5 = module_0._turbo_mapping(var_4, var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_update_with_predicate_false_on_new_key. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.m()
    var_2 = 3
    var_3 = module_0.m()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pmap_constructor_initializes_attributes. Retrieved 15/16 statements.
# Partially parsed test_pmap_constructor_is_not_direct_usage_compliant. Retrieved 3/6 statements.
# Partially parsed test_pmap_constructor_with_empty_data. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = []
    var_1 = 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_contains_evaluates_false_on_exception_during_unpacking. Retrieved 1/5 statements.
# Partially parsed test_contains_evaluates_false_on_non_iterable_arg. Retrieved 1/5 statements.
# Partially parsed test_contains_evaluates_false_on_single_element_tuple. Retrieved 2/6 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 1
    var_1 = (var_0,)



# Parsed testcases at query #13
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 5



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_contains_with_non_iterable_arg_returns_false. Retrieved 1/5 statements.
# Partially parsed test_contains_with_single_element_tuple_returns_false. Retrieved 2/6 statements.
# Partially parsed test_contains_with_non_tuple_arg_returns_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 1
    var_1 = (var_0,)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_multiple_elements. Retrieved 10/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_collisions. Retrieved 9/10 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = 2
    var_5 = (var_1, var_4)
    var_6 = [var_3, var_5]
    var_7 = [var_6]
    var_8 = 2



# Parsed testcases at query #18
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()
    var_4 = hash(var_2)
    var_5 = hash(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_update_with_does_not_trigger_if_condition. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.m()
    var_2 = 10
    var_3 = module_0.m()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_update_with_predicate_false_on_new_key. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()



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




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #23
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 5



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_via_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 10/11 statements.
# Partially parsed test_pmap_constructor_empty_state. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_none_buckets. Retrieved 3/4 statements.
# Partially parsed test_pmap_constructor_with_hashable_keys. Retrieved 7/8 statements.


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
    var_9 = 2

def test_case_0():
    var_0 = []
    var_1 = 0

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 1



# Parsed testcases at query #28
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = module_0.m()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_contains_predicate_evaluates_to_false_on_non_iterable_arg. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pmap_update_with_addition. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_logic. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 9/11 statements.
# Partially parsed test_pmap_update_with_no_maps. Retrieved 3/5 statements.
# Partially parsed test_pmap_update_with_replacement. Retrieved 6/8 statements.


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
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 'd'
    var_6 = 4
    var_7 = {var_5: var_6}
    var_8 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 10
    var_4 = module_0.m()
    var_5 = lambda l, r: r



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_contains_predicate_evaluates_to_false_with_invalid_tuple_format. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = (var_0,)



# Parsed testcases at query #32
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
    var_7 = len(var_6)
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 20
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 8
    var_8 = module_0._turbo_mapping(var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 99
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 4
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



# Parsed testcases at query #33
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = module_0.m()
    var_3 = 2
    var_4 = module_0.m()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pmap_update_with_merging_logic. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_logic. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_no_maps. Retrieved 3/5 statements.


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
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 4
    var_6 = module_0.m()
    var_7 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_multiple_items. Retrieved 10/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 5



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_turbo_mapping_with_pre_size. Retrieved 6/8 statements.
# Partially parsed test_turbo_mapping_with_empty_dict. Retrieved 4/6 statements.
# Partially parsed test_turbo_mapping_collision_handling. Retrieved 5/22 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'first'
    var_3 = 'second'
    var_4 = 8



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pmap_update_with_merging_values. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_precedence. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps_and_rightmost_precedence. Retrieved 11/13 statements.
# Partially parsed test_pmap_update_with_no_changes. Retrieved 4/6 statements.
# Partially parsed test_pmap_update_with_new_keys. Retrieved 5/7 statements.


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
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'd'
    var_7 = 17
    var_8 = 35
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = lambda l, r: r

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
    var_2 = 2
    var_3 = module_0.m()
    var_4 = lambda l, r: r



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty_state. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_multiple_items. Retrieved 10/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_contains_invalid_arg_type_returns_false. Retrieved 1/9 statements.
# Partially parsed test_contains_invalid_arg_structure_returns_false. Retrieved 2/10 statements.
# Partially parsed test_contains_non_iterable_arg_returns_false. Retrieved 1/9 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 1
    var_1 = (var_0,)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pmap_constructor_initializes_attributes. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_handles_empty_state. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_multiple_elements. Retrieved 10/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'a'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_pmap_update_with_addition. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_precedence. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 8/11 statements.
# Partially parsed test_pmap_update_with_dict. Retrieved 9/11 statements.
# Partially parsed test_pmap_update_with_no_changes. Retrieved 4/6 statements.


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
    var_2 = 2
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 4
    var_6 = 5
    var_7 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r
    var_4 = 'a'
    var_5 = 'c'
    var_6 = 10
    var_7 = 3
    var_8 = {var_4: var_6, var_5: var_7}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: r



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_pmap_values_eq_self.
# Failed to parse test_pmap_values_eq_different_instance.
# Failed to parse test_pmap_values_eq_with_value.




# Parsed testcases at query #2
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 8
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
    var_5 = 4
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2

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
    var_9 = len(var_8)
    assert var_9 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty_state. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_collisions. Retrieved 10/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pmap_constructor_initializes_slots. Retrieved 6/9 statements.
# Partially parsed test_pmap_constructor_with_empty_state. Retrieved 3/8 statements.
# Partially parsed test_pmap_constructor_equality_identity. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = None
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = [var_3]



# Parsed testcases at query #5
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty. Retrieved 2/3 statements.
# Partially parsed test_pmap_constructor_with_collisions. Retrieved 10/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = []
    var_1 = 0

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
    var_7 = len(var_6)
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2

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
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = None
    var_8 = module_0._turbo_mapping(var_6, var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._turbo_mapping(var_2, var_1)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #8
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = module_0.m()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pmap_update_with_add_logic. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_logic. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 12/14 statements.
# Partially parsed test_pmap_update_with_no_changes. Retrieved 6/8 statements.
# Partially parsed test_pmap_update_with_new_keys. Retrieved 8/10 statements.


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
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 10
    var_4 = 3
    var_5 = module_0.m()
    var_6 = 'c'
    var_7 = 'd'
    var_8 = 5
    var_9 = 4
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = lambda l, r: l + r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = lambda l, r: l
    var_4 = 0
    var_5 = module_0.m()

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



# Parsed testcases at query #10
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)



# Parsed testcases at query #11
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.PMapItems(var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pmap_constructor_initializes_attributes. Retrieved 11/14 statements.
# Partially parsed test_pmap_constructor_with_empty_buckets. Retrieved 3/8 statements.
# Partially parsed test_pmap_constructor_preserves_identity. Retrieved 6/9 statements.


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
    var_0 = 'k'
    var_1 = 'v'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pmap_update_with_basic_merge. Retrieved 11/13 statements.
# Partially parsed test_pmap_update_with_custom_function. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_leftmost_preference. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_no_overlapping_keys. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_empty_map. Retrieved 4/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'd'
    var_7 = 17
    var_8 = 35
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = lambda l, r: r

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
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()
    var_4 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = module_0.m()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmap_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 5



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_update_with_does_not_evaluate_key_in_evolver_as_true_for_new_keys. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pmap_constructor_initialization. Retrieved 7/8 statements.
# Partially parsed test_pmap_constructor_empty_state. Retrieved 3/4 statements.
# Partially parsed test_pmap_constructor_with_multiple_items. Retrieved 10/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4, var_0]
    var_6 = 1

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = 0

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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pmap_items_contains_valid_pair. Retrieved 6/25 statements.
# Partially parsed test_pmap_items_contains_invalid_key. Retrieved 3/14 statements.
# Partially parsed test_pmap_items_contains_mismatched_value. Retrieved 3/14 statements.
# Partially parsed test_pmap_items_contains_non_iterable_arg. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_eq_not_implemented_for_non_mapping. Retrieved 3/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 5



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_turbo_mapping_empty. Retrieved 4/6 statements.


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
    var_5 = 4
    var_6 = module_0._turbo_mapping(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = module_0._turbo_mapping(var_2, var_1)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3._buckets
    var_6 = len(var_5)
    assert var_6 == 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = iter(var_6)
    var_8 = 0
    var_9 = module_0._turbo_mapping(var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = var_4._buckets
    var_6 = len(var_5)
    assert var_6 == 8



# Parsed testcases at query #20
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_contains_raises_exception_on_ununpackable_arg. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pmap_update_with_add_values. Retrieved 4/7 statements.
# Partially parsed test_pmap_update_with_keep_leftmost. Retrieved 8/10 statements.
# Partially parsed test_pmap_update_with_multiple_maps. Retrieved 10/13 statements.
# Partially parsed test_pmap_update_with_no_overlap. Retrieved 5/7 statements.
# Partially parsed test_pmap_update_with_empty_maps. Retrieved 5/7 statements.


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
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 'a'
    var_5 = 3
    var_6 = {var_4: var_5}
    var_7 = lambda l, r: l

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.m()
    var_3 = 3
    var_4 = module_0.m()
    var_5 = 'a'
    var_6 = 'd'
    var_7 = 17
    var_8 = 35
    var_9 = {var_5: var_7, var_6: var_8}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()
    var_4 = lambda l, r: r

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = lambda l, r: r
    var_3 = module_0.m()
    var_4 = {}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_update_with_predicate_evaluates_to_false. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()
    var_4 = 5
    var_5 = module_0.m()



# Parsed testcases at query #24
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = module_0.m()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pmap_items_contains_valid_item. Retrieved 12/22 statements.
# Partially parsed test_pmap_items_contains_invalid_tuple_structure. Retrieved 10/19 statements.
# Partially parsed test_pmap_items_contains_wrong_value. Retrieved 10/19 statements.
# Partially parsed test_pmap_items_contains_missing_key. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = lambda x: x in var_5
    var_7 = lambda x: var_2
    var_8 = (var_0, var_2)
    var_9 = [var_8]
    var_10 = iter(var_9)
    var_11 = lambda : var_10

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = lambda x: var_3
    var_6 = (var_0, var_3)
    var_7 = [var_6]
    var_8 = iter(var_7)
    var_9 = lambda : var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = lambda x: var_3
    var_6 = (var_0, var_3)
    var_7 = [var_6]
    var_8 = iter(var_7)
    var_9 = lambda : var_8

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda x: x == var_0
    var_4 = lambda x: var_1
    var_5 = (var_0, var_1)
    var_6 = [var_5]
    var_7 = iter(var_6)
    var_8 = lambda : var_7



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_update_with_does_not_trigger_true_condition_for_new_key. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()



# Parsed testcases at query #28
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_contains_predicate_false_on_ununpackable_arg. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_eq_not_notimplemented_for_mapping. Retrieved 3/6 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = module_0.m()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_eq_not_mapping_returns_not_implemented. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_update_with_predicate_false. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_contains_evaluates_false_on_unhashable_arg. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_turbo_mapping_predicate_false_via_exception. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pmap_eq_not_mapping_returns_not_implemented. Retrieved 5/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 'a'
    var_3 = (var_2, var_0)
    var_4 = [var_3]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_update_with_leftmost_behavior_is_false. Retrieved 4/7 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.m()
    var_2 = 2
    var_3 = module_0.m()



# Parsed testcases at query #37
#--------------------------




import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PMapItems(var_4)



# Parsed testcases at query #38
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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 16
    var_4 = module_0._turbo_mapping(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4._buckets
    var_7 = len(var_6)
    assert var_7 == 16

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0._turbo_mapping(var_0, var_1)
    var_3 = len(var_2)
    assert var_3 == 0

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._turbo_mapping(var_4, var_2)
    var_6 = len(var_5)
    assert var_6 == 2



