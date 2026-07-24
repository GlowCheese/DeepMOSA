####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Partially parsed test_maybe_parse_user_type_with_mixed_list. Retrieved 1/3 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_empty_list. Retrieved 2/7 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_mixed_numeric_types. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_default_parameter. Retrieved 1/6 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_preserves_type. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_checked_map_type_meta_new_stores_key_types. Retrieved 1/4 statements.
# Partially parsed test_checked_map_type_meta_new_stores_value_types. Retrieved 1/4 statements.
# Partially parsed test_checked_map_type_meta_new_stores_invariants. Retrieved 1/8 statements.
# Partially parsed test_checked_map_type_meta_new_sets_default_serializer. Retrieved 1/6 statements.
# Partially parsed test_checked_map_type_meta_new_sets_empty_slots. Retrieved 1/4 statements.
# Partially parsed test_checked_map_type_meta_new_default_serializer_with_primitives. Retrieved 3/7 statements.
# Partially parsed test_checked_map_type_meta_new_inherits_key_types. Retrieved 1/6 statements.
# Partially parsed test_checked_map_type_meta_new_inherits_value_types. Retrieved 1/6 statements.
# Partially parsed test_checked_map_type_meta_new_inherits_invariants. Retrieved 1/10 statements.
# Failed to parse test_checked_map_type_meta_new_multiple_types.


def test_case_0():
    var_0 = '_checked_key_types'

def test_case_0():
    var_0 = '_checked_value_types'

def test_case_0():
    var_0 = '_checked_invariants'

def test_case_0():
    var_0 = '__serializer__'

def test_case_0():
    var_0 = '__slots__'

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'

def test_case_0():
    var_0 = '_checked_key_types'

def test_case_0():
    var_0 = '_checked_value_types'

def test_case_0():
    var_0 = '_checked_invariants'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_invariant_errors_all_pass. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_all_fail. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_mixed. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_single_pass. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_single_fail. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_with_different_data_types. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_uses_elem_parameter. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'expected'
    var_1 = 'unexpected'



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_checked_map_type_meta_line_3_predicate_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'regular_key'
    var_1 = 'regular_value'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_store_invariants_single_invariant_in_dct. Retrieved 4/12 statements.
# Partially parsed test_store_invariants_invariant_in_base_class. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_multiple_invariants. Retrieved 2/14 statements.
# Partially parsed test_store_invariants_wrapped_invariant_returns_bool_tuple. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_wrapped_invariant_merges_results. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_inheritance_order. Retrieved 2/14 statements.
# Partially parsed test_store_invariants_diamond_inheritance. Retrieved 5/16 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = var_0['dest']
    var_6 = bool(var_0['dest'] == ())
    assert var_6 is True

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 0

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True

def test_case_0():
    var_0 = 'src'
    var_1 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'dest'
    var_5 = 'src'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Invariants must be callable'

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 0
    var_4 = 10

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 0
    var_4 = 10

def test_case_0():
    var_0 = 'src'
    var_1 = 'dest'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_type_checking. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_value_type_checking. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'InvariantException'

def test_case_0():
    var_0 = 'key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not a float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_returns_correct_type. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_checked_pmap_new_with_empty_initial.
# Partially parsed test_checked_pmap_new_with_initial_dict. Retrieved 5/12 statements.
# Partially parsed test_checked_pmap_new_with_size_parameter. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_new_with_invalid_key_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_new_with_invalid_value_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_new_with_invariant. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_new_with_invariant_violation. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_new_with_multiple_entries. Retrieved 7/13 statements.
# Partially parsed test_checked_pmap_new_preserves_type. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 16

def test_case_0():
    var_0 = 1
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'not_an_int'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_list. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_values. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector. Retrieved 4/11 statements.
# Partially parsed test_checked_pvector_constructor_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_checked_pvector_constructor_with_mixed_numeric_types. Retrieved 4/8 statements.
# Partially parsed test_checked_pvector_constructor_type_error. Retrieved 4/8 statements.
# Partially parsed test_checked_pvector_constructor_invariant_error. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_single_element. Retrieved 2/6 statements.
# Partially parsed test_checked_pvector_constructor_preserves_type. Retrieved 4/10 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 42
    var_1 = [var_0]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_list. Retrieved 2/7 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_integers. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_mixed_numbers. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector. Retrieved 6/12 statements.
# Partially parsed test_checked_pvector_constructor_with_single_element. Retrieved 3/7 statements.
# Partially parsed test_checked_pvector_constructor_with_large_list. Retrieved 5/9 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.python_pvector(var_4)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 42
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 101
    var_3 = range(var_1, var_2)
    var_4 = list(var_3)



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_checkedtype_constructor.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_parameter.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_store_types_single_type_in_dct. Retrieved 3/5 statements.
# Partially parsed test_store_types_multiple_types_as_list. Retrieved 3/6 statements.
# Partially parsed test_store_types_from_base_class. Retrieved 3/8 statements.
# Partially parsed test_store_types_dct_overrides_base. Retrieved 2/8 statements.
# Partially parsed test_store_types_multiple_bases. Retrieved 3/11 statements.
# Partially parsed test_store_types_nested_iterables. Retrieved 3/7 statements.
# Partially parsed test_store_types_source_not_in_dct_or_bases. Retrieved 4/6 statements.
# Partially parsed test_store_types_mixed_types_and_strings. Retrieved 4/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'destination'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0['destination']
    var_6 = bool(var_0['destination'] == ())
    assert var_6 is True

def test_case_0():
    var_0 = 'source'
    var_1 = ()
    var_2 = 'destination'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 'str'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'destination'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['destination']
    var_7 = bool(var_2['destination'] == ('str',))
    assert var_7 is True

def test_case_0():
    var_0 = 'source'
    var_1 = ()
    var_2 = 'destination'

def test_case_0():
    var_0 = {}
    var_1 = 'destination'
    var_2 = 'source'
    var_3 = var_0['destination']

def test_case_0():
    var_0 = 'source'
    var_1 = 'destination'

def test_case_0():
    var_0 = {}
    var_1 = 'destination'
    var_2 = 'source'
    var_3 = var_0['destination']

def test_case_0():
    var_0 = 'source'
    var_1 = ()
    var_2 = 'destination'

def test_case_0():
    var_0 = 'other'
    var_1 = ()
    var_2 = 'destination'
    var_3 = 'source'

def test_case_0():
    var_0 = 'source'
    var_1 = 'CustomType'
    var_2 = ()
    var_3 = 'destination'



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_maybe_parse_user_type_line_18_predicate.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 7/12 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_checked_pmap_initial_items_iteration. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.5
    var_5 = 3.5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 1
    var_8 = 1.5
    var_9 = (var_7, var_8)
    var_10 = 2
    var_11 = 2.5
    var_12 = (var_10, var_11)
    var_13 = 3
    var_14 = 3.5
    var_15 = (var_13, var_14)



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Partially parsed test_maybe_parse_user_type_with_mixed_list. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Partially parsed test_maybe_parse_user_type_with_dict. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

def test_case_0():
    var_0 = 'type'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Type specifications must be types or strings'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checkedpset_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_negative_values. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_repr. Retrieved 6/11 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.7
    var_3 = 3.2
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.7
    var_7 = 3.2

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3, var_3, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'Positives'
    var_6 = 'Positives'



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_checked_pmap_new_with_empty_initial.
# Partially parsed test_checked_pmap_new_with_initial_dict. Retrieved 5/12 statements.
# Partially parsed test_checked_pmap_new_with_size_parameter. Retrieved 4/11 statements.
# Partially parsed test_checked_pmap_new_with_empty_and_size. Retrieved 2/9 statements.
# Partially parsed test_checked_pmap_new_validates_key_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_new_validates_value_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_new_with_invariant. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_new_invariant_violation. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_new_multiple_items. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 16

def test_case_0():
    var_0 = {}
    var_1 = 32

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_raises_error_on_invalid_type. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_none_in_expected_types. Retrieved 4/9 statements.
# Partially parsed test_check_types_error_message_format. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'invalid'
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1]

def test_case_0():
    var_0 = 1.5
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass'
    var_4 = 'int'
    var_5 = 'float'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_without_checked_type_subclass. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_subclass. Retrieved 5/15 statements.
# Partially parsed test_checked_type_create_with_matching_type_in_data. Retrieved 4/16 statements.
# Partially parsed test_checked_type_create_ignore_extra_parameter. Retrieved 6/16 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 42
    var_3 = [var_2]
    var_4 = 2
    var_5 = 3

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #26
#--------------------------

# Failed to parse test_checked_pmap_new_with_empty_initial.
# Partially parsed test_checked_pmap_new_with_initial_dict. Retrieved 5/12 statements.
# Partially parsed test_checked_pmap_new_with_size_parameter. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_new_invalid_key_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_new_invalid_value_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_new_with_invariant_valid. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_new_with_invariant_invalid. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_new_preserves_type. Retrieved 3/10 statements.
# Partially parsed test_checked_pmap_new_multiple_items. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16

def test_case_0():
    var_0 = 123
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = '4'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #28
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_default_parameter. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_class_type. Retrieved 3/10 statements.
# Partially parsed test_checked_pmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_wrong_value_type. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_string_type_names. Retrieved 6/17 statements.
# Partially parsed test_check_types_raises_with_wrong_string_type. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'invalid'
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass'
    var_6 = 'int'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'invalid'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()
    var_6 = set()



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #33
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #34
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_non_checked_type. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_matching_data. Retrieved 1/14 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_non_matching_data. Retrieved 4/11 statements.
# Partially parsed test_checked_type_create_with_mixed_data_types. Retrieved 2/11 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_flag. Retrieved 4/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'raw_data_1'
    var_3 = 'raw_data_2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'raw_data'

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'raw_data'
    var_3 = [var_2]
    var_4 = True



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/12 statements.
# Partially parsed test_checkedpmap_constructor_with_single_entry. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'hello'
    var_3 = 'world'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_list_data_no_checked_types. Retrieved 5/9 statements.
# Failed to parse test_get_type_with_type_object.
# Partially parsed test_get_types_with_multiple_types. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)

def test_case_0():
    var_0 = 'builtins.str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.get_types(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = module_0._get_class(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.dict'
    var_1 = module_0._get_class(var_0)



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_single_bool_result_false.
# Failed to parse test_wrap_invariant_with_multiple_results_all_pass.
# Failed to parse test_wrap_invariant_with_multiple_results_one_fails.
# Failed to parse test_wrap_invariant_with_multiple_results_all_fail.
# Partially parsed test_wrap_invariant_passes_args_and_kwargs. Retrieved 3/7 statements.
# Partially parsed test_wrap_invariant_with_multiple_results_and_args. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_single_element. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_elements. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_preserves_class_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_default_parameter. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 10
    var_1 = 10.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = {}



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 11/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_check_types_predicate_true_with_nonempty_expected_types.




# Parsed testcases at query #41
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = set()



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_failed_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_entries. Retrieved 11/16 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'string'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string_value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_default_argument. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = {var_0: var_0, var_1: var_1, var_2: var_2, var_3: var_3}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_checkedtype_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_default_size_parameter. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 10
    var_1 = 10.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 8



# Parsed testcases at query #47
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_no_checked_types. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_matching_data. Retrieved 2/10 statements.
# Partially parsed test_checked_type_create_returns_cls_instance_with_source_data. Retrieved 5/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'test_module.CheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = False

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.
# Partially parsed test_checked_pmap_constructor_preserves_class_type. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.5
    var_5 = 2.25
    var_6 = 3.75
    var_7 = 4.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_check_types_predicate_line_1_evaluates_to_false. Retrieved 6/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(var_0)
    assert var_5 is False



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 11/17 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_default_parameter. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_returns_correct_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 5
    var_1 = 5.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.1
    var_5 = 2.2
    var_6 = 3.3
    var_7 = 4.4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #55
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}



# Parsed testcases at query #56
#--------------------------

# Failed to parse test_checkedtype_constructor.




# Parsed testcases at query #57
#--------------------------

# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checkedpset_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_negative_raises_invariant_exception. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type_raises_exception. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_pmap_initial. Retrieved 2/8 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.5
    var_7 = 3.5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3, var_3, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = 0
    var_6 = 1
    var_7 = 2

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'string'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pmap()



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_restore_pickle_calls_create_with_data_and_empty_factory_fields. Retrieved 4/10 statements.
# Partially parsed test_restore_pickle_with_empty_dict. Retrieved 2/8 statements.
# Partially parsed test_restore_pickle_with_complex_data. Retrieved 11/17 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()

def test_case_0():
    var_0 = {}
    var_1 = set()

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = {var_0: var_4, var_1: var_8}
    var_10 = set()



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_list. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_integers. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_numbers. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_number. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_tuple_input. Retrieved 5/9 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.5
    var_7 = 3.5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3, var_3, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_wrap_invariant_predicate_line_3. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Partially parsed test_maybe_parse_user_type_with_mixed_list. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Failed to parse test_maybe_parse_user_type_with_complex_nested_structure.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_checked_pmap_constructor_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_single_entry. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 11/17 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #63
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Partially parsed test_maybe_parse_user_type_with_invalid_list. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

def test_case_0():
    var_0 = 42
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'



# Parsed testcases at query #64
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_non_checked_type. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_matching_data. Retrieved 1/9 statements.
# Failed to parse test_checked_type_create_returns_same_instance_when_already_correct_type.
# Failed to parse test_get_type_with_type_object.
# Partially parsed test_get_types_with_multiple_types. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)

def test_case_0():
    var_0 = 'builtins.float'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.get_types(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_check_types_predicate_evaluates_to_false. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'string'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'can only be used with'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_check_types_predicate_with_empty_expected_types. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = None



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 2/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'not an instance'



# Parsed testcases at query #68
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_key_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_value_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_element'
    var_1 = 'invalid_data_2'
    var_2 = 'invalid_data_3'
    var_3 = 'valid_data_1'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_invalid_type_raises_exception. Retrieved 6/18 statements.
# Partially parsed test_check_types_empty_iterable. Retrieved 1/5 statements.
# Partially parsed test_check_types_with_custom_exception. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'invalid'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = True
    assert var_5 is True
    var_6 = 'TestClass'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = True
    assert var_5 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_parameter.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 9/14 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

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



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_element'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_check_types_predicate_evaluates_to_true. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = False
    assert var_5 is True



# Parsed testcases at query #74
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 5
    var_1 = 5.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.1
    var_5 = 2.2
    var_6 = 3.3
    var_7 = 4.4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_values. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.5
    var_7 = 3.5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3, var_3, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3



# Parsed testcases at query #77
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_failed_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_list. Retrieved 2/6 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_integers. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_mixed_numeric_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector. Retrieved 6/11 statements.
# Partially parsed test_checked_pvector_constructor_with_tuple. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_single_element. Retrieved 3/7 statements.
# Partially parsed test_checked_pvector_constructor_with_default_argument. Retrieved 1/5 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.python_pvector(var_4)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 5
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test_elem'



# Parsed testcases at query #81
#--------------------------

# Failed to parse test_check_types_predicate_line_1.




# Parsed testcases at query #82
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_empty_factory_fields. Retrieved 5/14 statements.
# Partially parsed test_restore_pickle_calls_create_with_correct_parameters. Retrieved 5/15 statements.
# Partially parsed test_restore_pickle_returns_instance. Retrieved 1/8 statements.
# Partially parsed test_restore_pickle_with_empty_data. Retrieved 1/7 statements.
# Partially parsed test_restore_pickle_factory_fields_is_always_empty_set. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = set()

def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'any'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #83
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_dict_initial. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not a float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.5
    var_5 = 3.5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #85
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 42
    var_1 = 42.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #87
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #88
#--------------------------

# Failed to parse test_checked_type_create_predicate_line_1.




# Parsed testcases at query #89
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 8/30 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = [var_0, var_1, var_2]
    var_6 = None
    var_7 = [var_0, var_1, var_2]



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_wrap_invariant_predicate_line_3. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_various_data_types. Retrieved 13/14 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = 'data3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'error3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = 'error3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = True
    var_1 = 123
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = 2
    var_9 = 3
    var_10 = [var_0, var_8, var_9]
    var_11 = (var_3, var_10)
    var_12 = [var_2, var_7, var_11]



# Parsed testcases at query #92
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_items. Retrieved 9/14 statements.
# Partially parsed test_checkedpmap_constructor_repr. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.1
    var_5 = 2.2
    var_6 = 3.3
    var_7 = 4.4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap'
    var_4 = '1'
    var_5 = '1.5'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_checked_type_create_returns_same_instance_when_already_correct_type. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_returns_cls_instance_when_source_data_not_instance. Retrieved 5/10 statements.
# Partially parsed test_checked_type_create_with_empty_checked_types. Retrieved 4/9 statements.
# Partially parsed test_checked_type_create_returns_instance_when_source_is_list. Retrieved 5/13 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_parameter. Retrieved 6/11 statements.
# Partially parsed test_checked_type_create_with_factory_fields_parameter. Retrieved 7/12 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 'extra'
    var_2 = 'field'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'field1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}



# Parsed testcases at query #94
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_items. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'InvariantException'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'not an instance of TestClass'



# Parsed testcases at query #96
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_single_element. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_elements. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.1
    var_5 = 2.2
    var_6 = 3.3
    var_7 = 4.4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_wrap_invariant_predicate_line_3. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_check_types_predicate_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #102
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_source_data_list. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_calls_cls_constructor. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_parameter. Retrieved 6/10 statements.
# Partially parsed test_checked_type_create_with_factory_fields_parameter. Retrieved 8/12 statements.
# Failed to parse test_get_type_with_type_object.
# Failed to parse test_get_types_with_multiple_types.
# Partially parsed test_get_types_with_mixed_types. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'field'
    var_6 = 'value'
    var_7 = {var_5: var_6}

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_types(var_2)

def test_case_0():
    var_0 = 'builtins.str'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'not an instance of MockCheckedType'



# Parsed testcases at query #104
#--------------------------

# Failed to parse test_checked_type_create_predicate_line_1_false.




# Parsed testcases at query #105
#--------------------------

# Partially parsed test_check_types_predicate_false. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'string'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type TestClass can only be used with'



# Parsed testcases at query #106
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_no_checked_types. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_matching_data. Retrieved 5/7 statements.
# Failed to parse test_get_type_with_type_object.
# Partially parsed test_get_types_with_multiple_types. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)

def test_case_0():
    var_0 = 'builtins.str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_types(var_2)
    var_4 = var_3[0]
    var_5 = var_3[1]



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_true. Retrieved 6/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_pass. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_fail. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_with_default_initial.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_creates_correct_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 11/17 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #109
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_returns_correct_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_preserves_values. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 8/10 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 4/8 statements.
# Partially parsed test_check_types_raises_error_for_invalid_type. Retrieved 7/12 statements.
# Partially parsed test_check_types_with_string_type_name. Retrieved 9/11 statements.
# Partially parsed test_check_types_with_mixed_string_and_type. Retrieved 7/10 statements.
# Partially parsed test_check_types_raises_error_with_custom_exception_type. Retrieved 8/15 statements.
# Partially parsed test_check_types_error_contains_actual_type. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = []
    var_1 = 'TestClass'
    var_2 = ()
    var_3 = {}
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2.5
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Type TestClass can only be used with'
    var_10 = 'not float'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'builtins.int'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'builtins.int'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'CustomException'
    var_5 = {}
    var_6 = 1
    var_7 = 'invalid'
    var_8 = [var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 3.14
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'float'



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_wrap_invariant_with_boolean_result. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_tuple_of_tuples. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_all_true_results. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_all_false_results. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_kwargs. Retrieved 2/6 statements.
# Partially parsed test_wrap_invariant_multiple_args. Retrieved 3/7 statements.
# Failed to parse test_wrap_invariant_empty_error_list.
# Failed to parse test_wrap_invariant_single_false_result.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5
    var_1 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_raises_error_on_invalid_type. Retrieved 4/8 statements.
# Failed to parse test_check_types_with_class_types.
# Partially parsed test_check_types_raises_error_with_custom_exception_type. Retrieved 3/9 statements.
# Partially parsed test_check_types_error_message_format. Retrieved 3/7 statements.
# Failed to parse test_check_types_with_subclass.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Type'
    var_5 = 'can only be used with'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/6 statements.
# Partially parsed test_checkedpset_constructor_with_valid_elements. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_float_elements. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_mixed_valid_elements. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_duplicate_elements. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_negative_number. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_single_element. Retrieved 3/7 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.5
    var_7 = 3.5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3, var_3, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 42
    var_2 = [var_1]
    var_3 = 42



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_list.
# Partially parsed test_maybe_parse_user_type_with_mixed_types_and_strings. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_with_deeply_nested_list.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.5
    var_5 = 2.25
    var_6 = 3.75
    var_7 = 4.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_from_checked_pmap_instance. Retrieved 3/10 statements.
# Partially parsed test_checked_pmap_constructor_type_error_invalid_key. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_type_error_invalid_value. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_get_type_with_type_object.
# Failed to parse test_get_type_with_int_type.
# Failed to parse test_get_type_with_list_type.
# Failed to parse test_get_type_returns_type_for_type_input.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = module_0.get_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.list'
    var_1 = module_0.get_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = module_0.get_type(var_0)
    var_2 = var_1.__name__
    assert var_2 == 'OrderedDict'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_store_invariants_single_invariant_in_dct. Retrieved 4/12 statements.
# Partially parsed test_store_invariants_multiple_invariants_in_bases. Retrieved 5/17 statements.
# Partially parsed test_store_invariants_dct_overrides_base. Retrieved 2/14 statements.
# Partially parsed test_store_invariants_inherited_invariants_multiple_levels. Retrieved 5/17 statements.
# Partially parsed test_store_invariants_wrapped_invariant_with_bool_result. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_wrapped_invariant_with_tuple_result. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_duplicate_bases_not_repeated. Retrieved 5/12 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = var_0['dest']
    var_6 = bool(var_0['dest'] == ())
    assert var_6 is True

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 0

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = 'src'
    var_1 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'dest'
    var_5 = 'src'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Invariants must be callable'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 0

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 0

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 9/14 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.1
    var_5 = 2.2
    var_6 = 3.3
    var_7 = 4.4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_invariant_errors_all_valid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_all_invalid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_mixed. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_single_invalid. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_single_valid. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_with_different_data_types. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_store_invariants_with_source. Retrieved 4/12 statements.
# Partially parsed test_store_invariants_inherited. Retrieved 2/14 statements.
# Partially parsed test_store_invariants_multiple_bases. Retrieved 5/17 statements.
# Partially parsed test_store_invariants_wrapped_function. Retrieved 6/14 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = '_invariants'
    var_3 = 'invariant'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = '_invariants'
    var_6 = bool('_invariants' in var_0)
    assert var_6 is True
    var_7 = var_0['_invariants']
    var_8 = bool(var_0['_invariants'] == ())
    assert var_8 is True

def test_case_0():
    var_0 = 'my_invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = '_invariants'
    var_4 = 0

def test_case_0():
    var_0 = 'my_invariant'
    var_1 = '_invariants'
    var_2 = '_invariants'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'my_invariant'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = '_invariants'
    var_5 = 'my_invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Invariants must be callable'

def test_case_0():
    var_0 = 'my_invariant'
    var_1 = {}
    var_2 = '_invariants'
    var_3 = '_invariants'
    var_4 = bool('_invariants' in var_1)
    assert var_4 is True
    var_5 = var_1[var_2]
    var_6 = len(var_5)
    assert var_6 == 2

def test_case_0():
    var_0 = 'my_invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = 0
    var_4 = None
    var_5 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_list. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_integers. Retrieved 4/10 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_mixed_numeric_types. Retrieved 4/9 statements.
# Partially parsed test_checked_pvector_constructor_with_invariant_check. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector. Retrieved 4/12 statements.
# Partially parsed test_checked_pvector_constructor_with_tuple. Retrieved 4/9 statements.
# Partially parsed test_checked_pvector_constructor_with_generator. Retrieved 4/10 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'two'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #12
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/10 statements.
# Partially parsed test_invariant_errors_all_valid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_all_invalid. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test_element'
    var_1 = 'invalid1'
    var_2 = 'invalid2'

def test_case_0():
    var_0 = 'test_element'

def test_case_0():
    var_0 = 'test_element'



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 6/9 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 7/10 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 6/9 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 3/6 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 7/9 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 8/11 statements.
# Partially parsed test_check_types_first_element_invalid. Retrieved 6/9 statements.
# Partially parsed test_check_types_with_class_type_string. Retrieved 8/10 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    assert var_5 is False

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 1
    var_3 = 'string'
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'string'
    var_3 = 3.14
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    assert var_5 is False

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = True
    assert var_2 is False

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'string'
    var_3 = 3.14
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = True
    assert var_6 is False

def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = 1
    var_3 = 'string'
    var_4 = [var_2, var_3]
    var_5 = True
    var_6 = True
    assert var_6 is True
    var_7 = True
    assert var_7 is True

def test_case_0():
    var_0 = False
    var_1 = 'string'
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'builtins.int'
    var_6 = [var_5]
    var_7 = True
    assert var_7 is False



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_without_checked_types. Retrieved 2/6 statements.
# Partially parsed test_checked_type_create_with_checked_type_no_conversion_needed. Retrieved 1/9 statements.
# Partially parsed test_checked_type_create_with_matching_type_in_source_data. Retrieved 2/14 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_data'

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []

def test_case_0():
    var_0 = '__main__.ValidType'
    var_1 = [var_0]
    var_2 = 42



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = 'data3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'error2'
    var_7 = (var_0, var_6)
    var_8 = 'error3'
    var_9 = (var_0, var_8)
    var_10 = [var_2, var_5, var_7, var_9]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = 'error3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = set()



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/12 statements.
# Partially parsed test_checkedpmap_constructor_with_pmap_input. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_preserves_class_type. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 'key'
    var_1 = 42
    var_2 = {var_0: var_1}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 8/30 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_0, var_1, var_2]
    var_7 = None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_store_types_with_multiple_bases. Retrieved 6/10 statements.
# Partially parsed test_store_types_with_type_object. Retrieved 3/5 statements.
# Partially parsed test_store_types_with_list_of_types. Retrieved 3/6 statements.
# Partially parsed test_store_types_dct_takes_precedence. Retrieved 5/8 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '__annotations__'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'types'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = 'types'
    var_7 = bool('types' in var_2)
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'my_types'
    var_1 = 'int'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'parsed_types'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = 'parsed_types'
    var_7 = bool('parsed_types' in var_2)
    assert var_7 is True
    var_8 = var_2['parsed_types']
    var_9 = bool(var_2['parsed_types'] == ('int',))
    assert var_9 is True

def test_case_0():
    var_0 = 'source_attr'
    var_1 = 'str'
    var_2 = {var_0: var_1}
    var_3 = 'source_attr'
    var_4 = 'float'
    var_5 = {var_3: var_4}
    var_6 = 'source_attr'
    var_7 = 'int'
    var_8 = {var_6: var_7}
    var_9 = 'dest_attr'
    var_10 = 'dest_attr'
    var_11 = bool('dest_attr' in var_8)
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest_attr'
    var_5 = 'source_attr'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)
    var_7 = 'dest_attr'
    var_8 = bool('dest_attr' in var_2)
    assert var_8 is True
    var_9 = var_2['dest_attr']
    var_10 = bool(var_2['dest_attr'] == ())
    assert var_10 is True

def test_case_0():
    var_0 = 'my_types'
    var_1 = []
    var_2 = 'parsed_types'
    var_3 = 'parsed_types'

def test_case_0():
    var_0 = 'my_types'
    var_1 = []
    var_2 = 'parsed_types'
    var_3 = 'parsed_types'

def test_case_0():
    var_0 = 'source_attr'
    var_1 = 'base_value'
    var_2 = {var_0: var_1}
    var_3 = 'source_attr'
    var_4 = 'dct_value'
    var_5 = {var_3: var_4}
    var_6 = 'dest_attr'
    var_7 = 'dest_attr'
    var_8 = bool('dest_attr' in var_5)
    assert var_8 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checked_pmap_new_with_empty_initial. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_new_with_single_element. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_new_with_multiple_elements. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_new_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_new_with_size_parameter. Retrieved 6/11 statements.
# Failed to parse test_checked_pmap_new_default_parameters.
# Partially parsed test_checked_pmap_new_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_new_with_dict_initial. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_new_invalid_key_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_new_invalid_value_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_new_invariant_violation_raises_error. Retrieved 4/9 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_elem'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data1'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_list.
# Partially parsed test_maybe_parse_user_type_with_mixed_types_and_strings. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = set(var_3)
    var_5 = bool(var_4 == {'int', 'str'})
    assert var_5 is True

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 7/12 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_all_valid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_all_invalid. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test_elem'

def test_case_0():
    var_0 = 'test_elem'

def test_case_0():
    var_0 = 'test_elem'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_elem'
    var_2 = module_0._invariant_errors(var_1, var_0)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_maybe_parse_user_type_type_not_iterable.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_without_checked_type_subclass. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_subclass. Retrieved 6/22 statements.
# Partially parsed test_checked_type_create_with_matching_type. Retrieved 2/14 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = False

def test_case_0():
    var_0 = '__main__.InnerType'
    var_1 = [var_0]
    var_2 = 5



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_without_checked_type. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_matching_data. Retrieved 7/16 statements.
# Failed to parse test_get_type_with_type_object.
# Partially parsed test_get_types_with_multiple_types. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = '__main__.CheckedType'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_6]
    var_8 = False

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)

def test_case_0():
    var_0 = 'builtins.str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.get_types(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0._get_class(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = module_0._get_class(var_0)



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_failed_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 17/28 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = []
    var_3 = True
    var_4 = False
    assert var_4 is True
    var_5 = None
    var_6 = []
    var_7 = None
    var_8 = True
    var_9 = False
    assert var_9 is True
    var_10 = None
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = True
    var_16 = False
    assert var_16 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data1'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_wrap_invariant_with_single_boolean_result.
# Failed to parse test_wrap_invariant_with_multiple_results.
# Failed to parse test_wrap_invariant_all_passing.
# Failed to parse test_wrap_invariant_all_failing.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 1/7 statements.
# Failed to parse test_wrap_invariant_single_false_result.
# Failed to parse test_wrap_invariant_multiple_results_with_mixed_data_types.


def test_case_0():
    var_0 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_types_predicate_true. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Partially parsed test_maybe_parse_user_type_with_mixed_list. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Failed to parse test_maybe_parse_user_type_with_generator.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 4/12 statements.
# Partially parsed test_restore_pickle_with_empty_data. Retrieved 2/10 statements.
# Partially parsed test_restore_pickle_with_complex_data. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()

def test_case_0():
    var_0 = {}
    var_1 = set()

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = {var_0: var_4, var_1: var_8}
    var_10 = set()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 8/10 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 4/8 statements.
# Partially parsed test_check_types_raises_error_on_invalid_type. Retrieved 5/10 statements.
# Partially parsed test_check_types_raises_error_on_first_invalid_type. Retrieved 7/12 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 7/14 statements.
# Partially parsed test_check_types_error_message_format. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = []
    var_1 = 'TestClass'
    var_2 = ()
    var_3 = {}
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'invalid'
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'can only be used with'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = []
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'CustomException'
    var_1 = {}
    var_2 = 'TestClass'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = 'invalid'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 3.14
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestClass'
    var_8 = 'int'
    var_9 = 'float'



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_checked_type_constructor.




# Parsed testcases at query #42
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_without_checked_type. Retrieved 4/10 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_matching_data. Retrieved 4/19 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_non_matching_data. Retrieved 5/13 statements.
# Partially parsed test_checked_type_create_ignore_extra_parameter. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'created'
    var_1 = 'test.MockCheckedType'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = lambda x, ignore_extra=False: f'created_{id(x)}'
    var_1 = 'test.MockCheckedType'
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'created'
    var_1 = 'test.MockCheckedType'
    var_2 = 'data'
    var_3 = [var_2]
    var_4 = True



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_maybe_parse_user_type_type_not_iterable.




# Parsed testcases at query #44
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_predicate_line_18_evaluates_to_true.




# Parsed testcases at query #47
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_repr. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap'
    var_4 = '1'
    var_5 = '1.5'



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_wrap_invariant_with_bool_result.
# Failed to parse test_wrap_invariant_with_tuple_results.
# Failed to parse test_wrap_invariant_with_failed_results.
# Failed to parse test_wrap_invariant_with_all_failed_results.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 3/7 statements.
# Failed to parse test_wrap_invariant_with_tuple_results_mixed.
# Failed to parse test_wrap_invariant_preserves_bool_true.
# Failed to parse test_wrap_invariant_preserves_bool_false.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #49
#--------------------------

# Failed to parse test_checked_type_create_isinstance_returns_source_data.




# Parsed testcases at query #50
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_at_line_5_evaluates_to_true. Retrieved 6/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_checked_pmap_initial_items_iteration. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_checkedpvector_constructor_with_empty_initial.
# Partially parsed test_checkedpvector_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple_initial. Retrieved 4/8 statements.
# Partially parsed test_checkedpvector_constructor_with_pythonpvector_initial. Retrieved 4/11 statements.
# Partially parsed test_checkedpvector_constructor_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checkedpvector_constructor_with_invariant_violation. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_creates_correct_class. Retrieved 4/10 statements.
# Partially parsed test_checkedpvector_constructor_with_generator. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_float_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_raises_error. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_wrong_type_raises_error. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_default_no_args. Retrieved 1/6 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.5
    var_7 = 3.5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_1, var_2, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'string'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pmap()

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_without_checked_type. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_matching_data. Retrieved 6/21 statements.
# Partially parsed test_checked_type_create_with_checked_type_non_matching_data. Retrieved 9/23 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_flag. Retrieved 8/21 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
    var_1 = '__main__.CheckedType'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_5]
    var_7 = 0

def test_case_0():
    var_0 = []
    var_1 = '__main__.CheckedType'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = [var_5, var_8]

def test_case_0():
    var_0 = []
    var_1 = '__main__.CheckedType'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_5]
    var_7 = True
    var_8 = 0



# Parsed testcases at query #55
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'valid1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'valid2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test'
    var_9 = module_0._invariant_errors(var_8, var_7)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'valid'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'error_data'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_3, var_7]
    var_9 = 'test'
    var_10 = module_0._invariant_errors(var_9, var_8)
    var_11 = bool(var_10 == ['error_data'])
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = 'valid'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = 'error2'
    var_9 = (var_0, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_3, var_7, var_10]
    var_12 = 'test'
    var_13 = module_0._invariant_errors(var_12, var_11)
    var_14 = bool(var_13 == ['error1', 'error2'])
    assert var_14 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = 'error3'
    var_8 = (var_0, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_3, var_6, var_9]
    var_11 = 'test'
    var_12 = module_0._invariant_errors(var_11, var_10)
    var_13 = bool(var_12 == ['error1', 'error2', 'error3'])
    assert var_13 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = module_0._invariant_errors(var_1, var_0)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 42
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = 'valid'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = (var_0, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_3, var_7, var_12]
    var_14 = 'test'
    var_15 = module_0._invariant_errors(var_14, var_13)
    var_16 = bool(var_15 == [42, {'key': 'value'}])
    assert var_16 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = 'valid'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_3, var_7]
    var_9 = 'test'
    var_10 = module_0._invariant_errors(var_9, var_8)
    var_11 = bool(var_10 == [None])
    assert var_11 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checked_pmap_constructor_default_parameter.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_data. Retrieved 7/11 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'InvariantException'

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'CheckedKeyTypeError'

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'CheckedTypeError'

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 10.5
    var_4 = 20.75
    var_5 = 30.25
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #57
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_violates_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 11/17 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'not_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'not an instance of MockCheckedType'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_mixed_data_types. Retrieved 9/10 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = 'data3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'error3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = 'error3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 123
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data'
    var_5 = (var_3, var_4)
    var_6 = None
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = 'data3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'error2'
    var_7 = (var_0, var_6)
    var_8 = 'error3'
    var_9 = (var_0, var_8)
    var_10 = [var_2, var_5, var_7, var_9]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_with_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_returns_checkedpmap_instance. Retrieved 3/10 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = 'data3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'error2'
    var_7 = (var_0, var_6)
    var_8 = 'error3'
    var_9 = (var_0, var_8)
    var_10 = [var_2, var_5, var_7, var_9]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_checked_pmap_initial_items_iteration. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_evaluates_to_false. Retrieved 4/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'error_data'
    var_2 = (var_0, var_1)
    var_3 = [var_2]



# Parsed testcases at query #65
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_entries. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 5
    var_1 = 5.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_empty_list. Retrieved 1/6 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_values. Retrieved 5/11 statements.
# Partially parsed test_checkedpvector_constructor_with_mixed_numeric_types. Retrieved 4/9 statements.
# Partially parsed test_checkedpvector_constructor_with_generator. Retrieved 4/10 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 4/9 statements.
# Partially parsed test_checkedpvector_constructor_with_pythonpvector. Retrieved 4/12 statements.
# Partially parsed test_checkedpvector_constructor_invalid_type_raises_error. Retrieved 4/8 statements.
# Partially parsed test_checkedpvector_constructor_invariant_violation_raises_error. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_preserves_order. Retrieved 6/10 statements.
# Partially parsed test_checkedpvector_constructor_with_single_element. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = 'hello'
    var_1 = [var_0]



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_checkedpvector_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_floats. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_mixed_numeric_types. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_pythonpvector. Retrieved 5/12 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_generator. Retrieved 5/11 statements.
# Partially parsed test_checkedpvector_constructor_invalid_type. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_negative_value. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_preserves_class_type. Retrieved 5/10 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 8/10 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 4/8 statements.
# Partially parsed test_check_types_raises_error_on_invalid_type. Retrieved 7/12 statements.
# Partially parsed test_check_types_with_string_type_name. Retrieved 9/12 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 8/15 statements.
# Partially parsed test_check_types_error_message_format. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'hello'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = []
    var_1 = 'TestClass'
    var_2 = ()
    var_3 = {}
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2.5
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'can only be used with'
    var_10 = 'not float'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'builtins.int'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'CustomException'
    var_1 = {}
    var_2 = 'TestClass'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = 1
    var_7 = 'invalid'
    var_8 = [var_6, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2.5
    var_6 = [var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'TestClass'
    var_9 = 'int'
    var_10 = 'float'



# Parsed testcases at query #69
#--------------------------

# Failed to parse test_checked_type_create_isinstance_returns_source_data.




# Parsed testcases at query #70
#--------------------------

# Failed to parse test_check_types_predicate_true_with_non_empty_expected_types.




# Parsed testcases at query #71
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_failing_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_preserves_class_type. Retrieved 5/12 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'not_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data1'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 4/12 statements.
# Partially parsed test_restore_pickle_with_different_data. Retrieved 5/15 statements.
# Partially parsed test_restore_pickle_with_empty_data. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = set()

def test_case_0():
    var_0 = {}
    var_1 = set()



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test_elem'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_float_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_invariant_violation. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicate_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_default_empty. Retrieved 1/6 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.5
    var_7 = 3.5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3, var_3, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_evaluates_to_false. Retrieved 7/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data1'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #78
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'one'
    var_4 = 'two'
    var_5 = 'three'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checkedpset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_float_elements. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_duplicate_elements. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invariant_violation. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_zero_element. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_pmap_initial. Retrieved 7/12 statements.
# Partially parsed test_checkedpset_constructor_returns_correct_class_type. Retrieved 5/10 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 1.5
    var_6 = 2.5
    var_7 = 3.5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3, var_3, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.pmap(var_5)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 5
    var_2 = 10
    var_3 = 15
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #80
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #81
#--------------------------

# Partially parsed test_invariant_errors_all_pass. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_all_fail. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_mixed. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_single_pass. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_single_fail. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_with_different_data_types. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_single_entry. Retrieved 3/8 statements.
# Failed to parse test_checked_pmap_constructor_default_parameter.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_violates_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries_with_invariant. Retrieved 5/10 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = {var_1: var_1, var_2: var_2, var_3: var_3}



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 5/13 statements.
# Partially parsed test_restore_pickle_with_complex_data. Retrieved 9/16 statements.
# Partially parsed test_restore_pickle_factory_fields_is_empty_set. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'nested'
    var_4 = 'object'
    var_5 = {var_3: var_4}
    var_6 = [var_0, var_1, var_2, var_5]
    var_7 = set()
    var_8 = set()

def test_case_0():
    var_0 = 'test'
    var_1 = set()
    var_2 = set()



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_with_various_data_types. Retrieved 13/14 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = 'data3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'error3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = 'error3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = True
    var_1 = 123
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 2
    var_5 = 3
    var_6 = [var_0, var_4, var_5]
    var_7 = (var_3, var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = (var_3, var_10)
    var_12 = [var_2, var_7, var_11]



# Parsed testcases at query #86
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #87
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_returns_correct_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_items. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_checkedpmap_constructor_empty. Retrieved 1/6 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Failed to parse test_checkedpmap_constructor_default_initial.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/10 statements.
# Partially parsed test_checkedpmap_constructor_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 9/14 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.5
    var_5 = 2.25
    var_6 = 3.75
    var_7 = 4.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_checkedtype_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_at_line_5_evaluates_to_true. Retrieved 6/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_checkedpmap_initial_items_iteration. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #92
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_single_element. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_elements. Retrieved 9/14 statements.
# Partially parsed test_checkedpmap_constructor_returns_correct_type. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'InvariantException'

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'CheckedKeyTypeError'

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'CheckedTypeError'

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 1.1
    var_5 = 2.2
    var_6 = 3.3
    var_7 = 4.4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_invariant_errors_returns_empty_list_when_all_valid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_returns_all_errors_when_all_invalid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_with_single_valid_invariant. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_with_single_invalid_invariant. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_with_different_data_types. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #94
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #95
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #96
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/12 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'InvariantException'

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'CheckedKeyTypeError'

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'CheckedTypeError'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_violated_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_entries. Retrieved 11/18 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'key'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checkedpmap_constructor_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_failed_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_single_entry. Retrieved 3/9 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 5.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #99
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #100
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_default_parameter. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'not_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Failed to parse test_checkedpmap_constructor_default_parameter.
# Partially parsed test_checkedpmap_constructor_with_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 9/14 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'key'
    var_1 = 42
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #103
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_results_all_true.
# Failed to parse test_wrap_invariant_with_multiple_results_one_false.
# Failed to parse test_wrap_invariant_with_multiple_results_all_false.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 3/7 statements.
# Failed to parse test_wrap_invariant_with_false_bool_result.
# Failed to parse test_wrap_invariant_with_empty_result_list.
# Failed to parse test_wrap_invariant_with_mixed_error_data_types.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'



# Parsed testcases at query #104
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_without_checked_type. Retrieved 4/10 statements.
# Partially parsed test_checked_type_create_with_checked_type_matching_data. Retrieved 5/15 statements.
# Partially parsed test_checked_type_create_with_checked_type_not_matching_data. Retrieved 4/14 statements.
# Partially parsed test_checked_type_create_ignore_extra_parameter. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'data1'
    var_2 = 'data2'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 'created'
    var_1 = 'builtins.str'
    var_2 = 'data1'
    var_3 = [var_2]
    var_4 = False

def test_case_0():
    var_0 = 'created_data'
    var_1 = 'builtins.str'
    var_2 = 123
    var_3 = [var_2]

def test_case_0():
    var_0 = 'created'
    var_1 = 'builtins.str'
    var_2 = 456
    var_3 = [var_2]
    var_4 = True
    var_5 = True



