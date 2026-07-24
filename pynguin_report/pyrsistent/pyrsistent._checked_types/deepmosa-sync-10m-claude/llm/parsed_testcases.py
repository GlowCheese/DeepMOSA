####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_list.
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_invariant_errors_all_valid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_all_invalid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_mixed. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_single_valid. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_single_invalid. Retrieved 1/5 statements.
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



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
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
    var_0 = 'key'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Type specifications must be types or strings'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_store_invariants_basic. Retrieved 5/15 statements.
# Partially parsed test_store_invariants_multiple_inheritance. Retrieved 5/15 statements.
# Partially parsed test_store_invariants_override. Retrieved 3/14 statements.
# Partially parsed test_store_invariants_not_callable. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_wrapped_behavior. Retrieved 6/13 statements.
# Partially parsed test_store_invariants_diamond_inheritance. Retrieved 5/17 statements.
# Partially parsed test_store_invariants_mixed_callable_non_callable. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '__invariant__'
    var_1 = ()
    var_2 = '__wrapped_invariants__'
    var_3 = '__invariant__'
    var_4 = 0

def test_case_0():
    var_0 = {}
    var_1 = '__wrapped_invariants__'
    var_2 = '__invariant__'
    var_3 = bool(var_1 in var_0)
    assert var_3 is True
    var_4 = var_0[var_1]
    var_5 = len(var_4)
    assert var_5 == 2

def test_case_0():
    var_0 = '__invariant__'
    var_1 = '__wrapped_invariants__'
    var_2 = '__invariant__'

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = '__wrapped_invariants__'
    var_3 = '__invariant__'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invariants must be callable'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = '__wrapped_invariants__'
    var_3 = '__invariant__'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = bool(var_2 in var_0)
    assert var_5 is True
    var_6 = var_0[var_2]
    var_7 = len(var_6)
    assert var_7 == 0

def test_case_0():
    var_0 = '__invariant__'
    var_1 = ()
    var_2 = '__wrapped_invariants__'
    var_3 = '__invariant__'
    var_4 = 0
    var_5 = None

def test_case_0():
    var_0 = {}
    var_1 = '__wrapped_invariants__'
    var_2 = '__invariant__'
    var_3 = bool(var_1 in var_0)
    assert var_3 is True
    var_4 = var_0[var_1]
    var_5 = len(var_4)
    assert var_5 == 1

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'not callable'
    var_2 = {var_0: var_1}
    var_3 = '__wrapped_invariants__'
    var_4 = '__invariant__'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Invariants must be callable'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/6 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_entry. Retrieved 3/8 statements.
# Failed to parse test_checked_pmap_constructor_default_argument.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.


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



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_number. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_pmap. Retrieved 2/8 statements.
# Partially parsed test_checked_pset_constructor_preserves_class_type. Retrieved 5/11 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pmap()

def test_case_0():
    var_0 = lambda n: (n > 0, 'NotPositive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_checked_pmap_new_with_default_arguments.
# Partially parsed test_checked_pmap_new_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_new_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_new_with_size_argument. Retrieved 8/13 statements.
# Partially parsed test_checked_pmap_new_with_empty_dict_and_size. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_new_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_new_multiple_items. Retrieved 9/14 statements.
# Partially parsed test_checked_pmap_new_returns_instance_of_correct_class. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'hello'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 100
    var_4 = 200
    var_5 = 300
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 16

def test_case_0():
    var_0 = {}
    var_1 = 32

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

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_store_invariants_single_invariant. Retrieved 3/9 statements.
# Partially parsed test_store_invariants_multiple_invariants. Retrieved 3/10 statements.
# Partially parsed test_store_invariants_inherited. Retrieved 2/13 statements.
# Partially parsed test_store_invariants_wrapped_invariants. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_custom_destination_name. Retrieved 3/7 statements.
# Partially parsed test_store_invariants_multiple_inheritance_levels. Retrieved 5/15 statements.
# Partially parsed test_store_invariants_tuple_result. Retrieved 3/9 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = '_invariants'
    var_3 = 'invariant'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = '_invariants'
    var_6 = bool('_invariants' not in var_0)
    assert var_6 is True

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = '_invariants'

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = '_invariants'

def test_case_0():
    var_0 = 'invariant'
    var_1 = '_invariants'
    var_2 = '_invariants'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = '_invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Invariants must be callable'

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = 0
    var_4 = 5

def test_case_0():
    var_0 = 'my_inv'
    var_1 = ()
    var_2 = 'custom_dest'
    var_3 = 'custom_dest'
    var_4 = 'my_inv'

def test_case_0():
    var_0 = {}
    var_1 = '_invariants'
    var_2 = 'invariant'
    var_3 = '_invariants'
    var_4 = bool('_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 2

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = '_invariants'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/11 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/11 statements.
# Partially parsed test_check_types_raises_exception_on_invalid_type. Retrieved 4/12 statements.
# Partially parsed test_check_types_with_class_type_objects. Retrieved 4/11 statements.
# Partially parsed test_check_types_with_mixed_type_specifications. Retrieved 4/11 statements.
# Partially parsed test_check_types_exception_message_format. Retrieved 2/14 statements.


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
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type TestClass can only be used with'
    var_6 = 'not str'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 3.14
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_numbers. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_number. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_tuple. Retrieved 5/10 statements.


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
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_with_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_items. Retrieved 9/14 statements.


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
    var_0 = '1'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = '1.5'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

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



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 7/12 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_store_invariants_predicate_line_1. Retrieved 11/16 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = 'wrapped_invariants'
    var_6 = 'invariant'
    var_7 = module_0.store_invariants(var_3, var_4, var_5, var_6)
    var_8 = bool(var_5 in var_3)
    assert var_8 is True
    var_9 = var_3[var_5]
    var_10 = var_3[var_5]
    var_11 = len(var_10)
    assert var_11 == 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_types_single_type_in_dct. Retrieved 3/5 statements.
# Partially parsed test_store_types_list_of_types_in_dct. Retrieved 3/6 statements.
# Partially parsed test_store_types_from_base_class. Retrieved 3/8 statements.
# Partially parsed test_store_types_dct_overrides_base. Retrieved 2/8 statements.
# Partially parsed test_store_types_multiple_bases. Retrieved 3/11 statements.
# Partially parsed test_store_types_nested_iterables. Retrieved 3/7 statements.
# Partially parsed test_store_types_mixed_types_and_strings. Retrieved 4/7 statements.
# Partially parsed test_store_types_source_name_not_present. Retrieved 4/6 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0['dest']
    var_6 = bool(var_0['dest'] == ())
    assert var_6 is True

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'MyType'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'dest'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['dest']
    var_7 = bool(var_2['dest'] == ('MyType',))
    assert var_7 is True

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0['dest']

def test_case_0():
    var_0 = 'src'
    var_1 = 'dest'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0['dest']

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'

def test_case_0():
    var_0 = 'src'
    var_1 = 'CustomType'
    var_2 = ()
    var_3 = 'dest'

def test_case_0():
    var_0 = 'other_key'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'my_dest'
    var_3 = 'my_src'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = 'my_dest'
    var_6 = bool('my_dest' in var_0)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.InvariantException(**var_0)
    var_2 = var_1.invariant_errors
    var_3 = bool(var_1.invariant_errors == ())
    assert var_3 is True
    var_4 = var_1.missing_fields
    var_5 = bool(var_1.missing_fields == ())
    assert var_5 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'error2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.InvariantException(var_2, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ('error1', 'error2'))
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ())
    assert var_8 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.InvariantException(missing_fields=var_2, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ())
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ('field1', 'field2'))
    assert var_8 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'error2'
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = (var_3, var_4)
    var_6 = {}
    var_7 = module_0.InvariantException(var_2, var_5, **var_6)
    var_8 = var_7.invariant_errors
    var_9 = bool(var_7.invariant_errors == ('error1', 'error2'))
    assert var_9 is True
    var_10 = var_7.missing_fields
    var_11 = bool(var_7.missing_fields == ('field1', 'field2'))
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'callable_error_result'
    var_1 = lambda : var_0
    var_2 = 'static_error'
    var_3 = (var_1, var_2)
    var_4 = {}
    var_5 = module_0.InvariantException(var_3, **var_4)
    var_6 = var_5.invariant_errors
    var_7 = bool(var_5.invariant_errors == ('callable_error_result', 'static_error'))
    assert var_7 is True
    var_8 = var_5.missing_fields
    var_9 = bool(var_5.missing_fields == ())
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_default_parameter. Retrieved 3/8 statements.


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
    var_3 = 10

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 'answer'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = 2
    var_3 = {i: i * var_2 for i in var_1}

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_without_checked_type_subclass. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_subclass_matching_data. Retrieved 3/12 statements.
# Partially parsed test_checked_type_create_with_checked_type_subclass_non_matching_data. Retrieved 4/11 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_parameter. Retrieved 4/11 statements.
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
    var_0 = '__main__.MockCheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'raw_data_1'
    var_3 = 'raw_data_2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'data'
    var_3 = [var_2]
    var_4 = True

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



# Parsed testcases at query #19
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
    var_0 = 'key'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Type specifications must be types or strings'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_wrap_invariant_with_bool_result. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_bool_result_false. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_multiple_results. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_multiple_results_one_failure. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_multiple_results_multiple_failures. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_kwargs. Retrieved 2/6 statements.
# Partially parsed test_wrap_invariant_with_multiple_args. Retrieved 2/6 statements.
# Failed to parse test_wrap_invariant_empty_failures.
# Failed to parse test_wrap_invariant_all_failures.


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

def test_case_0():
    var_0 = 5
    var_1 = 20

def test_case_0():
    var_0 = 1
    var_1 = -1



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_default_initial. Retrieved 1/7 statements.
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
    var_2 = 2.5
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

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector. Retrieved 5/12 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_negative_value. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_preserves_type. Retrieved 5/11 statements.
# Partially parsed test_checked_pvector_constructor_with_tuple. Retrieved 5/9 statements.


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
    var_0 = lambda n: (n > 0, 'Not positive')
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_check_types_predicate_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_invariant_errors_all_valid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_some_invalid. Retrieved 1/9 statements.
# Partially parsed test_invariant_errors_all_invalid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_single_valid_invariant. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_single_invalid_invariant. Retrieved 1/5 statements.
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Failed to parse test_checked_pmap_constructor_with_default_argument.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_entries. Retrieved 7/12 statements.


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
    var_0 = 'key'
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



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_maybe_parse_user_type_line_18_predicate.




# Parsed testcases at query #29
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Partially parsed test_maybe_parse_user_type_with_mixed_list. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Failed to parse test_maybe_parse_user_type_with_dict.
# Failed to parse test_maybe_parse_user_type_with_deeply_nested_iterables.


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

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Failed to parse test_checkedpmap_constructor_default_parameter.
# Partially parsed test_checkedpmap_constructor_with_multiple_valid_entries. Retrieved 11/18 statements.


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



# Parsed testcases at query #31
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
    var_0 = False
    var_1 = 123
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data'
    var_5 = (var_3, var_4)
    var_6 = None
    var_7 = (var_0, var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = (var_0, var_10)
    var_12 = [var_2, var_5, var_7, var_11]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_parameter.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/10 statements.


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
    var_0 = 5
    var_1 = 5.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #34
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_wrong_value_type. Retrieved 3/8 statements.
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
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_list. Retrieved 2/7 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_mixed_numbers. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_pvector_input. Retrieved 5/12 statements.
# Partially parsed test_checked_pvector_constructor_with_tuple_input. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_default_parameter. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_single_element. Retrieved 3/7 statements.


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

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 42
    var_2 = [var_1]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_types_predicate_true_with_expected_types. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test_elem'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 11/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = bool(var_4)
    assert var_5 is False
    var_6 = [var_0, var_1, var_2]
    var_7 = bool(var_4)
    assert var_7 is True
    var_8 = [var_0, var_1, var_2]
    var_9 = []
    var_10 = bool(var_9)
    assert var_10 is False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_number. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_preserves_class_type. Retrieved 5/11 statements.


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
    var_6 = 1
    var_7 = 2

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
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_wrap_invariant_predicate_line_3. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Partially parsed test_maybe_parse_user_type_with_mixed_list. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.


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



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/10 statements.


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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_string_type_name. Retrieved 6/17 statements.
# Partially parsed test_check_types_raises_with_wrong_type_in_list. Retrieved 4/17 statements.


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
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = "Type TestClass can only be used with ('int',), not str"

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_empty_factory_fields. Retrieved 4/11 statements.
# Partially parsed test_restore_pickle_passes_correct_arguments. Retrieved 5/12 statements.
# Partially parsed test_restore_pickle_with_empty_data. Retrieved 2/9 statements.


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



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_from_pmap_instance. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 'answer'
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
    var_5 = 'one'
    var_6 = 'two'
    var_7 = 'three'
    var_8 = 'four'
    var_9 = 'five'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_empty_factory_fields. Retrieved 4/11 statements.
# Partially parsed test_restore_pickle_with_different_data_types. Retrieved 7/14 statements.
# Partially parsed test_restore_pickle_factory_fields_is_empty_set. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = set()

def test_case_0():
    var_0 = 'test_string'
    var_1 = set()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checked_pmap_constructor_with_default_empty_initial.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
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
    var_4 = bool('value' in str(e).lower() or 'type' in str(e).lower())
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_restore_pickle_calls_create_with_factory_fields_empty_set. Retrieved 8/12 statements.
# Partially parsed test_restore_pickle_with_empty_data. Retrieved 6/10 statements.
# Partially parsed test_restore_pickle_with_complex_data. Retrieved 14/18 statements.
# Partially parsed test_restore_pickle_factory_fields_is_empty_set. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = '_factory_fields'
    var_5 = {var_0: var_1}
    var_6 = set()
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = {}
    var_1 = 'data'
    var_2 = '_factory_fields'
    var_3 = {}
    var_4 = set()
    var_5 = {var_1: var_3, var_2: var_4}

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
    var_10 = 'data'
    var_11 = '_factory_fields'
    var_12 = set()
    var_13 = {var_10: var_9, var_11: var_12}

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_element'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_checked_pmap_new_with_empty_initial. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_new_with_single_element. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_new_with_multiple_elements. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_new_with_explicit_size. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_new_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_new_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_new_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_new_with_violated_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_new_returns_correct_type. Retrieved 5/10 statements.
# Failed to parse test_checked_pmap_new_default_argument.


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
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16

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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_float_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checked_pset_constructor_with_negative_invariant_violation. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_tuple_initial. Retrieved 5/9 statements.


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
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 'invalid'
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True

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
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3



# Parsed testcases at query #53
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



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_element'
    var_1 = 'invalid2'
    var_2 = 'invalid3'



# Parsed testcases at query #55
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #56
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_pythonpvector. Retrieved 5/12 statements.
# Partially parsed test_checkedpvector_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 5/9 statements.


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
    var_1 = 1
    var_2 = 2
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



# Parsed testcases at query #58
#--------------------------

# Failed to parse test_check_types_predicate_with_non_empty_expected_types.




# Parsed testcases at query #59
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checked_pmap_constructor_with_default_initial.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 11/16 statements.
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
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
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
    var_0 = 10
    var_1 = 10.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_with_simple_data. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_non_checked_type. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_matching_data. Retrieved 3/10 statements.
# Partially parsed test_checked_type_create_ignore_extra_parameter. Retrieved 5/12 statements.
# Partially parsed test_checked_type_create_factory_fields_parameter. Retrieved 7/11 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = [var_0]
    var_2 = 'hello'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 6/7 statements.
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

def test_case_0():
    var_0 = False
    var_1 = 123
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data'
    var_5 = (var_3, var_4)
    var_6 = None
    var_7 = (var_0, var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = (var_0, var_10)
    var_12 = [var_2, var_5, var_7, var_11]



# Parsed testcases at query #62
#--------------------------

# Failed to parse test_checkedpvector_constructor_with_empty_initial.
# Partially parsed test_checkedpvector_constructor_with_list. Retrieved 4/9 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_checkedpvector_constructor_with_pythonpvector. Retrieved 4/11 statements.
# Partially parsed test_checkedpvector_constructor_with_multiple_types. Retrieved 4/8 statements.
# Partially parsed test_checkedpvector_constructor_with_invariant. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_preserves_type. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 7
    var_1 = 8
    var_2 = 9
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #63
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_from_dict_literal. Retrieved 5/9 statements.


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
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #64
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 7/12 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/16 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 6/18 statements.
# Partially parsed test_check_types_with_multiple_expected_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_raises_on_first_invalid_element. Retrieved 7/19 statements.


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
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = True
    var_6 = 'Type TestClass can only be used with'
    var_7 = bool(var_5)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = []
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = True
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #66
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #67
#--------------------------

# Partially parsed test_wrap_invariant_predicate_at_line_3_evaluates_to_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_with_non_matching_data. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_false. Retrieved 5/12 statements.
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
    var_0 = '__main__.InnerCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = False

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
    var_0 = 'builtins.ValueError'
    var_1 = module_0._get_class(var_0)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_wrap_invariant_with_bool_result. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_maybe_parse_user_type_line_18_predicate. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 18 (is_type and not is_iterable) evaluates to True.'



# Parsed testcases at query #71
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/10 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

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



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 9/14 statements.


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
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #73
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_default_argument. Retrieved 1/7 statements.


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
    var_0 = {}



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_element'
    var_1 = 'invalid_data_2'
    var_2 = 'invalid_data_3'



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #76
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #77
#--------------------------

# Failed to parse test_checked_type_create_predicate_line_1_false.




# Parsed testcases at query #78
#--------------------------

# Failed to parse test_wrap_invariant_predicate_at_line_3.




# Parsed testcases at query #79
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_raises_exception_for_invalid_type. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_class_type_string. Retrieved 5/16 statements.
# Partially parsed test_check_types_raises_with_class_type_string. Retrieved 4/16 statements.


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
    var_1 = 'string'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass'
    var_6 = 'int'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'builtins.int'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'string'
    var_1 = [var_0]
    var_2 = 'builtins.int'
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checked_pmap_constructor_default_empty.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 7/13 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #81
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_default_argument. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_returns_correct_type. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_entries. Retrieved 7/12 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #82
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/10 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.


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



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #84
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_single_element. Retrieved 3/8 statements.
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

def test_case_0():
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0: var_0, var_1: var_1, var_2: var_2}
    var_4 = 8



# Parsed testcases at query #85
#--------------------------

# Failed to parse test_isinstance_predicate_evaluates_to_false.




# Parsed testcases at query #86
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_returns_pmap_instance. Retrieved 3/8 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

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
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #88
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_source_data_matching_cls. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_no_checked_types. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_returns_cls_instance. Retrieved 5/15 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_parameter. Retrieved 6/17 statements.


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
    var_0 = '__main__.MockCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '__main__.MockCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 5/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #91
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Partially parsed test_maybe_parse_user_type_with_mixed_types_and_strings. Retrieved 1/3 statements.


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

def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #92
#--------------------------

# Failed to parse test_line_18_predicate_evaluates_to_true.




# Parsed testcases at query #93
#--------------------------

# Partially parsed test_checked_type_create_returns_source_when_already_instance. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_returns_cls_instance_when_no_checked_type. Retrieved 5/10 statements.
# Partially parsed test_checked_type_create_with_checked_type_matching_data. Retrieved 5/17 statements.
# Partially parsed test_checked_type_create_with_checked_type_non_matching_data. Retrieved 5/17 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_flag. Retrieved 6/18 statements.
# Partially parsed test_checked_type_create_with_factory_fields. Retrieved 8/13 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.InnerCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '__main__.InnerCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '__main__.InnerCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}



# Parsed testcases at query #94
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 5/12 statements.


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
    var_4 = 1.5
    var_5 = 2.25
    var_6 = 3.75
    var_7 = 4.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = set()



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #97
#--------------------------

# Failed to parse test_isinstance_source_data_is_cls.




# Parsed testcases at query #98
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_failed_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 7/12 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



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

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #101
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_default_dict. Retrieved 1/6 statements.
# Partially parsed test_checked_pmap_constructor_type_error. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_value_type_error. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.


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
    var_0 = {}

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
    var_2 = 3
    var_3 = 4
    var_4 = 1.5
    var_5 = 2.25
    var_6 = 3.75
    var_7 = 4.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #103
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_with_default_empty_dict. Retrieved 1/7 statements.


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
    var_0 = {}



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checkedpmap_constructor_default_argument.
# Partially parsed test_checkedpmap_constructor_with_single_entry. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 7/13 statements.


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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_mixed. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_with_none_data. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_with_complex_data. Retrieved 15/16 statements.


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
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = 'error3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'data2'
    var_7 = (var_0, var_6)
    var_8 = 'error2'
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_5, var_7, var_9]

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
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = (var_3, var_1)
    var_5 = 'data'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = (var_0, var_3)
    var_5 = False
    var_6 = 'error'
    var_7 = 'details'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = 2
    var_11 = 3
    var_12 = [var_0, var_10, var_11]
    var_13 = (var_5, var_12)
    var_14 = [var_4, var_9, var_13]



# Parsed testcases at query #106
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 5/12 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 9/14 statements.


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
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

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



# Parsed testcases at query #107
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_single_element. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_elements. Retrieved 11/17 statements.


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



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/11 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/11 statements.
# Partially parsed test_check_types_raises_exception_on_invalid_type. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_single_invalid_element. Retrieved 2/13 statements.
# Partially parsed test_check_types_with_string_type_names. Retrieved 6/15 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/8 statements.


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
    var_5 = "Type TestClass can only be used with ('int',), not str"

def test_case_0():
    var_0 = 3.14
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]

def test_case_0():
    var_0 = []



# Parsed testcases at query #109
#--------------------------

# Failed to parse test_checked_type_create_predicate_false.




# Parsed testcases at query #110
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test'
    var_9 = module_0._invariant_errors(var_8, var_7)
    var_10 = bool(var_9 == ['error1', 'error2'])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'valid'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'error1'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = 'valid2'
    var_9 = (var_0, var_8)
    var_10 = lambda x: var_9
    var_11 = 'error2'
    var_12 = (var_4, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_3, var_7, var_10, var_13]
    var_15 = 'test'
    var_16 = module_0._invariant_errors(var_15, var_14)
    var_17 = bool(var_16 == ['error1', 'error2'])
    assert var_17 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = module_0._invariant_errors(var_1, var_0)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'valid'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = [var_3]
    var_5 = 'test'
    var_6 = module_0._invariant_errors(var_5, var_4)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'error'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = [var_3]
    var_5 = 'test'
    var_6 = module_0._invariant_errors(var_5, var_4)
    var_7 = bool(var_6 == ['error'])
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 42
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = (var_0, var_6)
    var_8 = lambda x: var_7
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = (var_0, var_12)
    var_14 = lambda x: var_13
    var_15 = [var_3, var_8, var_14]
    var_16 = 'test'
    var_17 = module_0._invariant_errors(var_16, var_15)
    var_18 = bool(var_17 == [42, {'key': 'value'}, [1, 2, 3]])
    assert var_18 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_float_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_value. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicate_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 7/11 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 8/11 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 4/8 statements.
# Partially parsed test_check_types_raises_on_invalid_type. Retrieved 6/11 statements.
# Partially parsed test_check_types_with_string_type_names. Retrieved 8/11 statements.
# Partially parsed test_check_types_raises_with_string_type_names. Retrieved 7/11 statements.
# Partially parsed test_check_types_custom_exception_type. Retrieved 6/13 statements.


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
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = None
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
    var_5 = 'invalid'
    var_6 = [var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'builtins.int'
    var_4 = [var_3]
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'invalid'
    var_5 = [var_4]
    var_6 = 'builtins.int'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 'invalid'
    var_6 = [var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_initial_data. Retrieved 5/10 statements.
# Failed to parse test_checkedpmap_constructor_with_default_empty_dict.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_single_element. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_returns_instance_of_correct_type. Retrieved 5/12 statements.


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
    var_1 = 42.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_store_types_single_type_in_dict. Retrieved 4/6 statements.
# Partially parsed test_store_types_multiple_types_in_dict. Retrieved 4/7 statements.
# Partially parsed test_store_types_with_base_class. Retrieved 3/9 statements.
# Partially parsed test_store_types_overwrites_existing_destination. Retrieved 6/8 statements.
# Partially parsed test_store_types_nested_list_of_types. Retrieved 4/8 statements.
# Partially parsed test_store_types_multiple_bases. Retrieved 3/12 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]
    var_6 = bool(var_0[var_2] == ())
    assert var_6 is True

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'MyType'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'dest'
    var_5 = 'src'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)
    var_7 = var_2[var_4]
    var_8 = bool(var_2[var_4] == ('MyType',))
    assert var_8 is True

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'

def test_case_0():
    var_0 = 'src'
    var_1 = 'dest'
    var_2 = 'src'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'nonexistent'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]
    var_6 = bool(var_0[var_2] == ())
    assert var_6 is True

def test_case_0():
    var_0 = 'dest'
    var_1 = 'src'
    var_2 = 'old_value'
    var_3 = ()
    var_4 = 'dest'
    var_5 = 'src'

def test_case_0():
    var_0 = 'src'
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'

def test_case_0():
    var_0 = 'src'
    var_1 = 'dest'
    var_2 = 'src'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_integers. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_invariant_violation. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_zero. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_preserves_type. Retrieved 5/10 statements.


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
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_types_predicate_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_with_multiple_items. Retrieved 11/18 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_store_invariants_single_invariant. Retrieved 4/12 statements.
# Partially parsed test_store_invariants_multiple_invariants. Retrieved 2/13 statements.
# Partially parsed test_store_invariants_inherited. Retrieved 2/13 statements.
# Partially parsed test_store_invariants_wrapped_invariant_bool. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_wrapped_invariant_multiple_results. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_multiple_bases. Retrieved 2/17 statements.
# Partially parsed test_store_invariants_destination_name. Retrieved 3/9 statements.


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
    var_0 = 'invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = '_invariants'
    var_4 = 0

def test_case_0():
    var_0 = 'invariant'
    var_1 = '_invariants'
    var_2 = '_invariants'

def test_case_0():
    var_0 = 'invariant'
    var_1 = '_invariants'
    var_2 = '_invariants'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = '_invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Invariants must be callable'

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = 0
    var_4 = 5

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = '_invariants'
    var_3 = 0
    var_4 = 5

def test_case_0():
    var_0 = 'invariant'
    var_1 = '_invariants'

def test_case_0():
    var_0 = 'check'
    var_1 = ()
    var_2 = 'custom_dest'
    var_3 = 'custom_dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = '_invariants'
    var_3 = 'invariant'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = var_0['_invariants']
    var_6 = bool(var_0['_invariants'] == ())
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_non_checked_type. Retrieved 5/10 statements.
# Partially parsed test_checked_type_create_with_string_type_name. Retrieved 5/12 statements.
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

def test_case_0():
    var_0 = 'collections.UserList'
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
    var_0 = []
    var_1 = module_0.get_types(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serialize_empty_set. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_integers. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_floats. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_mixed_numbers. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_format_none. Retrieved 6/10 statements.
# Partially parsed test_serialize_returns_set_type. Retrieved 5/10 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []
    var_2 = set()

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
    var_5 = None

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_string_type_names. Retrieved 5/15 statements.
# Partially parsed test_check_types_with_empty_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/15 statements.
# Partially parsed test_check_types_raises_on_invalid_type. Retrieved 4/16 statements.
# Partially parsed test_check_types_default_exception. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]
    var_3 = 'builtins.str'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass'
    var_6 = 'int'
    var_7 = 'str'

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_list. Retrieved 2/7 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_pvector_input. Retrieved 5/12 statements.
# Partially parsed test_checked_pvector_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_single_element. Retrieved 3/7 statements.
# Partially parsed test_checked_pvector_constructor_default_empty. Retrieved 1/6 statements.


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
    var_1 = 1
    var_2 = 2
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
    var_1 = 42
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Partially parsed test_maybe_parse_user_type_with_mixed_list. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_with_nested_list.
# Failed to parse test_maybe_parse_user_type_with_tuple.


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

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type specifications must be types or strings'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_default_argument.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_violates_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 11/16 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_store_invariants_callable_check. Retrieved 20/26 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = 'stored_invariants'
    var_6 = module_0.store_invariants(var_3, var_4, var_5, var_0)
    var_7 = 'stored_invariants'
    var_8 = bool('stored_invariants' in var_3)
    assert var_8 is True
    var_9 = var_3[var_5]
    var_10 = 'not_callable'
    var_11 = {var_0: var_10}
    var_12 = []
    var_13 = 'stored_invariants'
    var_14 = 'invariant'
    var_15 = module_0.store_invariants(var_11, var_12, var_13, var_14)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = lambda x: x > var_14
    var_18 = {var_13: var_17}
    var_19 = []
    var_20 = module_0.store_invariants(var_18, var_19, var_5, var_13)
    var_21 = var_18[var_5]
    var_22 = len(var_21)
    assert var_22 == 1



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_checked_type_create_predicate_line_2.




# Parsed testcases at query #19
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_iterable.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_invariant_errors_all_valid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_all_invalid. Retrieved 1/7 statements.
# Partially parsed test_invariant_errors_mixed. Retrieved 1/11 statements.
# Partially parsed test_invariant_errors_single_invariant_valid. Retrieved 1/5 statements.
# Partially parsed test_invariant_errors_single_invariant_invalid. Retrieved 1/5 statements.
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



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_returns_correct_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.


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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_none_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/16 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_string_type_reference. Retrieved 6/17 statements.
# Partially parsed test_check_types_with_mixed_type_references. Retrieved 6/18 statements.


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
    var_4 = None

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
    var_5 = 'TestClass'
    var_6 = 'int'
    var_7 = 'str'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = 'builtins.float'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_integers. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_types. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_duplicates. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_negative_invariant_violation. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 2/7 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pmap()



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_maybe_parse_user_type_line_18_predicate.




# Parsed testcases at query #25
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_single_element. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_elements. Retrieved 7/12 statements.


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
    var_0 = 'a'
    var_1 = 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty_default.
# Partially parsed test_checkedpmap_constructor_with_valid_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_class_type. Retrieved 3/10 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 7/12 statements.


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



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_default_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/10 statements.


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
    var_0 = {}

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
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_tuple_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterables.
# Partially parsed test_maybe_parse_user_type_with_invalid_iterable. Retrieved 1/4 statements.
# Failed to parse test_maybe_parse_user_type_with_custom_class.


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
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

def test_case_0():
    var_0 = 123
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
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_checkedpset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_float_elements. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_duplicate_elements. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_negative_number_raises_error. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_wrong_type_raises_error. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_default_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_checkedpset_constructor_preserves_class_type. Retrieved 5/9 statements.


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

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')

def test_case_0():
    var_0 = lambda n: (n > 0, 'NotPositive')
    var_1 = 5
    var_2 = 10
    var_3 = 15
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_check_types_predicate_true_with_non_empty_expected_types.




# Parsed testcases at query #31
#--------------------------

# Failed to parse test_checked_pmap_new_with_default_arguments.
# Partially parsed test_checked_pmap_new_with_initial_dict. Retrieved 5/13 statements.
# Partially parsed test_checked_pmap_new_with_empty_initial_dict. Retrieved 1/9 statements.
# Partially parsed test_checked_pmap_new_with_initial_and_size. Retrieved 7/14 statements.
# Partially parsed test_checked_pmap_new_with_size_only. Retrieved 5/12 statements.
# Partially parsed test_checked_pmap_new_with_invariant. Retrieved 6/13 statements.
# Partially parsed test_checked_pmap_new_with_invalid_invariant. Retrieved 4/11 statements.
# Partially parsed test_checked_pmap_new_multiple_items. Retrieved 11/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = {}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'p'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 1

def test_case_0():
    var_0 = lambda k, v: (v > 0, 'Value must be positive')
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (v > 0, 'Value must be positive')
    var_1 = 'a'
    var_2 = -1
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

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



# Parsed testcases at query #32
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
    var_0 = 'key'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Type specifications must be types or strings'



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_default_parameter. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 11/17 statements.
# Partially parsed test_checked_pmap_constructor_returns_same_instance_if_already_checked. Retrieved 3/9 statements.


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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = set()



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_key_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_invalid_value_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_initial_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 8/13 statements.


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
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 2.5
    var_5 = 3.75
    var_6 = 4.25
    var_7 = {var_0: var_0, var_1: var_4, var_2: var_5, var_3: var_6}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'not an instance of MockClass'



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_check_types_predicate_true.




# Parsed testcases at query #38
#--------------------------

# Failed to parse test_maybe_parse_user_type_line_18_predicate.




# Parsed testcases at query #39
#--------------------------

# Failed to parse test_wrap_invariant_with_bool_result.
# Failed to parse test_wrap_invariant_with_tuple_results.
# Failed to parse test_wrap_invariant_with_mixed_results.
# Failed to parse test_wrap_invariant_with_all_false_results.
# Failed to parse test_wrap_invariant_with_all_true_results.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 3/7 statements.
# Failed to parse test_wrap_invariant_with_single_result_list.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_checkedtype_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_check_types_predicate_true_with_non_empty_expected_types.




# Parsed testcases at query #42
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_default_parameter.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_key_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_value_type. Retrieved 3/9 statements.
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
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checkedpmap_constructor_with_default_empty.
# Partially parsed test_checkedpmap_constructor_multiple_entries. Retrieved 7/12 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_checkedtype_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'not a MockCls instance'



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty_default.
# Partially parsed test_checked_pmap_constructor_with_valid_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_single_valid_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_valid_items. Retrieved 9/14 statements.


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



# Parsed testcases at query #47
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_source_data_list. Retrieved 5/12 statements.
# Partially parsed test_checked_type_create_with_checked_type_in_types. Retrieved 4/12 statements.
# Partially parsed test_checked_type_create_ignore_extra_parameter. Retrieved 5/12 statements.
# Partially parsed test_checked_type_create_factory_fields_parameter. Retrieved 8/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = [var_2]
    var_4 = False

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'field'
    var_6 = 'value'
    var_7 = {var_5: var_6}



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_default_param. Retrieved 4/9 statements.


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
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 10



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_checked_pvector_constructor_empty. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_checked_pvector_constructor_with_pvector. Retrieved 5/11 statements.
# Partially parsed test_checked_pvector_constructor_preserves_type. Retrieved 4/10 statements.
# Partially parsed test_checked_pvector_constructor_with_generator. Retrieved 5/10 statements.


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
    var_3 = (var_1, var_2)

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

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #51
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #52
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/6 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Failed to parse test_checkedpmap_constructor_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items_with_invariant. Retrieved 5/10 statements.


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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_element'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_with_non_matching_data. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_subclass. Retrieved 5/16 statements.
# Partially parsed test_checked_type_create_with_matching_type_in_data. Retrieved 3/16 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_flag. Retrieved 5/18 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.ConcreteCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '__main__.ConcreteCheckedType'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 2

def test_case_0():
    var_0 = '__main__.ConcreteCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = True



# Parsed testcases at query #55
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
    var_0 = False
    var_1 = 123
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data'
    var_5 = (var_3, var_4)
    var_6 = None
    var_7 = (var_0, var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = (var_3, var_10)
    var_12 = [var_2, var_5, var_7, var_11]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_checked_type_create_predicate_line_1_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'not a MockCheckedType instance'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_serialize_with_integers. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_floats. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_mixed_numbers. Retrieved 5/9 statements.
# Partially parsed test_serialize_empty_set. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 6/10 statements.
# Partially parsed test_serialize_single_element. Retrieved 3/7 statements.


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
    var_1 = []
    var_2 = set()

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'json'

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 42
    var_2 = [var_1]



# Parsed testcases at query #58
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_results_all_true.
# Failed to parse test_wrap_invariant_with_multiple_results_one_false.
# Failed to parse test_wrap_invariant_with_multiple_results_all_false.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 3/7 statements.
# Partially parsed test_wrap_invariant_with_multiple_results_and_args. Retrieved 1/5 statements.
# Failed to parse test_wrap_invariant_preserves_false_bool_result.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 5



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_with_single_entry. Retrieved 3/8 statements.
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
    var_0 = 42
    var_1 = 3.14
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_empty_list. Retrieved 2/7 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_mixed_numbers. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_pythonpvector. Retrieved 5/12 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_preserves_class_type. Retrieved 5/11 statements.


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
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #61
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



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_checked_pmap_initial_items_iteration. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_checkedtype_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_element'
    var_1 = 'invalid_data_2'
    var_2 = 'invalid_data_3'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_serialize_returns_set. Retrieved 5/10 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #66
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



# Parsed testcases at query #67
#--------------------------

# Failed to parse test_checked_type_create_predicate_line_1_false.




# Parsed testcases at query #68
#--------------------------

# Partially parsed test_wrap_invariant_predicate_line_3. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #69
#--------------------------

# Failed to parse test_checked_type_create_predicate_line_1_false.




# Parsed testcases at query #70
#--------------------------

# Partially parsed test_serialize_empty_checked_pset. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_integers. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_floats. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_mixed_numbers. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 6/10 statements.
# Partially parsed test_serialize_with_single_element. Retrieved 3/7 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []
    var_2 = set()

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
    var_5 = 'json'

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 42
    var_2 = [var_1]



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 7/13 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #72
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_returns_correct_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_dict_input. Retrieved 7/12 statements.


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
    var_1 = 'invalid'
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
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_with_simple_data. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_no_matching_instance. Retrieved 5/12 statements.
# Partially parsed test_checked_type_create_with_matching_instance_type. Retrieved 3/12 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_flag. Retrieved 4/11 statements.
# Partially parsed test_checked_type_create_empty_checked_types. Retrieved 2/6 statements.


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
    var_4 = [var_2, var_3]
    var_5 = False

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = 2
    var_4 = False

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]
    var_4 = True

def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_checkedpmap_initial_items_iteration. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_checkedtype_constructor.




# Parsed testcases at query #76
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_empty_list. Retrieved 2/7 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_integers. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_with_valid_floats. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_python_pvector. Retrieved 5/12 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_with_generator. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_default_empty. Retrieved 1/6 statements.
# Partially parsed test_checkedpvector_constructor_invalid_type_raises_error. Retrieved 5/9 statements.
# Partially parsed test_checkedpvector_constructor_negative_value_raises_error. Retrieved 5/9 statements.


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



# Parsed testcases at query #77
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 11/16 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 'string_value'
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
    var_4 = 5
    var_5 = 1.1
    var_6 = 2.2
    var_7 = 3.3
    var_8 = 4.4
    var_9 = 5.5
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_default_size. Retrieved 6/11 statements.


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
    var_5 = 10



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_wrap_invariant_predicate_line_3. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 0



# Parsed testcases at query #80
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/11 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #81
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_single_element. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_multiple_elements. Retrieved 7/12 statements.


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



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Failed to parse test_checkedpmap_constructor_default_initial.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_failed_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_valid_entries. Retrieved 9/14 statements.


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
    var_3 = 4
    var_4 = 1.1
    var_5 = 2.2
    var_6 = 3.3
    var_7 = 4.4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}



# Parsed testcases at query #83
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/10 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

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



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_check_types_predicate_evaluates_to_false. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'string_value'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'can only be used with'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 5/13 statements.
# Partially parsed test_restore_pickle_with_empty_dict. Retrieved 3/10 statements.
# Partially parsed test_restore_pickle_with_complex_data. Retrieved 12/19 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = set()

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
    var_11 = set()



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_invariant_errors_returns_invalid_data. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_element'
    var_1 = 'invalid_data_2'
    var_2 = 'invalid_data_3'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 8/23 statements.


def test_case_0():
    var_0 = True
    assert var_0 is True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = 1.5
    var_7 = [var_6]



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 9/33 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = 2.5
    var_8 = [var_0, var_4, var_7]



# Parsed testcases at query #89
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checked_pmap_constructor_with_default_argument. Retrieved 5/10 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #90
#--------------------------

# Failed to parse test_wrap_invariant_with_bool_result.
# Failed to parse test_wrap_invariant_with_tuple_result_all_pass.
# Failed to parse test_wrap_invariant_with_tuple_result_one_fail.
# Failed to parse test_wrap_invariant_with_tuple_result_multiple_fails.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 3/7 statements.
# Failed to parse test_wrap_invariant_with_empty_tuple_result.
# Failed to parse test_wrap_invariant_with_all_false_results.


def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = 'test'



# Parsed testcases at query #91
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



# Parsed testcases at query #92
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 9/14 statements.


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



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_true. Retrieved 8/9 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = 'data3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_invariant. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/9 statements.
# Failed to parse test_checkedpmap_constructor_default_argument.
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



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'test'
    var_5 = None
    var_6 = [var_0, var_4, var_5]
    var_7 = []



# Parsed testcases at query #96
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_violates_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test_checked_type_create_without_checked_type. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_matching_data. Retrieved 1/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_non_matching_data. Retrieved 6/17 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_flag. Retrieved 7/18 statements.


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
    var_2 = []

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = 0

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = True
    var_7 = 0



# Parsed testcases at query #98
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_from_dict. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_multiple_items. Retrieved 9/14 statements.


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
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

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



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/18 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/19 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/9 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_types_with_none_value. Retrieved 4/18 statements.


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
    var_5 = 'str'

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
    var_1 = None
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #100
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_valid. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_invalid. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
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
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = '1'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = '1.5'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

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



# Parsed testcases at query #101
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_source_data_list. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_calls_cls_constructor. Retrieved 4/8 statements.
# Partially parsed test_checked_type_create_with_empty_checked_types. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_returns_source_when_already_instance. Retrieved 1/4 statements.


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



# Parsed testcases at query #102
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_without_checked_types. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_and_matching_data. Retrieved 5/9 statements.
# Failed to parse test_get_type_with_type_object.
# Partially parsed test_get_types_multiple_types. Retrieved 1/3 statements.


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
    var_0 = 'os.path.PathLike'
    var_1 = module_0._get_class(var_0)
    var_2 = var_1.__name__
    assert var_2 == 'PathLike'

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



# Parsed testcases at query #103
#--------------------------

# Failed to parse test_isinstance_source_data_cls_returns_source_data.




# Parsed testcases at query #104
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_preserves_type. Retrieved 3/10 statements.
# Partially parsed test_checked_pmap_constructor_with_large_dict. Retrieved 3/8 statements.


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
    var_1 = 'not a float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = {i: float(i) for i in var_1}



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_restore_pickle. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'number'
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()
    var_6 = set()



# Parsed testcases at query #106
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_multiple_items. Retrieved 11/17 statements.


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
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

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



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_restore_pickle_calls_create_with_empty_factory_fields. Retrieved 7/11 statements.
# Partially parsed test_restore_pickle_with_empty_data. Retrieved 5/9 statements.
# Partially parsed test_restore_pickle_with_complex_data. Retrieved 15/19 statements.
# Partially parsed test_restore_pickle_factory_fields_is_set. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'data'
    var_4 = '_factory_fields'
    var_5 = set()
    var_6 = {var_3: var_2, var_4: var_5}

def test_case_0():
    var_0 = {}
    var_1 = 'data'
    var_2 = '_factory_fields'
    var_3 = set()
    var_4 = {var_1: var_0, var_2: var_3}

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
    var_10 = 'data'
    var_11 = '_factory_fields'
    var_12 = set()
    var_13 = {var_10: var_9, var_11: var_12}
    var_14 = set()

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_check_types_predicate_line_1. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #109
#--------------------------

# Failed to parse test_checked_type_create_isinstance_predicate.




# Parsed testcases at query #110
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
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
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #111
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_from_dict. Retrieved 7/12 statements.
# Partially parsed test_checked_pmap_constructor_multiple_entries. Retrieved 9/14 statements.


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
    var_3 = 1.5
    var_4 = 2.25
    var_5 = 3.75
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

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



# Parsed testcases at query #112
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_multiple_items. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/11 statements.


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
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #113
#--------------------------

# Partially parsed test_check_types_predicate_evaluates_to_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'string_value'
    var_1 = [var_0]
    var_2 = False
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/8 statements.
# Failed to parse test_checked_pmap_constructor_with_default_initial.
# Partially parsed test_checked_pmap_constructor_preserves_class_type. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_single_entry. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_entries. Retrieved 11/17 statements.


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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 42
    var_1 = 'answer'
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



# Parsed testcases at query #115
#--------------------------

# Failed to parse test_checkedpmap_constructor_empty.
# Partially parsed test_checkedpmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_single_item. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_dict_conversion. Retrieved 7/12 statements.
# Partially parsed test_checkedpmap_constructor_preserves_type. Retrieved 5/11 statements.


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
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 10.5
    var_4 = 20.5
    var_5 = 30.5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #116
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_default_initial.
# Partially parsed test_checked_pmap_constructor_single_entry. Retrieved 3/8 statements.
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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #117
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 11/21 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 10/15 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 4/11 statements.
# Partially parsed test_check_types_with_invalid_type_raises_error. Retrieved 4/9 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 3/10 statements.
# Partially parsed test_check_types_with_multiple_expected_types_first_invalid. Retrieved 3/8 statements.
# Partially parsed test_check_types_with_multiple_expected_types_valid. Retrieved 5/9 statements.
# Partially parsed test_check_types_error_message_format. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_6, var_7]
    var_9 = []
    var_10 = [var_9]
    var_11 = [var_0, var_6]
    var_12 = []
    var_13 = [var_12]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = [var_5]
    var_7 = 'anything'
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = [var_10]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 1.5
    var_1 = [var_0]
    var_2 = []
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1.5
    var_1 = [var_0]
    var_2 = []
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = [var_4]

def test_case_0():
    var_0 = 1.5
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'list'
    var_4 = 'int'
    var_5 = 'float'



# Parsed testcases at query #118
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_dict. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.
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
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #119
#--------------------------

# Failed to parse test_checked_type_create_with_instance_of_cls.
# Partially parsed test_checked_type_create_with_source_data_already_correct_type. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_with_no_checked_types. Retrieved 5/9 statements.
# Partially parsed test_checked_type_create_returns_source_when_already_instance. Retrieved 1/4 statements.
# Failed to parse test_get_type_with_type_object.
# Partially parsed test_get_types_with_multiple_types. Retrieved 1/3 statements.
# Failed to parse test_get_types_with_all_type_objects.


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
    var_2 = 'builtins.float'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_types(var_3)



