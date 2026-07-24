####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_initial_as_same_type_instance. Retrieved 3/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_checked_key_type_and_create. Retrieved 3/12 statements.
# Partially parsed test_constructor_with_checked_value_type_and_create. Retrieved 1/12 statements.
# Partially parsed test_constructor_with_initial_as_dict. Retrieved 5/11 statements.
# Failed to parse test_constructor_with_no_arguments.
# Partially parsed test_constructor_with_positional_initial. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_keyword_initial. Retrieved 3/8 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}

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
    var_0 = 'raw_key'
    var_1 = 'raw_value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'raw_value'

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
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.
# Partially parsed test_maybe_parse_user_type_with_mixed_iterable. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = 'int'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'bool'
    var_1 = 'float'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)

def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_store_invariants_adds_wrapped_invariants. Retrieved 6/16 statements.
# Partially parsed test_store_invariants_merges_multiple_inherited_invariants. Retrieved 10/24 statements.
# Partially parsed test_store_invariants_handles_diamond_inheritance_without_duplicates. Retrieved 8/22 statements.
# Partially parsed test_store_invariants_raises_type_error_for_non_callable. Retrieved 3/9 statements.
# Partially parsed test_store_invariants_includes_invariant_from_current_dict. Retrieved 5/21 statements.
# Partially parsed test_store_invariants_wraps_invariant_returning_tuple_of_tuples. Retrieved 6/14 statements.
# Partially parsed test_store_invariants_with_no_invariants_found. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_wrap_invariant_preserves_simple_bool_result. Retrieved 6/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = '_invariants_'
    var_2 = '_invariant_'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = None

def test_case_0():
    var_0 = {}
    var_1 = '_invariants_'
    var_2 = '_invariant_'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = None
    var_8 = 1
    var_9 = var_3[var_8]

def test_case_0():
    var_0 = {}
    var_1 = '_invariants_'
    var_2 = '_invariant_'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = None

def test_case_0():
    var_0 = {}
    var_1 = '_invariants_'
    var_2 = '_invariant_'

def test_case_0():
    var_0 = '_invariant_'
    var_1 = '_invariants_'
    var_2 = 0
    var_3 = None
    var_4 = 1

def test_case_0():
    var_0 = {}
    var_1 = '_invariants_'
    var_2 = '_invariant_'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = None

def test_case_0():
    var_0 = {}
    var_1 = '_invariants_'
    var_2 = '_invariant_'
    var_3 = var_0[var_1]

def test_case_0():
    var_0 = {}
    var_1 = '_invariants_'
    var_2 = '_invariant_'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_with_valid_initial_dict. Retrieved 5/10 statements.
# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_invalid_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_type_key. Retrieved 3/11 statements.
# Partially parsed test_constructor_with_checked_type_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_with_checked_type_and_invariant. Retrieved 4/12 statements.
# Partially parsed test_constructor_with_checked_type_and_invariant_violation. Retrieved 4/13 statements.
# Partially parsed test_constructor_with_same_class_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_undefined_size_constant. Retrieved 3/7 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_inheritance. Retrieved 3/10 statements.
# Partially parsed test_constructor_with_no_type_specification. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_union_key_type. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_union_value_type. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_union_type_violation. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2.0
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 2.0
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (v > k, 'Value not greater than key')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (v > k, 'Value not greater than key')
    var_1 = 2
    var_2 = 1.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'b'
    var_2 = 'a'
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 'text'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1.5
    var_1 = {var_0: var_0}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_store_invariants_all_callable. Retrieved 13/15 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'invariants'
    var_3 = '_invariant'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_7 = var_0[var_2]
    var_8 = var_0[var_2]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_0[var_2][var_10]
    var_12 = callable(var_11)



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_checkedpmap_initial. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_key_violation. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value_violation. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_success. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_checked_key_type_create. Retrieved 3/10 statements.
# Partially parsed test_constructor_with_checked_value_type_create. Retrieved 3/10 statements.
# Partially parsed test_constructor_with_both_checked_types_create. Retrieved 3/13 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = '1'
    var_1 = 5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = '1'
    var_1 = 5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_store_invariants_adds_tuple_of_wrapped_invariants. Retrieved 9/25 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'invariant'
    var_3 = 'invariants'
    var_4 = lambda x: x
    var_5 = '_all_dicts'
    var_6 = None
    var_7 = []
    var_8 = module_0.store_invariants(var_0, var_1, var_3, var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_store_invariants_all_callable. Retrieved 11/32 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'invariant'
    var_3 = '_all_dicts'
    var_4 = None
    var_5 = 'dest'
    var_6 = module_0.store_invariants(var_0, var_1, var_5, var_2)
    var_7 = var_0[var_5]
    var_8 = var_0[var_5]
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_3._all_dicts



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_predicate_at_line_18_evaluates_to_true.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test__store_types_single_dict. Retrieved 3/5 statements.
# Partially parsed test__store_types_multiple_dicts. Retrieved 7/13 statements.
# Partially parsed test__store_types_with_iterable_source. Retrieved 3/6 statements.
# Partially parsed test__store_types_with_preserved_iterable_type. Retrieved 3/5 statements.
# Partially parsed test__store_types_combine_dict_and_bases. Retrieved 4/9 statements.
# Partially parsed test__store_types_nested_iterables. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'Base2'
    var_4 = ()
    var_5 = {}
    var_6 = 'dest'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 'MyClass'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'dest'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'source'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_CheckedType_constructor_initialization. Retrieved 3/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = '__slots__'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_predicate_at_line_18_evaluates_to_true.




# Parsed testcases at query #13
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'ok'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error2'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)

import builtins as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1._invariant_errors(var_0, var_1)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = 'data1'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = True
    var_9 = 'data2'
    var_10 = (var_8, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_7, var_11]
    var_13 = module_0._invariant_errors(var_3, var_12)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'err2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 'err3'
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0._invariant_errors(var_0, var_11)



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_correct_types. Retrieved 4/8 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 7/14 statements.
# Partially parsed test_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_wrong_value_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_checkedpmap_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_constructor_returns_same_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_nested_checked_types. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_inheritance. Retrieved 3/10 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 3.5
    var_3 = {var_0: var_0, var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.0
    var_5 = {var_3: var_4}
    var_6 = 1
    var_7 = -1.0
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'x'

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #15
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'ok'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error2'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)

import builtins as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1._invariant_errors(var_0, var_1)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = 'a'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = True
    var_9 = 'b'
    var_10 = (var_8, var_9)
    var_11 = lambda x: var_10
    var_12 = True
    var_13 = 'c'
    var_14 = (var_12, var_13)
    var_15 = lambda x: var_14
    var_16 = [var_7, var_11, var_15]
    var_17 = module_0._invariant_errors(var_3, var_16)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'err2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 'err3'
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0._invariant_errors(var_0, var_11)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__checked_type_create_with_same_class. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_without_checked_types. Retrieved 5/10 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_matching_data. Retrieved 3/11 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_non_matching_data. Retrieved 3/14 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_ignore_extra. Retrieved 4/15 statements.
# Partially parsed test__checked_type_create_with_multiple_checked_types_and_matching_data. Retrieved 1/14 statements.
# Partially parsed test__checked_type_create_with_multiple_checked_types_and_non_matching_data. Retrieved 3/18 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = module_0.CheckedType()
    var_3 = [var_2]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]
    var_4 = True

def test_case_0():
    var_0 = '__main__.CheckedTypeA'
    var_1 = '__main__.CheckedTypeB'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = '__main__.CheckedTypeA'
    var_1 = '__main__.CheckedTypeB'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = [var_3]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_with_valid_initial_dict. Retrieved 5/10 statements.
# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_type_check_failure_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_failure_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_success. Retrieved 3/7 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_checked_type_create_key. Retrieved 4/19 statements.
# Partially parsed test_constructor_with_checked_type_create_value. Retrieved 4/19 statements.
# Partially parsed test_constructor_persistent_returns_same_type. Retrieved 2/10 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_constructor_serialize. Retrieved 4/9 statements.
# Partially parsed test_constructor_create_with_same_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_reduce_for_pickling. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = 'raw_key'
    var_1 = 'raw_value'
    var_2 = {var_0: var_1}
    var_3 = 0

def test_case_0():
    var_0 = 'raw_key'
    var_1 = 'raw_value'
    var_2 = {var_0: var_1}
    var_3 = 0

def test_case_0():
    var_0 = 1
    var_1 = 1.5

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda format, k, v: (k, v)
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.
# Partially parsed test_maybe_parse_user_type_with_mixed_iterable. Retrieved 1/3 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = 'int'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = 'int'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)

def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_checking_valid. Retrieved 3/7 statements.
# Partially parsed test_constructor_type_checking_invalid_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_checking_invalid_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_valid. Retrieved 3/7 statements.
# Partially parsed test_constructor_invariant_invalid. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_multiple_invariants_failure. Retrieved 4/9 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_initial_as_checkedpmap. Retrieved 3/8 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key must be non-negative')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key must be non-negative')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_CheckedType_constructor. Retrieved 3/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = '__slots__'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_success. Retrieved 4/9 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_tuple_initial. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_generator_initial. Retrieved 2/7 statements.
# Partially parsed test_constructor_preserves_subclass_identity. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)

def test_case_0():
    var_0 = 7
    var_1 = 8
    var_2 = 9
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 'invalid'
    var_2 = 12
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2
    var_3 = 3.0
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 13
    var_1 = 14
    var_2 = 15
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_invariant_errors_passes_elem_to_each_invariant. Retrieved 2/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test_element'
    var_1 = True
    var_2 = 'ok1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 'ok'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error2'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.object()

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'bad'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'good'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'worse'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_type_error_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_error_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_checked_type_key. Retrieved 3/12 statements.
# Partially parsed test_constructor_with_checked_type_value. Retrieved 3/12 statements.
# Partially parsed test_constructor_with_checked_type_create. Retrieved 3/16 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/11 statements.
# Partially parsed test_constructor_multiple_invariants_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_self_as_initial. Retrieved 3/8 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_undefined_size. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = '1'
    var_1 = '1.5'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (k > 0, 'Key must be positive')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k > 0, 'Key must be positive')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_no_type_specified. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

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
    var_1 = 'a'
    var_2 = 3.5
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__restore_pickle. Retrieved 3/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test_data'
    var_1 = module_0._restore_pickle(var_0)
    var_2 = set()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 10/11 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_empty_input. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_mixed_data. Retrieved 12/13 statements.


def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = (var_0, var_3)
    var_5 = []
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = []
    var_8 = (var_0, var_7)
    var_9 = [var_2, var_6, var_8]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = 'error2'
    var_5 = [var_4]
    var_6 = (var_0, var_5)
    var_7 = 'error3'
    var_8 = [var_7]
    var_9 = (var_0, var_8)
    var_10 = [var_3, var_6, var_9]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'err1'
    var_5 = 'err2'
    var_6 = [var_4, var_5]
    var_7 = (var_3, var_6)
    var_8 = 'err3'
    var_9 = [var_8]
    var_10 = (var_3, var_9)
    var_11 = [var_2, var_7, var_10]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_initial_as_same_class_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter_and_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_constructor_with_initial_as_dict. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_initial_as_iterable_of_pairs. Retrieved 9/14 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}
    var_1 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 2.5
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = dict(var_6)
    var_8 = dict(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_type_check_failure_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_failure_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_success. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_key_type_create. Retrieved 5/13 statements.
# Partially parsed test_constructor_with_checked_value_type_create. Retrieved 5/13 statements.
# Partially parsed test_constructor_with_existing_checkedpmap. Retrieved 3/8 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_store_invariants_stores_wrapped_invariants. Retrieved 8/20 statements.
# Partially parsed test_store_invariants_inherits_from_multiple_bases. Retrieved 9/21 statements.
# Partially parsed test_store_invariants_merges_local_and_inherited. Retrieved 4/19 statements.
# Partially parsed test_store_invariants_raises_typeerror_for_non_callable. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_handles_empty_invariants. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_wraps_invariant_returning_tuple_of_results. Retrieved 5/12 statements.
# Partially parsed test_store_invariants_avoids_duplicate_inheritance. Retrieved 7/19 statements.
# Partially parsed test_store_invariants_preserves_wrapping_for_bool_result. Retrieved 5/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = callable(var_6)

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 1
    var_8 = var_3[var_7]

def test_case_0():
    var_0 = 'invariant'
    var_1 = 'invariants'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant'

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = 0
    var_4 = var_0[var_1][var_3]

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = 0
    var_4 = var_0[var_1][var_3]



# Parsed testcases at query #2
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error2'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'err2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import builtins as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1._invariant_errors(var_0, var_1)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = False
    var_2 = 123
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = (var_1, var_7)
    var_9 = lambda x: var_8
    var_10 = True
    var_11 = 'skip'
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_4, var_9, var_13]
    var_15 = module_0._invariant_errors(var_0, var_14)



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_complex_nested_iterable.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_correct_types. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_incorrect_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_incorrect_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_key_type.
# Failed to parse test_constructor_with_checked_value_type.
# Partially parsed test_constructor_with_checked_types_and_create. Retrieved 3/20 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_self_as_initial. Retrieved 3/8 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_0, var_1: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'raw_key'
    var_1 = 'raw_value'
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
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_predicate_at_line_18_evaluates_to_true.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_without_checked_types. Retrieved 5/10 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_mismatched_data. Retrieved 4/17 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_matching_data. Retrieved 2/14 statements.
# Partially parsed test__checked_type_create_with_ignore_extra_true. Retrieved 5/18 statements.


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
    var_2 = 5
    var_3 = 6
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 10

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 7
    var_3 = [var_2]
    var_4 = True
    var_5 = 0



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_single_false_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_results.
# Failed to parse test_wrap_invariant_with_all_true_multiple_results.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 5/12 statements.
# Failed to parse test_wrap_invariant_with_empty_multiple_results.


def test_case_0():
    var_0 = 5
    var_1 = 4
    var_2 = 2
    var_3 = 1
    var_4 = 3



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_no_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 2/7 statements.
# Partially parsed test_check_types_with_type_string. Retrieved 6/8 statements.
# Partially parsed test_check_types_with_mixed_type_and_string. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_type_string. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'hello'
    var_3 = 3.14
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'invalid'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 'hello'
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 1
    var_2 = 'hello'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'invalid'
    var_3 = [var_2]



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_no_type_specified. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

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
    var_1 = 'a'
    var_2 = 3.5
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_types_with_expected_types_and_matching_element. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'MockSourceClass'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_type_string. Retrieved 6/8 statements.
# Partially parsed test_check_types_with_type_string_invalid. Retrieved 7/10 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 2/8 statements.
# Partially parsed test_check_types_with_single_expected_type. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_single_expected_type_invalid. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_none_values. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_none_values_invalid. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 'hello'
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 'hello'
    var_5 = []
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 'hello'
    var_3 = []
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'not an int'
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'three'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_0, var_2]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'invalid'
    var_3 = [var_1, var_0, var_2]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_store_invariants_with_no_inheritance. Retrieved 3/7 statements.
# Partially parsed test_store_invariants_with_single_invariant. Retrieved 8/14 statements.
# Partially parsed test_store_invariants_with_multiple_inheritance. Retrieved 8/18 statements.
# Partially parsed test_store_invariants_with_diamond_inheritance. Retrieved 8/20 statements.
# Partially parsed test_store_invariants_with_local_override. Retrieved 12/18 statements.
# Partially parsed test_store_invariants_with_multiple_invariants_in_chain. Retrieved 8/16 statements.
# Partially parsed test_store_invariants_with_non_callable_raises_typeerror. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_wrapped_invariant_returning_list. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_with_wrapped_invariant_returning_single_bool. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_destination_name_different. Retrieved 8/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = None

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]
    var_6 = None
    var_7 = [inv(var_6) for inv in var_5]

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = None

def test_case_0():
    var_0 = 'invariant'
    var_1 = False
    var_2 = 'local'
    var_3 = (var_2,)
    var_4 = (var_1, var_3)
    var_5 = lambda self: var_4
    var_6 = {var_0: var_5}
    var_7 = 'invariants'
    var_8 = var_6[var_7]
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_6[var_7][var_1]
    var_11 = None

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]
    var_6 = None
    var_7 = [inv(var_6) for inv in var_5]

def test_case_0():
    var_0 = 'not a function'
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant'

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0
    var_4 = None

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0
    var_4 = None

def test_case_0():
    var_0 = {}
    var_1 = 'checks'
    var_2 = 'check'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checked_type_create_with_checked_type_and_mismatched_data. Retrieved 5/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 'data1'
    var_2 = 'data2'
    var_3 = [var_1, var_2]
    var_4 = False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 5/20 statements.


def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_invariants_adds_tuple_of_wrapped_invariants. Retrieved 6/12 statements.
# Partially parsed test_store_invariants_inherits_from_bases. Retrieved 5/18 statements.
# Partially parsed test_store_invariants_skips_missing_keys. Retrieved 5/14 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = lambda x: f'wrapped_{x}'
    var_1 = {}
    var_2 = []
    var_3 = 'dest'
    var_4 = 'src'
    var_5 = module_0.store_invariants(var_1, var_2, var_3, var_4)

def test_case_0():
    var_0 = lambda x: f'wrapped_{x}'
    var_1 = 'src'
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'src'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)

def test_case_0():
    var_0 = lambda x: f'wrapped_{x}'
    var_1 = {}
    var_2 = 'src'
    var_3 = 'dest'
    var_4 = 'src'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_store_invariants_adds_destination. Retrieved 10/4 statements.
# Partially parsed test_store_invariants_collects_from_dct. Retrieved 3/9 statements.
# Partially parsed test_store_invariants_collects_from_base_classes. Retrieved 5/10 statements.
# Partially parsed test_store_invariants_collects_from_multiple_bases. Retrieved 5/13 statements.
# Partially parsed test_store_invariants_inherits_from_base_hierarchy. Retrieved 5/12 statements.
# Partially parsed test_store_invariants_avoids_duplicates. Retrieved 5/12 statements.
# Partially parsed test_store_invariants_combines_dct_and_bases. Retrieved 2/12 statements.
# Partially parsed test_store_invariants_wraps_invariants. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_raises_on_non_callable_in_base. Retrieved 4/8 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = 'invariants'
    var_5 = 'inv'
    var_6 = module_0.store_invariants(var_3, var_0, var_4, var_5)
    var_7 = var_3[var_4]
    var_8 = var_3[var_4]
    var_9 = len(var_8)
    assert var_9 == 0

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = 'invariants'
    var_5 = 'inv'
    var_6 = module_0.store_invariants(var_3, var_0, var_4, var_5)
    var_7 = var_3[var_4]
    var_8 = var_3[var_4]
    var_9 = len(var_8)
    assert var_9 == 0

def test_case_0():
    var_0 = 'inv'
    var_1 = ()
    var_2 = 'invariants'

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'inv'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'inv'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'inv'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'inv'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1

def test_case_0():
    var_0 = 'inv'
    var_1 = 'invariants'

def test_case_0():
    var_0 = 'inv'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0
    var_4 = None

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'inv'
    var_1 = 'not a function'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'invariants'
    var_5 = 'inv'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)

def test_case_0():
    var_0 = 'not a function'
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'inv'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_valid_key_value_pairs. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises_error. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 3/8 statements.
# Failed to parse test_constructor_with_checked_key_type_create.
# Partially parsed test_constructor_with_initial_as_same_type_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_constructor_repr_output. Retrieved 3/8 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_invalid_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_invalid_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_key_type.
# Partially parsed test_constructor_with_initial_as_same_class_instance. Retrieved 3/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_size_parameter. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_undefined_size_constant. Retrieved 5/10 statements.
# Partially parsed test_constructor_repr_output. Retrieved 3/8 statements.
# Partially parsed test_constructor_str_output. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}

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
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5

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
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_checked_pmap_constructor_with_empty_initial.
# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_type_check_failure_key. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_type_check_failure_value. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invariant_success. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_checked_type_source. Retrieved 1/10 statements.
# Partially parsed test_checked_pmap_constructor_repr. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_passing_checked_pmap_instance. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap({1: 1.5})'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_invariant_errors_passes_elem_to_each_invariant. Retrieved 2/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test_element'
    var_1 = True
    var_2 = 'ok1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 'ok'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error2'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.object()

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'ignore1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'include1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'ignore2'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_checkedpmap_initial. Retrieved 3/10 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_size_and_initial. Retrieved 3/9 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_checked_key_type. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_checked_value_type. Retrieved 1/10 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.0
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = {}
    var_1 = 0

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_respects_key_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_respects_value_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_enforces_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_checked_type_creation. Retrieved 3/20 statements.
# Partially parsed test_constructor_returns_same_instance_if_already_checked_type. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

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
    var_0 = 'raw_key'
    var_1 = 'raw_value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_store_types_single_dict. Retrieved 3/5 statements.
# Partially parsed test_store_types_multiple_dicts. Retrieved 7/13 statements.
# Partially parsed test_store_types_with_iterable_source. Retrieved 3/6 statements.
# Partially parsed test_store_types_missing_source. Retrieved 5/9 statements.
# Partially parsed test_store_types_preserved_iterable_type. Retrieved 3/5 statements.
# Partially parsed test_store_types_nested_iterables. Retrieved 3/7 statements.
# Partially parsed test_store_types_duplicate_sources. Retrieved 4/9 statements.
# Partially parsed test_store_types_mixed_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'Base2'
    var_4 = ()
    var_5 = {}
    var_6 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = {}
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'source'
    var_4 = 'destination'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'destination'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 'CustomType'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'str'
    var_2 = []
    var_3 = 'destination'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__restore_pickle. Retrieved 5/9 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0._restore_pickle(var_2)
    var_4 = set()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test___new___creates_checked_pmap_from_dict. Retrieved 5/10 statements.
# Failed to parse test___new___creates_empty_checked_pmap.
# Partially parsed test___new___creates_checked_pmap_with_size. Retrieved 5/10 statements.
# Partially parsed test___new___enforces_key_type. Retrieved 3/8 statements.
# Partially parsed test___new___enforces_value_type. Retrieved 3/8 statements.
# Partially parsed test___new___enforces_invariant. Retrieved 4/9 statements.
# Partially parsed test___new___accepts_valid_invariant. Retrieved 3/8 statements.
# Partially parsed test___new___creates_from_checked_pmap_instance. Retrieved 3/8 statements.
# Partially parsed test___new___creates_from_iterable_of_pairs. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 2.25
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_empty_dict. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_key_value_pairs. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_invariants. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Failed to parse test_checked_pmap_constructor_with_checked_type_key.
# Partially parsed test_checked_pmap_constructor_with_initial_as_same_type_instance. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_undefined_size_constant. Retrieved 3/8 statements.


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
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 10/11 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 12/13 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_empty_input. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 5/6 statements.
# Partially parsed test_merge_invariant_results_false_with_empty_data. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_mixed_data_types. Retrieved 11/12 statements.


def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = (var_0, var_3)
    var_5 = []
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = []
    var_8 = (var_0, var_7)
    var_9 = [var_2, var_6, var_8]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = []
    var_6 = (var_4, var_5)
    var_7 = 'error2'
    var_8 = 'error3'
    var_9 = [var_7, var_8]
    var_10 = (var_0, var_9)
    var_11 = [var_3, var_6, var_10]

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = (var_0, var_5)
    var_7 = 'c'
    var_8 = [var_7]
    var_9 = (var_0, var_8)
    var_10 = [var_3, var_6, var_9]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'only error'
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = [var_3]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = (var_3, var_5)
    var_7 = 'text'
    var_8 = [var_7]
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_6, var_9]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_mixed_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_single_true. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_single_false. Retrieved 4/5 statements.
# Partially parsed test_merge_invariant_results_data_types. Retrieved 11/12 statements.


def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = (var_0, var_3)
    var_5 = []
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = []
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
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = 'error2'
    var_7 = (var_3, var_6)
    var_8 = []
    var_9 = (var_0, var_8)
    var_10 = [var_2, var_5, var_7, var_9]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = False
    var_1 = 'error'
    var_2 = (var_0, var_1)
    var_3 = [var_2]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 123
    var_5 = (var_3, var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_5, var_9]



# Parsed testcases at query #30
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test_data'
    var_1 = module_0._restore_pickle(var_0)
    var_2 = 'test_data'
    var_3 = set()
    var_4 = (var_2, var_3)



