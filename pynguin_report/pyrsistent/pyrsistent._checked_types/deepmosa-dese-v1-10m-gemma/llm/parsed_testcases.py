####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'error'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'must be positive'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = False
    var_2 = 'not greater than zero'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = False
    var_10 = 'even number required'
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = [var_4, var_8, var_12]
    var_14 = module_0._invariant_errors(var_0, var_13)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'missing key a'
    var_4 = lambda x: (var_0 in x, var_3)
    var_5 = 0
    var_6 = 'value must be positive'
    var_7 = lambda x: (x[var_0] > var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_2, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = -1
    var_2 = {var_0: var_1}
    var_3 = 'missing key a'
    var_4 = lambda x: (var_0 in x, var_3)
    var_5 = 0
    var_6 = 'value must be positive'
    var_7 = lambda x: (x[var_0] > var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_2, var_8)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_wrap_invariant_single_bool_result.
# Failed to parse test_wrap_invariant_single_bool_false_result.
# Failed to parse test_wrap_invariant_multiple_results_all_true.
# Failed to parse test_wrap_invariant_multiple_results_with_false.
# Partially parsed test_wrap_invariant_preserves_arguments. Retrieved 2/6 statements.
# Partially parsed test_wrap_invariant_preserves_keyword_arguments. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 5

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_store_invariants_basic_functionality. Retrieved 10/19 statements.
# Partially parsed test_store_invariants_inheritance. Retrieved 16/29 statements.
# Partially parsed test_store_invariants_raises_type_error. Retrieved 8/13 statements.
# Partially parsed test_store_invariants_no_source_found. Retrieved 3/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = 10

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = 'base_val'
    var_3 = (var_2,)
    var_4 = (var_1, var_3)
    var_5 = False
    var_6 = 'derived_val'
    var_7 = (var_6,)
    var_8 = (var_5, var_7)
    var_9 = 'dest'
    var_10 = 'src'
    var_11 = var_0[var_9]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_0[var_9]
    var_14 = False
    var_15 = True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'not_a_callable'
    var_2 = {var_0: var_1}
    var_3 = 'dest'
    var_4 = 'src'
    var_5 = module_0.store_invariants(var_2, var_0, var_3, var_4)
    var_6 = 'TypeError not raised'
    var_7 = AssertionError(var_6)

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'non_existent'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/12 statements.
# Partially parsed test_check_types_invalid_input_raises_error. Retrieved 8/18 statements.
# Partially parsed test_check_types_empty_expected_types. Retrieved 5/11 statements.
# Partially parsed test_check_types_empty_iterator. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = True
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = [var_2, var_1, var_4]
    var_6 = 'Exception was not raised'
    var_7 = AssertionError(var_6)

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = []
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = []



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_inheritance. Retrieved 5/17 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 5/12 statements.
# Partially parsed test_store_invariants_no_source_found. Retrieved 3/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = 'TypeError not raised'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'non_existent'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_iterable.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'my_type'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_types_success. Retrieved 4/11 statements.
# Partially parsed test_check_types_failure_raises_exception. Retrieved 3/13 statements.
# Partially parsed test_check_types_empty_expected_types_does_nothing. Retrieved 5/11 statements.
# Partially parsed test_check_types_empty_iterator_does_nothing. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 'string'
    var_3 = 3.14
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_store_types_single_source. Retrieved 5/7 statements.
# Partially parsed test_store_types_multiple_bases. Retrieved 3/9 statements.
# Partially parsed test_store_types_with_iterable_input. Retrieved 5/7 statements.
# Partially parsed test_store_types_no_matching_source. Retrieved 5/9 statements.
# Partially parsed test_store_types_overwrites_destination. Retrieved 10/14 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'MyType'
    var_2 = []
    var_3 = 'dest'
    var_4 = 'source'
    var_5 = module_0._store_types(var_0, var_2, var_3, var_4)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = 'existing'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'dest'
    var_4 = 'source'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'dest'
    var_1 = 'old'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = 'Base'
    var_6 = ()
    var_7 = 'source'
    var_8 = []
    var_9 = module_0._store_types(var_3, var_8, var_0, var_7)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/12 statements.
# Partially parsed test_checkedpmap_constructor_with_explicit_size. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'string_key'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checkedpset_constructor_with_iterable. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_empty_iterable. Retrieved 2/7 statements.
# Partially parsed test_checkedpset_constructor_with_pmap. Retrieved 6/11 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'not_an_int'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_checked_type_create_returns_instance_if_already_correct_type.
# Partially parsed test_checked_type_create_creates_new_instance_from_data. Retrieved 4/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_recursion. Retrieved 4/15 statements.
# Partially parsed test_checked_type_create_skips_recursion_if_data_matches_existing_type. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'alpha'
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (v > k, 'Value must be greater than key')
    var_1 = 1
    var_2 = 0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_restore_pickle_calls_create_with_correct_arguments. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_checked_type_create_returns_instance_if_already_correct_type.
# Partially parsed test_checked_type_create_wraps_data_in_constructor. Retrieved 4/9 statements.
# Partially parsed test_checked_type_create_uses_checked_type_recursion. Retrieved 7/24 statements.
# Partially parsed test_checked_type_create_skips_recursion_if_data_is_already_correct_type. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '__main__.InnerCheckedType'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 0

def test_case_0():
    var_0 = []
    var_1 = '__main__.InnerCheckedType'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_store_types_single_source. Retrieved 5/7 statements.
# Partially parsed test_store_types_multiple_bases_inheritance. Retrieved 3/9 statements.
# Partially parsed test_store_types_with_iterable_input. Retrieved 6/8 statements.
# Partially parsed test_store_types_no_matching_source. Retrieved 5/9 statements.
# Partially parsed test_store_types_overwriting_existing. Retrieved 7/9 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'MyType'
    var_2 = []
    var_3 = 'dest'
    var_4 = 'source'
    var_5 = module_0._store_types(var_0, var_2, var_3, var_4)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = []
    var_3 = 'dest'
    var_4 = 'source'
    var_5 = module_0._store_types(var_0, var_2, var_3, var_4)

def test_case_0():
    var_0 = 'other'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'dest'
    var_4 = 'source'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'dest'
    var_1 = 'old'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = 'source'
    var_6 = module_0._store_types(var_3, var_4, var_0, var_5)



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_checked_type_instantiation_error.
# Failed to parse test_checked_type_abstract_methods_raise_error.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_list. Retrieved 4/9 statements.
# Partially parsed test_checkedpvector_constructor_with_tuple. Retrieved 3/8 statements.
# Failed to parse test_checkedpvector_constructor_empty.
# Partially parsed test_checkedpvector_constructor_type_validation_error. Retrieved 5/10 statements.
# Partially parsed test_checkedpvector_constructor_invariant_validation_error. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1.5
    var_1 = 2.5
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = 1
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = 'Constructor should have failed type check'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'Constructor should have failed invariant check'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 9/23 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 8/13 statements.
# Partially parsed test_store_invariants_no_matches. Retrieved 3/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest_inv'
    var_2 = 'src_inv'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = 1
    var_8 = var_0[var_1][var_7]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src_inv'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = 'dest_inv'
    var_4 = 'src_inv'
    var_5 = module_0.store_invariants(var_2, var_0, var_3, var_4)
    var_6 = 'TypeError not raised'
    var_7 = AssertionError(var_6)

def test_case_0():
    var_0 = {}
    var_1 = 'dest_inv'
    var_2 = 'src_inv'



# Parsed testcases at query #20
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'msg'
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
    var_6 = 'error_1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'fine'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = 'error_2'
    var_13 = (var_5, var_12)
    var_14 = lambda x: var_13
    var_15 = [var_4, var_8, var_11, var_14]
    var_16 = module_0._invariant_errors(var_0, var_15)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = False
    var_2 = 'fail_a'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'fail_b'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 'empty'
    var_3 = lambda x: (len(x) > var_1, var_2)
    var_4 = 't'
    var_5 = 'wrong_start'
    var_6 = lambda x: (x.startswith(var_4), var_5)
    var_7 = 'wrong_end'
    var_8 = lambda x: (x.endswith(var_4), var_7)
    var_9 = [var_3, var_6, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = 'too_short'
    var_3 = lambda x: (len(x) > var_1, var_2)
    var_4 = 'z'
    var_5 = 'wrong_start'
    var_6 = lambda x: (x.startswith(var_4), var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0._invariant_errors(var_0, var_7)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_checked_type_instantiation_fails_due_to_abstract_method.


def test_case_0():
    pass



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_maybe_parse_user_type_evaluates_line_18.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_types_predicate_is_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_explicit_size. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = 'Should have raised InvariantException'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_check_types_predicate_true. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_checked_type_create_returns_source_data_when_instance_of_cls.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_wrap_invariant_returns_bool_directly. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_merges_list_of_results. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_handles_all_true_results. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_handles_empty_list. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_preserves_args_kwargs. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_check_types_predicate_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_iterable. Retrieved 4/9 statements.
# Failed to parse test_checked_pvector_constructor_empty.
# Partially parsed test_checked_pvector_constructor_with_existing_pvector. Retrieved 3/9 statements.
# Partially parsed test_checked_pvector_constructor_type_validation. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 'not_an_int'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_invariant_errors_evaluates_to_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 10



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_invariant_errors_returns_true_when_all_pass. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = []



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 3/8 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 3/8 statements.
# Partially parsed test_checkedpset_constructor_with_empty_iterable. Retrieved 2/6 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 'not_an_int'
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []



# Parsed testcases at query #34
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source_name'
    var_1 = 'not_a_callable'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'source_name'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/12 statements.
# Partially parsed test_checkedpset_constructor_with_empty. Retrieved 1/5 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type_raises. Retrieved 4/8 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant_raises. Retrieved 3/7 statements.


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
    var_2 = 'string'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_size. Retrieved 2/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = {}
    var_1 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_empty. Retrieved 1/5 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 3/7 statements.


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
    var_1 = 'not_a_number'
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Partially parsed test_maybe_parse_user_type_nested_iterable. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'float'
    var_1 = 'bool'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #39
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_0, var_3)
    var_5 = 'c'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0._merge_invariant_results(var_7)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = 'c'
    var_7 = (var_0, var_6)
    var_8 = 'd'
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_5, var_7, var_9]
    var_11 = module_0._merge_invariant_results(var_10)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'x'
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0._merge_invariant_results(var_5)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_invariant_results(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'only_one'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0._merge_invariant_results(var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'only_one'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0._merge_invariant_results(var_3)



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_checked_type_create_returns_instance_if_already_correct_type.
# Partially parsed test_checked_type_create_creates_new_instance_from_source_data. Retrieved 3/8 statements.
# Partially parsed test_checked_type_create_with_checked_type_recursion. Retrieved 3/5 statements.
# Partially parsed test_checked_type_create_with_already_correct_subclass_data. Retrieved 2/12 statements.
# Partially parsed test_checked_type_create_passes_ignore_extra_flag. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'val1'
    var_1 = 'val2'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'item1'
    var_1 = 'item2'

def test_case_0():
    var_0 = 'data'
    var_1 = [var_0]
    var_2 = True
    var_3 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_empty. Retrieved 1/5 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 'not an int'
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_store_types_basic. Retrieved 3/9 statements.
# Partially parsed test_store_types_string_input. Retrieved 4/9 statements.
# Partially parsed test_store_types_iterable_input. Retrieved 3/9 statements.
# Partially parsed test_store_types_multiple_bases. Retrieved 3/11 statements.
# Partially parsed test_store_types_no_matching_key. Retrieved 3/9 statements.
# Partially parsed test_store_types_overwriting_existing. Retrieved 2/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'

def test_case_0():
    var_0 = 'str'
    var_1 = {}
    var_2 = 'dest'
    var_3 = 'source'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'

def test_case_0():
    var_0 = 'dest'
    var_1 = 'source'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = 'InvariantException was not raised for invalid data'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'CheckedKeyTypeError was not raised for invalid key type'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_checked_type_create_identity.
# Partially parsed test_checked_type_create_constructor_call. Retrieved 4/8 statements.
# Partially parsed test_checked_type_create_with_checked_types_recursion. Retrieved 3/5 statements.
# Partially parsed test_checked_type_create_with_checked_types_no_transformation_needed. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'raw'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 6/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = 2.5
    var_5 = {var_1: var_4}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_restore_pickle_returns_created_object. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_data'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 'not_an_int'
    var_2 = {var_0: var_1}



# Parsed testcases at query #8
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_0, var_3)
    var_5 = 'c'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0._merge_invariant_results(var_7)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = 'c'
    var_7 = (var_0, var_6)
    var_8 = 'd'
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_5, var_7, var_9]
    var_11 = module_0._merge_invariant_results(var_10)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'x'
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0._merge_invariant_results(var_5)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_invariant_results(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_merge_invariant_results_predicate_true. Retrieved 6/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 8/14 statements.
# Partially parsed test_exec. Retrieved 2/5 statements.
# Partially parsed test_store_invariants_inheritance. Retrieved 3/5 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_no_source_found. Retrieved 3/7 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 'dest'
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = 'dest'
    var_2 = None

def test_case_0():
    var_0 = 'not a callable'
    var_1 = {}
    var_2 = 'dest'
    var_3 = 'src'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'non_existent'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_type_violation. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_empty_initial. Retrieved 1/6 statements.


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
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_restore_pickle_success. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 9/21 statements.
# Partially parsed test_store_invariants_multiple_inheritance. Retrieved 11/20 statements.
# Partially parsed test_store_invariants_all_callables. Retrieved 7/10 statements.
# Partially parsed test_store_invariants_no_source_found. Retrieved 3/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'src_inv'
    var_2 = 'dst_inv'
    var_3 = lambda : (True, 'parent')
    var_4 = {}
    var_5 = var_4[var_2]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_4[var_2][var_7]

def test_case_0():
    var_0 = lambda : (True, 'A')
    var_1 = lambda : (False, 'B')
    var_2 = {}
    var_3 = 'res'
    var_4 = 'src_inv'
    var_5 = var_2[var_3]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_2[var_3][var_7]
    var_9 = 1
    var_10 = var_2[var_3][var_9]

def test_case_0():
    var_0 = 'not a callable'
    var_1 = {}

def test_case_0():
    var_0 = lambda : True
    var_1 = {}
    var_2 = 'out'
    var_3 = 'src_inv'
    var_4 = 0
    var_5 = var_1[var_2][var_4]
    var_6 = callable(var_5)

def test_case_0():
    var_0 = {}
    var_1 = 'out'
    var_2 = 'src_inv'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'not a TargetClass instance'



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_checked_type_create_returns_source_if_already_instance.
# Partially parsed test_checked_type_create_wraps_list_using_checked_type_logic. Retrieved 8/24 statements.
# Partially parsed test_checked_type_create_with_direct_constructor_no_checked_types. Retrieved 4/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'builtins.int'
    var_2 = [var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 'not_int'
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_checked_type_is_not_abstract_instantiable_due_to_abstractmethods.


def test_case_0():
    pass



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serialize_with_default_format. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_specific_format. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'upper'



# Parsed testcases at query #19
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source_name'
    var_1 = 'not_a_callable'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'source_name'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_checked_type_instantiation_error.
# Failed to parse test_checked_type_is_abstract.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/11 statements.
# Partially parsed test_check_types_invalid_input_raises_error. Retrieved 4/14 statements.
# Partially parsed test_check_types_empty_expected_types. Retrieved 5/11 statements.
# Partially parsed test_check_types_empty_iterator. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = []



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_iterable. Retrieved 4/9 statements.
# Failed to parse test_checked_pvector_constructor_empty.
# Partially parsed test_checked_pvector_constructor_from_existing_pvector. Retrieved 3/9 statements.
# Partially parsed test_checked_pvector_constructor_type_validation_on_init. Retrieved 4/9 statements.
# Partially parsed test_checked_pvector_constructor_invariant_validation. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1.5
    var_1 = 2.5
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'not_an_int'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = -2
    var_3 = [var_1, var_2]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_size_and_initial_dict. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_violation. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = {var_0: var_2, var_1: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 1.9
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'float'
    var_1 = 'bool'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_wrap_invariant_returns_bool_directly. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_merges_list_of_results. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_returns_true_if_all_are_true. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_handles_empty_list. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_with_complex_types. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/12 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_type_violation. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_explicit_size. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = 10



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_argument. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'not a dummy type'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checked_type_is_instance_of_object. Retrieved 1/2 statements.


def test_case_0():
    pass

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'not an instance of MockClass'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_success. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_failure. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_with_type_mismatch. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.9
    var_4 = {var_1: var_3, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = 'Should have raised InvariantException'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_with_explicit_size. Retrieved 3/7 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = 'Should have raised InvariantException'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #34
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'some_string'
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_checked_type_create_returns_instance_if_already_correct_type.
# Partially parsed test_checked_type_create_creates_new_instance_from_source_data. Retrieved 4/9 statements.
# Partially parsed test_checked_type_create_with_checked_type_recursion. Retrieved 7/19 statements.
# Partially parsed test_checked_type_create_with_ignore_extra_flag_passed. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = '__main__.BaseCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = '__main__.ActualCheckedType'
    var_6 = [var_5]
    var_7 = 'abc'
    var_8 = [var_2, var_7]

def test_case_0():
    var_0 = '__main__.SimpleCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]
    var_4 = True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_types_predicate_false_when_expected_types_is_empty. Retrieved 2/3 statements.


def test_case_0():
    var_0 = []
    var_1 = []



