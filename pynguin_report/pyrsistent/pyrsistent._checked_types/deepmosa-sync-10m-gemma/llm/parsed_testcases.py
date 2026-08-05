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
    var_3 = bool(var_2 == [])
    assert var_3 is True

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
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'not positive'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = (var_1, var_2)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_8, var_10]
    var_12 = module_0._invariant_errors(var_0, var_11)
    var_13 = bool(var_12 == ['not positive'])
    assert var_13 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = -5
    var_1 = 0
    var_2 = 'must be positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 'must be negative'
    var_5 = lambda x: (x < var_1, var_4)
    var_6 = [var_3, var_5]
    var_7 = module_0._invariant_errors(var_0, var_6)
    var_8 = bool(var_7 == ['must be positive'])
    assert var_8 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'err2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['err1', 'err2'])
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.
# Failed to parse test_maybe_parse_tuple_preservation.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'float'
    var_1 = 'bool'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('float', 'bool'))
    assert var_4 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'float'
    var_1 = 'bool'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('float', 'bool'))
    assert var_4 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checkedpset_constructor_with_invariant_failure. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/6 statements.


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
    var_2 = 'string'
    var_3 = [var_1, var_2]
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]
    var_3 = bool(True)
    assert var_3 is True
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'float'
    var_1 = 'bool'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('float', 'bool'))
    assert var_4 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 10



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_maybe_parse_user_type_evaluates_string_branch.




# Parsed testcases at query #8
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserves_iterable_type.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/11 statements.
# Partially parsed test_check_types_invalid_input_raises_error. Retrieved 4/17 statements.
# Partially parsed test_check_types_empty_expected_types_passes. Retrieved 5/11 statements.
# Partially parsed test_check_types_multiple_valid_types. Retrieved 4/11 statements.


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
    var_4 = "Type MockSource can only be used with ('int'), not str"

def test_case_0():
    var_0 = 'any'
    var_1 = 123
    var_2 = True
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = True
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'err1'
    var_1 = 'err2'
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = (var_3,)
    var_5 = 'base error'
    var_6 = 'msg'
    var_7 = {var_6: var_5}
    var_8 = module_0.InvariantException(var_2, var_4, **var_7)
    var_9 = var_8.invariant_errors
    var_10 = bool(var_8.invariant_errors == ('err1', 'err2'))
    assert var_10 is True
    var_11 = var_8.missing_fields
    var_12 = bool(var_8.missing_fields == ('field1',))
    assert var_12 is True
    var_13 = str(var_8)
    var_14 = 'base error'
    var_15 = bool('base error' in var_13)
    assert var_15 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'dynamic_err'
    var_1 = lambda : var_0
    var_2 = 'static_err'
    var_3 = (var_1, var_2)
    var_4 = ()
    var_5 = {}
    var_6 = module_0.InvariantException(var_3, var_4, **var_5)
    var_7 = var_6.invariant_errors
    var_8 = bool(var_6.invariant_errors == ('dynamic_err', 'static_err'))
    assert var_8 is True
    var_9 = var_6.missing_fields
    var_10 = bool(var_6.missing_fields == ())
    assert var_10 is True

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
    var_6 = str(var_1)
    var_7 = 'invariant_errors=[], missing_fields=[]'
    var_8 = bool('invariant_errors=[], missing_fields=[]' in var_6)
    assert var_8 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'e1'
    var_1 = (var_0,)
    var_2 = 'f1'
    var_3 = 'f2'
    var_4 = (var_2, var_3)
    var_5 = 'msg'
    var_6 = 'msg'
    var_7 = {var_6: var_5}
    var_8 = module_0.InvariantException(var_1, var_4, **var_7)
    var_9 = 'msg, invariant_errors=[e1], missing_fields=[f1, f2]'
    var_10 = str(var_8)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_checked_type_instantiation_fails_due_to_abstract_methods.


def test_case_0():
    pass



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 9/16 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 6/11 statements.
# Partially parsed test_store_invariants_multiple_inheritance. Retrieved 12/31 statements.
# Partially parsed test_store_invariants_no_matches. Retrieved 3/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'check_base'
    var_2 = 'wrapped_checks'
    var_3 = bool(var_2 in var_0)
    assert var_3 is True
    var_4 = var_0[var_2]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_0[var_2][var_6]
    var_8 = 10
    var_9 = var_7(var_8)
    var_10 = bool(var_9 == (True, (10,)))
    assert var_10 is True

def test_case_0():
    var_0 = 'not a callable'
    var_1 = {}
    var_2 = 'dest'
    var_3 = 'check_base'
    var_4 = 'TypeError not raised'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = {}
    var_1 = 'checks'
    var_2 = 'inv_a'
    var_3 = {}
    var_4 = 'dest'
    var_5 = 'shared_inv'
    var_6 = var_3[var_4]
    var_7 = len(var_6)
    var_8 = bool(var_7 >= 1)
    assert var_8 is True
    var_9 = 0
    var_10 = var_3[var_4][var_9]
    var_11 = 5
    var_12 = var_10(var_11)
    var_13 = bool(var_12 == (True, (5,)))
    assert var_13 is True

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'non_existent'
    var_3 = 'dest'
    var_4 = bool('dest' in var_0)
    assert var_4 is True
    var_5 = var_0['dest']
    var_6 = bool(var_0['dest'] == ())
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 7/13 statements.
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
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = 1.5
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_checkedpset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checkedpset_constructor_with_invariant_violation. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/6 statements.


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
    var_0 = lambda n: (True, None)
    var_1 = 1
    var_2 = 'string'
    var_3 = [var_1, var_2]
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]
    var_3 = 'Negative'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (True, None)
    var_1 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_types_single_source. Retrieved 3/7 statements.
# Partially parsed test_store_types_multiple_bases_and_overrides. Retrieved 2/9 statements.
# Partially parsed test_store_types_no_matching_source. Retrieved 3/7 statements.
# Partially parsed test_store_types_with_string_types. Retrieved 6/10 statements.
# Partially parsed test_store_types_with_iterable_input. Retrieved 3/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = var_0['dest']

def test_case_0():
    var_0 = 'source'
    var_1 = 'dest'

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = 'dest'
    var_4 = bool('dest' not in var_0)
    assert var_4 is True

def test_case_0():
    var_0 = 'source'
    var_1 = 'int'
    var_2 = 'str'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'dest'
    var_6 = var_4['dest']
    var_7 = bool(var_4['dest'] == ('int', 'str'))
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = var_0['dest']



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_checked_type_create_direct_instance. Retrieved 1/3 statements.
# Partially parsed test_checked_type_create_conversion_int. Retrieved 2/4 statements.
# Partially parsed test_checked_type_create_list_conversion. Retrieved 7/18 statements.
# Partially parsed test_checked_type_create_no_checked_types. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = '123'
    var_1 = 123

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1]
    var_5 = 1
    var_6 = 2

def test_case_0():
    var_0 = ()
    var_1 = 'raw_string'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_store_invariants_predicate_true. Retrieved 9/22 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'dst'
    var_2 = lambda x: x
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = module_0.store_invariants(var_3, var_4, var_1, var_0)
    var_6 = bool(var_1 in var_3)
    assert var_6 is True
    var_7 = 0
    var_8 = var_3[var_1][var_7]
    var_9 = callable(var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_restore_pickle_returns_instance_with_correct_data. Retrieved 6/10 statements.
# Partially parsed test_restore_pickle_passes_empty_set_to_factory_fields. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()

def test_case_0():
    var_0 = 'some_data'
    var_1 = set()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_iterable. Retrieved 4/9 statements.
# Failed to parse test_checkedpvector_constructor_empty.
# Partially parsed test_checkedpvector_constructor_with_existing_pvector. Retrieved 3/10 statements.
# Partially parsed test_checkedpvector_constructor_type_validation. Retrieved 4/8 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/8 statements.
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
    var_1 = 'a'
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
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = 'Should have raised InvariantException'
    var_5 = AssertionError(var_4)
    var_6 = 'Invalid mapping'

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/11 statements.
# Partially parsed test_check_types_valid_multiple_types. Retrieved 4/11 statements.
# Partially parsed test_check_types_invalid_type_raises_exception. Retrieved 3/13 statements.
# Partially parsed test_check_types_empty_expected_types_does_nothing. Retrieved 5/11 statements.
# Partially parsed test_check_types_empty_iterator. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = "Type MyClass can only be used with ('int'), not str"

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_iterable. Retrieved 4/9 statements.
# Failed to parse test_checkedpvector_constructor_with_empty.
# Partially parsed test_checkedpvector_constructor_with_existing_pvector. Retrieved 3/10 statements.
# Partially parsed test_checkedpvector_constructor_with_invalid_type_raises. Retrieved 4/8 statements.
# Partially parsed test_checkedpvector_constructor_with_invariant_violation_raises. Retrieved 5/8 statements.


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
    var_1 = 'string'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(True)
    assert var_5 is True
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_checked_type_create_predicate_is_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'not an instance of MockType'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_store_types_predicate_evaluates_to_true. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = 'dest_key'
    var_2 = 'value_in_base'
    var_3 = 'other_key'
    var_4 = 'ignore'
    var_5 = {var_3: var_4}
    var_6 = 'value_in_base'
    var_7 = [var_6]
    var_8 = var_5[var_1]
    var_9 = bool(var_5[var_1] == ['parsed_value'])
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_types_predicate_true. Retrieved 7/9 statements.


import builtins as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'Source'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_checkedpvector_constructor_from_iterable. Retrieved 4/9 statements.
# Failed to parse test_checkedpvector_constructor_empty.
# Partially parsed test_checkedpvector_constructor_from_existing_pvector. Retrieved 3/9 statements.
# Partially parsed test_checkedpvector_constructor_type_validation_fails. Retrieved 2/7 statements.
# Partially parsed test_checkedpvector_constructor_invariant_validation_fails. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = [var_0]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_success. Retrieved 6/11 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 5/10 statements.


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
    var_0 = lambda k, v: (v > k, 'Value must be greater than key')
    var_1 = 1
    var_2 = 5
    var_3 = 2
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = None
    var_1 = lambda k, v: (v > k, 'Value must be greater than key')
    var_2 = 1
    var_3 = 0
    var_4 = {var_2: var_3}
    var_5 = 'Value must be greater than key'



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_checked_type_instantiation_error.
# Failed to parse test_checked_type_abstract_methods_raise_error.




# Parsed testcases at query #32
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'ok'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'fine'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'ok'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error_found'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['error_found'])
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'ok'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'err2'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)
    var_14 = bool(var_13 == ['err1', 'err2'])
    assert var_14 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'a_is_one'
    var_4 = lambda x: (x.get(var_0) == var_1, var_3)
    var_5 = 'b'
    var_6 = None
    var_7 = 'b_missing'
    var_8 = lambda x: (x.get(var_5) is not var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_2, var_9)
    var_11 = bool(var_10 == ['b_missing'])
    assert var_11 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 4/9 statements.


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
    var_3 = 5

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_store_types_predicate_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'old_name'
    var_1 = 'val1'
    var_2 = {var_0: var_1}
    var_3 = 'val2'
    var_4 = 'new_name'
    var_5 = 'old_name'
    var_6 = 'new_name'
    var_7 = bool('new_name' in var_2)
    assert var_7 is True
    var_8 = var_2['new_name']
    var_9 = bool(var_2['new_name'] == ['type1'])
    assert var_9 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serialize_basic_types. Retrieved 4/9 statements.
# Partially parsed test_serialize_with_custom_format. Retrieved 4/9 statements.
# Failed to parse test_serialize_empty_set.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'repr'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_serialize_basic. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_strings. Retrieved 3/7 statements.
# Partially parsed test_serialize_different_format. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 'b'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'json'
    var_4 = 1
    var_5 = 2



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_invariant_errors_multiple_invalid. Retrieved 8/11 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

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
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'is negative'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['is negative'])
    assert var_11 is True

def test_case_0():
    var_0 = -5
    var_1 = 0
    var_2 = 'must be positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 10
    var_5 = 'must be less than 10'
    var_6 = lambda x: (x < var_4, var_5)
    var_7 = 'must be int'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'missing key a'
    var_4 = lambda x: (var_0 in x, var_3)
    var_5 = 0
    var_6 = 'a must be positive'
    var_7 = lambda x: (x[var_0] > var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_2, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = -1
    var_2 = {var_0: var_1}
    var_3 = 'missing key a'
    var_4 = lambda x: (var_0 in x, var_3)
    var_5 = 0
    var_6 = 'a must be positive'
    var_7 = lambda x: (x[var_0] > var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_2, var_8)
    var_10 = bool(var_9 == ['a must be positive'])
    assert var_10 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_serialize_with_default_format. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_custom_format. Retrieved 4/9 statements.
# Failed to parse test_serialize_empty_set.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'str'
    var_4 = 'a'
    var_5 = 'b'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_iterable.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'float'
    var_1 = 'bool'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('float', 'bool'))
    assert var_4 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_maybe_parse_user_type_evaluates_string_branch.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/12 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_type_mismatch. Retrieved 5/11 statements.


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
    var_2 = 1.9
    var_3 = {var_1: var_2}
    var_4 = 'Should have raised InvariantException'
    var_5 = AssertionError(var_4)
    var_6 = 'Invalid mapping'

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_store_types_single_source. Retrieved 5/7 statements.
# Partially parsed test_store_types_multiple_bases_inheritance. Retrieved 3/9 statements.
# Partially parsed test_store_types_preserves_iterable_type. Retrieved 5/8 statements.
# Partially parsed test_store_types_no_matching_keys. Retrieved 5/9 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0['dest']

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = var_0['dest']

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'int'
    var_2 = []
    var_3 = 'dest'
    var_4 = 'source'
    var_5 = module_0._store_types(var_0, var_2, var_3, var_4)
    var_6 = var_0['dest']
    var_7 = bool(var_0['dest'] == ['int'])
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0['dest']

def test_case_0():
    var_0 = 'existing'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'dest'
    var_4 = 'source'
    var_5 = 'dest'
    var_6 = bool('dest' not in var_2)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_checkedpset_constructor_with_valid_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/6 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (True, '')
    var_1 = 'not an int'
    var_2 = [var_1]
    var_3 = bool(True)
    assert var_3 is True
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda n: (True, '')
    var_1 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_store_types_predicate_true. Retrieved 6/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'value_in_base'
    var_2 = 'dest'
    var_3 = 'source_key'
    var_4 = 'value_in_base'
    var_5 = [var_4]
    var_6 = var_0[var_2]
    var_7 = bool(var_0[var_2] == ['parsed_type'])
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_checked_type_instantiation_error.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/11 statements.
# Partially parsed test_check_types_invalid_input_raises_error. Retrieved 4/17 statements.
# Partially parsed test_check_types_empty_expected_types_passes. Retrieved 5/11 statements.
# Partially parsed test_check_types_empty_iterator_passes. Retrieved 1/8 statements.


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
    var_4 = "Type MyClass can only be used with ('int'), not str"

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = []



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_checkedpmap_new_with_initial_data. Retrieved 7/8 statements.
# Partially parsed test_checkedpmap_new_with_predefined_size. Retrieved 4/6 statements.
# Partially parsed test_checkedpmap_new_empty. Retrieved 3/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'initial'
    var_7 = {var_6: var_4}
    var_8 = module_0.CheckedPMap(*var_5, **var_7)
    var_9 = dict(var_8)
    var_10 = bool(var_9 == var_4)
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = 10

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'initial'
    var_3 = {var_2: var_0}
    var_4 = module_0.CheckedPMap(*var_1, **var_3)
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #10
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

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
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'is negative'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = (var_1, var_2)
    var_10 = lambda x: var_9
    var_11 = 'is too small'
    var_12 = (var_5, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_4, var_8, var_10, var_13]
    var_15 = module_0._invariant_errors(var_0, var_14)
    var_16 = bool(var_15 == ['is negative', 'is too small'])
    assert var_16 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = False
    var_2 = 'error 1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error 2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['error 1', 'error 2'])
    assert var_11 is True

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
    var_10 = bool(var_9 == [])
    assert var_10 is True

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
    var_10 = bool(var_9 == ['value must be positive'])
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_restore_pickle_calls_create_with_correct_args. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 8/17 statements.
# Partially parsed test_store_invariants_inheritance. Retrieved 14/27 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 5/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'inv'
    var_2 = 'wrapped_inv'
    var_3 = bool(var_2 in var_0)
    assert var_3 is True
    var_4 = var_0[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_0[var_2][var_6]
    var_8 = var_7()
    var_9 = bool(var_8 == (True, (1, 2)))
    assert var_9 is True

def test_case_0():
    var_0 = {}
    var_1 = 'gp_inv'
    var_2 = 'dest'
    var_3 = True
    var_4 = False
    var_5 = lambda : var_4
    var_6 = 'p'
    var_7 = (var_6,)
    var_8 = 'dest'
    var_9 = 'gp_inv'
    var_10 = var_0[var_8]
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = var_0[var_8][var_4]
    var_13 = var_12()
    var_14 = bool(var_13 == (True,))
    assert var_14 is True

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'not_callable'
    var_3 = 'TypeError not raised'
    var_4 = AssertionError(var_3)

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



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_checked_type_create_returns_instance_if_already_correct_type.
# Partially parsed test_checked_type_create_wraps_list_using_cls_constructor. Retrieved 4/10 statements.
# Partially parsed test_checked_type_create_uses_checked_type_factory_for_elements. Retrieved 3/11 statements.
# Partially parsed test_checked_type_create_skips_factory_if_element_already_matches_type. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.CheckedType(*var_0, **var_1)
    var_3 = 'new_item'
    var_4 = [var_2, var_3]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_basic_types. Retrieved 4/9 statements.
# Partially parsed test_serialize_with_custom_format. Retrieved 4/8 statements.
# Partially parsed test_serialize_reproducibility. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'str'

def test_case_0():
    var_0 = 1.5
    var_1 = 2.5
    var_2 = [var_0, var_1]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_validation. Retrieved 4/12 statements.
# Partially parsed test_checkedpmap_constructor_type_validation. Retrieved 3/8 statements.


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
    var_0 = lambda k, v: (k == v, 'Key must equal value')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_checked_pvector_constructor_from_iterable. Retrieved 4/9 statements.
# Partially parsed test_checked_pvector_constructor_from_existing_pvector. Retrieved 3/9 statements.
# Failed to parse test_checked_pvector_constructor_empty.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]



# Parsed testcases at query #18
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

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
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'is_negative'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['is_negative'])
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = -5
    var_1 = 0
    var_2 = 'must_be_positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 2
    var_5 = 'must_be_even'
    var_6 = lambda x: (x % var_4 == var_1, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0._invariant_errors(var_0, var_7)
    var_9 = bool(var_8 == ['must_be_positive', 'must_be_even'])
    assert var_9 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'missing_key'
    var_4 = lambda x: (var_0 in x, var_3)
    var_5 = 'wrong_value'
    var_6 = lambda x: (x[var_0] == var_1, var_5)
    var_7 = [var_4, var_6]
    var_8 = module_0._invariant_errors(var_2, var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = 'missing_key'
    var_4 = lambda x: (var_0 in x, var_3)
    var_5 = 1
    var_6 = 'wrong_value'
    var_7 = lambda x: (x[var_0] == var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_2, var_8)
    var_10 = bool(var_9 == ['wrong_value'])
    assert var_10 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.


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
    var_3 = 5

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.
# Partially parsed test_maybe_parse_string_element_in_iterable. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_preserved_iterable_returns_list.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'float'
    var_1 = 'bool'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('float', 'bool'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_explicit_size. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invariant_failure. Retrieved 8/14 statements.
# Partially parsed test_checked_pmap_constructor_type_error. Retrieved 5/11 statements.


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
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.9
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Should have raised InvariantException'
    var_7 = AssertionError(var_6)

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_checked_type_instantiation_error.
# Partially parsed test_checked_type_abstract_methods_raise_error. Retrieved 1/14 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_wrap_invariant_returns_boolean_directly.
# Failed to parse test_wrap_invariant_returns_false_verdict_on_failure.
# Failed to parse test_wrap_invariant_returns_true_verdict_on_all_success.
# Partially parsed test_wrap_invariant_passes_arguments. Retrieved 2/6 statements.
# Failed to parse test_wrap_invariant_handles_empty_list.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #24
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
    var_8 = ()
    var_9 = (var_0, var_8)
    var_10 = module_0._merge_invariant_results(var_7)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'x'
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = (var_0, var_3)
    var_5 = 'z'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]
    var_8 = (var_1, var_3, var_5)
    var_9 = (var_0, var_8)
    var_10 = module_0._merge_invariant_results(var_7)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'skip'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'keep'
    var_5 = (var_3, var_4)
    var_6 = 'ignore'
    var_7 = (var_0, var_6)
    var_8 = 'save'
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_5, var_7, var_9]
    var_11 = (var_4, var_8)
    var_12 = (var_3, var_11)
    var_13 = module_0._merge_invariant_results(var_10)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = ()
    var_3 = (var_1, var_2)
    var_4 = module_0._merge_invariant_results(var_0)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'only_one'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = ()
    var_5 = (var_0, var_4)
    var_6 = module_0._merge_invariant_results(var_3)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'only_one'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = (var_1,)
    var_5 = (var_0, var_4)
    var_6 = module_0._merge_invariant_results(var_3)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True



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
# Partially parsed test_checkedpmap_constructor_with_fixed_size. Retrieved 3/7 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = 'Invalid mapping'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_store_types_predicate_true. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'old_name'
    var_1 = 'existing'
    var_2 = {var_0: var_1}
    var_3 = 'value'
    var_4 = 'new_name'
    var_5 = 'source_key'
    var_6 = 'new_name'
    var_7 = bool('new_name' in var_2)
    assert var_7 is True
    var_8 = var_2['new_name']
    var_9 = bool(var_2['new_name'] == ['type1'])
    assert var_9 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 8/14 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'IntToFloatMutatingMap'
    var_6 = globals()
    var_7 = var_5 in var_6

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (k == v, 'Key must equal value')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_2}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = 'Invalid mapping'
    var_5 = 'InvariantException not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'CheckedKeyTypeError not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_store_invariants_predicate_true. Retrieved 17/33 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = globals()
    var_1 = '_all_dicts'
    var_2 = globals()
    var_3 = 'wrap_invariant'
    var_4 = 'src'
    var_5 = 'dst'
    var_6 = lambda x: x
    var_7 = {var_4: var_6}
    var_8 = []
    var_9 = module_0.store_invariants(var_7, var_8, var_5, var_4)
    var_10 = bool(var_5 in var_7)
    assert var_10 is True
    var_11 = 0
    var_12 = var_7[var_5][var_11]
    var_13 = callable(var_12)
    var_14 = bool(var_13)
    assert var_14 is True
    var_15 = '_all_dicts'
    var_16 = globals()[var_15]
    var_17 = 'wrap_invariant'
    var_18 = globals()[var_17]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_store_types_predicate_evaluates_to_true. Retrieved 4/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'value'
    var_2 = 'dest'
    var_3 = 'source_key'
    var_4 = bool('source_key' in var_0.__dict__ or True)
    assert var_4 is True
    var_5 = var_0[var_2]
    var_6 = bool(var_0[var_2] == ['value', 'exists'])
    assert var_6 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_check_types_success. Retrieved 5/9 statements.
# Partially parsed test_check_types_failure_raises_exception. Retrieved 4/13 statements.
# Partially parsed test_check_types_empty_expected_types_does_nothing. Retrieved 6/9 statements.
# Partially parsed test_check_types_empty_iterator_does_nothing. Retrieved 1/5 statements.
# Partially parsed test_check_types_with_string_type_references. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'not_an_int'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 'a'
    var_3 = 2
    var_4 = [var_1, var_3]
    var_5 = [var_1, var_2, var_4]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1.0
    var_1 = 2.5
    var_2 = [var_0, var_1]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_checkedpvector_constructor_from_iterable. Retrieved 4/9 statements.
# Partially parsed test_checkedpvector_constructor_from_pythonpvector. Retrieved 4/9 statements.
# Failed to parse test_checkedpvector_constructor_empty.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = module_0.python_pvector(var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 5

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/11 statements.
# Partially parsed test_check_types_invalid_input_raises_exception. Retrieved 4/17 statements.
# Partially parsed test_check_types_empty_expected_types_passes. Retrieved 5/11 statements.
# Partially parsed test_check_types_empty_iterator_passes. Retrieved 1/8 statements.


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
    var_4 = "Type MyClass can only be used with ('int'), not str"

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = None
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = []



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_types_predicate_true. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_checked_type_instantiation_error.




# Parsed testcases at query #38
#--------------------------

# Partially parsed test_restore_pickle_returns_new_instance_with_correct_data. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Partially parsed test_maybe_parse_user_type_nested_list. Retrieved 1/4 statements.
# Failed to parse test_maybe_parse_user_type_tuple_of_types.


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
    var_2 = 'Type specifications must be types or strings'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'not an instance of MockClass'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_checkedpvector_constructor_with_iterable. Retrieved 4/9 statements.
# Failed to parse test_checkedpvector_constructor_empty.
# Partially parsed test_checkedpvector_constructor_with_pythonpvector. Retrieved 3/10 statements.
# Partially parsed test_checkedpvector_constructor_type_validation. Retrieved 4/8 statements.


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



