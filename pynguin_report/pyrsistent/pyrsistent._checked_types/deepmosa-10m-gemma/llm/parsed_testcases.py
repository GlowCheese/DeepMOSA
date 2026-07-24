####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.
# Failed to parse test_maybe_parse_user_type_single_element_list.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(True)
    assert var_2 is True
    var_3 = 'Should have raised TypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #2
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
    var_6 = 'error_msg'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['error_msg'])
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
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
    var_0 = 5
    var_1 = 0
    var_2 = 'must be positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 10
    var_5 = 'must be less than 10'
    var_6 = lambda x: (x < var_4, var_5)
    var_7 = 2
    var_8 = 'must be even'
    var_9 = lambda x: (x % var_7 == var_1, var_8)
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0._invariant_errors(var_0, var_10)
    var_12 = bool(var_11 == ['must be even'])
    assert var_12 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.
# Failed to parse test_maybe_parse_user_type_single_tuple.
# Failed to parse test_maybe_parse_user_type_preserved_type.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.
# Failed to parse test_maybe_parse_tuple_of_types.


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



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserves_iterable_type.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/12 statements.
# Partially parsed test_check_types_invalid_input_raises_error. Retrieved 9/19 statements.
# Partially parsed test_check_types_empty_expected_types_does_nothing. Retrieved 6/12 statements.
# Partially parsed test_check_types_empty_iterator_does_nothing. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = True
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = [var_2, var_4]
    var_6 = [var_2, var_1, var_5]
    var_7 = "Type DummySource can only be used with ('int', 'str'), not list"
    var_8 = 'CheckedValueTypeError was not raised'
    var_9 = AssertionError(var_8)

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = []

def test_case_0():
    var_0 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_merging_logic. Retrieved 6/13 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 6/11 statements.
# Partially parsed test_store_invariants_inheritance_chain. Retrieved 11/21 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'target_inv'
    var_2 = 'invariant_source'
    var_3 = 'target_inv'
    var_4 = bool('target_inv' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = var_8()
    var_10 = bool(var_9 == (True, (1,)))
    assert var_10 is True

def test_case_0():
    var_0 = {}
    var_1 = 'target_inv'
    var_2 = 'inv_src'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = var_4()
    var_6 = bool(var_5 == (False, ('b',)))
    assert var_6 is True

def test_case_0():
    var_0 = 'not a callable'
    var_1 = {}
    var_2 = 'target_inv'
    var_3 = 'invariant_source'
    var_4 = 'TypeError not raised'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = {}
    var_1 = 'target_inv'
    var_2 = 'inv_src'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 1
    var_6 = var_0[var_1][var_5]
    var_7 = var_6()
    var_8 = bool(var_7 == (False, 'child_data'))
    assert var_8 is True
    var_9 = 0
    var_10 = var_0[var_1][var_9]
    var_11 = var_10()
    assert var_11 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 8/16 statements.


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
    var_3 = 16

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = lambda k, v: (k == v, 'Mismatch')
    var_5 = 1
    var_6 = 2
    var_7 = {var_5: var_6}



# Parsed testcases at query #11
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
    var_9 = bool(var_8 == (True, ()))
    assert var_9 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_0, var_3)
    var_5 = 'c'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0._merge_invariant_results(var_7)
    var_9 = bool(var_8 == (False, ('a', 'b', 'c')))
    assert var_9 is True

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
    var_12 = bool(var_11 == (False, ('b', 'd')))
    assert var_12 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_invariant_results(var_0)
    var_2 = bool(var_1 == (True, ()))
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.
# Partially parsed test_checkedpmap_constructor_with_invalid_invariant_raises_error. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_type_mismatch_raises_error. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_explicit_size. Retrieved 3/7 statements.


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
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_invariant_failure. Retrieved 6/12 statements.


def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.5
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = 10

def test_case_0():
    var_0 = lambda k, v: (v == k, 'Value must equal key')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = 'Should have raised InvariantException'
    var_5 = AssertionError(var_4)
    var_6 = 'Value must equal key'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_checked_pvector_constructor_from_iterable. Retrieved 4/9 statements.
# Partially parsed test_checked_pvector_constructor_from_existing_pvector. Retrieved 4/9 statements.
# Failed to parse test_checked_pvector_constructor_empty.
# Partially parsed test_checked_pvector_constructor_type_validation_fails. Retrieved 2/7 statements.


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

def test_case_0():
    var_0 = 'not an int'
    var_1 = [var_0]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 11/21 statements.
# Partially parsed test_store_invariants_merging_logic. Retrieved 7/13 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 6/11 statements.
# Partially parsed test_store_invariants_inheritance_chain. Retrieved 11/23 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = 'dest'
    var_4 = bool('dest' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = var_8()
    var_10 = bool(var_9 == (True, (1,)))
    assert var_10 is True
    var_11 = 1
    var_12 = var_0[var_1][var_11]
    var_13 = var_12()
    var_14 = bool(var_13 == (True, (1,)))
    assert var_14 is True

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = 0
    var_5 = var_0[var_2][var_4]
    var_6 = var_5()
    var_7 = bool(var_6 == (False, (('b',),)))
    assert var_7 is True

def test_case_0():
    var_0 = 'not a callable'
    var_1 = {}
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = 'TypeError not raised'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = var_6()
    var_8 = bool(var_7 == (False,))
    assert var_8 is True
    var_9 = 1
    var_10 = var_0[var_1][var_9]
    var_11 = var_10()
    var_12 = bool(var_11 == (True,))
    assert var_12 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_maybe_parse_user_type_evaluates_is_type_and_not_is_iterable.




# Parsed testcases at query #17
#--------------------------

# Failed to parse test_checked_type_instantiation_fails_due_to_abstractmethod.
# Partially parsed test_checked_type_is_instance_of_object. Retrieved 1/2 statements.


def test_case_0():
    pass

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.CheckedType(*var_0, **var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_store_types_single_source. Retrieved 5/7 statements.
# Partially parsed test_store_types_multiple_bases_inheritance. Retrieved 3/9 statements.
# Partially parsed test_store_types_with_iterable_input. Retrieved 13/15 statements.
# Partially parsed test_store_types_overwriting_existing_key. Retrieved 7/9 statements.
# Partially parsed test_store_types_no_source_found. Retrieved 3/7 statements.


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
    var_1 = 'MyType'
    var_2 = []
    var_3 = 'dest'
    var_4 = 'source'
    var_5 = module_0._store_types(var_0, var_2, var_3, var_4)
    var_6 = var_0['dest']
    var_7 = bool(var_0['dest'] == ['MyType'])
    assert var_7 is True

import builtins as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'Iterable'
    var_3 = ()
    var_4 = '__iter__'
    var_5 = []
    var_6 = iter(var_5)
    var_7 = lambda self: var_6
    var_8 = {var_4: var_7}
    var_9 = [var_2, var_3, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'dest'
    var_13 = 'source'
    var_14 = module_1._store_types(var_0, var_1, var_12, var_13)
    var_15 = var_0['dest']

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'dest'
    var_1 = 'old'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = 'source'
    var_6 = module_0._store_types(var_3, var_4, var_0, var_5)
    var_7 = var_3['dest']

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = 'dest'
    var_4 = bool('dest' not in var_0)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_argument. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 6/12 statements.


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
    var_4 = 2.5
    var_5 = {var_1: var_4}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 6/12 statements.


def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.5
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = 10

def test_case_0():
    var_0 = lambda k, v: (v == k, 'Value must equal key')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = 'Value must equal key'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_restore_pickle_calls_create_with_correct_arguments. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_type_error_on_key. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.5
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = {}
    var_2 = 10

def test_case_0():
    var_0 = lambda k, v: (v == k, 'Value must equal key')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Value must equal key'

def test_case_0():
    var_0 = lambda k, v: (True, None)
    var_1 = 'not_an_int'
    var_2 = 1.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_invariant_errors_predicate_evaluates_to_true. Retrieved 6/19 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = [var_2]
    var_4 = module_0._invariant_errors(var_0, var_3)
    var_5 = bool(var_4 == [None])
    assert var_5 is True
    var_6 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_store_invariants_predicate_true. Retrieved 10/15 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_dest'
    var_2 = True
    var_3 = lambda : var_2
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = module_0.store_invariants(var_4, var_5, var_1, var_0)
    var_7 = bool(var_1 in var_4)
    assert var_7 is True
    var_8 = 0
    var_9 = var_4[var_1][var_8]
    var_10 = callable(var_9)
    var_11 = bool(var_10)
    assert var_11 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_empty_iterable. Retrieved 2/6 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type_raises_error. Retrieved 4/8 statements.
# Partially parsed test_checkedpset_constructor_with_invariant_violation_raises_error. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_pmap_direct_initialization. Retrieved 7/11 statements.


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
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'string'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = 1
    var_10 = 2



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict_and_invariants. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_invariant_failure. Retrieved 5/10 statements.
# Partially parsed test_checkedpmap_constructor_type_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}
    var_5 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = {var_1: var_1, var_2: var_3}
    var_5 = 'Invalid mapping'

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.


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



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_serialize_raises_not_implemented_error.
# Partially parsed test_serialize_with_argument_passing_to_subclass. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'json'
    var_1 = 'xml'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation_during_evolution. Retrieved 4/9 statements.


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
    var_4 = 'Invalid mapping'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serialize_basic_dict_conversion. Retrieved 5/11 statements.
# Partially parsed test_serialize_empty_map. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_custom_serializer_logic. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = '1'
    var_3 = '2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'string'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list.
# Failed to parse test_maybe_parse_user_type_preserved_type.


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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checkedpmap_constructor_invariant_violation. Retrieved 4/9 statements.


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_empty. Retrieved 2/6 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'string'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size_argument. Retrieved 4/8 statements.
# Partially parsed test_checkedpmap_constructor_with_invariant_violation. Retrieved 6/12 statements.
# Partially parsed test_checkedpmap_constructor_with_type_violation. Retrieved 3/8 statements.


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
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = 2.5
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 4/10 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_types_valid_input. Retrieved 4/11 statements.
# Partially parsed test_check_types_invalid_input_raises_error. Retrieved 4/17 statements.
# Partially parsed test_check_types_empty_expected_types_does_nothing. Retrieved 5/11 statements.
# Partially parsed test_check_types_empty_iterator_does_nothing. Retrieved 1/8 statements.


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
    var_4 = "Type SourceClass can only be used with ('int'), not str"

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = []



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checked_type_is_abstract_and_raises_not_implemented_on_create. Retrieved 2/10 statements.
# Partially parsed test_checked_type_raises_not_implemented_on_serialize. Retrieved 1/9 statements.


def test_case_0():
    pass

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = {}
    var_1 = bool(True)
    assert var_1 is True
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 7/15 statements.
# Partially parsed test_store_invariants_inheritance_and_merging. Retrieved 9/26 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 6/11 statements.
# Partially parsed test_store_invariants_multiple_invariants_merging. Retrieved 6/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'invariant_source'
    var_3 = 'dest'
    var_4 = bool('dest' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'invariant_source'
    var_3 = {}
    var_4 = var_3[var_1]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_1][var_6]
    var_8 = var_7()
    var_9 = bool(var_8 == (True, (1,)))
    assert var_9 is True

def test_case_0():
    var_0 = 'not_callable'
    var_1 = {}
    var_2 = 'dest'
    var_3 = 'invariant_source'
    var_4 = 'TypeError not raised'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'invariant_source'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = var_4()
    var_6 = bool(var_5 == (True, (1,)))
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_checked_type_create_returns_instance_if_already_correct_type.
# Partially parsed test_checked_type_create_wraps_data_in_new_instance_for_simple_type. Retrieved 4/9 statements.
# Partially parsed test_checked_type_create_uses_checked_type_recursion. Retrieved 7/14 statements.
# Partially parsed test_checked_type_create_skips_checked_type_creation_if_data_already_matches_type_in_list. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 0

def test_case_0():
    var_0 = 'raw_string'



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_list_of_types.
# Failed to parse test_maybe_parse_user_type_nested_list_of_types.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_invariants_success. Retrieved 11/19 statements.
# Partially parsed test_store_invariants_type_error. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_merging_logic. Retrieved 6/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source'
    var_3 = 'dest'
    var_4 = bool('dest' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = var_8()
    var_10 = bool(var_9 == (False, (2,)))
    assert var_10 is True
    var_11 = 1
    var_12 = var_0[var_1][var_11]
    var_13 = var_12()
    var_14 = bool(var_13 == (True, (1,)))
    assert var_14 is True

def test_case_0():
    var_0 = 'I am a string'
    var_1 = {}
    var_2 = 'dest'
    var_3 = 'not_callable'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'source_func'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = var_4()
    var_6 = bool(var_5 == (False, ('b',)))
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_maybe_parse_user_type_evaluates_true_at_line_18.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_and_initial_dict. Retrieved 4/9 statements.


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_checkedpset_constructor_with_list. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checkedpset_constructor_with_invariant_failure. Retrieved 4/8 statements.
# Partially parsed test_checkedpset_constructor_with_empty_input. Retrieved 2/6 statements.


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
    var_2 = 'not_an_int'
    var_3 = [var_1, var_2]
    var_4 = bool(True)
    assert var_4 is True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -5
    var_3 = [var_1, var_2]
    var_4 = 'Negative'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_initial_dict. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_invariant_violation. Retrieved 8/14 statements.
# Partially parsed test_checked_pmap_constructor_type_error. Retrieved 5/11 statements.


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
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = 2.5
    var_5 = {var_1: var_4}
    var_6 = 'Should have raised InvariantException'
    var_7 = AssertionError(var_6)

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 1.0
    var_2 = {var_0: var_1}
    var_3 = 'Should have raised CheckedKeyTypeError'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #20
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
    var_9 = bool(var_8 == (True, ()))
    assert var_9 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_0, var_3)
    var_5 = 'c'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0._merge_invariant_results(var_7)
    var_9 = bool(var_8 == (False, ('a', 'b', 'c')))
    assert var_9 is True

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
    var_12 = bool(var_11 == (False, ('b', 'd')))
    assert var_12 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_invariant_results(var_0)
    var_2 = bool(var_1 == (True, ()))
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'only_one'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0._merge_invariant_results(var_3)
    var_5 = bool(var_4 == (True, ()))
    assert var_5 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'only_one'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0._merge_invariant_results(var_3)
    var_5 = bool(var_4 == (False, ('only_one',)))
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_iterable. Retrieved 4/9 statements.
# Failed to parse test_checked_pvector_constructor_with_empty.
# Partially parsed test_checked_pvector_constructor_with_existing_pvector. Retrieved 4/9 statements.
# Partially parsed test_checked_pvector_constructor_type_validation. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1.1
    var_1 = 2.2
    var_2 = [var_0, var_1]
    var_3 = module_0.python_pvector(var_2)

def test_case_0():
    var_0 = 'not'
    var_1 = 'an'
    var_2 = 'int'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_invariant_errors_with_complex_data. Retrieved 7/10 statements.
# Partially parsed test_invariant_errors_with_complex_data_failure. Retrieved 7/10 statements.


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
    var_1 = False
    var_2 = 'error_1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'error_2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['error_1'])
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'error_A'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'skip'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error_B'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = (var_5, var_12)
    var_14 = lambda x: var_13
    var_15 = [var_4, var_8, var_11, var_14]
    var_16 = module_0._invariant_errors(var_0, var_15)
    var_17 = bool(var_16 == ['error_A', 'error_B'])
    assert var_17 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'must be positive'
    var_5 = lambda x: (x.get(var_0) > var_3, var_4)
    var_6 = 'not a dict'

def test_case_0():
    var_0 = 'a'
    var_1 = -1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'must be positive'
    var_5 = lambda x: (x.get(var_0) > var_3, var_4)
    var_6 = 'not a dict'



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_checked_type_create_returns_source_data_when_is_instance.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_checkedpmap_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_checkedpmap_constructor_with_size. Retrieved 3/9 statements.
# Partially parsed test_checkedpmap_constructor_with_invariants_success. Retrieved 6/11 statements.
# Partially parsed test_checkedpmap_constructor_with_invariants_failure. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (v > k, 'Value must be greater than key')
    var_1 = 1
    var_2 = 2
    var_3 = 5
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (v > k, 'Value must be greater than key')
    var_1 = 1
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'Value must be greater than key'



